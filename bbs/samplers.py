from bbs.utils import is_in_bounds, match_input, which_in_bounds
import jax
import jax.numpy as jnp
import jax.random as jr

import blackjax
from blackjax.types import ArrayTree

import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params

import optax
from optax._src.base import OptState

from jaxtyping import Array
from typing import Any, Union, Callable, NamedTuple
from dataclasses import dataclass, replace
import tqdm
import time
import datetime

def get_gradient_estimator(logdensity_fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
    return lambda x, _: jax.grad(logdensity_fn)(x)  # First argument is for minibatching, which we dont use

def param_clip(x, bounds):
    return jnp.clip(x, min=bounds[:,0], max=bounds[:,1])

def bounds_resizer(bounds:Array, resize: Array) -> Array:
    mu = bounds.mean(axis=1)
    pivot = jnp.repeat(mu, 2).reshape(-1,2)
    centered_bounds = bounds - pivot
    centered_rescaled_bounds = jnp.multiply(centered_bounds, resize)
    out =  centered_rescaled_bounds + pivot
    assert(bounds.shape == out.shape)
    return out

def time_inference_loop(func):
    def _wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        if 'show_progress_bar' in kwargs and kwargs['show_progress_bar']: 
            seconds = time.perf_counter()-t0

            iters_second = '?'
            iters_second = args[2]/seconds
            print(f'Time: {datetime.timedelta(seconds=seconds)}, {iters_second} it/s')
        return out
    return _wrapper

@dataclass
class ConfigSampler:
    samples_budget: int
    num_chains: int = 1 # NOTE(Javier): Some methods dont support this
    use_param_clip: bool = False
    
    
@time_inference_loop
def blackjax_multiple_loop(
    key, kernel, num_samples, bounds, num_chains=1, state=None, use_param_clip=False, show_progress_bar=False, jax_compile=True
):
    if state is None:
        key, init_key = jax.random.split(key)
        initial_position = jr.uniform(key=init_key, minval=bounds[:,0], maxval=bounds[:,1], shape=(num_chains, bounds.shape[0],))   
        state = jax.vmap(kernel.init)(initial_position)

    @jax.jit
    def one_step(states, k):
        keys = jax.random.split(k, num_chains)
        states, infos = jax.vmap(kernel.step)(keys, states)
        if use_param_clip:
            states = states._replace(position=param_clip(state.position, bounds))
        return states, (states, infos)

    keys = jax.random.split(key, num_samples)

    final_states, (all_states, all_infos) = jax.lax.scan(one_step, state, keys)  
    
    # Swap the axes corresponding to num_samples and num_chain
    all_chains = jnp.swapaxes(all_states.position, 0, 1)  # shape [num_samples, num_chains, dim] -> [num_chains, num_samples, num_dim]
    all_acceptance_rate = jnp.swapaxes(all_infos.acceptance_rate, 0, 1) # shape [num_samples, num_chains] -> [num_chains, num_samples]

    assert(all_chains.shape == (num_chains, num_samples, bounds.shape[0]))
    assert(all_acceptance_rate.shape == (num_chains, num_samples))

    return final_states.position, all_chains, all_infos.acceptance_rate

@time_inference_loop
def blackjax_single_loop(key, kernel, num_samples, bounds, num_chains=1, state=None, use_param_clip=False, show_progress_bar=False, jax_compile=True):
    # TODO(Javier): Technically is the same as the multiple, but for reproducibility, this stays here as the num_chains==1 case until we rerun all experiments from scratch
    assert(num_chains == 1)

    if state is None:
        key, init_key = jax.random.split(key)
        initial_position = jr.uniform(key=init_key, minval=bounds[:,0], maxval=bounds[:,1], shape=(bounds.shape[0],))   
        state = kernel.init(initial_position)

    step_fn = jax.jit(kernel.step) if jax_compile else kernel.step
    if show_progress_bar and jax_compile: step_fn(key, state) # Force compile before loop

    tqdm_interval = lambda x: tqdm.trange(x, mininterval=1)
    iter_range = tqdm_interval if show_progress_bar else jnp.arange

    all_samples = [None] * num_samples
    acceptance_rates = [None] * num_samples
    for i in iter_range(num_samples):
        key, sampler_key = jax.random.split(key)
        state, info = step_fn(sampler_key, state)
        if use_param_clip:
            state = state._replace(position=param_clip(state.position, bounds))
        all_samples[i] = state.position.reshape(-1)
        acceptance_rates[i] = info.acceptance_rate
    return state.position, jnp.array(all_samples), jnp.array(acceptance_rates)

def blackjax_window_adaptation_warmup(key, hmc_method, bounds, num_warmup, target_acceptance_rate, logdensity_fn, **hmc_method_args):
    key, init_key = jax.random.split(key)
    initial_position = jr.uniform(key=init_key, minval=bounds[:,0], maxval=bounds[:,1], shape=(bounds.shape[0],))   

    adapt = blackjax.window_adaptation(
        hmc_method, logdensity_fn, target_acceptance_rate=target_acceptance_rate, **hmc_method_args
    )
    key, warmup_key = jax.random.split(key)
    (last_state, parameters), _ = adapt.run(warmup_key, initial_position, num_warmup)
    return last_state, parameters

def blackjax_simple_loop(*args, **kwargs):
    if not 'num_chains' in kwargs or kwargs['num_chains'] == 1:
        return blackjax_single_loop(*args, **kwargs)
    else:
        return blackjax_multiple_loop(*args, **kwargs)


@dataclass
class ConfigMH(ConfigSampler):
    proposal_distribution: Callable = None

class MHState(NamedTuple):
    position: Array
    loglikelihood: float

class MHInfo(NamedTuple):
    acceptance_rate: float
    is_accepted: bool

class KernelMH(NamedTuple):
    init: Callable
    step: Callable

def get_mh_kernel(loglikelihood_fn, proposal_distribution, bounds):
    def _init(position):
        return MHState(position=position, loglikelihood=loglikelihood_fn(position))

    def _step(key: jr.PRNGKey, state: MHState):
        key, proposal_key, accept_key = jr.split(key, 3)
        prop_position = proposal_distribution(proposal_key, state.position, bounds)

        prop_loglikelihood = loglikelihood_fn(prop_position)
        p_accept = jnp.exp(prop_loglikelihood - state.loglikelihood)

        accept = (is_in_bounds(prop_position, bounds)) & (p_accept > jr.uniform(accept_key))
        new_state = MHState(position=prop_position, loglikelihood=prop_loglikelihood)

        info = MHInfo(acceptance_rate=p_accept, is_accepted=accept)
        return (
        jax.lax.cond(
            accept,
            lambda _: new_state,
            lambda _: state,
            operand=None,
        ),
        info,
    )
        return MHState(position=ret_position, loglikelihood=ret_loglikelihood), 
    return KernelMH(init=_init, step=_step)

def metropolis_hastings(key: jr.PRNGKey, logdensity_fn: Callable[..., Array], bounds: Array, config: Union[dict, ConfigMH], show_progress_bar=False, jax_compile=True) -> tuple[Array, Array]:
    if isinstance(config, dict): config = ConfigMH(**config)
    if show_progress_bar: print(config)

    state = None

   # Build the kernel
    kernel = get_mh_kernel(logdensity_fn, config.proposal_distribution, bounds)
    return blackjax_simple_loop(key, kernel, config.samples_budget, bounds, num_chains=config.num_chains, 
                                state=state, use_param_clip=config.use_param_clip,
                                show_progress_bar=show_progress_bar, jax_compile=jax_compile)

@dataclass
class ConfigBlackjaxHMC(ConfigSampler):
    step_size: float = 1e-2
    num_integration_steps: int = 5
    inverse_mass_matrix: Array = jnp.array([1., 1.])
    adapt_warmup_steps: Union[float,int] = 0 
    adapt_target_acceptance_rate: float = 0.8


def blackjax_hmc(key: jr.PRNGKey, logdensity_fn: Callable[..., Array], bounds: Array, config: Union[dict, ConfigBlackjaxHMC], show_progress_bar=False, jax_compile=True) -> tuple[Array, Array]:
    if isinstance(config, dict): config = ConfigBlackjaxHMC(**config)
    method = blackjax.hmc
    if show_progress_bar: print(config)

    state = None
    num_warmup = 0
    if config.adapt_warmup_steps > 0:
        num_warmup = config.adapt_warmup_steps
        if isinstance(num_warmup, float):
            assert(num_warmup < 1.0)
            num_warmup = int(config.samples_budget * num_warmup)

        state, parameters = blackjax_window_adaptation_warmup(key, method, bounds, 
                                                              num_warmup=num_warmup, 
                                                              target_acceptance_rate=config.adapt_target_acceptance_rate, 
                                                              logdensity_fn=logdensity_fn,
                                                              num_integration_steps=config.num_integration_steps
                                                              )  
        config = replace(config, **parameters)
        if show_progress_bar: print('Updated', config)

    # Build the kernel
    kernel = method(logdensity_fn, config.step_size, config.inverse_mass_matrix, config.num_integration_steps)
    return blackjax_simple_loop(key, kernel, config.samples_budget-num_warmup, bounds, num_chains=config.num_chains, 
                                state=state, use_param_clip=config.use_param_clip,
                                show_progress_bar=show_progress_bar, jax_compile=jax_compile)

@dataclass
class ConfigBlackjaxMALA(ConfigSampler):
    step_size: float = 1e-2

def blackjax_mala(key: jr.PRNGKey, logdensity_fn: Callable[..., Array], bounds: Array, config: Union[dict, ConfigBlackjaxMALA], show_progress_bar=False, jax_compile=True) -> tuple[Array, Array]:
    if isinstance(config, dict): config = ConfigBlackjaxMALA(**config)
    method = blackjax.mala
    if show_progress_bar: print(config)

    state = None

    kernel = method(logdensity_fn, config.step_size)
    return blackjax_simple_loop(key, kernel, config.samples_budget, bounds, state=state, num_chains=config.num_chains, show_progress_bar=show_progress_bar, jax_compile=jax_compile)


@dataclass
class ConfigBlackjaxNUTS(ConfigSampler):
    step_size: float = 1e-2
    inverse_mass_matrix: Array = jnp.array([1., 1.])
    adapt_warmup_steps: int = 0 
    adapt_target_acceptance_rate: float = 0.8

def blackjax_nuts(key: jr.PRNGKey, logdensity_fn: Callable[..., Array], bounds: Array, config: Union[dict, ConfigBlackjaxNUTS], show_progress_bar=False, jax_compile=True) -> tuple[Array, Array]:
    if isinstance(config, dict): config = ConfigBlackjaxNUTS(**config)
    method = blackjax.nuts
    if show_progress_bar: print(config)

    # Warmup adapt if neccesary
    state = None
    num_warmup = 0
    if config.adapt_warmup_steps > 0:
        num_warmup = config.adapt_warmup_steps
        if isinstance(num_warmup, float):
            assert(num_warmup < 1.0)
            num_warmup = int(config.samples_budget * num_warmup)

        state, parameters = blackjax_window_adaptation_warmup(key, method, bounds, 
                                                              num_warmup=num_warmup, 
                                                              target_acceptance_rate=config.adapt_target_acceptance_rate, 
                                                              logdensity_fn=logdensity_fn)  
        config = replace(config, **parameters)
        if show_progress_bar: print('Updated', config)

    # Build the kernel
    kernel = method(logdensity_fn, config.step_size, config.inverse_mass_matrix)
    return blackjax_simple_loop(key, kernel, config.samples_budget-num_warmup, bounds, num_chains=config.num_chains,
                                state=state, use_param_clip=config.use_param_clip,
                                show_progress_bar=show_progress_bar, jax_compile=jax_compile)

@dataclass
class ConfigTemperedSMC(ConfigSampler):
    step_size: float = 1e-2
    num_integration_steps: int = 5
    inverse_mass_matrix: Array = jnp.array([1., 1.])
    num_mcmc_steps: int = 1

def blackjax_temperedsmc(key: jr.PRNGKey, logdensity_fn: Callable[..., Array], bounds: Array, config: Union[dict, ConfigTemperedSMC], show_progress_bar=False, jax_compile=True) -> tuple[Array, Array]:
    if isinstance(config, dict): config = ConfigTemperedSMC(**config)
    if show_progress_bar: print(config)

    def prior_log_prob(x):
        return 0.0
    
    prior_log_prob
    tempered = blackjax.adaptive_tempered_smc(
        prior_log_prob,
        logdensity_fn,
        blackjax.hmc.build_kernel(),
        blackjax.hmc.init,
        extend_params({
            'step_size': config.step_size, 
            'num_integration_steps': config.num_integration_steps,
            'inverse_mass_matrix': config.inverse_mass_matrix,
        }),
        resampling.systematic,
        0.75,
        num_mcmc_steps=config.num_mcmc_steps,
    )

    num_iters = int( config.samples_budget / (config.num_chains * config.num_mcmc_steps) )

    key, init_key = jax.random.split(key)
    initial_position = jr.uniform(key=init_key, minval=bounds[:,0], maxval=bounds[:,1], shape=(config.num_chains, bounds.shape[0],))   
    state = tempered.init(initial_position)

    key, sample_key = jax.random.split(key)
    final_particles, all_chains = smc_inference_loop(sample_key, tempered.step, state, num_iters, bounds=bounds, use_param_clip=config.use_param_clip,
                                                     show_progress_bar=show_progress_bar, jax_compile=jax_compile)
    return final_particles, all_chains[:,0,:], jnp.array([])

def smc_inference_loop(rng_key, smc_kernel, initial_state, num_samples, bounds=None, use_param_clip=False, show_progress_bar=False, jax_compile=True):
    assert(not use_param_clip or bounds is not None)

    def one_step(carry):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        state, _ = smc_kernel(subk, state)
        return i + 1, state, k

    # n_iter, final_state, _ = jax.lax.while_loop(
    #     cond, one_step, (0, initial_state, rng_key)
    # )

    all_chains = [] # shape = [step, num_chains, num_dim]
    tuple_state = (0, initial_state, rng_key)

    step_fn = jax.jit(one_step) if jax_compile else one_step
    if show_progress_bar and jax_compile: step_fn(tuple_state) # Force compile before loop

    tqdm_interval = lambda x: tqdm.trange(x, mininterval=1)
    iter_range = tqdm_interval if show_progress_bar else jnp.arange
    for i in iter_range(num_samples):
        tuple_state = step_fn(tuple_state)
        _, state, k = tuple_state  # state.particles.shape = [num_chains, num_dim]
        if use_param_clip:
            state = state._replace(particles=param_clip(state.particles, bounds))
    
        all_chains.append(state.particles) # shape = [step, num_chains, num_dim]

    all_chains = jnp.swapaxes(jnp.array(all_chains), 0, 1)  # shape = [num_chains, step, num_dim]
    return tuple_state[1].particles, all_chains


class ScheduleState(NamedTuple):
    step_size: float
    do_sample: bool

def build_schedule(
    num_training_steps,
    num_cycles=4,
    initial_step_size=1e-3,
    exploration_ratio=0.25,
):
    cycle_length = num_training_steps // num_cycles

    def schedule_fn(step_id):
        do_sample = False
        if ((step_id % cycle_length)/cycle_length) >= exploration_ratio:
            do_sample = True

        cos_out = jnp.cos(jnp.pi * (step_id % cycle_length) / cycle_length) + 1
        step_size = 0.5 * cos_out * initial_step_size

        return ScheduleState(step_size, do_sample)

    return schedule_fn


@dataclass
class ConfigBlackjaxCyclicalSGLD(ConfigSampler):
    num_cycles: int = 30
    step_size: float = 1e-3
    ratio_exploration: float = 0.25
    sgd_learning_rate: float = 1e-4
    sgd_grad_clipping: float = 1.0

def blackjax_cyclical_sgld(key: jr.PRNGKey, logdensity_fn: Callable[..., Array], bounds: Array, config: Union[dict, ConfigBlackjaxCyclicalSGLD], show_progress_bar=False, jax_compile=True) -> tuple[Array, Array]:
    if isinstance(config, dict): config = ConfigBlackjaxCyclicalSGLD(**config)
    grad_estimator_fn = get_gradient_estimator(logdensity_fn)

    num_training_steps = config.samples_budget
    schedule_fn = build_schedule(num_training_steps, config.num_cycles, config.step_size, config.ratio_exploration)
    schedule = [schedule_fn(i) for i in range(num_training_steps)]

    init, step = cyclical_sgld(grad_estimator_fn, config.sgd_learning_rate, config.sgd_grad_clipping)

    key, init_key = jax.random.split(key)
    initial_position = jr.uniform(key=init_key, minval=bounds[:,0], maxval=bounds[:,1], shape=(bounds.shape[0],))   
    state = init(initial_position)

    step_fn = jax.jit(step) if jax_compile else step
    if show_progress_bar and jax_compile: step_fn(init_key, state, 0, schedule[0]) # Force compile before loop

    cyclical_samples = []
    tqdm_interval = lambda x: tqdm.trange(x, mininterval=1)
    iter_range = tqdm_interval if show_progress_bar else jnp.arange
    for i in iter_range(num_training_steps):
        prev_pos = state.position
        key, sampler_key = jax.random.split(key)
        state = step_fn(sampler_key, state, 0, schedule[i])

        # if jnp.isnan(state.position).any() or not is_in_bounds(state.position, bounds):
        #     print(f'info chain at step {i} (do_sample:{schedule[i].do_sample}) nan or outofbounds in this position: {state.position}')
        #     print(f'nan or outofbounds in this position: {state.position}')
        #     print(f'log_density_fn prev value: {logdensity_fn(prev_pos)}')
        #     #print(f'Last 10 samples:', jnp.array(cyclical_samples[-10:]))

        if jnp.isnan(state.position).any():
            key, init_key = jr.split(key)
            initial_position = jr.uniform(key=init_key, minval=bounds[:,0], maxval=bounds[:,1], shape=(bounds.shape[0],))   
            state = init(initial_position)
            print(f'Error: chain failed at step {i} (do_sample:{schedule[i].do_sample}) doing {prev_pos}->{state.position}, resetting chain.')
            
        else:
            in_bounds_mask = is_in_bounds(state.position, bounds)  
            if config.use_param_clip and not jnp.all(in_bounds_mask):
                key, resize_key = jr.split(key)
                
                in_bounds_mask = is_in_bounds(state.position, bounds)
                resize = jr.uniform(resize_key, minval=0.96, maxval=1)
                new_bounds = bounds_resizer(bounds, resize)
                bounded_position = param_clip(state.position, new_bounds)
                new_position = jnp.where(in_bounds_mask, state.position, bounded_position)

                #print(f'shaking bounds from {state.position} to {new_position} by resize={resize} and mask={in_bounds_mask}')
                state = state._replace(position=new_position)


        # if config.use_param_clip:
        #     state = state._replace(position=param_clip(state.position, bounds))

        if schedule[i].do_sample:
            cyclical_samples.append(state.position.reshape(-1))
    return state.position, jnp.array(cyclical_samples), jnp.array([])

class CyclicalSGMCMCState(NamedTuple):
    """State of the Cyclical SGMCMC sampler.
    """
    position: ArrayTree
    opt_state: OptState

def cyclical_sgld(grad_estimator_fn, learning_rate, grad_clipping):
    # Initialize the SgLD step function
    sgld = blackjax.sgld(grad_estimator_fn)

    sgd = optax.sgd(learning_rate)
    if grad_clipping > 0:
        sgd = optax.chain(
            optax.clip_by_global_norm(grad_clipping),
            optax.sgd(learning_rate),
        )

    def init_fn(position):
        opt_state = sgd.init(position)
        return CyclicalSGMCMCState(position, opt_state)

    def step_fn(rng_key, state, minibatch, schedule_state):
        """Cyclical SGLD kernel."""

        def step_with_sgld(current_state):
            rng_key, state, minibatch, step_size = current_state
            new_position = sgld.step(rng_key, state.position, minibatch, step_size)
            return CyclicalSGMCMCState(new_position, state.opt_state)

        def step_with_sgd(current_state):
            _, state, minibatch, step_size = current_state
            grads = grad_estimator_fn(state.position, 0)
            rescaled_grads = - 1. * step_size * grads
            updates, new_opt_state = sgd.update(
                rescaled_grads, state.opt_state, state.position
            )
            new_position = optax.apply_updates(state.position, updates)
            return CyclicalSGMCMCState(new_position, new_opt_state)

        new_state = jax.lax.cond(
            schedule_state.do_sample,
            step_with_sgld,
            step_with_sgd,
            (rng_key, state, minibatch, schedule_state.step_size)
        )

        return new_state

    return init_fn, step_fn

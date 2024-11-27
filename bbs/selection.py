import jax
import jax.numpy as jnp
import jax.random as jr
import jaxopt

import scipy

from bbs.utils import is_in_bounds, negative, match_input, soft_bounds_weighting, stack_field_from_list

def filter_within_bounds(bounds, x, y=None):
    '''
    Removes the entries from x,y where x is out of bounds
    '''
    valid_indices = is_in_bounds(x, bounds)
    x = x[valid_indices]

    if y is None:
        return x
    
    y = y[valid_indices]
    return x, y

def thompson_sampling(key, posterior_sample_fn, bounds, num_queries=1, num_points_sample=None):
    if num_points_sample is None: num_points_sample = 50 + (bounds.shape[0] - 1) * 30

    # TODO(Javier): Implement with a ngrid
    sample_at = jr.uniform(key, minval=bounds[:,0], maxval=bounds[:,1], shape=(num_points_sample, bounds.shape[0]))

    post_samples = posterior_sample_fn(key, sample_at, shape=(num_queries,))
    i_best = jnp.argmin(post_samples, axis=1)
    x, y = sample_at[i_best], post_samples[i_best]
    return x, y

def acquisition_maximization(key, func, bounds, method='L-BFGS-B', num_retrials=10, maxiter=2_000):
    key, init_key = jr.split(key)
    init_positions = jr.uniform(key=init_key, minval=bounds[:,0], maxval=bounds[:,1], shape=(num_retrials, bounds.shape[0]))

    res_list = []
    for i in range(num_retrials):
        res = scipy.optimize.minimize(negative(func), init_positions[i], bounds=bounds, method=method)
        res_list.append(res)
    x = stack_field_from_list('x', res_list)
    y = -1.0 * stack_field_from_list('fun', res_list) # Negate the values since they come from negative(fn)
    return x, y

def acquisition_vectorized_maximization(key, func, bounds, method='L-BFGS-B', stepsize=1e-3, num_retrials=10, maxiter=2_000):
    key, init_key = jr.split(key)
    init_positions = jr.uniform(key=init_key, minval=bounds[:,0], maxval=bounds[:,1], shape=(num_retrials, bounds.shape[0]))

    solver = jaxopt.LBFGS(fun=negative(func), maxiter=maxiter, unroll=False)
    def solver_run_wrapper(x):
        return solver.run(x, bounds)

    run_solver = jax.vmap(solver_run_wrapper, in_axes=(0,))


    sols = run_solver(init_positions)
    return sols

def boltzmann_sampling(key, log_func, bounds, sampler, sampler_config, 
    num_queries=1, num_warmup=0.3, 
    temperature=None, normalize=True, 
    jax_compile=True
):
    soft_bounds_fn = lambda x: soft_bounds_weighting(x, bounds, min_val=1e-20) # min value is non-zero to avoid zeros when sampling

    logdensity_fn = None

    if temperature is None:
        @match_input
        def acq_logdensity_fn(xtest):
            return log_func(xtest) + jnp.log(soft_bounds_fn(xtest))
        
        logdensity_fn = acq_logdensity_fn
        
    else:
        max_log_val = 0.0
        if normalize:
            x, val = acquisition_maximization(key, log_func, bounds)
            x, val = filter_within_bounds(bounds, x, val)
            max_log_val = jnp.max(val)

        @match_input
        def boltz_log_density_fn(xtest):
            normalized_val = jnp.exp(log_func(xtest) - max_log_val)
            return jnp.log(jnp.exp(normalized_val / temperature)) + jnp.log(soft_bounds_fn(xtest))
                           
        logdensity_fn = boltz_log_density_fn
    print('DEBUG temperature', temperature)
    
    key, sampler_key = jr.split(key)
    _, chains, _ = sampler(sampler_key, logdensity_fn, bounds, sampler_config, jax_compile=jax_compile)
    assert(chains.shape[-1] == bounds.shape[0])

    if chains.ndim == 2: 
        chains = chains[jnp.newaxis,:,:]

    key, choice_key = jr.split(key)
    if num_queries <= chains.shape[0]: # Take end of chains
        samples = chains[:,-1,:]
        samples = filter_within_bounds(bounds, samples)
        if num_queries <= samples.shape[0]:
            i_sample = jr.choice(choice_key, jnp.arange(samples.shape[0]), shape=(num_queries,), replace=False)
            return samples[i_sample]
    
    # Fallback behaviour, sample from chains
    num_ignore = num_warmup
    if isinstance(num_warmup, float):
        num_ignore = int(chains.shape[1] * num_warmup)
    assert(num_warmup < chains.shape[1])

    # Filter samples within bounds
    samples = jnp.reshape(chains[:,num_ignore:,:], (-1,chains.shape[-1]))
    samples = filter_within_bounds(bounds, samples)
    if num_queries > samples.shape[0]:
        print('DEBUG not enough samples for num_queries')
        print('num ignore', num_ignore)
        print('num warmup', num_warmup)

        orig_samples = jnp.reshape(chains[:,num_ignore:,:], (-1,chains.shape[-1]))
        filter_samples = filter_within_bounds(bounds, orig_samples)
        print('with ignore: orig_samples -> filter_smaples', orig_samples.shape[0], '->', filter_samples.shape[0])
        
        orig_samples = jnp.reshape(chains[:,:,:], (-1,chains.shape[-1]))
        filter_samples = filter_within_bounds(bounds, orig_samples)
        print('using all: orig_samples -> filter_smaples', orig_samples.shape[0], '->', filter_samples.shape[0])

        samples = filter_samples

    if num_queries > samples.shape[0]:
        print('Debug 10 init chain', chains[:,:10,:])
        print('...')
        print('Debug 10 final chain', chains[:,-10:,:])

    assert(num_queries <= samples.shape[0])

    # Randomly pick
    key, choice_key = jr.split(key)
    i_query = jr.choice(choice_key, jnp.arange(samples.shape[0]), shape=(num_queries,), replace=False)
    return samples[i_query]

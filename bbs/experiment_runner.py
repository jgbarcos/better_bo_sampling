
import jax.random as jr
import jax.numpy as jnp
import numpy as np

from dataclasses import dataclass, asdict, replace, is_dataclass
from typing import Any, Union, Callable, NamedTuple, Optional
from multiprocessing import pool
from jaxtyping import Array

import tqdm
import time
import copy

import bbs.bayes_opt as bo
from bbs.bayes_opt import ConfigBO, StateBO, get_bo_functions
from bbs.datasets import get_problem
from bbs.gaussian_process import get_gp_functions
from bbs.utils import stack_field_from_list

@dataclass
class ExperimentResults:
    x: Array
    y: Array
    fit_time: Array
    select_time: Array

def ignore_keys(data, *fields):
    return {k:v for k,v in data.items() if k not in fields}
    
def stacked_results(all_exp: list[ExperimentResults]):
    return ExperimentResults(
        x = stack_field_from_list('x', all_exp),
        y = stack_field_from_list('y', all_exp),
        fit_time = stack_field_from_list('fit_time', all_exp),
        select_time = stack_field_from_list('select_time', all_exp),
    )

def plot_optimization(ax, experiments: list[ExperimentResults], experiments_plot_options: list[dict], 
        use_quantiles=None, 
        plot_confidence=True,
        plot_log_min=False, 
        sort_labels=True,

    ):

    experiments = copy.deepcopy(experiments)
    experiments_plot_options = copy.deepcopy(experiments_plot_options)

    if plot_log_min:
        global_incumbent = min([jnp.min(exp.y) for exp in experiments])

    if sort_labels:
        def sort_by_mean(index):
            incumbents = np.minimum.accumulate(experiments[index].y, axis=-1)
            mu = incumbents.mean(axis=0)
            return -mu[-1]

        sorted_indices = sorted(range(len(experiments)), key=sort_by_mean)
        experiments = [experiments[i] for i in sorted_indices]
        experiments_plot_options = [experiments_plot_options[i] for i in sorted_indices]
        for i, plot_options in enumerate(experiments_plot_options):
            total = len(experiments_plot_options)
            plot_options['label'] = f'{total-i} - ' + plot_options['label']

    for exp, plot_options in zip(experiments, experiments_plot_options):
        incumbents = np.minimum.accumulate(exp.y, axis=-1)

        mu = incumbents.mean(axis=0)

        conf = 2 * incumbents.std(axis=0)
        lower = mu-conf
        upper = mu+conf

        if use_quantiles is not None:
            lower = jnp.quantile(incumbents, use_quantiles, axis=0)
            upper = jnp.quantile(incumbents, 1-use_quantiles, axis=0)

        if plot_log_min:
            apply_log = lambda x: jnp.log(jnp.abs(x - global_incumbent) + 1e-12)
            mu = apply_log(mu)
            lower = apply_log(lower)
            upper = apply_log(upper)

        x_steps = np.arange(exp.x.shape[1])

        ax.plot(x_steps, mu, **plot_options)
        if plot_confidence: ax.fill_between(x_steps, lower, upper, alpha=0.4, linestyle='', **ignore_keys(plot_options, 'label','marker', 'linestyle'))
        ax.set_xlabel('opt steps')
        ax.set_ylabel('incumbent')
    #ax.legend()

def plot_field(ax, field_name, experiments: list[ExperimentResults], experiments_plot_options: list[dict]):
    for exp, plot_options in zip(experiments, experiments_plot_options):
        values = getattr(exp, field_name)
        
        mu = values.mean(axis=0)
        conf = 2 * values.std(axis=0)

        x_axis = np.arange(values.shape[1])

        ax.plot(x_axis, mu, **plot_options)
        ax.fill_between(x_axis, mu-conf, mu+conf, alpha=0.5, **plot_options | none_dict('label','marker'))
        ax.set_ylabel(field_name)
    ax.legend()

def get_args_run_repeat_experiment(work_identifier=None, num_trials=1, seed=0, **kwargs):
    list_args = []
    for trial in range(num_trials):
        args = {
            'work_identifier': work_identifier,
            'seed': trial + seed,
            **kwargs,
        }
        list_args.append(args)
    return list_args

def task_run_experiment(d: dict):
    return _task_run_experiment(**d)

def _task_run_experiment(work_identifier=None, config_bo: dict={}, problem_name: Optional[str]=None, problem_args: dict={}, seed=0, 
            **kwargs): #num_bo_steps=None, selection_fn=None):
    config_with_f = {}
    if problem_name is not None:
        problem = get_problem(problem_name, args=problem_args)
        config_with_f = {
            'f': problem.f,
            'bounds': problem.bounds,
        }
    config = ConfigBO(**config_bo|config_with_f)
    
    res = run_experiment(config=config, seed=seed, **kwargs) #num_bo_steps=num_bo_steps, selection_fn=selection_fn, 
    if work_identifier is not None:
        return work_identifier, res
    else:
        return res

def run_repeat_experiment(config: ConfigBO, num_bo_steps, selection_fn, num_trials=1, seed=0):
    all_exp = []
    for trial in range(num_trials):
        new_seed = trial + seed
        single_experiment = run_experiment(config, num_bo_steps=num_bo_steps, selection_fn=selection_fn, seed=new_seed)
        all_exp.append(single_experiment)

    return stacked_results(all_exp)

def run_experiment(config: ConfigBO, num_bo_steps, selection_fn, selection_args: dict={}, seed=0, other_seed = None, show_progress_bar=False):
    if other_seed is None: other_seed = seed + 1_000

    # The idea is to have 2 different keys, so that we can easily replicate the state of bo from a record of StateBO.
    bo_key = jr.PRNGKey(seed)
    other_key = jr.PRNGKey(other_seed)

    update_model_fn, eval_observation_fn = bo.get_bo_functions(config)
    state = StateBO(key=bo_key)

    fit_times, select_times = [], []
    range_fn = tqdm.trange if show_progress_bar else range
    for _ in range_fn(num_bo_steps):
        assert((state.x is None and state.y is None) or state.x.shape[0] == state.y.shape[0])

        t0 = time.perf_counter()
        state, model, x, y= update_model_fn(state)
        predict_fn, posterior_sample_fn = get_gp_functions(model, x, y)
        fit_times.append(time.perf_counter() - t0)

        other_key, selection_key = jr.split(other_key)

        predict_fn(jnp.array([[0.5] * config.bounds.shape[0]]))

        t0 = time.perf_counter()
        x_query = selection_fn(selection_key, predict_fn, posterior_sample_fn, config.bounds, x, y, **selection_args)
        select_times.append(time.perf_counter() - t0)

        state, _ = eval_observation_fn(state, x_query)

    return ExperimentResults(
        x=jnp.copy(state.x), 
        y=jnp.copy(state.y),
        fit_time = jnp.array(fit_times),
        select_time = jnp.array(select_times)
    )
import context

import jax
import jax.numpy as jnp
import jax.random as jr
from tinygp import GaussianProcess, kernels
from dataclasses import dataclass, asdict, replace, is_dataclass
from jaxtyping import Array
from typing import Any, Union, Callable, NamedTuple
from typing import Sequence
import tqdm

import bbs.samplers as samplers
from bbs.selection import acquisition_maximization, boltzmann_sampling, filter_within_bounds, thompson_sampling
from bbs.gaussian_process import train_gp_params
from bbs.experiment_runner import run_repeat_experiment, plot_optimization, plot_field, task_run_experiment, get_args_run_repeat_experiment
from bbs.criteria import get_ei_fn, get_log_ei_fn

# Config and Info
jax.config.update("jax_enable_x64", True)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pprint
import itertools
import argparse

def random_jump(key, _pos, bounds):
    return jr.uniform(key=key, minval=bounds[:,0], maxval=bounds[:,1], shape=(bounds.shape[0],))   

def get_add_normal_jump(sigma):
    def _fn(key, pos, bounds):
        return pos + sigma * jr.normal(key=key, shape=(bounds.shape[0],))   
    return _fn

def get_multiproposal(jump_funcs, weights=None):
    indices = jnp.arange(len(jump_funcs))
    def _fn(key, pos, bounds):
        key, choice_key = jr.split(key)
        i_choice = jr.choice(choice_key, indices, p=weights)
        return jax.lax.switch(i_choice, jump_funcs, key, pos, bounds)
    return _fn

def generate_multiproposal(tuples):
    list_functions = []
    list_weights = []
    for t in tuples:
        func = None
        weight = None
        if t[0] == 'random':
            _, weight = t
            func = random_jump
        elif t[0] == 'normal':
            _, weight, sigma = t
            func = get_add_normal_jump(sigma)
        else:
            print(f'Error wrong proposal distribution {t[0]}')
        list_functions.append(func)
        list_weights.append(weight)

    return get_multiproposal(list_functions, jnp.array(list_weights))


def select_with_ei_maximization(key, crit_fn, bounds, num_retrials=40, num_iters=5_000):
    x, val = acquisition_maximization(key, crit_fn, bounds, num_retrials=num_retrials, maxiter=num_iters)
    x, val = filter_within_bounds(bounds, x, val)

    # Return the best query
    i_best = jnp.argmax(val)
    return jnp.atleast_2d(x[i_best]), val[i_best]

def select_with_thompson_sampling(key, posterior_sample_fn, bounds, num_queries=1, num_points=None):
    x, y = thompson_sampling(key, posterior_sample_fn, bounds, num_queries, num_points)
    return x, y

def select_with_boltzmann_sampling(key, log_crit_fn, bounds, num_queries=1, sampler=None, sampler_config=None, temperature=None, normalize=True, jax_compile=True):
    x = boltzmann_sampling(key, log_crit_fn, bounds, sampler, sampler_config, num_queries, temperature=temperature, normalize=normalize, jax_compile=jax_compile)
    val = log_crit_fn(x) #TODO(Javier): This is not really needed

    return x, val

def max_ei(selection_key, predict_fn, posterior_sample_fn, bounds, x, y):
    x_query, _ = select_with_ei_maximization(selection_key, get_ei_fn(predict_fn, y), bounds, num_retrials=40, num_iters=3_000)
    return x_query

def max_log_ei(selection_key, predict_fn, posterior_sample_fn, bounds, x, y):
    x_query, _ = select_with_ei_maximization(selection_key, get_log_ei_fn(predict_fn, y), bounds, num_retrials=40, num_iters=3_000)
    return x_query

def sample_thompson(selection_key, predict_fn, posterior_sample_fn, bounds, x, y,
        num_queries=1,
        num_points=5_000,
    ):
    x_query, _ = select_with_thompson_sampling(selection_key, posterior_sample_fn, bounds, num_queries=num_queries, num_points=num_points)
    return x_query

def sample_boltzmann(selection_key, predict_fn, posterior_sample_fn, bounds, x, y, 
        num_queries=1,
        crit_name: str = 'ei',
        sampler_name: str = 'NUTS',
        sampler_args: dict = {},
        temperature=None, 
        normalize=True, 
        jax_compile: bool = True,
    ):
    log_crit_fn = None
    if crit_name == 'ei':
        crit_fn = get_ei_fn(predict_fn, y)
        log_crit_fn = lambda x: jnp.log(crit_fn(x))
    elif crit_name == 'log_ei':
        log_crit_fn = get_log_ei_fn(predict_fn, y)

    default_budget = dict(samples_budget=2_000)
    default_hmc = dict(inverse_mass_matrix=jnp.ones(shape=(bounds.shape[0],)))

    sampler, sampler_config = None, None
    if sampler_name == 'NUTS':
        sampler = samplers.blackjax_nuts
        sampler_config = default_budget |  default_hmc | sampler_args
    elif sampler_name == 'HMC':
        sampler = samplers.blackjax_hmc
        sampler_config = default_budget |  default_hmc | sampler_args
    elif sampler_name == 'CyclicalSGLD':
        sampler = samplers.blackjax_cyclical_sgld
        sampler_config = default_budget | {'step_size': 1e-3} | sampler_args
    elif sampler_name == 'MALA':
        sampler = samplers.blackjax_mala
        sampler_config = default_budget | sampler_args
    elif sampler_name == 'TemperedSMC':
        sampler = samplers.blackjax_temperedsmc
        sampler_config = default_budget | default_hmc | sampler_args
    elif sampler_name == 'MH':
        sampler = samplers.metropolis_hastings
        if 'multiproposal' in sampler_args:
            sampler_args['proposal_distribution'] = generate_multiproposal(sampler_args.pop('multiproposal'))
        sampler_config = default_budget | sampler_args
    else:
        print(f'Sampler {sampler_name} not found.')
    assert(sampler is not None)
    x_query, _ = select_with_boltzmann_sampling(selection_key, log_crit_fn, bounds, num_queries=num_queries, 
        sampler=sampler, sampler_config=sampler_config, 
        temperature=temperature, normalize=normalize, 
        jax_compile=jax_compile)
    return x_query

parser = argparse.ArgumentParser()
parser.add_argument('--reversed', action='store_true', help='Runs the experiments in reverse order.')
parser.add_argument('--diverged', action='store_true', help='Runs the experiments in diverging from center order.')
parser.add_argument('--stoperr', action='store_true', help='Stops running experiments on exception.')
parser.add_argument('--cpu', action='store_true', help='Forces cpu device.')
parser.add_argument('--rerun', action='store_true', help='When experiment filename already exists, it reruns and overwrites the existing experiment')
args = parser.parse_args()

if args.cpu:
    jax.config.update("jax_platform_name", "cpu")

# Main BO configuration
def build_gp(params, x):
    kernel = jnp.exp(params["log_gp_amp"]) * kernels.Matern52(
        jnp.exp(params["log_gp_scale"]), distance=kernels.L2Distance()
    )
    return GaussianProcess(
        kernel,
        x,
        diag=1e-3,
        mean=params["gp_mean"],
    )
init_params = {
    'log_gp_amp': jnp.log(0.1),
    'log_gp_scale': jnp.log(1.0),
    'gp_mean': jnp.float64(0.0),
    #'log_gp_diag': jnp.log(0.1),
}

bo_config = dict(
    init_model_params=init_params, 
    build_model=build_gp,
    train_model=train_gp_params,
    num_init_points=6
)
experiment_args = {
    'config_bo': bo_config, 'num_bo_steps': 30, 'num_trials': 10, 'seed': 0
}


# Samplers config
use_param_clip = {'use_param_clip': True}
default_multiproposal = {
    'multiproposal': [
        ['random', 0.25],
        ['normal', 0.25, 0.3],
        ['normal', 0.25, 0.1],
        ['normal', 0.25, 0.01],
    ]
}

adapt_params = {
    'adapt_warmup_steps': 0.2
}
do_chains = lambda n: {'num_chains': n}
list_mcmc_samplers = [
    ###('TemperedSMC', None, {}),
    ('HMC', 'HMCx5_paramclip', use_param_clip|do_chains(5)),
    ('MALA', 'MALAx5_paramclip', use_param_clip|do_chains(5)),
    #('NUTS', 'NUTSx5_paramclip', use_param_clip|do_chains(5)),
    ('CyclicalSGLD', 'CyclicalSGLD_paramclip', {
        'step_size': 1e-3,
        'use_param_clip': True,
    }),
    # ('CyclicalSGLD', 'CyclicalSGLD_default', {
    #     'step_size': 1e-3,
    #     'use_param_clip': False,
    # }),
    ('MH', 'MultiproposalMHx5', default_multiproposal|do_chains(5)),
    #('MH', 'MultiproposalMH', default_multiproposal),
]
list_mcmc_samplers = [
    ###('TemperedSMC', None, {}),
    ('HMC', 'HMCx5_paramclip', use_param_clip|do_chains(5)),
    ('MALA', 'MALAx5_paramclip', use_param_clip|do_chains(5)),
    ('NUTS', 'NUTSx5_paramclip', use_param_clip|do_chains(5)),
    ('CyclicalSGLD', 'CyclicalSGLD_paramclip', {
        'step_size': 1e-3,
        'use_param_clip': True,
    }),
    # ('CyclicalSGLD', 'CyclicalSGLD_default', {
    #     'step_size': 1e-3,
    #     'use_param_clip': False,
    # }),
    ('MH', 'MultiproposalMHx5', default_multiproposal|do_chains(5)),
    #('MH', 'MultiproposalMH', default_multiproposal),
]

list_sampler_criteria = ['log_ei'] #['ei', 'log_ei']
list_sampler_iters = [4_000] #[2_000, 4_000]
list_num_queries = [5]

select_experiments = []

if True: # MAX EI
    select_methods = [max_log_ei] # [max_ei, max_log_ei]
    select_names = ['max_log_ei'] # ['max_ei', 'max_log_ei']
    select_args = [{} for _ in select_methods]
    select_experiments += [
    {
        'name': sel_name, 
        'func': sel_func, 
        'args': sel_args, 
        'exp_overwrite': {'num_bo_steps':experiment_args['num_bo_steps']*5}, # Since it is sequential, multiply x5 observations
    } for sel_func, sel_name, sel_args in zip(select_methods, select_names, select_args)]



if True: # TS
    for num_queries in list_num_queries:
        for samples_budget in list_sampler_iters:

            formatted_name = f'sample_thompson_-_queries_{num_queries}_-_x{samples_budget}'

            select_experiments += [
                {
                    'name': formatted_name, 
                    'func': sample_thompson, 
                    'args': {
                        'num_queries': num_queries,
                        'num_points': samples_budget,
                    }
                }
            ]

if True: #AS
    for num_queries in list_num_queries:
        for crit_name in list_sampler_criteria:
            for samples_budget in list_sampler_iters:
                for sampler_name, sampler_tag, sampler_args in list_mcmc_samplers:
                    final_sampler_args = sampler_args | {'samples_budget': samples_budget}
                    if sampler_tag is None: sampler_tag = sampler_name

                    formatted_name = f'sample_boltzmann_-_queries_{num_queries}_-_crit_{crit_name}_-_sampler_{sampler_tag}_x{samples_budget}'

                    select_experiments += [
                        {
                            'name': formatted_name,
                            'func': sample_boltzmann,
                            'args': {
                                'num_queries': num_queries,
                                'crit_name': crit_name,
                                'sampler_name': sampler_name,
                                'sampler_args': final_sampler_args,
                                'jax_compile': True,
                            },
                        },
                    ]

if False: # Botlzmann with temperature
    for num_queries in list_num_queries:
        for crit_name in list_sampler_criteria:
            for samples_budget in list_sampler_iters:
                for temp in [1e-2, 1e-1, 1.0]:
                    for sampler_name, sampler_tag, sampler_args in [('MH', 'MultiproposalMHx5', default_multiproposal|do_chains(5))]:
                        final_sampler_args = sampler_args | {'samples_budget': samples_budget}
                        if sampler_tag is None: sampler_tag = sampler_name

                        formatted_name = f'sample_temperature_boltzmann_-_temp_{temp}_-_queries_{num_queries}_-_crit_{crit_name}_-_sampler_{sampler_tag}_x{samples_budget}'

                        select_experiments += [
                            {
                                'name': formatted_name,
                                'func': sample_boltzmann,
                                'args': {
                                    'num_queries': num_queries,
                                    'crit_name': crit_name,
                                    'sampler_name': sampler_name,
                                    'sampler_args': final_sampler_args,
                                    'temperature': temp,
                                    'jax_compile': True,
                                },
                            },
                        ]

# print('#### DEBUG RUNNING LESS METHODS')
# select_methods = [select_methods[3], select_methods[4]]
# select_names = [select_names[3], select_names[4]]
pp = pprint.PrettyPrinter(depth=4)
print('Running following methods')
pp.pprint(select_experiments)

list_problem_names = []
list_problem_args = []

if False:
    list_problem_names += [
        'rosenbrock',
        'ackley-3D',
        'ackley-5D',
        'ackley-10D',
        'alpine1-5D',
        'alpine1-10D',
        'alpine2-3D',
        'alpine2-5D',
        'alpine2-10D',
        # 'forrester',
        # 'beale',
        # 'dropwave',
        # 'cosines',
        # 'branin',
        # 'goldstein',
        # 'sixhumpcamel',
        # 'mccormick',
        # 'powers',
        #'alpine1-3D',
    ]
    list_problem_args += [{} for _ in list_problem_names]

if True:
    list_problem_names += ['rover-default', 'rover-hill_map']
    list_problem_args += [{'img_map': 'default_map.png'}, {'img_map': 'hill_map.png'}]

work_to_run = []
for p_name, p_args in zip(list_problem_names, list_problem_args):
    #for sel_fn, sel_name, sel_args in zip(select_methods, select_names, select_args):
    for sel_data in select_experiments:
        sel_fn, sel_name, sel_args = sel_data['func'], sel_data['name'], sel_data['args']
        exp_overwrite = {} if not 'exp_overwrite' in sel_data else sel_data['exp_overwrite']
        work_identifier = (p_name, sel_name)
        arg_list = get_args_run_repeat_experiment(
            work_identifier=work_identifier, 
            problem_name=p_name, 
            problem_args=p_args, 
            **experiment_args|exp_overwrite|{'selection_fn': sel_fn, 'selection_args': sel_args},
        )
        work_to_run += arg_list

#work_to_run = sorted(work_to_run, key=lambda a: (a['seed'], a['work_identifier'][0], a['work_identifier'][1]))
work_to_run = sorted(work_to_run, key=lambda a: (a['work_identifier'][0], a['seed'], a['work_identifier'][1]))

if args.reversed:
    work_to_run = list(reversed(work_to_run))

def diverged_sort(arr):
    # Sort in diverging order from center
    # Example: [1,2,3,4,5] -> [3,2,4,1,5]
    indices = jnp.arange(len(arr))
    pivot_index = indices[int(len(indices)/2)]
    pivot_dist = lambda i: abs(pivot_index-i)
    i_sort = sorted(indices, key=pivot_dist)
    return [arr[i] for i in i_sort]

if args.diverged:
    work_to_run = diverged_sort(work_to_run)

from multiprocessing import Pool
from functools import reduce
import traceback

from pathlib import Path
import pickle
import os

def save_opt_result(res, directory, filename: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as fp:
        pickle.dump(res, fp)

# print('Tasks to run:')
# pp.pprint(work_tasks)

# Run the tasks
res_list = []

for i in tqdm.trange(len(work_to_run)):
    
    work_id = work_to_run[i]['work_identifier']
    seed = work_to_run[i]['seed']

    directory = os.path.join('./','results', *work_id)
    filename = f'opt_res_{seed}.pkl'
    error_filename = f'error_log_{seed}.txt'
    info_filename = f'info_log_{seed}.txt'

    path_filename = os.path.join(directory, filename)
    path_error_filename = os.path.join(directory, error_filename)
    path_info_filename = os.path.join(directory, info_filename)

    if args.rerun and os.path.isfile(path_filename):
        print(f'File already exists {path_filename}, skipping')
        continue


    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path_info_filename, "w") as fp:
        debug_pp = pprint.PrettyPrinter(depth=4, stream=fp)
        debug_pp.pprint(work_to_run[i])

    #pp.pprint(work_tasks[i])
    try:
        work_id, res = task_run_experiment(work_to_run[i])
        print(f'Saving file {path_filename}.')
        save_opt_result(asdict(res), directory, filename)

        if os.path.isfile(path_error_filename):
            os.remove(path_error_filename)

    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print(f'Task {i} failed!')
        pp.pprint(work_to_run[i])
        traceback.print_exc()

        with open(path_error_filename, "w") as fp:
            fp.write(str(e))
            fp.write(traceback.format_exc())

        if os.path.isfile(path_filename):
            os.remove(path_filename)

        if args.stoperr:
            break


    jax.clear_caches()

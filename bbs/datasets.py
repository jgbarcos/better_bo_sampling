import jax
from jaxtyping import Array
import numpy as np

from typing import Any, Union, Callable, NamedTuple, Optional
from dataclasses import dataclass, replace
import copy

from bbs.utils import vectorized_wrapper

from bbs.problems.rover.rover_function import RoverProblem
from bbs.problems import experiments1d, experiments2d, experimentsNd


GPYOPT_PROBLEMS = {
    'forrester': experiments1d.forrester,
    'rosenbrock': experiments2d.rosenbrock,
    'beale': experiments2d.beale,
    'dropwave': experiments2d.dropwave,
    'cosines': experiments2d.cosines,
    'branin': experiments2d.branin,
    'goldstein': experiments2d.goldstein,
    'sixhumpcamel': experiments2d.sixhumpcamel,
    'mccormick': experiments2d.mccormick,
    'powers': experiments2d.powers,
    'eggholder': experiments2d.eggholder,
    'alpine1': experimentsNd.alpine1,
    'alpine2': experimentsNd.alpine2,
    'ackley': experimentsNd.ackley,
}

@dataclass
class Problem:
    f: Callable
    bounds: Array
    norm_fn: Callable = lambda x: x
    unorm_fn: Callable = lambda x: x
    raw_f: Callable = None
    raw_bounds: Array = None
    
    def __post_init__(self):
        if self.raw_f is None:
            self.raw_f = self.f
        if self.raw_bounds is None:
            self.raw_bounds = self.bounds

def parse_function_name(full_name: str):
    fields = full_name.split('-')
    func_name = fields[0]
    fields = fields[1:]

    input_dim = None
    for f in fields:
        if f.endswith('D'):
            input_dim = int(f.split('D')[0])

    return func_name, input_dim

def get_problem(full_name: str, normalize_x: bool=True, args: dict = {}) -> Problem:
    problem_name, input_dim = parse_function_name(full_name)

    if problem_name in GPYOPT_PROBLEMS:
        prob_args = {'input_dim': input_dim}|args
        prob = GPYOPT_PROBLEMS[problem_name](**prob_args)
        raw_f = prob.f
        raw_bounds = np.array(prob.bounds)

    if problem_name.startswith('rover'):
        image_path = './bbs/problems/rover/'
        rover_args = copy.deepcopy(args)

        if 'img_map' in args and args['img_map'] is not None:
            rover_args =  args | {'img_map': image_path +  args['img_map']}

        rover = RoverProblem(**(dict(num_points=2, k=2, s=0.0, extra_domain=0.0, img_map=None)|rover_args))

        raw_f = vectorized_wrapper(rover.__call__)
        raw_bounds = np.vstack((rover.lb, rover.ub)).T

    assert(raw_f is not None and raw_bounds is not None)

    f = raw_f
    bounds = raw_bounds
    norm_fn: Callable = lambda x: x
    unorm_fn: Callable = lambda x: x
    if normalize_x:
        def normalize(x: Array) -> Union[Array]:
            return (x - raw_bounds[:,0]) / (raw_bounds[:,1] - raw_bounds[:,0])
        def denormalize(x: Array) -> Union[Array]:
            return x * (raw_bounds[:,1] - raw_bounds[:,0]) + raw_bounds[:,0]
        def normalized_f(x: Array) -> Union[Array, float]:
            return raw_f(denormalize(x))

        bounds = np.zeros_like(raw_bounds)
        bounds[:,1] = 1.0
        f = normalized_f
        norm_fn = normalize
        unorm_fn = denormalize

    return Problem(f=f, bounds=bounds, norm_fn=norm_fn, unorm_fn=unorm_fn, raw_f=raw_f, raw_bounds=raw_bounds)
    


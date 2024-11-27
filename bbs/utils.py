import jax
import jax.numpy as jnp
from jaxtyping import Array

from typing import Any, Union, Callable, NamedTuple, Optional
from operator import attrgetter

def match_input(func: Callable, index: int=0):
    '''
    Decorator that given the arg at index corresponding to an Array of input points, transforms the output to match the input.
    If func is called with a single point (1D) it returns a float. If it is called with a 2D array, it returns an array of floats.
    This is required since some functions always return a 1D output despite using a 1D input, when it should be a float.
        f([0,1]) -> float
        f([[0,1]]) -> [float]
        f([[0,1],[0,1]]) -> [float,float]
    '''
    def wrapper(*args, **kwargs) -> Union[Array, float, bool]:
        x_in = args[index]
        out = func(*args, **kwargs)
        if x_in.ndim == 1 and out.ndim == 1 and len(out) == 1:
            out = out[0]
        return out
    return wrapper


def elem_gauss_func(x, a=1.0,b=0.0,c=1/jnp.sqrt(2), dome=0.0):
    center = x-b
    recenter = jax.lax.cond(
        jnp.abs(center) < dome,
        lambda: 0.0,
        lambda: jax.lax.cond(
            center > dome,
            lambda: center - dome,
            lambda: center + dome,
        )
    )
    return a * jnp.exp(-(recenter)**2 / (2*c**2))# / ( a * jnp.exp(-(0.0)**2 / (2*c**2)) )

def _single_soft_bounds(elem, bound, min_val, max_val, resize, c):
    center = (bound[0] + bound[1]) / 2.0
    diff = (bound[1] - bound[0]) * resize
    return min_val + elem_gauss_func(elem, a=max_val-min_val, b=center, c=c, dome=diff/2.0)
_soft_bounds = jax.vmap(_single_soft_bounds, in_axes=[0, None, None, None, None, None], out_axes=0)


@match_input
def soft_bounds_weighting(x_in: Array, bounds: Array, min_val: float=0.0, max_val: float=1.0, resize=0.98, c=1e-2) -> Union[Array]:
    '''
    Changes the value from max_val smoothly to min_val when going out of the axis-aligned bounds.
    '''
    x = jnp.atleast_2d(x_in)
    w = 1.0
    for i in range(x.shape[-1]):
        w *= _soft_bounds(x[:,i], bounds[i], min_val, max_val, resize, c)
    return w

@match_input
def is_in_bounds(x: Array, bounds: Array):
    x = jnp.atleast_2d(x)
    return jnp.all(jnp.logical_and(bounds[:,0] <= x, x <= bounds[:,1]), axis=1)

@match_input
def which_in_bounds(x: Array, bounds: Array):
    x = jnp.atleast_2d(x)
    return jnp.logical_and(bounds[:,0] <= x, x <= bounds[:,1])

def negative(fn: Callable) -> Callable:
    '''
    Negates the function output values, useful when maximizing a function using a minimization optimizer
    '''
    def wrapper(*args, **kwargs):
        return -fn(*args, **kwargs)
    return wrapper


def _stack_field_from_list(field_name: str, elems: list) -> Array:
    return jnp.array(list(map(attrgetter(field_name), elems)))

def stack_field_from_list(field_names: Union[str,list[str]], elems: list):
    if isinstance(field_names, str):
        return _stack_field_from_list(field_names, elems)

    stacked_fields = []
    for name in field_names:
        stacked_fields.append(_stack_field_from_list(name, elems))
    return tuple(stacked_fields)

def dict_without_keys(d, discard):
    return {k:v for k,v in d if k not in discard}

def vectorized_wrapper(f):
    '''
    For functions that dont accept multiple x inputs, this wrapper calls them in sequential order.
    '''
    @match_input
    def _func(x):
        x = jnp.atleast_2d(x)
        return jnp.array([f(x[i]) for i in range(x.shape[0])])
    return _func
        

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm
from jaxtyping import Array

from typing import Any, Union, Callable, NamedTuple, Optional
from dataclasses import dataclass, replace

from bbs.gaussian_process import train_gp_params
from bbs.utils import match_input, soft_bounds_weighting

@dataclass(frozen=True)
class StateBO:
    key: jr.PRNGKey = jr.PRNGKey(123)
    x: Array = None
    y: Array = None
    model_params: dict = None

@dataclass
class ConfigBO:
    f: Callable
    bounds: Array
    init_model_params: dict
    build_model: Callable
    train_model: Callable[..., dict] = train_gp_params
    num_init_points: int = 10

def get_bo_functions(config: ConfigBO) -> tuple[Callable, Callable]:
    if isinstance(config, dict): config = ConfigBO(**config)

    step_fn = lambda state: update_model(state, config)
    observe_fn = lambda state, x: evaluate_observation(state, config, x)
    return step_fn, observe_fn

# class BayesOpt():
#     '''
#     Class based interface for functional calls
#     '''
#     def __init__(self, config:Union[dict, ConfigBO]):
#         self.update_config(config)
#         self.state = StateBO
        
#     def update_config(self, config): 
#         self._step_fn, self._observe_fn = get_bo_functions(config)

#     def update_model(self):
#         self.state, self.predict, self.crit, self.logdensity = self._step_fn(self.state)

#     def evaluate_observation(self, x):
#         state, y = self._observe_fn(self.state, x)
#         return y

def update_model(state:StateBO, config:ConfigBO) -> tuple[StateBO, Callable, Callable, Callable]:
    # State
    model_params = state.model_params
    x, y = state.x, state.y
    key = state.key

    # Config
    f = config.f
    bounds = config.bounds
    num_dims = bounds.shape[0]
    build_model = config.build_model

    key, init_key, train_key = jr.split(key, 3)
    
    # Initialize x and y if needed
    if x is None or y is None:
        x = jr.uniform(key=init_key, minval=bounds[:,0], maxval=bounds[:,1], shape=(config.num_init_points, num_dims))
        y = f(x).reshape(-1)

    # Initialize model params if needed
    if model_params is None:
        model_params = config.init_model_params
    
    # Train the model params according to current data x,y
    model_params = config.train_model(train_key, model_params, x, y, build_model)
    model = build_model(model_params, x)

    return replace(state, key=key, x=x, y=y, model_params=model_params), model, jnp.copy(x), jnp.copy(y) #predict_fn, crit_fn, logdensity_fn, posterior_sample_fn

def evaluate_observation(state:StateBO, config:ConfigBO, x_query: Array, y_query: Optional[Array] = None) -> tuple[StateBO, Array]:
    if y_query is None:
        y_query = config.f(x_query).reshape(-1)

    x = jnp.vstack((state.x, x_query))
    y = jnp.concatenate((state.y, y_query))

    return replace(state, x=x, y=y), y_query
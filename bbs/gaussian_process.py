from tinygp import kernels, GaussianProcess

import jax
import jax.numpy as jnp
from jaxtyping import Array

import flax.linen as nn
from flax.linen.initializers import zeros

import optax
import matplotlib.pyplot as plt
import jaxopt

from bbs.utils import match_input


def default_build_gp(params: dict, x: Array) -> GaussianProcess:
    kernel = jnp.exp(params["log_gp_amp"]) * kernels.Matern52(
        jnp.exp(params["log_gp_scale"])
    )
    return GaussianProcess(
        kernel,
        x,
        diag=jnp.exp(params["log_gp_diag"]),
        mean=params["gp_mean"],
    )

def get_negll_loss(build_gp, x, y):
    def _fn(params):
        gp = build_gp(params, x)
        return -gp.log_probability(y)
    return _fn

def get_gp_functions(gp, x, y):
    @match_input
    def predict_fn(xtest):
        mu, var = gp.predict(y, jnp.atleast_2d(xtest), return_var=True)
        return mu, var
    
    def posterior_sample_fn(key, xtest, shape):
        return gp.condition(y, xtest).gp.sample(key, shape)
    
    return predict_fn, posterior_sample_fn

def train_gp_params(_key, params, x, y, build_gp=None, return_final_loss=False):
    if build_gp is None:
        build_gp = default_build_gp
    loss = get_negll_loss(build_gp, x, y)

    solver = jaxopt.ScipyMinimize(fun=loss, method='L-BFGS-B')
    soln = solver.run(params)

    if return_final_loss:
        return soln.params, soln.state.fun_val
    else:
        return soln.params

    optimizer = optax.sgd(learning_rate=1e-7)
    opt_state = optimizer.init(params)
    loss_grad_fn = jax.jit(jax.value_and_grad(loss))

    losses = []
    for i in range(n_opt_iters):
        loss_val, grads = loss_grad_fn(params)
        if i < 10:
            print(loss_val, grads)
        losses.append(loss_val)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    print('losses:', losses)

    if plot_loss:
        plt.plot(losses)
        plt.ylabel("negative log likelihood")
        _ = plt.xlabel("step number")
        plt.show()
        print('showing plot')
    print('exiting')
    print('params:', params)

    return params
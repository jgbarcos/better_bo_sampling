import jax
from jax import random
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import blackjax
import tinygp
from tensorflow_probability.substrates import jax as tfp

from print_versions import print_versions
print_versions(globals())

import time
import matplotlib.pyplot as plt
import tqdm

# Config and Info
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import multiprocessing
num_cores = multiprocessing.cpu_count()
print(f'NUM_CPU_CORES: {num_cores}')

import sys
sys.path.append('../') # Make main module visible

jnp.array(1.)

def get_test_dataset():
    import numpy as np
    random = np.random.default_rng(42)
    x = np.sort(np.append(random.uniform(0, 3.8, 28), random.uniform(5.5, 10, 18),))
    yerr = random.uniform(0.08, 0.22, len(x))
    y = (0.2 * (x - 5) + np.sin(3 * x + 0.1 * (x - 5) ** 2) + yerr * random.normal(size=len(x)))
    true_x = np.linspace(0, 10, 100)
    true_y = 0.2 * (true_x - 5) + np.sin(3 * true_x + 0.1 * (true_x - 5) ** 2)
    return x, y, yerr, true_x, true_y

from bbs.gaussian_process import train_gp_params, default_build_gp

from tinygp import kernels, GaussianProcess


def TEST_GP():
    x, y, yerr, true_x, true_y = get_test_dataset()
    build_gp = default_build_gp
    init_params = {
        'log_gp_amp': jnp.log(0.1),
        'log_gp_scale': jnp.log(1),
        'gp_mean': jnp.float64(0.0),
        'log_gp_diag': jnp.log(0.1),
    }

    print('before training')
    #params = train_gp_params(None, init_params, x, y, build_gp)
    params = init_params
    #gp_cond = build_gp(params, x).condition(y, true_x).gp
    #pred, var = gp_cond.loc, gp_cond.variance

    print('after training')
    pred, var = build_gp(params, x).predict(y, true_x, return_var=True)
    
    interval = 2 * jnp.sqrt(var)

    plt.plot(true_x, true_y, "k", lw=1.5, alpha=0.3, label="truth")
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(true_x, pred, color='b', label="max likelihood model")
    plt.fill_between(true_x, pred-interval, pred+interval, color='b', alpha=0.5)
    plt.xlabel("x [day]")
    plt.ylabel("y [ppm]")
    plt.xlim(0, 10)
    plt.ylim(-2.5, 2.5)
    plt.legend()
    _ = plt.title("maximum likelihood")
TEST_GP()
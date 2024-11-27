import jax
from tensorflow_probability.substrates import jax as tfp
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.special import ndtr as Phi
from jaxtyping import Array

from typing import Any, Union, Callable, NamedTuple
import math

from bbs.utils import match_input

def get_ei_fn(predict_fn: Callable, y: Array) -> Callable:
    @match_input
    def crit_fn(xtest: Array) -> Array:
        mu, var = predict_fn(jnp.atleast_2d(xtest))
        return expected_improvement(mu, jnp.sqrt(var), jnp.min(y))
    return crit_fn

def get_log_ei_fn(predict_fn: Callable, y: Array) -> Callable:
    @match_input
    def crit_fn(xtest: Array) -> Array:
        mu, var = predict_fn(jnp.atleast_2d(xtest))
        return log_expected_improvement(mu, jnp.sqrt(var), jnp.min(y))
    return crit_fn


def expected_improvement(mu: Array, sigma: Array, incumbent: float):
    improvement = incumbent - mu
    z = improvement / sigma
    ei = improvement * Phi(z) + sigma * norm.pdf(z)
    return ei

def log_expected_improvement(mu: Array, sigma: Array, incumbent: float):
    u = _scaled_improvement(mu, sigma, incumbent, False)
    return _log_ei_helper(u) + jnp.log(sigma)

def _scaled_improvement(
    mean: Array, sigma: Array, best_f: Array, maximize: bool
) -> Array:
    """From botorch.criteria.analytic
    Returns `u = (mean - best_f) / sigma`, -u if maximize == True.
    """
    u = (mean - best_f) / sigma
    return u if maximize else -u

def _ei_helper(u: Array) -> Array:
    """ Jax implementation based on _ei_helper() from botorch.criteria.analytic
    Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return norm.pdf(u) + u * Phi(u)

def _log_ei_helper(u: Array) -> Array:
    """Jax implementation based on _log_ei_helper() from botorch.criteria.analytic
    Accurately computes log(phi(u) + u * Phi(u)) in a differentiable manner for u in
    [-10^100, 10^100] in double precision, and [-10^20, 10^20] in single precision.
    Beyond these intervals, a basic squaring of u can lead to floating point overflow.
    In contrast, the implementation in _ei_helper only yields usable gradients down to
    u ~ -10. As a consequence, _log_ei_helper improves the range of inputs for which a
    backward pass yields usable gradients by many orders of magnitude.
    """
    if not (u.dtype == jnp.float32 or u.dtype == jnp.float64):
        raise TypeError(
            f"LogExpectedImprovement only supports jnp.float32 and jnp.float64 "
            f"dtypes, but received {u.dtype}."
        )
    # The function has two branching decisions. The first is u < bound, and in this
    # case, just taking the logarithm of the naive _ei_helper implementation works.
    bound = -1
    u_upper = jnp.where(u < bound, bound, u)  # mask u to avoid NaNs in gradients
    log_ei_upper = jnp.log(_ei_helper(u_upper))

    # When u <= bound, we need to be more careful and rearrange the EI formula as
    # log(phi(u)) + log(1 - exp(w)), where w = log(abs(u) * Phi(u) / phi(u)).
    # To this end, a second branch is necessary, depending on whether or not u is
    # smaller than approximately the negative inverse square root of the machine
    # precision. Below this point, numerical issues in computing log(1 - exp(w)) occur
    # as w approaches zero from below, even though the relative contribution to log_ei
    # vanishes in machine precision at that point.
    neg_inv_sqrt_eps = -1e6 if u.dtype == jnp.float64 else -1e3

    # mask u for to avoid NaNs in gradients in first and second branch
    u_lower = jnp.where(u > bound, bound, u) #u.masked_fill(u > bound, bound)
    u_eps = jnp.where(u < neg_inv_sqrt_eps, neg_inv_sqrt_eps, u) #u_lower.masked_fill(u < neg_inv_sqrt_eps, neg_inv_sqrt_eps)
    # compute the logarithm of abs(u) * Phi(u) / phi(u) for moderately large negative u
    w = _log_abs_u_Phi_div_phi(u_eps)

    # 1) Now, we use a special implementation of log(1 - exp(w)) for moderately
    # large negative numbers, and
    # 2) capture the leading order of log(1 - exp(w)) for very large negative numbers.
    # The second special case is technically only required for single precision numbers
    # but does "the right thing" regardless.
    log_ei_lower = norm.logpdf(u) + (
        jnp.where(
            u > neg_inv_sqrt_eps,
            tfp.math.log1mexp(w),
            # The contribution of the next term relative to log_phi vanishes when
            # w_lower << eps but captures the leading order of the log1mexp term.
            -2 * jnp.log(jnp.abs(u_lower)),
        )
    )
    return jnp.where(u > bound, log_ei_upper, log_ei_lower) #torch.where(u > bound, log_ei_upper, log_ei_lower)

_neg_inv_sqrt2 = -(2**-0.5)
_log_sqrt_pi_div_2 = math.log(math.pi / 2) / 2
def _log_abs_u_Phi_div_phi(u: Array) -> Array:
    """Jax implementation based on _log_abs_u_Phi_div_phi() from botorch.criteria.analytic
    Computes log(abs(u) * Phi(u) / phi(u)), where phi and Phi are the normal pdf
    and cdf, respectively. The function is valid for u < 0.

    NOTE: In single precision arithmetic, the function becomes numerically unstable for
    u < -1e3. For this reason, a second branch in _log_ei_helper is necessary to handle
    this regime, where this function approaches -abs(u)^-2 asymptotically.

    The implementation is based on the following implementation of the logarithm of
    the scaled complementary error function (i.e. erfcx). Since we only require the
    positive branch for _log_ei_helper, _log_abs_u_Phi_div_phi does not have a branch,
    but is only valid for u < 0 (so that _neg_inv_sqrt2 * u > 0).

        def logerfcx(x: Tensor) -> Tensor:
            return torch.where(
                x < 0,
                torch.erfc(x.masked_fill(x > 0, 0)).log() + x**2,
                torch.special.erfcx(x.masked_fill(x < 0, 0)).log(),
        )

    Further, it is important for numerical accuracy to move u.abs() into the
    logarithm, rather than adding u.abs().log() to logerfcx. This is the reason
    for the rather complex name of this function: _log_abs_u_Phi_div_phi.
    """
    # get_constants_like allocates tensors with the appropriate dtype and device and
    # caches the result, which improves efficiency.
    a, b = _neg_inv_sqrt2, _log_sqrt_pi_div_2
    return jnp.log(tfp.math.erfcx(a * u) * jnp.abs(u)) + b

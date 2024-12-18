import math
from math import pi
import torch
from torch import Tensor

from collections.abc import Iterator
from functools import lru_cache
from numbers import Number
from typing import Optional, Union

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

import bbs.criteria as jax_criteria

def log_expected_improvement(mu: Tensor, sigma: Tensor, incumbent: float):
    u = _scaled_improvement(mu, sigma, incumbent, False)
    return _log_ei_helper(u) + sigma.log()


def jax_compare(jax_func, name):
    def decorator(func):
        def call_jax(x: Tensor):
            jax_array = jnp.array(x.cpu().detach().numpy())
            jax_res = jax_func(jax_array)
            this_res = func(x)

            print(f'Function:{name}')
            print(' - jax', jax_res)
            print(' - torch', this_res.cpu().detach().numpy())
            return this_res
        return call_jax
    return decorator
  
_log2 = math.log(2)  
_inv_sqrt_2 = 1 / math.sqrt(2)
_neg_inv_sqrt_2 = -_inv_sqrt_2
_inv_sqrt_2pi = 1 / math.sqrt(2 * pi)
_log_sqrt_2pi = math.log(2 * pi) / 2
_neg_inv_sqrt_2 = -_inv_sqrt_2
_log_sqrt_2pi = math.log(2 * pi) / 2
_neg_inv_sqrt2 = -(2**-0.5)
_log_sqrt_pi_div_2 = math.log(math.pi / 2) / 2

@lru_cache(maxsize=None)
def get_constants(
    values: Union[Number, Iterator[Number]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    r"""Returns scalar-valued Tensors containing each of the given constants.
    Used to expedite tensor operations involving scalar arithmetic. Note that
    the returned Tensors should not be modified in-place."""
    if isinstance(values, Number):
        return torch.full((), values, dtype=dtype, device=device)

    return tuple(torch.full((), val, dtype=dtype, device=device) for val in values)

def get_constants_like(
    values: Union[Number, Iterator[Number]],
    ref: Tensor,
) -> Union[Tensor, Iterator[Tensor]]:
    return get_constants(values, device=ref.device, dtype=ref.dtype)

def _scaled_improvement(
    mean: Tensor, sigma: Tensor, best_f: Tensor, maximize: bool
) -> Tensor:
    """From botorch.criteria.analytic
    Returns `u = (mean - best_f) / sigma`, -u if maximize == True.
    """
    u = (mean - best_f) / sigma
    return u if maximize else -u

@jax_compare(jax_criteria._ei_helper, '_ei_helper')
def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)

@jax_compare(tfp.math.log1mexp, 'log1mexp')
def log1mexp(x: Tensor) -> Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    log2 = get_constants_like(values=_log2, ref=x)
    is_small = -log2 < x  # x < 0
    return torch.where(
        is_small,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )

@jax_compare(jax_criteria._log_ei_helper, '_log_ei_helper')
def _log_ei_helper(u: Tensor) -> Tensor:
    """Accurately computes log(phi(u) + u * Phi(u)) in a differentiable manner for u in
    [-10^100, 10^100] in double precision, and [-10^20, 10^20] in single precision.
    Beyond these intervals, a basic squaring of u can lead to floating point overflow.
    In contrast, the implementation in _ei_helper only yields usable gradients down to
    u ~ -10. As a consequence, _log_ei_helper improves the range of inputs for which a
    backward pass yields usable gradients by many orders of magnitude.
    """
    if not (u.dtype == torch.float32 or u.dtype == torch.float64):
        raise TypeError(
            f"LogExpectedImprovement only supports torch.float32 and torch.float64 "
            f"dtypes, but received {u.dtype}."
        )
    # The function has two branching decisions. The first is u < bound, and in this
    # case, just taking the logarithm of the naive _ei_helper implementation works.
    bound = -1
    u_upper = u.masked_fill(u < bound, bound)  # mask u to avoid NaNs in gradients
    log_ei_upper = _ei_helper(u_upper).log()

    # When u <= bound, we need to be more careful and rearrange the EI formula as
    # log(phi(u)) + log(1 - exp(w)), where w = log(abs(u) * Phi(u) / phi(u)).
    # To this end, a second branch is necessary, depending on whether or not u is
    # smaller than approximately the negative inverse square root of the machine
    # precision. Below this point, numerical issues in computing log(1 - exp(w)) occur
    # as w approaches zero from below, even though the relative contribution to log_ei
    # vanishes in machine precision at that point.
    neg_inv_sqrt_eps = -1e6 if u.dtype == torch.float64 else -1e3

    # mask u for to avoid NaNs in gradients in first and second branch
    u_lower = u.masked_fill(u > bound, bound)
    u_eps = u_lower.masked_fill(u < neg_inv_sqrt_eps, neg_inv_sqrt_eps)
    # compute the logarithm of abs(u) * Phi(u) / phi(u) for moderately large negative u
    w = _log_abs_u_Phi_div_phi(u_eps)

    # 1) Now, we use a special implementation of log(1 - exp(w)) for moderately
    # large negative numbers, and
    # 2) capture the leading order of log(1 - exp(w)) for very large negative numbers.
    # The second special case is technically only required for single precision numbers
    # but does "the right thing" regardless.
    log_ei_lower = log_phi(u) + (
        torch.where(
            u > neg_inv_sqrt_eps,
            log1mexp(w),
            # The contribution of the next term relative to log_phi vanishes when
            # w_lower << eps but captures the leading order of the log1mexp term.
            -2 * u_lower.abs().log(),
        )
    )
    return torch.where(u > bound, log_ei_upper, log_ei_lower)


@jax_compare(jax_criteria._log_abs_u_Phi_div_phi, '_log_abs_u_Phi_div_phi')
def _log_abs_u_Phi_div_phi(u: Tensor) -> Tensor:
    """Computes log(abs(u) * Phi(u) / phi(u)), where phi and Phi are the normal pdf
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
    a, b = get_constants_like(values=(_neg_inv_sqrt2, _log_sqrt_pi_div_2), ref=u)
    return torch.log(torch.special.erfcx(a * u) * u.abs()) + b


def phi(x: Tensor) -> Tensor:
    r"""Standard normal PDF."""
    inv_sqrt_2pi, neg_half = get_constants_like((_inv_sqrt_2pi, -0.5), x)
    return inv_sqrt_2pi * (neg_half * x.square()).exp()


def log_phi(x: Tensor) -> Tensor:
    r"""Logarithm of standard normal pdf"""
    log_sqrt_2pi, neg_half = get_constants_like((_log_sqrt_2pi, -0.5), x)
    return neg_half * x.square() - log_sqrt_2pi

def Phi(x: Tensor) -> Tensor:
    r"""Standard normal CDF."""
    half, neg_inv_sqrt_2 = get_constants_like((0.5, _neg_inv_sqrt_2), x)
    return half * torch.erfc(neg_inv_sqrt_2 * x)

"""Black-76 on the forward: price and Greeks, vectorised.

Everything in volsurf is quoted against the forward `F` of the expiry, never spot: calibrating in
log(K/S) folds dividends, funding and the 16:15-vs-16:00 non-synchronicity of US equity option
marks into the smile as skew that is not there. `forward.py` reads F (and the discount `D`) out of
put-call parity; this module prices given them.

Units and signs (stated because vendors differ and factor-of-100 slips are silent):
  price      = D · Black(F, K, T, sigma), D = discount factor to expiry (1.0 → undiscounted)
  delta      = dPV/dF (per unit of forward); `delta_spot(...)` converts to dPV/dS with S = F·D/D_q
  gamma      = d²PV/dF²
  vega       = dPV/dsigma per 1.00 of volatility (divide by 100 for "per vol point")
  theta      = dPV/dt per CALENDAR DAY (YEAR = 365), holding F and D fixed
  vanna      = d²PV/(dF dsigma), volga = d²PV/dsigma²
  sigma = 0 prices at intrinsic; T <= 0 raises (an expired option has no Black price); a NaN sigma (the
  inverter's "no information" answer, or an SVI slice with w < 0) propagates as a NaN price / vega, never as
  intrinsic or 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

YEAR = 365.0
_SQRT_2PI = np.sqrt(2.0 * np.pi)

__all__ = ["YEAR", "Greeks", "d1d2", "price", "greeks", "vega", "delta_spot"]


def _arrays(*xs):
    return np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in xs))


def _check(F, K, T):
    if np.any(F <= 0):
        raise ValueError("forward must be positive")
    if np.any(K <= 0):
        raise ValueError("strike must be positive")
    if np.any(T <= 0):
        raise ValueError("time to expiry must be positive")


def _is_call(right) -> np.ndarray:
    """"C"/"P" strings (numpy str, or a pandas object column) or booleans is_call → boolean array."""
    r = np.asarray(right)
    if r.dtype.kind == "O" and r.size and all(isinstance(v, (bool, np.bool_)) for v in r.flat):
        return r.astype(bool)          # an object column of Python booleans (str() of True is not "C")
    if r.dtype.kind in "USO":          # 'O': a pandas object column would otherwise read as all-True booleans
        return np.char.upper(r.astype(str)) == "C"
    return r.astype(bool)


def d1d2(F, K, T, sigma):
    """The two Black arguments; sigma = 0 gives ±inf (handled by the callers)."""
    F, K, T, sigma = _arrays(F, K, T, sigma)
    vt = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / vt
    return d1, d1 - vt


def price(F, K, T, sigma, right="C", D=1.0):
    """D · Black-76 price; `right` is "C"/"P" (scalar or array, or booleans is_call). Broadcasts."""
    F, K, T, sigma, D = _arrays(F, K, T, sigma, D)
    _check(F, K, T)
    if np.any(sigma < 0):
        raise ValueError("volatility must be non-negative")
    call = _is_call(right)
    d1, d2 = d1d2(F, K, T, sigma)
    with np.errstate(invalid="ignore"):
        c = F * ndtr(d1) - K * ndtr(d2)
        p = K * ndtr(-d2) - F * ndtr(-d1)
    intrinsic = np.where(call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    out = np.where(sigma == 0, intrinsic, np.where(call, c, p))  # `sigma > 0` would price a NaN vol at intrinsic
    return D * out


def vega(F, K, T, sigma, D=1.0):
    """dPV/dsigma per 1.00 vol, same for calls and puts; 0 at sigma = 0."""
    F, K, T, sigma, D = _arrays(F, K, T, sigma, D)
    _check(F, K, T)
    d1, _ = d1d2(F, K, T, sigma)
    with np.errstate(invalid="ignore"):
        v = D * F * np.exp(-0.5 * d1 * d1) / _SQRT_2PI * np.sqrt(T)
    return np.where(sigma == 0, 0.0, v)  # NaN sigma stays NaN


@dataclass(frozen=True)
class Greeks:
    """Vectorised Greek set (arrays broadcast to the input shape). Units in the module docstring."""

    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    vanna: np.ndarray
    volga: np.ndarray


def greeks(F, K, T, sigma, right="C", D=1.0) -> Greeks:
    """Analytic Greeks against the forward; sigma must be > 0 everywhere."""
    F, K, T, sigma, D = _arrays(F, K, T, sigma, D)
    _check(F, K, T)
    if np.any(sigma <= 0):
        raise ValueError("greeks need sigma > 0 everywhere")
    call = _is_call(right)
    d1, d2 = d1d2(F, K, T, sigma)
    pdf = np.exp(-0.5 * d1 * d1) / _SQRT_2PI
    sq = np.sqrt(T)
    px = price(F, K, T, sigma, right, D)
    delta = D * np.where(call, ndtr(d1), ndtr(d1) - 1.0)
    gamma = D * pdf / (F * sigma * sq)
    veg = D * F * pdf * sq
    theta = -D * F * pdf * sigma / (2.0 * sq) / YEAR
    vanna = -D * pdf * d2 / sigma
    volga = veg * d1 * d2 / sigma
    return Greeks(px, delta, gamma, veg, theta, vanna, volga)


def delta_spot(delta_forward, F, S):
    """dPV/dS from dPV/dF when F = S · carry: dF/dS = F/S (continuous carry, no discrete dividends)."""
    delta_forward, F, S = _arrays(delta_forward, F, S)
    return delta_forward * F / S

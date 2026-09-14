"""Black implied volatility, vectorised: the inverse of `black.price` on the forward.

Every quote is normalised by the discounted forward before anything else happens:
  p = price / (D F),  x = ln(F / K),  s = sigma sqrt(T)
and inverted as the OUT-of-the-money option via parity (c - p = 1 - e^{-x} in these units), so the
solver only ever sees a price that is pure time value. Corrado-Miller (1996) supplies the start,
Halley steps in s do the work, and a bracket [lo, hi] on s (updated by the sign of model - p, with a
bisection whenever a step leaves it or is not finite) keeps every step honest. Convergence is
  |delta s| <= 1e-14 s   OR   |model - p| <= 1e-15 p
and is tested BEFORE the bracket test: a converged point whose next step underflows must not be
bisected away (that ordering and an absolute price tolerance were the two bugs the research pass hit).

NaN, never a guess, where the price carries no information about sigma:
  price < intrinsic - 1e-12 F,  price > cap + 1e-12 F (cap = D F for calls, D K for puts),
  T <= 0, non-finite inputs, or discounted time value = price - D max(theta (F - K), 0) < 1e-10 F
  (the information floor: d price / d sigma ~ 1e-5 F there and the map is numerically flat).
Above the floor the solver is conditioning-limited, not tolerance-limited: |iv - sigma| is ~1e-14
for tv/F >= 1e-4, ~1e-12 for [1e-6, 1e-4), ~1e-10 for [1e-8, 1e-6) (see tests/test_iv.py).

Units and signs: sigma is annualised Black volatility (0.20 = 20 %), F and K in the same currency,
T in years, D the discount factor to expiry (1.0 = undiscounted), `right` "C"/"P" or booleans is_call.
Speed (Apple M1 laptop, Python 3.12, numpy 2.5, 1e5 random points F = 100, k ~ U(-0.5, 0.5),
sigma ~ U(0.05, 1), T ~ U(0.01, 2), half of them ITM): 0.5-0.6 microseconds per point end to end,
mean 3.7 Halley iterations, max 16 near the floor. Measured (tests/test_iv.py::test_speed_report
prints it), not asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from volsurf import black
from volsurf.quotes import Slice

__all__ = ["implied_vol", "implied_vols", "from_spot", "TIME_VALUE_FLOOR", "S_MAX"]

TIME_VALUE_FLOOR = 1e-10  # of F: below this the price says nothing about sigma
INTRINSIC_TOL = 1e-12  # of F: slack on the no-arbitrage bounds
S_MAX = 10.0  # bracket top in s = sigma sqrt(T); prices closer to the cap than model(S_MAX) are NaN
MAX_ITER = 200
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _normalised(x, s, theta):
    """OTM Black price in forward units, its vega and volga with respect to s = sigma sqrt(T).

    theta = +1 (call) / -1 (put); price = theta (N(theta d1) - e^{-x} N(theta d2)), d1,2 = x/s +- s/2.
    vega = phi(d1) for both rights; volga = vega d1 d2 / s.
    """
    d1 = x / s + 0.5 * s
    d2 = d1 - s
    model = theta * (ndtr(theta * d1) - np.exp(-x) * ndtr(theta * d2))
    vega = np.exp(-0.5 * d1 * d1) / _SQRT_2PI
    volga = vega * d1 * d2 / s
    return model, vega, volga


def _is_call(right) -> np.ndarray:
    """black._is_call with pandas object arrays of "C"/"P" accepted too."""
    r = np.asarray(right)
    return black._is_call(r.astype(str) if r.dtype.kind == "O" else r)


def _corrado_miller(p, x, theta, T):
    """Corrado-Miller (1996) start in normalised units; 0.2 sqrt(T) where it is not usable."""
    d = -np.expm1(-x) * theta  # (1 - e^{-x}) q, <= 0 for an OTM option
    a = p - 0.5 * d
    with np.errstate(invalid="ignore"):
        s0 = _SQRT_2PI / (1.0 + np.exp(-x)) * (a + np.sqrt(np.maximum(a * a - d * d / np.pi, 0.0)))
    usable = np.isfinite(s0) & (s0 > 0.0) & (s0 < S_MAX)
    return np.where(usable, s0, 0.2 * np.sqrt(T))


def _halley(p, x, theta, s):
    """Bracketed Halley iteration on the valid points; returns s. Convergence is tested before the bracket."""
    lo = np.zeros_like(s)
    hi = np.full_like(s, S_MAX)
    active = np.ones(s.shape, dtype=bool)
    for _ in range(MAX_ITER):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        si, xi, pi, ti = s[idx], x[idx], p[idx], theta[idx]
        model, vega, volga = _normalised(xi, si, ti)
        diff = model - pi
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            newton = diff / vega
            step = newton / (1.0 - 0.5 * newton * volga / vega)
        done = (np.abs(step) <= 1e-14 * si) | (np.abs(diff) <= 1e-15 * pi)
        lo_i, hi_i = lo[idx], hi[idx]
        above = diff > 0.0
        hi_i = np.where(above, si, hi_i)
        lo_i = np.where(above, lo_i, si)
        done |= hi_i - lo_i <= 1e-14 * si  # bracket exhausted: the root is pinned to the same tolerance
        new = si - step
        bad = ~np.isfinite(new) | (new <= lo_i) | (new >= hi_i)
        new = np.where(bad, 0.5 * (lo_i + hi_i), new)
        s[idx] = np.where(done, si, new)
        lo[idx], hi[idx] = lo_i, hi_i
        active[idx] = ~done
    return s


def implied_vol(price, F, K, T, right="C", D=1.0):
    """Black implied volatility of `price` = D Black(F, K, T, sigma); broadcasts over numpy arrays.

    Returns sigma (annualised) with NaN where the price is below intrinsic, above the cap, has time
    value under the information floor (1e-10 F), or T <= 0. Scalars in, numpy float out.
    """
    price, F, K, T, D = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (price, F, K, T, D)))
    if np.any(F <= 0) or np.any(K <= 0) or np.any(D <= 0):
        raise ValueError("F, K and D must be positive")
    call = np.broadcast_to(_is_call(right), price.shape)
    shape = price.shape
    price, F, K, T, D, call = (np.ravel(v) for v in (price, F, K, T, D, call))
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.log(F / K)
        p = price / (D * F)
        # OTM via parity: an ITM call becomes the put at the same strike and vice versa
        otm_call = x <= 0.0
        p_otm = np.where(call == otm_call, p, p - np.where(call, -np.expm1(-x), np.expm1(-x)))
        theta = np.where(otm_call, 1.0, -1.0)
        cap = np.where(otm_call, 1.0, np.exp(-x))
        tv_ok = D * p_otm >= TIME_VALUE_FLOOR
        cap_ok = p_otm <= cap + INTRINSIC_TOL / D
    ok = (T > 0.0) & np.isfinite(price) & np.isfinite(x) & tv_ok & cap_ok
    out = np.full(price.shape, np.nan)
    if ok.any():
        pk, xk, tk, Tk = p_otm[ok], x[ok], theta[ok], T[ok]
        model_top, _, _ = _normalised(xk, np.full_like(pk, S_MAX), tk)
        inside = pk < model_top
        s0 = _corrado_miller(pk[inside], xk[inside], tk[inside], Tk[inside])
        s = _halley(pk[inside], xk[inside], tk[inside], s0)
        res = np.full(pk.shape, np.nan)
        res[inside] = s / np.sqrt(Tk[inside])
        out[ok] = res
    return out.reshape(shape)[()]


def implied_vols(slice: Slice, F: float, D: float) -> pd.DataFrame:
    """Bid, mid and ask implied vols of every quote in a slice against its forward and discount.

    Columns: strike, right, k = ln(K/F), bid, ask, iv_bid, iv_mid, iv_ask, tv_mid (discounted time
    value of the mid, in price units). Row order follows `slice.df`. One-sided quotes (bid = 0) get
    NaN on the bid side by the floor rule; nothing is dropped here (that is `weights.select`).
    """
    df = slice.df
    K = df["strike"].to_numpy(dtype=float)
    right = df["right"].to_numpy().astype(str)
    bid = df["bid"].to_numpy(dtype=float)
    ask = df["ask"].to_numpy(dtype=float)
    mid = 0.5 * (bid + ask)
    call = _is_call(right)
    intrinsic = D * np.where(call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    out = pd.DataFrame({
        "strike": K,
        "right": right,
        "k": np.log(K / F),
        "bid": bid,
        "ask": ask,
        "iv_bid": implied_vol(bid, F, K, slice.T, right, D),
        "iv_mid": implied_vol(mid, F, K, slice.T, right, D),
        "iv_ask": implied_vol(ask, F, K, slice.T, right, D),
        "tv_mid": mid - intrinsic,
    })
    return out


def from_spot(price, S, K, T, right="C", r=0.0, q=0.0):
    """Spot-convention adapter: F = S e^{(r - q) T}, D = e^{-r T}; matches `pricers.bs.implied_vol`."""
    S, T, r, q = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (S, T, r, q)))
    F = S * np.exp((r - q) * T)
    D = np.exp(-r * T)
    return implied_vol(price, F, K, T, right, D)

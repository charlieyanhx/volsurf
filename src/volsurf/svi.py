"""Raw SVI slices: the curve, its analytic derivatives, Gatheral's g(k), the SVI-JW and SSVI maps,
and a fitter that uses Zeliade's (m, sigma) reduction as the initialiser and one constrained polish.

Units and conventions:
  k      = ln(K / F), log-moneyness against the forward of the expiry (never spot)
  w(k)   = total implied variance sigma_BS(k)^2 * T; iv(k) = sqrt(w(k) / T) per 1.00 (0.20 = 20 %)
  raw SVI (Gatheral 2004):  w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ],  b >= 0, |rho| < 1, sigma > 0
  vol points (vp) = 100 x vol; `SVIFit.rmse_vol` and `max_abs_vol` are in vp.

Invariants kept here (each pinned by a test):
  min_k w(k) = a + b sigma sqrt(1 - rho^2)                      (analytic, `min_total_variance`)
  w'(k) -> b (1 +/- rho) as k -> +/-inf                          (`wing_slopes`)
  g(k)  = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2        (Gatheral-Jacquier 2014 eq 2.1; g >= 0 for all
                                                                  k <=> no butterfly arbitrage, Lemma 2.2)
  LEE_BOUND = 2: Lee's moment formula bounds the asymptotic slope of TOTAL variance by 2, so b (1 + |rho|) <= 2.
      Gatheral 2004 and Zeliade 2009 print 4 because they bound T * d(sigma^2)/dk with sigma^2 the annualised
      variance and then drop the factor; in total-variance units the bound is 2 (GJ 2014 Remark 4.3;
      Martini-Mingone 2021). Shipping 4 lets wings twice as steep as any martingale measure allows through the fit.
  The fitter constrains w_min >= 0 and b (1 + |rho|) <= LEE_BOUND ONLY. Zeliade's 0 <= a box is not a no-arbitrage
  condition (Vogt's own arbitrage example has a < 0) and it degrades real SPY fits 18x in RMSE.
  Weights multiply the SQUARED residual: objective = sum_i weights_i (w_model(k_i) - w_i)^2.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize

__all__ = [
    "LEE_BOUND",
    "SVIParams",
    "SVISlice",
    "SVIFit",
    "to_jw",
    "from_jw",
    "from_ssvi",
    "fit_svi",
    "fit_svi_multistart",
]

LEE_BOUND = 2.0
_RHO_MAX = 1.0 - 1e-6
_SIGMA_MIN = 1e-4


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters (a, b, rho, m, sigma) of one slice; validated on construction."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self):
        if not all(np.isfinite(v) for v in self.as_tuple()):
            raise ValueError("SVI parameters must be finite")
        if self.b < 0:
            raise ValueError("b must be >= 0")
        if abs(self.rho) >= 1:
            raise ValueError("|rho| must be < 1")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.a, self.b, self.rho, self.m, self.sigma)


class SVISlice:
    """One raw-SVI slice at maturity T (years): w, w', w'', g, iv."""

    def __init__(self, params: SVIParams, T: float):
        if T <= 0:
            raise ValueError("T must be positive")
        self.params = params
        self.T = float(T)

    def _r(self, k):
        a, b, rho, m, sigma = self.params.as_tuple()
        k = np.asarray(k, dtype=float)
        return a, b, rho, m, sigma, k - m, np.sqrt((k - m) ** 2 + sigma**2)

    def w(self, k) -> np.ndarray:
        a, b, rho, m, sigma, x, R = self._r(k)
        return a + b * (rho * x + R)

    def dw(self, k) -> np.ndarray:
        a, b, rho, m, sigma, x, R = self._r(k)
        return b * (rho + x / R)

    def d2w(self, k) -> np.ndarray:
        a, b, rho, m, sigma, x, R = self._r(k)
        return b * sigma**2 / R**3

    def g(self, k) -> np.ndarray:
        """GJ2014 eq 2.1; -inf where w <= 0 (no density exists there)."""
        k = np.asarray(k, dtype=float)
        w, wp, wpp = self.w(k), self.dw(k), self.d2w(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (1.0 - k * wp / (2.0 * w)) ** 2 - (wp**2 / 4.0) * (1.0 / w + 0.25) + wpp / 2.0
        return np.where(w > 0, out, -np.inf)

    def iv(self, k) -> np.ndarray:
        """Implied vol per 1.00; NaN where w < 0."""
        w = self.w(k)
        with np.errstate(invalid="ignore"):
            return np.sqrt(w / self.T)

    def min_total_variance(self) -> float:
        a, b, rho, m, sigma = self.params.as_tuple()
        return a + b * sigma * math.sqrt(1.0 - rho**2)

    def wing_slopes(self) -> tuple[float, float]:
        """(left, right) asymptotic |slopes| of w: (b(1 - rho), b(1 + rho)); w falls at rate b(1 - rho) as k -> -inf."""
        b, rho = self.params.b, self.params.rho
        return (b * (1.0 - rho), b * (1.0 + rho))

    def __repr__(self) -> str:
        p = self.params
        return f"SVISlice(T={self.T:.4f}, a={p.a:.6f}, b={p.b:.6f}, rho={p.rho:.4f}, m={p.m:.6f}, sigma={p.sigma:.6f})"


# ---------------------------------------------------------------------------------------------
# parameter maps
# ---------------------------------------------------------------------------------------------


def to_jw(params: SVIParams, T: float) -> tuple[float, float, float, float, float]:
    """Raw -> SVI-JW (v, psi, p, c, vtilde), GJ2014 eq 3.5: v = ATM variance w(0)/T, psi = d sqrt(w)/dk at k = 0
    (= sqrt(T) x the ATM vol skew), p/c = left/right wing slopes of sqrt(w), vtilde = min variance."""
    a, b, rho, m, sigma = params.as_tuple()
    R = math.sqrt(m * m + sigma * sigma)
    w0 = a + b * (-rho * m + R)
    v = w0 / T
    sq = math.sqrt(w0)
    psi = (b / 2.0) * (-m / R + rho) / sq
    p = b * (1.0 - rho) / sq
    c = b * (1.0 + rho) / sq
    vt = (a + b * sigma * math.sqrt(1.0 - rho * rho)) / T
    return (v, psi, p, c, vt)


def from_jw(v: float, psi: float, p: float, c: float, vtilde: float, T: float) -> SVIParams:
    """SVI-JW -> raw, GJ2014 Lemma 3.2 (needs -p <= 2 psi <= c, i.e. a convex smile)."""
    wt = v * T
    sq = math.sqrt(wt)
    b = sq / 2.0 * (c + p)
    rho = 1.0 - p * sq / b
    beta = rho - 2.0 * psi * sq / b
    if abs(beta) > 1.0 + 1e-12:
        raise ValueError("JW parameters violate -p <= 2 psi <= c (beta outside [-1, 1])")
    beta = max(-1.0, min(1.0, beta))
    one_m_rho2 = math.sqrt(max(1.0 - rho * rho, 0.0))
    if abs(abs(beta) - 1.0) < 1e-14:  # alpha = 0 <=> m = 0
        m = 0.0
        sigma = (v - vtilde) * T / (b * (1.0 - one_m_rho2))
    else:
        alpha = math.copysign(math.sqrt(1.0 / beta**2 - 1.0), beta)
        m = (v - vtilde) * T / (b * (-rho + math.copysign(math.sqrt(1.0 + alpha**2), alpha) - alpha * one_m_rho2))
        sigma = alpha * m
    a = vtilde * T - b * sigma * one_m_rho2
    return SVIParams(a, b, rho, m, sigma)


def from_ssvi(theta: float, rho: float, phi: float) -> SVIParams:
    """SSVI slice w = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)) as raw SVI (Mingone 2022):
    a = theta (1 - rho^2) / 2, b = theta phi / 2, m = -rho / phi, sigma = sqrt(1 - rho^2) / phi."""
    return SVIParams(theta * (1.0 - rho * rho) / 2.0, theta * phi / 2.0, rho, -rho / phi, math.sqrt(1.0 - rho * rho) / phi)


# ---------------------------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SVIFit:
    """Result of one slice fit. `objective` = sum weights (w_model - w)^2; `rmse_vol`/`max_abs_vol` in vol points
    on sqrt(w/T); `wing_slopes` = (b(1-rho), b(1+rho)); `starts_tried` = polishes run; `method` names the path."""

    params: SVIParams
    objective: float
    rmse_vol: float
    max_abs_vol: float
    n: int
    converged: bool
    wall_time: float
    wing_slopes: tuple[float, float]
    starts_tried: int
    method: str


def _model(p, k):
    a, b, rho, m, sigma = p
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


def _jac(p, k):
    """d w_model / d(a, b, rho, m, sigma), shape (n, 5)."""
    a, b, rho, m, sigma = p
    x = k - m
    R = np.sqrt(x * x + sigma * sigma)
    return np.column_stack([np.ones_like(k), rho * x + R, b * x, -b * (rho + x / R), b * sigma / R])


def _check_inputs(k, w, T, weights):
    k = np.asarray(k, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    if k.shape != w.shape:
        raise ValueError("k and w must have the same length")
    if k.size < 5:
        raise ValueError("need at least 5 quotes to identify 5 SVI parameters")
    if not (np.all(np.isfinite(k)) and np.all(np.isfinite(w))):
        raise ValueError("k and w must be finite")
    if T <= 0:
        raise ValueError("T must be positive")
    ww = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float).ravel()
    if ww.shape != w.shape or np.any(ww < 0) or not np.all(np.isfinite(ww)):
        raise ValueError("weights must be finite, non-negative and match w")
    return k, w, float(T), ww


def _constraints():
    return [
        {"type": "ineq", "fun": lambda p: p[0] + p[1] * p[4] * math.sqrt(max(1.0 - p[2] ** 2, 0.0)),
         "jac": lambda p: _wmin_jac(p)},
        {"type": "ineq", "fun": lambda p: LEE_BOUND - p[1] * (1.0 + abs(p[2])),
         "jac": lambda p: np.array([0.0, -(1.0 + abs(p[2])), -p[1] * np.sign(p[2]), 0.0, 0.0])},
    ]


def _wmin_jac(p):
    s = math.sqrt(max(1.0 - p[2] ** 2, 1e-300))
    return np.array([1.0, p[4] * s, -p[1] * p[4] * p[2] / s, 0.0, p[1] * s])


def _feasible(p, tol=1e-9) -> bool:
    a, b, rho, m, sigma = p
    return (a + b * sigma * math.sqrt(max(1.0 - rho * rho, 0.0)) >= -tol) and (b * (1.0 + abs(rho)) <= LEE_BOUND + tol)


_BOUNDS_LO = np.array([-np.inf, 0.0, -_RHO_MAX, -np.inf, _SIGMA_MIN])
_BOUNDS_HI = np.array([np.inf, np.inf, _RHO_MAX, np.inf, np.inf])


def _grid_candidates(k, w, ww, grid, n_keep):
    """Zeliade reduction: for each (m, sigma) solve the unconstrained weighted linear LS for (a, d, c) in
    w = a + d y + c sqrt(y^2 + 1), y = (k - m)/sigma (c = b sigma, d = rho b sigma). Returns raw-parameter
    starts sorted by objective, feasible-in-shape ones (c > 0, |d| < c) first."""
    nm, ns = grid
    span = max(float(k.max() - k.min()), 1e-3)
    margin = 0.1 * span
    ms = np.linspace(k.min() - margin, k.max() + margin, nm)
    sigmas = np.geomspace(0.005, 1.0, ns)
    M, S = np.meshgrid(ms, sigmas, indexing="ij")
    y = (k[None, None, :] - M[..., None]) / S[..., None]  # (nm, ns, n)
    X = np.stack([np.ones_like(y), y, np.sqrt(y * y + 1.0)], axis=-1)  # (nm, ns, n, 3)
    Xw = X * ww[None, None, :, None]
    A = np.einsum("...ni,...nj->...ij", Xw, X)
    rhs = np.einsum("...ni,n->...i", Xw, w)
    A = A + 1e-14 * np.eye(3)
    beta = np.linalg.solve(A, rhs[..., None])[..., 0]  # (nm, ns, 3)
    resid = np.einsum("...ni,...i->...n", X, beta) - w[None, None, :]
    obj = np.einsum("n,...n->...", ww, resid * resid)
    a_, d_, c_ = beta[..., 0], beta[..., 1], beta[..., 2]
    shape_ok = (c_ > 0) & (np.abs(d_) < c_)
    order = np.lexsort((obj.ravel(), ~shape_ok.ravel()))  # shape-feasible first, then by objective
    starts = []
    for idx in order[: max(n_keep, 1)]:
        i, j = np.unravel_index(idx, obj.shape)
        c = max(float(c_[i, j]), 1e-8)
        d = float(np.clip(d_[i, j], -c * _RHO_MAX, c * _RHO_MAX))
        starts.append(np.array([float(a_[i, j]), c / S[i, j], d / c, M[i, j], S[i, j]]))
    return starts


def _polish(start, k, w, ww, T):
    """Bounded least_squares with the analytic Jacobian; SLSQP with the two constraints if that lands infeasible."""
    sw = np.sqrt(ww)

    def resid(p):
        return sw * (_model(p, k) - w)

    def jac(p):
        return sw[:, None] * _jac(p, k)

    x0 = np.clip(start, _BOUNDS_LO + 1e-12, _BOUNDS_HI - 1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ls = least_squares(resid, x0, jac=jac, bounds=(_BOUNDS_LO, _BOUNDS_HI), xtol=1e-15, ftol=1e-15, gtol=1e-15,
                           max_nfev=400)
    if _feasible(ls.x):
        return ls.x, float(np.sum(ww * (_model(ls.x, k) - w) ** 2)), bool(ls.status > 0), "grid+lsq"
    x, obj, ok = _slsqp(ls.x, k, w, ww)
    return x, obj, ok, "grid+lsq+slsqp"


def _slsqp(x0, k, w, ww):
    def obj(p):
        return float(np.sum(ww * (_model(p, k) - w) ** 2))

    def grad(p):
        return 2.0 * (ww * (_model(p, k) - w)) @ _jac(p, k)

    bounds = list(zip(_BOUNDS_LO, _BOUNDS_HI, strict=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = minimize(obj, np.clip(x0, _BOUNDS_LO, _BOUNDS_HI), jac=grad, bounds=bounds, constraints=_constraints(),
                       method="SLSQP", options={"maxiter": 500, "ftol": 1e-15})
    return res.x, float(res.fun), bool(res.success) and _feasible(res.x)


def _finish(x, obj, ok, k, w, T, n_starts, method, t0) -> SVIFit:
    params = SVIParams(*(float(v) for v in x))
    sl = SVISlice(params, T)
    with np.errstate(invalid="ignore"):
        dv = 100.0 * (np.sqrt(np.maximum(sl.w(k), 0.0) / T) - np.sqrt(np.maximum(w, 0.0) / T))
    return SVIFit(params, obj, float(np.sqrt(np.mean(dv * dv))), float(np.max(np.abs(dv))), int(k.size), ok,
                  time.perf_counter() - t0, sl.wing_slopes(), n_starts, method)


def fit_svi(k, w, T, weights=None, grid=(41, 41), polish=True, n_keep=3) -> SVIFit:
    """Fit a raw SVI slice to (k, w) at maturity T.

    Zeliade's reduction on a `grid` = (n_m, n_sigma) of (m, sigma) — m over [k_min, k_max] widened by 10 % of the
    range, sigma over geomspace(0.005, 1.0) — with an unconstrained linear inner solve for (a, d, c); the best
    `n_keep` grid points seed one constrained polish each (bounded least_squares with the analytic Jacobian, then
    SLSQP with w_min >= 0 and b(1+|rho|) <= LEE_BOUND only if the least-squares optimum is infeasible). The best
    polished result is returned. `polish=False` returns the best grid point projected into the bounds (for the
    bench). Constraints: w_min >= 0, b(1+|rho|) <= 2, b >= 0, sigma >= 1e-4, |rho| < 1; NO a >= 0 box.
    """
    t0 = time.perf_counter()
    k, w, T, ww = _check_inputs(k, w, T, weights)
    starts = _grid_candidates(k, w, ww, grid, n_keep)
    if not polish:
        x = np.clip(starts[0], _BOUNDS_LO + 1e-12, _BOUNDS_HI - 1e-12)
        return _finish(x, float(np.sum(ww * (_model(x, k) - w) ** 2)), _feasible(x), k, w, T, 0, "grid", t0)
    best = None
    for s in starts:
        x, obj, ok, method = _polish(s, k, w, ww, T)
        if np.all(np.isfinite(x)) and _feasible(x, 1e-7) and (best is None or obj < best[1]):
            best = (x, obj, ok, method)
    if best is None:
        raise RuntimeError("SVI fit failed from every grid start")
    x, obj, ok, method = best
    return _finish(x, obj, ok, k, w, T, len(starts), method, t0)


def _seed_starts(k, w):
    """surfacemcp-style 36 seeds: (b0 x rho0 x sigma0) around the minimum of w."""
    w_min, k_at_min = float(np.min(w)), float(k[int(np.argmin(w))])
    spread = max(float(np.ptp(k)), 1e-3)
    return [np.array([max(w_min * 0.5, 1e-6), b0, rho0, k_at_min, max(sig0, 1e-4)])
            for b0 in (0.05, 0.2, 0.5) for rho0 in (-0.7, -0.3, 0.0, 0.3)
            for sig0 in (0.05 * spread, 0.2 * spread, 0.5 * spread)]


def fit_svi_multistart(k, w, T, weights=None, n_starts=36) -> SVIFit:
    """Reference fitter: constrained SLSQP over the 5 raw parameters from up to `n_starts` seeds (surfacemcp-style).
    Same objective and constraints as `fit_svi`; kept for the bench and the equal-optimum test."""
    t0 = time.perf_counter()
    k, w, T, ww = _check_inputs(k, w, T, weights)
    best = None
    seeds = _seed_starts(k, w)[:n_starts]
    for s in seeds:
        try:
            x, obj, ok = _slsqp(s, k, w, ww)
        except (ValueError, FloatingPointError):
            continue
        if np.all(np.isfinite(x)) and _feasible(x, 1e-7) and (best is None or obj < best[1]):
            best = (x, obj, ok)
    if best is None:
        raise RuntimeError("SVI multistart fit failed from every seed")
    return _finish(*best, k, w, T, len(seeds), "multistart-slsqp", t0)

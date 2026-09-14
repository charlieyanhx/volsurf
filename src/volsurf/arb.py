"""Static-arbitrage checks on fitted SVI slices and on raw quotes: counted and located, never repaired.

Three checks, each with its own report:
  butterfly(slice, k_lo, k_hi)   g(k) >= 0 (GJ2014 Lemma 2.2) on a dense grid over [k_lo - margin, k_hi + margin],
                                 violations counted INSIDE [k_lo, k_hi] (the quoted range) and OUTSIDE (extrapolation)
                                 separately, PLUS the analytic wing limits of g (a grid misses wing arbitrage):
                                   raw SVI, k -> +inf:  w ~ b(1+rho) k, so k w'/(2w) -> 1/2, w'^2/(4w) -> 0, w'' -> 0 and
                                                        g -> 1/4 - b^2 (1+rho)^2 / 16 = (4 - b^2 (1+rho)^2) / 16
                                   k -> -inf:           g -> (4 - b^2 (1-rho)^2) / 16
                                 which with b = theta phi / 2 is GJ Lemma 4.2's SSVI form (16 - (theta phi)^2 (1+/-rho)^2)/64.
                                 The limit is >= 0 iff b(1 +/- rho) <= 2 (Lee's bound per wing); a negative limit is a
                                 butterfly violation at some finite |k| even when every grid point passes.
  calendar(slices)               total variance non-decreasing in T at every grid k for each adjacent pair (sorted by T);
                                 `crossings` counts grid points, `worst_gap` = min(w_later - w_earlier) (negative = crossing).
  quote_level(pairs, F, D, T)    call-price convexity by second divided differences on consecutive quoted strikes, once on
                                 mids (`butterfly_raw_mid`) and once on the most-convex-favourable prices inside the bands
                                 (ask, bid, ask for the outer, middle, outer strike: `butterfly_within_band`). Raw-mid
                                 violations on real chains are 15-29 % of triplets and mean nothing; the within-band count
                                 is the one that is evidence (~0 on SPY). Puts are checked the same way; both sides sum.

Sign conventions: g < 0, gap < 0, second difference < 0 are the violations; `worst_*` is the most negative value seen,
reported even when it is positive (then `ok` is True). Tolerance `tol` (default -1e-10) absorbs round-off.
`repair_jw` (GJ 2014 section 5.1) is opt-in: the fitter never calls it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .svi import LEE_BOUND, SVIParams, SVISlice, from_jw, to_jw

__all__ = [
    "Evidence",
    "ButterflyReport",
    "CalendarReport",
    "QuoteLevelReport",
    "ArbitrageReport",
    "butterfly",
    "asymptotes",
    "calendar",
    "quote_level",
    "check_arbitrage",
    "repair_jw",
]

TOL = -1e-10


class Evidence(str, Enum):
    analytic = "analytic"
    numerical_scan = "numerical_scan"


@dataclass(frozen=True)
class ButterflyReport:
    ok: bool
    worst_g: float
    worst_k: float
    n_inside: int
    n_outside: int
    k_range_checked: tuple[float, float]
    asymptote_left: float
    asymptote_right: float
    evidence: Evidence
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalendarReport:
    ok: bool
    crossings: int
    worst_gap: float
    worst_k: float | None
    pair: tuple[float, float] | None
    k_range_checked: tuple[float, float] | None


@dataclass(frozen=True)
class QuoteLevelReport:
    butterfly_raw_mid: int
    butterfly_within_band: int
    calendar_raw: int | None
    n_triplets: int


@dataclass(frozen=True)
class ArbitrageReport:
    ok: bool
    butterfly: list[ButterflyReport]
    calendar: CalendarReport
    notes: tuple[str, ...] = field(default_factory=tuple)


def asymptotes(slice_: SVISlice) -> tuple[float, float]:
    """(lim_{k->-inf} g, lim_{k->+inf} g) = ((4 - b^2(1-rho)^2)/16, (4 - b^2(1+rho)^2)/16)."""
    left, right = slice_.wing_slopes()
    return ((4.0 - left * left) / 16.0, (4.0 - right * right) / 16.0)


def butterfly(slice_: SVISlice, k_lo: float, k_hi: float, margin: float | None = None, n: int = 2001,
              tol: float = TOL) -> ButterflyReport:
    """g(k) on [k_lo - margin, k_hi + margin] (margin default max(0.5, 2 sigma_atm sqrt(T))) plus the wing limits."""
    if k_hi < k_lo:
        raise ValueError("k_lo must be <= k_hi")
    if margin is None:
        w0 = float(slice_.w(0.0))
        sigma_atm = np.sqrt(max(w0, 0.0) / slice_.T)
        margin = max(0.5, 2.0 * sigma_atm * np.sqrt(slice_.T))
    lo, hi = k_lo - margin, k_hi + margin
    k = np.linspace(lo, hi, n)
    g = slice_.g(k)
    bad = g < tol
    inside = (k >= k_lo) & (k <= k_hi)
    n_in, n_out = int(np.sum(bad & inside)), int(np.sum(bad & ~inside))
    worst_g, worst_k = _refine_min(slice_, k, g)
    a_left, a_right = asymptotes(slice_)
    notes = []
    if n_in:
        notes.append(f"g < 0 at {n_in} grid points inside the quoted range [{k_lo:.4f}, {k_hi:.4f}]")
    if n_out:
        notes.append(f"g < 0 at {n_out} grid points outside the quoted range (extrapolation arbitrage)")
    if a_left < tol:
        notes.append(f"left-wing asymptote of g is {a_left:.4f} < 0: b(1-rho) = {slice_.wing_slopes()[0]:.4f} > {LEE_BOUND}")
    if a_right < tol:
        notes.append(f"right-wing asymptote of g is {a_right:.4f} < 0: b(1+rho) = {slice_.wing_slopes()[1]:.4f} > {LEE_BOUND}")
    if slice_.min_total_variance() < 0:
        notes.append(f"minimum total variance {slice_.min_total_variance():.3e} < 0")
    grid_ok = n_in == 0 and n_out == 0
    asym_ok = a_left >= tol and a_right >= tol
    evidence = Evidence.analytic if (grid_ok and not asym_ok) else Evidence.numerical_scan
    return ButterflyReport(grid_ok and asym_ok and slice_.min_total_variance() >= 0, worst_g, worst_k,
                           n_in, n_out, (float(lo), float(hi)), float(a_left), float(a_right), evidence, tuple(notes))


def _refine_min(slice_: SVISlice, k: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """Bounded scalar minimisation of g between the grid neighbours of the grid argmin (locates worst_k to ~1e-8)."""
    i = int(np.argmin(g))
    if not np.isfinite(g[i]) or i == 0 or i == k.size - 1:
        return float(g[i]), float(k[i])
    res = minimize_scalar(lambda x: float(slice_.g(x)), bounds=(k[i - 1], k[i + 1]), method="bounded",
                          options={"xatol": 1e-10})
    if res.fun <= g[i]:
        return float(res.fun), float(res.x)
    return float(g[i]), float(k[i])


def calendar(slices: list[tuple[float, SVISlice]], k_grid=None, tol: float = TOL) -> CalendarReport:
    """w(k, T2) >= w(k, T1) at every grid k for every adjacent pair after sorting by T. Fewer than two slices: ok."""
    if len(slices) < 2:
        return CalendarReport(True, 0, 0.0, None, None, None)
    for T, s in slices:
        if T <= 0:
            raise ValueError("T must be positive")
        if abs(T - s.T) > 1e-12 * max(1.0, T):
            raise ValueError("listed T disagrees with the slice's T")
    ordered = sorted(slices, key=lambda ts: ts[0])
    k = np.linspace(-1.0, 1.0, 401) if k_grid is None else np.asarray(k_grid, dtype=float)
    crossings, worst, worst_k, pair = 0, np.inf, None, None
    for (T1, s1), (T2, s2) in zip(ordered[:-1], ordered[1:], strict=True):
        if T2 <= T1:
            raise ValueError("two slices share a maturity")
        gap = s2.w(k) - s1.w(k)
        crossings += int(np.sum(gap < tol))
        i = int(np.argmin(gap))
        if gap[i] < worst:
            worst, worst_k, pair = float(gap[i]), float(k[i]), (float(T1), float(T2))
    return CalendarReport(crossings == 0, crossings, worst, worst_k, pair, (float(k.min()), float(k.max())))


def _second_differences(K: np.ndarray, lo: np.ndarray, mid: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Divided second differences on consecutive strikes with the outer prices `lo`/`hi` and the middle price `mid`."""
    K0, K1, K2 = K[:-2], K[1:-1], K[2:]
    return (hi[2:] - mid[1:-1]) / (K2 - K1) - (mid[1:-1] - lo[:-2]) / (K1 - K0)


def _count_convexity(K, bid, ask, tol) -> tuple[int, int, int]:
    mid = 0.5 * (bid + ask)
    raw = _second_differences(K, mid, mid, mid)
    band = _second_differences(K, ask, bid, ask)
    return int(np.sum(raw < tol)), int(np.sum(band < tol)), int(raw.size)


def quote_level(pairs: pd.DataFrame, F: float, D: float, T: float, tol: float = TOL) -> QuoteLevelReport:
    """Convexity of call and put prices across consecutive quoted strikes of one expiry, on mids and within the bands.

    `pairs` is `Slice.pairs()` (strike, call_bid, call_ask, put_bid, put_ask). F, D, T are carried for the report's
    provenance (the convexity check itself is model-free). `calendar_raw` is None: one expiry has no calendar."""
    if not (F > 0 and D > 0 and T > 0):
        raise ValueError("F, D, T must be positive")
    df = pairs.sort_values("strike").reset_index(drop=True)
    if len(df) < 3:
        return QuoteLevelReport(0, 0, None, 0)
    K = df["strike"].to_numpy(float)
    if np.any(np.diff(K) <= 0):
        raise ValueError("strikes must be strictly increasing (duplicate strikes in pairs)")
    raw_c, band_c, n_c = _count_convexity(K, df["call_bid"].to_numpy(float), df["call_ask"].to_numpy(float), tol)
    raw_p, band_p, n_p = _count_convexity(K, df["put_bid"].to_numpy(float), df["put_ask"].to_numpy(float), tol)
    return QuoteLevelReport(raw_c + raw_p, band_c + band_p, None, n_c + n_p)


def check_arbitrage(fits: list[tuple[float, SVISlice, float, float]], margin: float | None = None, n: int = 2001,
                    k_grid=None) -> ArbitrageReport:
    """Butterfly per slice on its declared (k_lo, k_hi) plus calendar across slices."""
    reports = [butterfly(s, k_lo, k_hi, margin=margin, n=n) for _, s, k_lo, k_hi in fits]
    cal = calendar([(T, s) for T, s, _, _ in fits], k_grid=k_grid)
    notes: list[str] = []
    for (T, _, _, _), r in zip(fits, reports, strict=True):
        notes.extend(f"T={T:.4f}: {msg}" for msg in r.notes)
    if not cal.ok:
        notes.append(f"calendar: {cal.crossings} crossings, worst gap {cal.worst_gap:.3e} at k={cal.worst_k:.4f} "
                     f"between T={cal.pair[0]:.4f} and T={cal.pair[1]:.4f}")
    if len(fits) < 2:
        notes.append("single slice: calendar condition not applicable")
    return ArbitrageReport(all(r.ok for r in reports) and cal.ok, reports, cal, tuple(notes))


def repair_jw(params: SVIParams, T: float) -> SVIParams:
    """GJ 2014 section 5.1: keep (v, psi, p) and set c' = p + 2 psi, vtilde' = v 4 p c' / (p + c')^2, which is
    guaranteed free of butterfly arbitrage (Lemma 4.1 via the SSVI form). OPT-IN: the fitter never calls this; a
    repaired slice is a different smile (it moves the call wing and the minimum variance), so report the violation
    and let the caller decide. GJ's Example 5.1 'optimal' (c*, vtilde*) = (0.8564763, 0.0116249) comes from a
    penalised price-distance optimisation, not from this closed form (which gives (0.3493158, 0.0154818))."""
    v, psi, p, _, _ = to_jw(params, T)
    c_new = p + 2.0 * psi
    if c_new <= 0:
        raise ValueError("p + 2 psi <= 0: the repair has no positive call wing")
    vt_new = v * 4.0 * p * c_new / (p + c_new) ** 2
    return from_jw(v, psi, p, c_new, vt_new, T)

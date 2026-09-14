"""One surface = one raw-SVI slice per (root, expiry) of a chain, fitted against that expiry's parity forward, with
the static-arbitrage checks and the fit-quality numbers carried on every slice and never repaired.

Pipeline per slice (`fit_slice`), each stage's decisions kept on the `SliceFit`:
  forward.fit_forward   -> F, D (mode auto: parity line for European, discount pinned from `rate` for American)
  iv.implied_vols       -> iv_bid / iv_mid / iv_ask per quote against (F, D), k = ln(K/F)
  weights.select        -> mask + ledger (two-sided, bid >= 2 ticks, finite iv, OTM only, relative spread, |k|)
  weights.*_weights     -> 1/(iv_ask - iv_bid)^2 (default), vega, or unit; rows with a non-finite weight are dropped
                           and counted
  svi.fit_svi           -> params on w = iv_mid^2 T with the weights in the objective
  arb.butterfly         -> g(k) on the SELECTED k-range [k_min, k_max] plus margin and the wing asymptotes
  arb.quote_level       -> raw-mid and within-band convexity counts on the slice's two-sided pairs (model-free)
  coverage_pct          -> share of the SELECTED quotes whose model price D Black(F, K, T, iv_model(k)) lies in
                           [bid, ask] (1e-9 slack for round-off); the number the README quotes as "inside bid/ask"
A slice is SKIPPED (recorded with its reason on `Surface.skipped`, never silently) when the forward fit raises or fewer
than `min_quotes` quotes survive selection. `arb.calendar` then runs across adjacent fitted slices on the k-range
both of them quote (a calendar crossing at a k neither slice has a quote at is extrapolation, reported separately by
`Surface.arbitrage()`, which uses the wide [-1, 1] grid).

Units: vols per 1.00 (0.20 = 20 %), `rmse_vp` / `max_abs_vp` in vol points (100 x vol), k = ln(K/F), T in years by
the chain's rule, coverage in percent, forward residuals in price units. `skew(expiry, delta)` = iv at the strike
whose undiscounted Black call delta N(d1) is +delta minus iv at the strike whose put delta N(d1) - 1 is -delta
(put minus call: positive for the usual equity smile), both found by root-finding on the fitted slice.

Nothing here reads spot, the source's IV column, or an external forward: F and D come from the quotes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

from . import black
from . import weights as weights_mod
from .arb import (
    ArbitrageReport,
    ButterflyReport,
    CalendarReport,
    QuoteLevelReport,
    butterfly,
    calendar,
    check_arbitrage,
    quote_level,
)
from .forward import ForwardFit, fit_forward
from .iv import implied_vols
from .quotes import Chain, Ledger, Slice
from .svi import SVIFit, SVIParams, SVISlice, fit_svi

__all__ = ["K_BUCKETS", "SliceFit", "Surface", "fit_slice", "fit_surface", "coverage_by_k"]

K_BUCKETS = ((0.0, 0.05), (0.05, 0.15), (0.15, 0.35), (0.35, math.inf))
_INSIDE_TOL = 1e-9  # price units of slack on the band edges (round-off, not a fill assumption)
WEIGHTINGS = ("spread", "vega", "unit")


@dataclass(frozen=True)
class SliceFit:
    """One fitted (root, expiry). `ivs` has one row per quote of the slice (iv.implied_vols); `mask` marks the rows
    the fit used; `weights` and `inside` are over the selected rows; `k_range` is the selected [k_min, k_max]."""

    slice: Slice
    forward: ForwardFit
    ivs: pd.DataFrame
    mask: np.ndarray
    weights: np.ndarray
    svi: SVIFit
    butterfly: ButterflyReport
    quote_level: QuoteLevelReport
    coverage_pct: float
    inside: np.ndarray
    ledger: Ledger
    k_range: tuple[float, float]
    weighting: str

    @property
    def T(self) -> float:
        return self.slice.T

    @property
    def root(self) -> str:
        return self.slice.root

    @property
    def expiry(self) -> pd.Timestamp:
        return self.slice.expiry

    @property
    def params(self) -> SVIParams:
        return self.svi.params

    @property
    def svi_slice(self) -> SVISlice:
        return SVISlice(self.svi.params, self.slice.T)

    @property
    def n(self) -> int:
        return int(self.mask.sum())

    def iv(self, k) -> np.ndarray:
        return self.svi_slice.iv(k)

    def selected(self) -> pd.DataFrame:
        return self.ivs[self.mask].reset_index(drop=True)


def _weights_for(kind: str, sel: pd.DataFrame, F: float, T: float) -> np.ndarray:
    if kind == "spread":
        return weights_mod.spread_weights(sel["iv_bid"].to_numpy(float), sel["iv_ask"].to_numpy(float))
    if kind == "vega":
        return weights_mod.vega_weights(F, sel["strike"].to_numpy(float), T, sel["iv_mid"].to_numpy(float))
    if kind == "unit":
        return weights_mod.unit_weights(len(sel))
    raise ValueError(f"weighting must be one of {WEIGHTINGS}")


def _inside_band(sel: pd.DataFrame, sl: SVISlice, F: float, D: float, T: float) -> np.ndarray:
    K = sel["strike"].to_numpy(float)
    model = black.price(F, K, T, sl.iv(np.log(K / F)), sel["right"].to_numpy().astype(str), D)
    return (model >= sel["bid"].to_numpy(float) - _INSIDE_TOL) & (model <= sel["ask"].to_numpy(float) + _INSIDE_TOL)


def fit_slice(slice: Slice, rate: float | None = None, mode: str = "auto", weighting: str = "spread",
              min_quotes: int = 8, **select_kw) -> SliceFit:
    """Forward -> IVs -> selection -> weights -> SVI -> butterfly / quote-level / coverage for one slice.
    Raises ValueError (with the reason) when the forward cannot be fitted or too few quotes survive."""
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    fwd = fit_forward(slice, rate=rate, mode=mode)
    F, D, T = fwd.forward, fwd.discount, slice.T
    ivs = implied_vols(slice, F, D)
    mask, ledger = weights_mod.select(ivs, T, **select_kw)
    w_sel = _weights_for(weighting, ivs[mask], F, T)
    finite = np.isfinite(w_sel) & (w_sel > 0)
    n_in = int(mask.sum())
    idx = np.flatnonzero(mask)[~finite]
    mask = mask.copy()
    mask[idx] = False
    ledger.record(f"finite {weighting} weight", n_in, int(mask.sum()))
    w_sel = w_sel[finite]
    if int(mask.sum()) < min_quotes:
        raise ValueError(f"{slice.label()}: {int(mask.sum())} quotes after selection < min_quotes={min_quotes} "
                         f"({ledger})")
    sel = ivs[mask]
    k = sel["k"].to_numpy(float)
    w = sel["iv_mid"].to_numpy(float) ** 2 * T
    fit = fit_svi(k, w, T, weights=w_sel)
    sl = SVISlice(fit.params, T)
    k_lo, k_hi = float(k.min()), float(k.max())
    inside = _inside_band(sel, sl, F, D, T)
    return SliceFit(slice, fwd, ivs, mask, w_sel, fit, butterfly(sl, k_lo, k_hi), quote_level(slice.pairs(), F, D, T),
                    100.0 * float(inside.mean()), inside, ledger, (k_lo, k_hi), weighting)


def coverage_by_k(fits: list[SliceFit] | tuple[SliceFit, ...]) -> pd.DataFrame:
    """Selected quotes pooled over the fitted slices, bucketed by |k|: columns bucket, n, inside, coverage_pct."""
    if fits:
        k = np.concatenate([f.selected()["k"].to_numpy(float) for f in fits])
        inside = np.concatenate([f.inside for f in fits])
    else:
        k, inside = np.zeros(0), np.zeros(0, dtype=bool)
    rows = []
    for lo, hi in K_BUCKETS:
        m = (np.abs(k) >= lo) & (np.abs(k) < hi)
        n = int(m.sum())
        label = f"[{lo:.2f}, {hi:.2f})" if math.isfinite(hi) else f"[{lo:.2f}, inf)"
        rows.append({"bucket": label, "n": n, "inside": int(inside[m].sum()),
                     "coverage_pct": 100.0 * float(inside[m].mean()) if n else float("nan")})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Surface:
    """Fitted slices in expiry order plus the calendar check across them and the list of skipped (label, reason)."""

    symbol: str
    quote_date: date
    fits: tuple[SliceFit, ...]
    calendar: CalendarReport
    skipped: tuple[tuple[str, str], ...] = ()
    discount_source: str = ""
    wall_time: float = 0.0
    source: str = ""
    exercise: str = ""
    quotes_in: int = 0
    chain_ledger: Ledger = field(default_factory=Ledger, compare=False)

    def expiries(self) -> list[tuple[str, pd.Timestamp]]:
        return [(f.root, f.expiry) for f in self.fits]

    def fit(self, expiry, root: str | None = None) -> SliceFit:
        e = pd.Timestamp(expiry)
        hits = [f for f in self.fits if f.expiry == e and (root is None or f.root == root)]
        if not hits:
            raise KeyError(f"no fitted slice for {e:%Y-%m-%d}" + (f" root {root}" if root else ""))
        if len(hits) > 1:
            raise KeyError(f"{len(hits)} roots fitted on {e:%Y-%m-%d} ({[f.root for f in hits]}): pass root=")
        return hits[0]

    def implied_vol(self, expiry, K, root: str | None = None) -> np.ndarray:
        f = self.fit(expiry, root)
        return f.iv(np.log(np.asarray(K, dtype=float) / f.forward.forward))

    def atm_vol(self, expiry, root: str | None = None) -> float:
        return float(self.fit(expiry, root).iv(0.0))

    def skew(self, expiry, delta: float = 0.25, root: str | None = None) -> dict:
        """Put-minus-call iv at the strikes of undiscounted Black delta -delta (put) and +delta (call), by root-finding
        N(d1(k)) = target on the fitted slice. Keys: expiry, root, T, k_put, k_call, strike_put, strike_call,
        iv_put, iv_call, skew."""
        if not 0.0 < delta < 0.5:
            raise ValueError("delta must be in (0, 0.5)")
        f = self.fit(expiry, root)
        sl, F = f.svi_slice, f.forward.forward
        k_call = _k_at_delta(sl, delta)  # N(d1) = delta
        k_put = _k_at_delta(sl, 1.0 - delta)  # N(d1) - 1 = -delta
        iv_p, iv_c = float(sl.iv(k_put)), float(sl.iv(k_call))
        return {"expiry": f.expiry, "root": f.root, "T": f.T, "k_put": k_put, "k_call": k_call,
                "strike_put": F * math.exp(k_put), "strike_call": F * math.exp(k_call), "iv_put": iv_p,
                "iv_call": iv_c, "skew": iv_p - iv_c}

    def term_structure(self) -> pd.DataFrame:
        rows = []
        for f in self.fits:
            try:
                sk = self.skew(f.expiry, 0.25, f.root)["skew"]
            except ValueError:
                sk = float("nan")
            rows.append({"root": f.root, "expiry": f.expiry, "T": f.T, "forward": f.forward.forward,
                         "discount": f.forward.discount, "implied_rate": f.forward.implied_rate, "mode": f.forward.mode,
                         "forward_rms": f.forward.residual_rms, "n_quotes": f.n, "atm_vol": float(f.iv(0.0)),
                         "skew_25d": sk, "rmse_vp": f.svi.rmse_vol, "max_abs_vp": f.svi.max_abs_vol,
                         "coverage_pct": f.coverage_pct, "g_min": f.butterfly.worst_g, "g_neg_inside": f.butterfly.n_inside,
                         "g_neg_outside": f.butterfly.n_outside, "asymptote_left": f.butterfly.asymptote_left,
                         "asymptote_right": f.butterfly.asymptote_right, "within_band_violations":
                         f.quote_level.butterfly_within_band, "raw_mid_violations": f.quote_level.butterfly_raw_mid})
        return pd.DataFrame(rows)

    def arbitrage(self) -> ArbitrageReport:
        return check_arbitrage([(f.T, f.svi_slice, *f.k_range) for f in self.fits])

    def report(self) -> dict:
        """The dict `volsurf fit` / `volsurf report --data` print; every number labelled by its basis in the key."""
        fits = self.fits
        rmse = np.array([f.svi.rmse_vol for f in fits])
        cov = np.array([f.coverage_pct for f in fits])
        fres = np.array([f.forward.residual_rms for f in fits])
        buckets = coverage_by_k(fits)
        n_sel = int(sum(f.n for f in fits))
        return {
            "symbol": self.symbol, "quote_date": str(self.quote_date), "source": self.source, "exercise": self.exercise,
            "expiries_fitted": len(fits), "expiries_skipped": len(self.skipped), "skipped": list(self.skipped),
            "quotes_in": int(self.quotes_in), "quotes_selected": n_sel,
            "quotes_dropped": {rule: int(n) for rule, n in _pool_ledgers(self, fits).items()},
            "rmse_vp_median": float(np.median(rmse)) if len(fits) else float("nan"),
            "rmse_vp_worst": float(rmse.max()) if len(fits) else float("nan"),
            "coverage_pct_median": float(np.median(cov)) if len(fits) else float("nan"),
            "coverage_pct_worst": float(cov.min()) if len(fits) else float("nan"),
            "coverage_by_k": {r.bucket: (int(r.n), float(r.coverage_pct)) for r in buckets.itertuples(index=False)},
            "slices_g_neg_inside": int(sum(f.butterfly.n_inside > 0 for f in fits)),
            "slices_g_neg_outside": int(sum((f.butterfly.n_outside > 0 or f.butterfly.asymptote_left < 0
                                              or f.butterfly.asymptote_right < 0) and f.butterfly.n_inside == 0
                                             for f in fits)),
            "within_band_violations": int(sum(f.quote_level.butterfly_within_band for f in fits)),
            "raw_mid_violations": int(sum(f.quote_level.butterfly_raw_mid for f in fits)),
            "n_triplets": int(sum(f.quote_level.n_triplets for f in fits)),
            "calendar_crossings": int(self.calendar.crossings), "calendar_worst_gap": float(self.calendar.worst_gap),
            "forward_residual_rms_median": float(np.median(fres)) if len(fits) else float("nan"),
            "forward_residual_rms_worst": float(fres.max()) if len(fits) else float("nan"),
            "discount_source": self.discount_source, "wall_time": float(self.wall_time),
        }


def _pool_ledgers(surface: Surface, fits) -> dict[str, int]:
    """Rows dropped per rule, summed over the chain ledger and the per-slice selection ledgers."""
    out: dict[str, int] = {}
    for rule, n in surface.chain_ledger.dropped().items():
        out[rule] = out.get(rule, 0) + n
    for f in fits:
        for rule, n in f.ledger.dropped().items():
            out[rule] = out.get(rule, 0) + n
    return out


def _k_at_delta(sl: SVISlice, target: float, k_max: float = 3.0, n: int = 601) -> float:
    """k with N(d1(k)) = target, d1 = (-k + w(k)/2) / sqrt(w(k)); the sign change nearest k = 0 is refined by Brent."""

    def f(k):
        w = float(sl.w(k))
        if w <= 0:
            return math.nan
        return float(ndtr((-k + 0.5 * w) / math.sqrt(w))) - target

    grid = np.linspace(-k_max, k_max, n)
    vals = np.array([f(k) for k in grid])
    ok = np.isfinite(vals)
    sign = np.sign(vals)
    change = np.flatnonzero(ok[:-1] & ok[1:] & (sign[:-1] * sign[1:] <= 0))
    if change.size == 0:
        raise ValueError(f"no strike with N(d1) = {target:.3f} on the fitted slice in |k| <= {k_max}")
    i = change[np.argmin(np.abs(grid[change]))]
    if vals[i] == 0.0:
        return float(grid[i])
    return float(brentq(f, grid[i], grid[i + 1], xtol=1e-14))


def _calendar_on_quoted_ranges(fits: list[SliceFit], n: int = 401) -> CalendarReport:
    """Calendar check per adjacent pair (by T) on the k-range BOTH slices quote (the intersection of their selected
    ranges), aggregated: crossings summed, worst gap kept with its k and pair; k_range_checked = the union covered.
    A pair whose quoted ranges do not overlap is not checked (nothing to compare without extrapolating)."""
    ordered = sorted(fits, key=lambda f: f.T)
    if len(ordered) < 2:
        return CalendarReport(True, 0, 0.0, None, None, None)
    crossings, worst, worst_k, pair, lo_all, hi_all = 0, math.inf, None, None, math.inf, -math.inf
    for f1, f2 in zip(ordered[:-1], ordered[1:], strict=True):
        lo, hi = max(f1.k_range[0], f2.k_range[0]), min(f1.k_range[1], f2.k_range[1])
        if hi <= lo:
            continue
        r = calendar([(f1.T, f1.svi_slice), (f2.T, f2.svi_slice)], k_grid=np.linspace(lo, hi, n))
        crossings += r.crossings
        lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
        if r.worst_gap < worst:
            worst, worst_k, pair = r.worst_gap, r.worst_k, r.pair
    if not math.isfinite(lo_all):
        return CalendarReport(True, 0, 0.0, None, None, None)
    return CalendarReport(crossings == 0, crossings, float(worst), worst_k, pair, (float(lo_all), float(hi_all)))


def fit_surface(chain: Chain, rate: float | None = None, mode: str = "auto", weighting: str = "spread",
                min_quotes: int = 8, calendar_k_grid=None, **select_kw) -> Surface:
    """Fit every slice of the chain (skips recorded, not raised) and run the calendar check across the fits.
    `select_kw` go to `weights.select` (otm_only, min_bid_ticks, tick, max_rel_spread, k_max). The calendar check
    runs per adjacent pair on the k-range both slices quote unless `calendar_k_grid` is given (then one grid for all;
    `Surface.arbitrage()` is the wide [-1, 1] version)."""
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    t0 = time.perf_counter()
    fits: list[SliceFit] = []
    skipped: list[tuple[str, str]] = []
    for sl in chain.slices():
        try:
            fits.append(fit_slice(sl, rate=rate, mode=mode, weighting=weighting, min_quotes=min_quotes, **select_kw))
        except (ValueError, RuntimeError) as e:
            skipped.append((sl.label(), str(e)))
    cal = _calendar_on_quoted_ranges(fits) if calendar_k_grid is None else \
        calendar([(f.T, f.svi_slice) for f in fits], k_grid=calendar_k_grid)
    modes = sorted({f.forward.mode for f in fits})
    if rate is not None and "fixed_discount" in modes:
        src = f"discount pinned from rate {rate:g} (fixed_discount)" + (
            f"; parity line on {modes}" if len(modes) > 1 else "")
    else:
        src = "discount fitted on the parity line" if fits else "no slice fitted"
    return Surface(chain.symbol, chain.quote_date, tuple(fits), cal, tuple(skipped), src, time.perf_counter() - t0,
                   chain.source, chain.exercise, len(chain.df), chain.ledger)

"""Forward and discount of an expiry from put-call parity on the quoted mids.

Model, per two-sided strike pair (bid > 0 on both legs):  C - P = D (F - K),  with C, P the mids.
Two estimators, the mode recorded on the result:
  parity_line     joint WLS of y = C - P on K: D = -slope, F = -intercept / slope. European only
                  (SPX, XSP, VIX): on American chains ITM puts carry early-exercise premium and the
                  free line returns D > 1 (implied r of -1 % to -54 % on SPY when r = 5 %).
                  Refused for exercise == "american" unless force=True; rejected when
                  D is outside (0.5, 1.02] or fewer than min_pairs pairs survive.
  fixed_discount  D = exp(-rate T) pinned, F = sum w (C - P + D K) / (D sum w). The American
                  default; also fine for European chains when a rate is known.
Weights w = 1 / (hw_C^2 + hw_P^2) with hw the half-widths floored at one tick (0.01), because the
mid's error variance scales with the width squared and a locked quote must not get infinite weight.
Pairs are restricted to |K / K_atm - 1| < window with K_atm the strike of smallest |C - P|, refitted
once with the window recentred on the first F; the window doubles (noted) while it holds fewer than
min_pairs pairs. Then a Huber-IRLS pass (c = 3 on the MAD scale, at most 10 rounds) when `huber`,
which is what removes a stale far quote the window let through. Exact data has MAD at rounding level
(below a thousandth of a tick) and the pass is a no-op.

Units: F, K, C, P in price units; T in years (from the slice); implied_rate = -ln(D) / T, continuous,
per year. residual_rms / residual_max are unweighted, in price units, over the pairs used.
Spot is never read: on end-of-day US chains the 16:15 option marks and the 16:00 stock close are not
synchronous (SPY 2020-03-16: parity F / S - 1 = +1.5 %), and VIX options settle on the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volsurf.quotes import Chain, Slice

__all__ = ["ForwardFit", "fit_forward", "forward_table", "TICK", "HUBER_C", "DISCOUNT_BAND"]

TICK = 0.01
HUBER_C = 3.0
HUBER_ROUNDS = 10
MAD_TO_SIGMA = 1.4826
_EXACT_SCALE = 1e-3 * TICK  # a MAD below a thousandth of a tick is rounding, not a residual
DISCOUNT_BAND = (0.5, 1.02)
MODES = ("auto", "parity_line", "fixed_discount")
_WIDEN_STEPS = 8


@dataclass(frozen=True)
class ForwardFit:
    """One expiry's parity fit. `notes` records every decision (mode, window, pairs, Huber) as text."""

    forward: float
    discount: float
    mode: str
    n_pairs: int
    window: float | None
    residual_rms: float
    residual_max: float
    implied_rate: float
    notes: tuple[str, ...] = ()


def _resolve_mode(slice: Slice, rate, mode: str, force: bool) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    american = slice.exercise == "american"
    if mode == "auto":
        if american and rate is None:
            raise ValueError("American chain needs a rate: the discount is not identifiable from parity")
        mode = "fixed_discount" if american else "parity_line"
    if mode == "parity_line" and american and not force:
        raise ValueError(
            f"parity_line refused on {slice.label()}: exercise is 'american' (ITM puts carry early-exercise "
            "premium, the free discount comes out > 1); use fixed_discount with a rate, or force=True"
        )
    if mode == "fixed_discount" and rate is None:
        raise ValueError("fixed_discount needs a rate")
    return mode


def _two_sided_pairs(slice: Slice) -> tuple[pd.DataFrame, int]:
    """Pairs with bid > 0 on both legs, and the number of pairs quoted at all."""
    pairs = slice.pairs()
    keep = (pairs["call_bid"] > 0) & (pairs["put_bid"] > 0)
    return pairs[keep].reset_index(drop=True), len(pairs)


def _weights(pairs: pd.DataFrame) -> np.ndarray:
    hw_c = np.maximum(0.5 * (pairs["call_ask"] - pairs["call_bid"]).to_numpy(), TICK)
    hw_p = np.maximum(0.5 * (pairs["put_ask"] - pairs["put_bid"]).to_numpy(), TICK)
    return 1.0 / (hw_c**2 + hw_p**2)


def _solve(K, y, w, mode: str, D_fixed: float | None) -> tuple[float, float]:
    """(F, D) from weighted pairs; the closed forms in the module docstring."""
    sw = w.sum()
    if mode == "fixed_discount":
        D = float(D_fixed)
        F = float(np.sum(w * (y + D * K)) / (D * sw))
        return F, D
    Kbar = np.sum(w * K) / sw
    ybar = np.sum(w * y) / sw
    sxx = np.sum(w * (K - Kbar) ** 2)
    if sxx <= 0:
        raise ValueError("parity_line needs at least two distinct strikes")
    beta = np.sum(w * (K - Kbar) * (y - ybar)) / sxx
    D = float(-beta)
    if not (DISCOUNT_BAND[0] < D <= DISCOUNT_BAND[1]):
        raise ValueError(f"parity_line rejected: fitted discount {D:.4f} outside ({DISCOUNT_BAND[0]}, {DISCOUNT_BAND[1]}]")
    return float(Kbar + ybar / D), D


def _select_window(K: np.ndarray, centre: float, window: float | None, min_pairs: int, notes: list[str]) -> np.ndarray:
    """Mask of pairs with |K / centre - 1| < window, doubling the window until min_pairs are inside."""
    if window is None:
        return np.ones(K.shape, dtype=bool)
    rel = np.abs(K / centre - 1.0)
    wdw = window
    for _ in range(_WIDEN_STEPS):
        mask = rel < wdw
        if mask.sum() >= min_pairs or mask.all():
            break
        wdw *= 2.0
    if wdw != window:
        notes.append(f"window widened {window:g} -> {wdw:g} to reach {int(mask.sum())} pairs")
    return mask


def _huber_pass(K, y, w, mode, D_fixed, F, D, notes: list[str]) -> tuple[float, float]:
    """IRLS with Huber weights min(1, c s / |r|), s = 1.4826 MAD; a no-op on exact data (MAD at rounding level)."""
    n_down = 0
    for _ in range(HUBER_ROUNDS):
        r = y - D * (F - K)
        scale = MAD_TO_SIGMA * np.median(np.abs(r - np.median(r)))
        if scale <= _EXACT_SCALE:  # residuals at rounding level: nothing to downweight
            break
        with np.errstate(divide="ignore"):
            h = np.minimum(1.0, HUBER_C * scale / np.abs(r))
        F_new, D_new = _solve(K, y, w * h, mode, D_fixed)
        n_down = int((h < 1.0).sum())
        converged = abs(F_new - F) <= 1e-12 * abs(F) and abs(D_new - D) <= 1e-14
        F, D = F_new, D_new
        if converged:
            break
    if n_down:
        notes.append(f"huber: {n_down} pairs downweighted")
    return F, D


def fit_forward(slice: Slice, rate: float | None = None, mode: str = "auto", window: float | None = 0.04,
                huber: bool = True, force: bool = False, min_pairs: int = 6) -> ForwardFit:
    """Fit (F, D) of one slice from parity on its two-sided pairs. Raises ValueError when it cannot.

    rate: continuously compounded, per year; required for fixed_discount (and for 'auto' on American).
    window: relative strike window around the ATM strike, then around the first F; None = all pairs.
    """
    notes: list[str] = []
    mode_used = _resolve_mode(slice, rate, mode, force)
    if mode_used != mode:
        notes.append(f"mode {mode} -> {mode_used} ({slice.exercise})")
    pairs, n_quoted = _two_sided_pairs(slice)
    notes.append(f"pairs quoted {n_quoted}, two-sided {len(pairs)}")
    if len(pairs) < min_pairs:
        raise ValueError(f"{slice.label()}: {len(pairs)} two-sided pairs < min_pairs={min_pairs}")
    K = pairs["strike"].to_numpy(dtype=float)
    y = (pairs["call_mid"] - pairs["put_mid"]).to_numpy(dtype=float)
    w_all = _weights(pairs)
    D_fixed = float(np.exp(-rate * slice.T)) if mode_used == "fixed_discount" else None
    if D_fixed is not None:
        notes.append(f"discount pinned from rate {rate:g}: D = {D_fixed:.10f}")

    k_atm = float(K[np.argmin(np.abs(y))])
    mask = _select_window(K, k_atm, window, min_pairs, notes)
    F, D = _solve(K[mask], y[mask], w_all[mask], mode_used, D_fixed)
    if window is not None:
        mask = _select_window(K, F, window, min_pairs, notes)
        notes.append(f"window {window:g} recentred from K_atm {k_atm:g} to F {F:.4f}: {int(mask.sum())} pairs")
        F, D = _solve(K[mask], y[mask], w_all[mask], mode_used, D_fixed)
    if huber:
        F, D = _huber_pass(K[mask], y[mask], w_all[mask], mode_used, D_fixed, F, D, notes)
    resid = y[mask] - D * (F - K[mask])
    return ForwardFit(
        forward=F, discount=D, mode=mode_used, n_pairs=int(mask.sum()), window=window,
        residual_rms=float(np.sqrt(np.mean(resid**2))), residual_max=float(np.max(np.abs(resid))),
        implied_rate=float(-np.log(D) / slice.T), notes=tuple(notes),
    )


def forward_table(chain: Chain, rate: float | None = None, **kw) -> pd.DataFrame:
    """One row per (root, expiry) in expiry order; a slice whose fit raises gets NaNs and the reason.

    drop_vs_previous = F minus the previous row's F: the ex-dividend jump diagnostic (a drop of about
    one dividend across an ex-date is expected; anything larger is a fit to look at).
    """
    rows = []
    for s in chain.slices():
        try:
            f = fit_forward(s, rate, **kw)
            rows.append({"root": s.root, "expiry": s.expiry, "T": s.T, "forward": f.forward, "discount": f.discount,
                         "implied_rate": f.implied_rate, "n_pairs": f.n_pairs, "residual_rms": f.residual_rms,
                         "mode": f.mode, "notes": "; ".join(f.notes)})
        except ValueError as e:
            rows.append({"root": s.root, "expiry": s.expiry, "T": s.T, "forward": np.nan, "discount": np.nan,
                         "implied_rate": np.nan, "n_pairs": 0, "residual_rms": np.nan, "mode": "failed",
                         "notes": f"failed: {e}"})
    cols = ["root", "expiry", "T", "forward", "discount", "implied_rate", "n_pairs", "residual_rms", "mode",
            "drop_vs_previous", "notes"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out["drop_vs_previous"] = out["forward"].diff()
    return out[cols]

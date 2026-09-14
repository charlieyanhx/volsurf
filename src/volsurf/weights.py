"""Quote selection for a slice fit, with every dropped row counted, and the three weighting schemes.

Input frame (the output shape of `iv.implied_vols`): columns strike, right ("C"/"P"), k = ln(K/F), bid, ask (dollars),
iv_bid, iv_mid, iv_ask (per 1.00, NaN where the price has no Black inverse).

`select` applies the rules IN THIS ORDER, each recorded in the returned `Ledger` as (rule, n_in, n_out) with n_in equal
to the previous rule's n_out (the ledger conserves rows):
  1. two-sided:      bid > 0
  2. min bid:        bid >= min_bid_ticks * tick   (Corbetta et al. 2019 drop quotes under 2 ticks)
  3. finite iv_mid
  4. OTM only:       puts with k < 0, calls with k >= 0   (when otm_only; ATM k = 0 goes to the call side)
  5. rel. spread:    (ask - bid) / mid <= max_rel_spread, mid = (bid + ask)/2
  6. |k| <= k_max    (when k_max is given)
The mask is over the input rows (True = kept); rules are applied to rows still alive, so counts add up.

Weights multiply the SQUARED residual in `svi.fit_svi` (objective = sum w_i r_i^2):
  spread_weights = 1 / max(iv_ask - iv_bid, floor)^2  — inverse variance of a mid whose error is ~ the band width;
                   NaN where the band is not finite (mask those rows first)
  vega_weights   = Black vega per 1.00 vol (`black.vega`); weights the fit toward where a vol error costs dollars
  unit_weights   = ones
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import black
from .quotes import Ledger

__all__ = ["REQUIRED_COLUMNS", "select", "spread_weights", "vega_weights", "unit_weights"]

REQUIRED_COLUMNS = ("strike", "right", "k", "bid", "ask", "iv_bid", "iv_mid", "iv_ask")


def _rule(ledger: Ledger, name: str, alive: np.ndarray, keep: np.ndarray) -> np.ndarray:
    n_in = int(alive.sum())
    out = alive & keep
    ledger.record(name, n_in, int(out.sum()))
    return out


def select(df_ivs: pd.DataFrame, T: float, otm_only: bool = True, min_bid_ticks: int = 2, tick: float = 0.01,
           max_rel_spread: float = 0.2, k_max: float | None = None) -> tuple[np.ndarray, Ledger]:
    """Boolean mask over the rows of `df_ivs` plus the ledger of what each rule removed (rules in the module docstring)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df_ivs.columns]
    if missing:
        raise ValueError(f"df_ivs is missing columns {missing}")
    if T <= 0:
        raise ValueError("T must be positive")
    if tick <= 0 or min_bid_ticks < 0 or max_rel_spread < 0:
        raise ValueError("tick > 0, min_bid_ticks >= 0 and max_rel_spread >= 0 are required")
    bid = df_ivs["bid"].to_numpy(float)
    ask = df_ivs["ask"].to_numpy(float)
    k = df_ivs["k"].to_numpy(float)
    right = df_ivs["right"].astype(str).str.upper().to_numpy()
    iv_mid = df_ivs["iv_mid"].to_numpy(float)
    ledger = Ledger()
    alive = np.ones(len(df_ivs), dtype=bool)
    alive = _rule(ledger, "two-sided (bid > 0)", alive, bid > 0)
    alive = _rule(ledger, f"bid >= {min_bid_ticks} ticks of {tick:g}", alive, bid >= min_bid_ticks * tick - 1e-12)
    alive = _rule(ledger, "finite iv_mid", alive, np.isfinite(iv_mid))
    if otm_only:
        alive = _rule(ledger, "OTM only (puts k < 0, calls k >= 0)", alive, np.where(right == "P", k < 0, k >= 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (ask - bid) / (0.5 * (ask + bid))
    alive = _rule(ledger, f"relative spread <= {max_rel_spread:g}", alive, rel <= max_rel_spread)
    if k_max is not None:
        alive = _rule(ledger, f"|k| <= {k_max:g}", alive, np.abs(k) <= k_max)
    return alive, ledger


def spread_weights(iv_bid, iv_ask, floor: float = 1e-4) -> np.ndarray:
    """1 / max(iv_ask - iv_bid, floor)^2; NaN where either vol is not finite."""
    if floor <= 0:
        raise ValueError("floor must be positive")
    iv_bid = np.asarray(iv_bid, dtype=float)
    iv_ask = np.asarray(iv_ask, dtype=float)
    spread = iv_ask - iv_bid
    with np.errstate(invalid="ignore"):
        w = 1.0 / np.maximum(spread, floor) ** 2
    return np.where(np.isfinite(spread), w, np.nan)


def vega_weights(F, K, T, sigma) -> np.ndarray:
    """Black vega per 1.00 vol at each quote (undiscounted, D = 1) as the weight."""
    return np.asarray(black.vega(F, K, T, sigma), dtype=float)


def unit_weights(n: int) -> np.ndarray:
    return np.ones(int(n), dtype=float)

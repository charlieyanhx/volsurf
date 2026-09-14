"""Synthetic chains from known SVI slices, so every test and README table has an answer key.

`synthetic_chain` prices each (expiry, strike, right) with Black-76 at the expiry's forward F and discount D on the
slice's total variance w(k), k = ln(K/F), and quotes bid/ask = mid -/+ h with the half-spread
    h = half_spread(price, k), default  max(0.05, 0.01 * price + 3 |k|)   (dollars; the widening in |k| stands in for
    the wider wings of real chains; the same rule was used in the forward-fit research so the numbers are comparable),
capped at the mid so bids never go below zero (h_eff = min(h, mid), bid = mid - h_eff, ask = mid + h_eff; deep wings
quote 0 x 2 mid, as real chains do).
With `noise_in_band` > 0 the mid is jittered uniformly inside +/- noise_in_band * h before the band is placed around
it (seeded), so parity C - P = D (F - K) holds EXACTLY on the mids only at noise 0. The jittered mid is floored at
price/2 so that, with the cap above, the model price always stays inside [bid, ask] (a within-band convexity check on
a noisy synthetic chain must still count 0). Sizes/volume/open interest are placeholders (10 / 10 / 0 / 100) because
nothing downstream reads them for a synthetic chain.

`spot_offset` is recorded in `Chain.source` as "spot=F_front*exp(spot_offset)" so a reader that carries a spot can be
exercised with spot != forward (the VIX-like case); the quotes never depend on it.

SYNTH_SURFACE: four expiries from an SSVI power-law surface (Gatheral-Jacquier 2014 eq 4.5, eta = 0.8, gamma = 0.5,
rho = -0.7 — GJ's Example A, which satisfies Thm 4.1 and Thm 4.2) with ATM total variance
theta(T) = (0.17 + 0.05 (1 - e^{-2T}))^2 T, mapped to raw SVI by `svi.from_ssvi` and rounded to 8 decimals.
Quote date 2026-06-15, expiries +18/+55/+146/+365 days (T = 0.04929, 0.15066, 0.39997, 0.99997 by the chain rule),
F = 100 exp(0.04 T) (a 4 % carry), D = exp(-0.045 T). Verified in tests/test_synth.py: negative skew, calendar-monotone
on k in [-1.5, 1.5], g > 0 on [-1, 1] with min g = 0.268 / 0.282 / 0.307 / 0.348 and both wing asymptotes ~ 0.25.

`synthetic_cboe_json(chain)` renders a chain in the exact shape of Cboe's delayed-quote JSON (keys listed in the
function) so `io.read_cboe_json` can be tested without real data; iv/greeks/theo fields are filled with placeholders
that a reader must not consume (bid/ask/sizes/volume/open_interest/last_trade_time are the real content).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import black
from .quotes import COLUMNS, Chain, time_to_expiry
from .svi import SVIParams, SVISlice

__all__ = [
    "SynthExpiry",
    "SYNTH_QUOTE_DATE",
    "SYNTH_SURFACE",
    "default_half_spread",
    "synth_strikes",
    "synthetic_chain",
    "occ_symbol",
    "synthetic_cboe_json",
]


@dataclass(frozen=True)
class SynthExpiry:
    """One synthetic expiry: settlement date, T (None -> the chain's ACT/365 rule), the slice, its forward and discount."""

    expiry: date
    T: float | None
    params: SVIParams
    forward: float
    discount: float


def default_half_spread(price, k) -> np.ndarray:
    """max(0.05, 0.01 price + 3 |k|) dollars (vectorised)."""
    price = np.asarray(price, dtype=float)
    k = np.asarray(k, dtype=float)
    return np.maximum(0.05, 0.01 * price + 3.0 * np.abs(k))


SYNTH_QUOTE_DATE = date(2026, 6, 15)

SYNTH_SURFACE: tuple[SynthExpiry, ...] = (
    SynthExpiry(date(2026, 7, 3), None, SVIParams(0.00038355, 0.01550153, -0.7, 0.03396060, 0.03464674),
                100.0 * math.exp(0.04 * 0.04929), math.exp(-0.045 * 0.04929)),
    SynthExpiry(date(2026, 8, 9), None, SVIParams(0.00128667, 0.02834198, -0.7, 0.06231091, 0.06356984),
                100.0 * math.exp(0.04 * 0.15066), math.exp(-0.045 * 0.15066)),
    SynthExpiry(date(2026, 11, 8), None, SVIParams(0.00397965, 0.04958494, -0.7, 0.11015985, 0.11238553),
                100.0 * math.exp(0.04 * 0.39997), math.exp(-0.045 * 0.39997)),
    SynthExpiry(date(2027, 6, 15), None, SVIParams(0.01159407, 0.08341665, -0.7, 0.19077046, 0.19462480),
                100.0 * math.exp(0.04 * 0.99997), math.exp(-0.045 * 0.99997)),
)


def synth_strikes(forward: float, T: float, sigma_atm: float, n: int = 41, width: float = 3.0,
                  step: float = 1.0) -> np.ndarray:
    """Strikes on a `step` grid covering k in [-width, +width] * sigma_atm * sqrt(T), at most n distinct values."""
    half = width * sigma_atm * math.sqrt(T)
    K = forward * np.exp(np.linspace(-half, half, n))
    return np.unique(np.round(K / step) * step)


def _expiry_T(quote_date: date, e: SynthExpiry, settlement: str) -> float:
    if e.T is not None:
        if e.T <= 0:
            raise ValueError("T must be positive")
        return float(e.T)
    return float(time_to_expiry(quote_date, "16:15", [np.datetime64(e.expiry)], [settlement])[0])


def _quotes_for_expiry(e: SynthExpiry, T: float, K: np.ndarray, half_spread, noise_in_band: float,
                       rng: np.random.Generator) -> dict:
    """Prices, mids and bands for one expiry, calls and puts stacked (calls first)."""
    if e.forward <= 0 or e.discount <= 0:
        raise ValueError("forward and discount must be positive")
    k = np.log(K / e.forward)
    sigma = SVISlice(e.params, T).iv(k)
    if np.any(~np.isfinite(sigma)):
        raise ValueError("the slice has negative total variance at a requested strike")
    right = np.concatenate([np.full(K.size, "C"), np.full(K.size, "P")])
    KK, kk, ss = np.tile(K, 2), np.tile(k, 2), np.tile(sigma, 2)
    price = black.price(e.forward, KK, T, ss, right, e.discount)
    h = np.asarray(half_spread(price, kk), dtype=float)
    mid = price + (rng.uniform(-1.0, 1.0, price.size) * noise_in_band * h if noise_in_band > 0 else 0.0)
    mid = np.maximum(mid, 0.5 * price)  # keeps the model price inside [bid, ask] once the band is capped at the mid
    h_eff = np.minimum(h, mid)
    return {"strike": KK, "right": right, "bid": mid - h_eff, "ask": mid + h_eff}


def synthetic_chain(symbol: str, quote_date: date, expiries: Sequence[SynthExpiry], strikes,
                    half_spread: Callable = default_half_spread, noise_in_band: float = 0.0,
                    exercise: str = "european", root: str | None = None, settlement: str = "PM",
                    spot_offset: float = 0.0, seed: int = 0, multiplier: float = 100.0) -> Chain:
    """A `quotes.Chain` priced from SVI slices. `strikes` is one array for every expiry or a list with one per expiry."""
    if len(expiries) == 0:
        raise ValueError("need at least one expiry")
    if noise_in_band < 0 or noise_in_band > 1:
        raise ValueError("noise_in_band must be in [0, 1] (a fraction of the half-spread)")
    per_expiry = list(strikes) if isinstance(strikes, (list, tuple)) and len(strikes) == len(expiries) \
        and np.ndim(strikes[0]) > 0 else [strikes] * len(expiries)
    if len(per_expiry) != len(expiries):
        raise ValueError("strikes must be one array or one array per expiry")
    rng = np.random.default_rng(seed)
    root = symbol if root is None else root
    frames = []
    for e, K in zip(expiries, per_expiry, strict=True):
        K = np.unique(np.asarray(K, dtype=float))
        if K.size == 0 or np.any(K <= 0):
            raise ValueError("strikes must be positive and non-empty")
        T = _expiry_T(quote_date, e, settlement)
        q = _quotes_for_expiry(e, T, K, half_spread, noise_in_band, rng)
        n = q["strike"].size
        frames.append(pd.DataFrame({
            "expiry": np.full(n, np.datetime64(e.expiry, "ns")), "strike": q["strike"], "right": q["right"],
            "bid": q["bid"], "ask": q["ask"], "bid_size": np.full(n, 10.0), "ask_size": np.full(n, 10.0),
            "volume": np.zeros(n), "open_interest": np.full(n, 100.0), "root": np.full(n, root),
            "settlement": np.full(n, settlement)}))
    df = pd.concat(frames, ignore_index=True)[list(COLUMNS)]
    spot = expiries[0].forward * math.exp(spot_offset)
    source = f"synthetic seed={seed} noise_in_band={noise_in_band:g} spot={spot:.6f}"
    return Chain(symbol, quote_date, df, exercise=exercise, multiplier=multiplier, source=source)


def occ_symbol(root: str, expiry, right: str, strike: float) -> str:
    """OCC option symbol ROOT + YYMMDD + C/P + 8-digit strike x 1000 (e.g. SPX260918C00200000)."""
    exp = pd.Timestamp(expiry)
    return f"{root}{exp:%y%m%d}{right.upper()}{int(round(float(strike) * 1000)):08d}"


_OPTION_KEYS = ("option", "bid", "bid_size", "ask", "ask_size", "iv", "open_interest", "volume", "delta", "gamma",
                "vega", "theta", "rho", "theo", "change", "open", "high", "low", "tick", "last_trade_price",
                "last_trade_time", "percent_change", "prev_day_close")


def synthetic_cboe_json(chain: Chain) -> dict:
    """The chain in Cboe's delayed_quotes JSON shape: {timestamp, symbol, data: {..., options: [rows]}} with every
    option row carrying the 23 keys of the real feed. last_trade_time = quote_date 16:14:59 (the reader's date rule)."""
    qd = chain.quote_date
    ltt = datetime(qd.year, qd.month, qd.day, 16, 14, 59).strftime("%Y-%m-%dT%H:%M:%S")
    rows = []
    for r in chain.df.itertuples(index=False):
        mid = 0.5 * (r.bid + r.ask)
        rows.append({
            "option": occ_symbol(r.root, r.expiry, r.right, r.strike), "bid": float(r.bid),
            "bid_size": _num(r.bid_size), "ask": float(r.ask), "ask_size": _num(r.ask_size), "iv": 0.0,
            "open_interest": _num(r.open_interest), "volume": _num(r.volume), "delta": 0.0, "gamma": 0.0,
            "vega": 0.0, "theta": 0.0, "rho": 0.0, "theo": float(mid), "change": 0.0, "open": float(mid),
            "high": float(mid), "low": float(mid), "tick": "up", "last_trade_price": float(mid),
            "last_trade_time": ltt, "percent_change": 0.0, "prev_day_close": float(mid)})
    spot = _spot_from_source(chain.source)
    next_day = (datetime(qd.year, qd.month, qd.day) + timedelta(days=1)).strftime("%Y-%m-%d 03:44:40")
    data = {"symbol": chain.symbol, "security_type": "index" if chain.exercise == "european" else "stock",
            "exchange_id": 0, "current_price": spot, "price_change": 0.0, "price_change_percent": 0.0, "bid": spot,
            "ask": spot, "bid_size": 0, "ask_size": 0, "open": spot, "high": spot, "low": spot, "close": spot,
            "prev_day_close": spot, "volume": 0, "iv30": 0.0, "iv30_change": 0.0, "iv30_change_percent": 0.0,
            "seqno": 0, "last_trade_time": ltt, "tick": "up", "options": rows}
    return {"timestamp": next_day, "symbol": chain.symbol, "data": data}


def _num(x) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _spot_from_source(source: str) -> float:
    for tok in source.split():
        if tok.startswith("spot="):
            return float(tok[5:])
    return float("nan")

"""volsurf.iv: round trips, the information floor, the NaN regions, and external oracles.

Oracle values (plan v2 / theory report `iv_roundtrip_and_floor`): on 1e5 random Black prices with
F = 100, k ~ U(-0.5, 0.5), sigma ~ U(0.05, 1), T ~ U(0.01, 2): |price(iv) - price| <= 1e-10 F;
|iv - sigma| <= 1e-13 for tv/F >= 1e-4, <= 1e-11 on [1e-6, 1e-4), <= 1e-9 on [1e-8, 1e-6);
NaN exactly where tv < 1e-10 F. Those bars are met on quotes priced as the OTM option. A quote priced
as the ITM option (converted through parity inside the inverter) carries the intrinsic's own rounding,
~1e-16 F, which is 1e-13 in sigma once divided by a vega of 1e-3 F (short T, |k| ~ 0.3, high sigma):
there the test asserts max(bar, 8 ulp(price) / vega), the bound the input allows, not the solver.
Speed is measured in `test_speed_report` and printed, not asserted (about 0.6-1 us per point here).
"""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from volsurf import black, iv
from volsurf.quotes import COLUMNS, Slice

F0 = 100.0
BUCKETS = ((1e-4, np.inf, 1e-13), (1e-6, 1e-4, 1e-11), (1e-8, 1e-6, 1e-9), (1e-10, 1e-8, 1e-8))


def _sample(n: int, seed: int, rights: str):
    rng = np.random.default_rng(seed)
    k = rng.uniform(-0.5, 0.5, n)
    K = F0 * np.exp(k)
    sigma = rng.uniform(0.05, 1.0, n)
    T = rng.uniform(0.01, 2.0, n)
    if rights == "otm":
        right = np.where(k < 0, "P", "C")
    else:
        right = np.where(rng.random(n) < 0.5, "C", "P")
    px = black.price(F0, K, T, sigma, right)
    intrinsic = np.where(right == "C", np.maximum(F0 - K, 0.0), np.maximum(K - F0, 0.0))
    tv = (px - intrinsic) / F0
    return K, T, sigma, right, px, tv


def _check_round_trip(K, T, sigma, right, px, tv, use_ulp_bound: bool):
    got = iv.implied_vol(px, F0, K, T, right)
    ok = ~np.isnan(got)
    # NaN exactly where the time value is under the floor
    assert np.array_equal(~ok, tv < iv.TIME_VALUE_FLOOR)
    assert ok.sum() > 0.95 * len(px)
    back = black.price(F0, K[ok], T[ok], got[ok], right[ok])
    assert np.max(np.abs(back - px[ok])) <= 1e-10 * F0
    err = np.abs(got - sigma)
    bar = np.full(px.shape, np.nan)
    for lo, hi, tol in BUCKETS:
        bar[(tv >= lo) & (tv < hi)] = tol
    if use_ulp_bound:
        with np.errstate(divide="ignore"):  # vega underflows exactly where the floor already says NaN
            bar = np.maximum(bar, 8.0 * np.spacing(px) / black.vega(F0, K, T, sigma))
    assert np.all(err[ok] <= bar[ok]), f"worst ratio {np.max(err[ok] / bar[ok])}"


def test_round_trip_otm_quotes_meets_contract_bars():
    """1e5 quotes priced as the OTM option: the plan's bars hold exactly as written."""
    _check_round_trip(*_sample(100_000, 0, "otm"), use_ulp_bound=False)


def test_round_trip_random_rights_within_input_conditioning():
    """1e5 quotes with random rights (half ITM, converted by parity): bars plus the input's ulp bound."""
    _check_round_trip(*_sample(100_000, 1, "random"), use_ulp_bound=True)


def test_nan_regions():
    K, T = 90.0, 0.5
    c_intr, p_intr = F0 - K, 0.0
    assert np.isnan(iv.implied_vol(c_intr - 1e-6, F0, K, T, "C"))  # below intrinsic
    assert np.isnan(iv.implied_vol(p_intr - 1e-6, F0, K, T, "P"))
    assert np.isnan(iv.implied_vol(F0 + 1e-6, F0, K, T, "C"))  # above the call cap D F
    assert np.isnan(iv.implied_vol(K + 1e-6, F0, K, T, "P"))  # above the put cap D K
    assert np.isnan(iv.implied_vol(5.0, F0, K, 0.0, "C"))  # T <= 0
    assert np.isnan(iv.implied_vol(5.0, F0, K, -0.1, "C"))
    assert np.isnan(iv.implied_vol(np.nan, F0, K, T, "C"))
    assert np.isnan(iv.implied_vol(np.inf, F0, K, T, "C"))
    with pytest.raises(ValueError):
        iv.implied_vol(5.0, -F0, K, T, "C")
    with pytest.raises(ValueError):
        iv.implied_vol(5.0, F0, K, T, "C", D=0.0)


def test_cap_with_discount():
    D = 0.97
    assert np.isnan(iv.implied_vol(D * F0 + 1e-6, F0, 90.0, 0.5, "C", D=D))
    assert np.isfinite(iv.implied_vol(D * F0 * 0.999, F0, 90.0, 0.5, "C", D=D))
    assert np.isnan(iv.implied_vol(D * 90.0 + 1e-6, F0, 90.0, 0.5, "P", D=D))


def test_information_floor_boundary():
    """Time value 0.9e-10 F is NaN, 1.1e-10 F is a number (the floor is 1e-10 F, not fuzzy)."""
    K, T = 80.0, 0.5
    floor = iv.TIME_VALUE_FLOOR * F0
    assert np.isnan(iv.implied_vol(F0 - K + 0.9 * floor, F0, K, T, "C"))
    assert np.isfinite(iv.implied_vol(F0 - K + 1.1 * floor, F0, K, T, "C"))
    assert np.isnan(iv.implied_vol(0.9 * floor, F0, K, T, "P"))
    assert np.isfinite(iv.implied_vol(1.1 * floor, F0, K, T, "P"))


def test_deep_otm_price_only_stopping_rule_is_wrong():
    """Theory report §4: a price-only absolute stop declares convergence at the wrong sigma deep OTM.

    Put F = 100, K = 78, T = 0.05, sigma = 0.25 has tv/F = 4.5e-8. Bisection stopped on
    |model - price| <= 1e-10 F returns sigma off by 4e-6; the relative rules return it to 1e-9.
    """
    K, T, sigma = 78.0, 0.05, 0.25
    px = black.price(F0, K, T, sigma, "P")
    assert 1e-8 <= px / F0 < 1e-6
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        s = 0.5 * (lo + hi)
        diff = black.price(F0, K, T, s, "P") - px
        if abs(diff) <= 1e-10 * F0:
            break
        lo, hi = (lo, s) if diff > 0 else (s, hi)
    assert abs(s - sigma) > 1e-6
    assert abs(iv.implied_vol(px, F0, K, T, "P") - sigma) <= 1e-9


def test_discount_factor_is_honoured():
    D = 0.95
    K = np.array([80.0, 100.0, 125.0])
    for right in ("C", "P"):
        px = black.price(F0, K, 0.75, 0.3, right, D)
        got = iv.implied_vol(px, F0, K, 0.75, right, D)
        assert np.max(np.abs(got - 0.3)) <= 1e-13
        # the same price read undiscounted is a different (higher) vol for the OTM leg
        assert not np.allclose(iv.implied_vol(px, F0, K, 0.75, right), 0.3)


def test_broadcasting_and_scalar_return():
    K = np.array([[90.0], [100.0], [110.0]])
    T = np.array([0.1, 0.5, 1.0, 2.0])
    px = black.price(F0, K, T, 0.22, "C")
    got = iv.implied_vol(px, F0, K, T, "C")
    assert got.shape == (3, 4)
    assert np.max(np.abs(got - 0.22)) <= 1e-13
    # booleans for right, array of rights
    got_bool = iv.implied_vol(px, F0, K, T, np.ones((3, 4), dtype=bool))
    assert np.array_equal(got, got_bool)
    rights = np.array(["C", "P", "C", "P"])
    px2 = black.price(F0, 95.0, T, 0.4, rights)
    assert np.max(np.abs(iv.implied_vol(px2, F0, 95.0, T, rights) - 0.4)) <= 1e-13
    scalar = iv.implied_vol(black.price(F0, 90.0, 0.5, 0.3, "P"), F0, 90.0, 0.5, "P")
    assert isinstance(scalar, np.floating)
    assert abs(float(scalar) - 0.3) <= 1e-14


def test_itm_and_otm_of_same_strike_agree():
    """Parity inside the inverter: the ITM call and the OTM put at one strike give the same sigma."""
    K, T, sigma, D = 85.0, 0.4, 0.35, 0.98
    c = black.price(F0, K, T, sigma, "C", D)
    p = black.price(F0, K, T, sigma, "P", D)
    assert abs(c - p - D * (F0 - K)) <= 1e-12
    assert abs(iv.implied_vol(c, F0, K, T, "C", D) - iv.implied_vol(p, F0, K, T, "P", D)) <= 1e-13


def _slice(F, D, T, strikes, sigma, hw):
    rows = []
    for right in ("C", "P"):
        mid = black.price(F, strikes, T, sigma, right, D)
        rows.append(pd.DataFrame({"expiry": pd.Timestamp("2026-12-18"), "strike": strikes, "right": right,
                                  "bid": np.maximum(mid - hw, 0.0), "ask": mid + hw, "bid_size": np.nan,
                                  "ask_size": np.nan, "volume": np.nan, "open_interest": np.nan,
                                  "root": "SPX", "settlement": "PM"}))
    df = pd.concat(rows, ignore_index=True)[list(COLUMNS)]
    return Slice("SPX", date(2026, 9, 18), "SPX", pd.Timestamp("2026-12-18"), T, df)


def test_implied_vols_dataframe():
    F, D, T = 4500.0, 0.99, 0.25
    strikes = np.array([3000.0, 4200.0, 4500.0, 4800.0])
    sigma = np.array([0.35, 0.22, 0.18, 0.17])
    sl = _slice(F, D, T, strikes, sigma, hw=np.array([0.0, 0.5, 0.5, 0.5]))
    out = iv.implied_vols(sl, F, D)
    assert list(out.columns) == ["strike", "right", "k", "bid", "ask", "iv_bid", "iv_mid", "iv_ask", "tv_mid"]
    assert len(out) == len(sl.df)
    assert np.allclose(out["k"], np.log(out["strike"] / F))
    mid = 0.5 * (out["bid"] + out["ask"])
    call = out["right"] == "C"
    intrinsic = D * np.where(call, np.maximum(F - out["strike"], 0), np.maximum(out["strike"] - F, 0))
    assert np.allclose(out["tv_mid"], mid - intrinsic)
    sig_by_strike = dict(zip(strikes, sigma, strict=True))
    expect = out["strike"].map(sig_by_strike).to_numpy()
    assert np.max(np.abs(out["iv_mid"] - expect)) <= 1e-12
    wide = out[out["strike"] > 3000.0]
    assert (wide["iv_bid"] < wide["iv_mid"]).all() and (wide["iv_mid"] < wide["iv_ask"]).all()
    locked = out[out["strike"] == 3000.0]  # hw = 0: bid = mid = ask, so the three vols coincide
    assert np.allclose(locked["iv_bid"], locked["iv_mid"]) and np.allclose(locked["iv_ask"], locked["iv_mid"])


def test_zero_bid_gives_nan_on_the_bid_side():
    F, D, T = 100.0, 1.0, 0.5
    strikes = np.array([60.0, 100.0])
    sl = _slice(F, D, T, strikes, np.array([0.3, 0.2]), hw=np.array([1e-3, 0.05]))
    df = sl.df.copy()
    df.loc[(df["strike"] == 60.0) & (df["right"] == "P"), "bid"] = 0.0
    sl = Slice(sl.symbol, sl.quote_date, sl.root, sl.expiry, sl.T, df)
    out = iv.implied_vols(sl, F, D)
    row = out[(out["strike"] == 60.0) & (out["right"] == "P")].iloc[0]
    assert np.isnan(row["iv_bid"]) and np.isfinite(row["iv_ask"])


def test_from_spot_conventions():
    """F = S e^{(r-q)T}, D = e^{-rT}: the Black-Scholes-Merton price inverts back to sigma."""
    S, K, T, r, q, sigma = 100.0, 95.0, 0.7, 0.04, 0.015, 0.27
    F, D = S * np.exp((r - q) * T), np.exp(-r * T)
    for right in ("C", "P"):
        px = black.price(F, K, T, sigma, right, D)
        assert abs(iv.from_spot(px, S, K, T, right, r, q) - sigma) <= 1e-13
    K = np.array([80.0, 100.0, 120.0])
    px = black.price(F, K, T, sigma, "P", D)
    assert np.max(np.abs(iv.from_spot(px, S, K, T, "P", r, q) - sigma)) <= 1e-13


def test_agrees_with_pricers_scalar_brent():
    """pricers.bs.implied_vol (Brent, xtol 1e-10) on 2000 random spot-convention quotes: |diff| <= 1e-9
    for tv/F >= 1e-8 (measured 7.8e-11); in [1e-10, 1e-8) both are conditioning-limited at the 1e-8 bar."""
    bs = pytest.importorskip("pricers.bs")
    rng = np.random.default_rng(3)
    n = 2000
    S, r, q = 100.0, 0.03, 0.01
    K = S * np.exp(rng.uniform(-0.5, 0.5, n))
    sigma = rng.uniform(0.05, 1.0, n)
    T = rng.uniform(0.01, 2.0, n)
    right = np.where(rng.random(n) < 0.5, "C", "P")
    px = np.array([bs.price(S, K[i], T[i], sigma[i], right[i], r, q) for i in range(n)])
    ours = iv.from_spot(px, S, K, T, right, r, q)
    theirs = np.array([bs.implied_vol(px[i], S, K[i], T[i], right[i], r, q) for i in range(n)])
    both = np.isfinite(ours) & np.isfinite(theirs)
    assert both.sum() > 0.95 * n
    F, D = S * np.exp((r - q) * T), np.exp(-r * T)
    intrinsic = D * np.where(right == "C", np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    tv = (px - intrinsic) / F
    diff = np.abs(ours - theirs)
    assert np.max(diff[both & (tv >= 1e-8)]) <= 1e-9
    assert np.max(diff[both]) <= 2e-8
    # Brent has no information floor: it returns a number wherever we return NaN, never the reverse
    assert not np.any(np.isfinite(ours) & np.isnan(theirs))


def test_agrees_with_lets_be_rational():
    """Jaeckel's `lets_be_rational` (scalar, undiscounted Black on F): 1e-13 for tv/F >= 1e-4, 1e-9 above 1e-8."""
    lbr = pytest.importorskip("lets_be_rational")
    K, T, sigma, right, px, tv = _sample(400, 5, "otm")
    ours = iv.implied_vol(px, F0, K, T, right)
    q = np.where(right == "C", 1.0, -1.0)
    theirs = np.full(px.shape, np.nan)
    for i in range(len(px)):
        if tv[i] >= 1e-8:
            theirs[i] = lbr.implied_volatility_from_a_transformed_rational_guess(px[i], F0, K[i], T[i], q[i])
    m = tv >= 1e-8
    assert np.max(np.abs(ours[m] - theirs[m])) <= 1e-9
    m4 = tv >= 1e-4
    assert np.max(np.abs(ours[m4] - theirs[m4])) <= 1e-13


def test_speed_report():
    """Measured, not asserted: microseconds per point on 1e5 random quotes (see the module docstring)."""
    K, T, sigma, right, px, tv = _sample(100_000, 7, "random")
    t0 = time.perf_counter()
    iv.implied_vol(px, F0, K, T, right)
    us = (time.perf_counter() - t0) / len(px) * 1e6
    print(f"\nimplied_vol: {us:.2f} us/pt on {len(px)} points")
    assert us < 1000.0  # only guards against a pathological regression (a Python loop)

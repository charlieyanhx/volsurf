"""volsurf.forward: parity forward and discount on synthetic chains priced by `black` around known (F, D).

Oracles (plan v2 / theory report `pcp_forward_*`):
  exact       European chain S = 4480, r = 0.05, q = 0.012, T = 0.25, strikes 3800..5175 step 25, skewed
              smile: F = 4522.7628016966 to |dF| <= 1e-9 (achieved 2.7e-12), D = 0.987577800494 to 1e-12.
  weighting   mid noise U(-hw/2, hw/2), hw = max(0.05, 0.01 price + 3|k|): sd(F_hat - F) WLS 0.139 vs
              OLS 0.186, ratio 1.34 over 2000 trials (assert > 1.2).
  stale       three far-OTM puts +3.00: Huber-IRLS or the |K/F - 1| < 4 % window gives |F_hat - F| < 1e-3;
              plain all-pair WLS is off by 0.04-0.14.
  american    parity_line refuses an American-flagged chain; fixed_discount recovers F within 1 bp.
  rate        +/-50 bp of rate moves F by < 2 bp at 30 DTE (the window is centred on F, so the pinned
              discount multiplies F - K_bar, not F).
  vix         quoted spot 15.46, parity forward 16.95: the fit returns 16.95 and never reads a spot.
  table       forward_table.drop_vs_previous = -1.40 when the second expiry's F is 1.40 lower.
"""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from volsurf import black
from volsurf.forward import ForwardFit, fit_forward, forward_table
from volsurf.quotes import COLUMNS, Chain, Slice

S0, R0, Q0, T0 = 4480.0, 0.05, 0.012, 0.25
F_REF = 4522.7628016966
D_REF = 0.987577800494
STRIKES_REF = np.arange(3800.0, 5175.0 + 1e-9, 25.0)


def smile_ref(k):
    return 0.18 - 0.35 * k + 0.6 * k * k


def hw_flat(mid, k):
    return np.full_like(mid, 0.05)


def hw_noise(mid, k):
    return np.maximum(0.05, 0.01 * mid + 3.0 * np.abs(k))


def quote_frame(F, D, T, strikes, smile, hw_fn, expiry, root="SPX", settlement="PM", rng=None):
    """Black quotes around (F, D): bid/ask = mid -/+ hw, with the quoted mid jittered by U(-hw/2, hw/2) if rng."""
    K = np.asarray(strikes, dtype=float)
    k = np.log(K / F)
    sigma = smile(k)
    rows = []
    for right in ("C", "P"):
        mid = black.price(F, K, T, sigma, right, D)
        hw = hw_fn(mid, k)
        noise = rng.uniform(-hw / 2, hw / 2) if rng is not None else 0.0
        rows.append(pd.DataFrame({"expiry": pd.Timestamp(expiry), "strike": K, "right": right,
                                  "bid": np.maximum(mid + noise - hw, 0.0), "ask": mid + noise + hw,
                                  "bid_size": np.nan, "ask_size": np.nan, "volume": np.nan,
                                  "open_interest": np.nan, "root": root, "settlement": settlement}))
    return pd.concat(rows, ignore_index=True)[list(COLUMNS)]


def make_slice(F, D, T, strikes=STRIKES_REF, smile=smile_ref, hw_fn=hw_flat, exercise="european", rng=None,
               symbol="SPX"):
    df = quote_frame(F, D, T, strikes, smile, hw_fn, "2026-12-18", root=symbol, rng=rng)
    return Slice(symbol, date(2026, 9, 18), symbol, pd.Timestamp("2026-12-18"), T, df, exercise)


def test_reference_forward_and_discount_are_the_oracle_values():
    assert abs(S0 * np.exp((R0 - Q0) * T0) - F_REF) <= 1e-9
    assert abs(np.exp(-R0 * T0) - D_REF) <= 1e-12


def test_exact_recovery_parity_line():
    """European exact chain: F to 1e-9 and D to 1e-12, with and without the strike window."""
    sl = make_slice(F_REF, D_REF, T0)
    for kw in (dict(), dict(window=None), dict(huber=False)):
        fit = fit_forward(sl, **kw)
        assert isinstance(fit, ForwardFit)
        assert fit.mode == "parity_line"
        assert abs(fit.forward - F_REF) <= 1e-9
        assert abs(fit.discount - D_REF) <= 1e-12
        assert abs(fit.implied_rate - R0) <= 1e-10
        assert fit.residual_rms <= 1e-9 and fit.residual_max <= 1e-8
        assert fit.n_pairs >= 6
    assert fit_forward(sl, window=None).n_pairs == len(STRIKES_REF)


def test_exact_recovery_fixed_discount():
    sl = make_slice(F_REF, D_REF, T0)
    fit = fit_forward(sl, rate=R0, mode="fixed_discount")
    assert fit.mode == "fixed_discount"
    assert abs(fit.forward - F_REF) <= 1e-9
    assert abs(fit.discount - D_REF) <= 1e-12
    assert any("pinned" in n for n in fit.notes)


def test_weighted_beats_unweighted_on_mid_noise():
    """sd ratio OLS / WLS > 1.2 over 2000 noisy trials (the report measured 1.34). Runs in a few seconds."""
    rng = np.random.default_rng(2)
    template = make_slice(F_REF, D_REF, T0, hw_fn=hw_noise)
    K = template.df["strike"].to_numpy()
    k = np.log(K / F_REF)
    sigma = smile_ref(k)
    mid = black.price(F_REF, K, T0, sigma, template.df["right"].to_numpy().astype(str), D_REF)
    hw = hw_noise(mid, k)
    is_call = template.df["right"].to_numpy() == "C"
    order_c, order_p = np.argsort(K[is_call]), np.argsort(K[~is_call])
    wls, ols = [], []
    t0 = time.perf_counter()
    for _ in range(2000):
        noise = rng.uniform(-hw / 2, hw / 2)
        df = template.df.assign(bid=np.maximum(mid + noise - hw, 0.0), ask=mid + noise + hw)
        sl = Slice("SPX", template.quote_date, "SPX", template.expiry, T0, df)
        wls.append(fit_forward(sl, window=None, huber=False).forward - F_REF)
        y = (mid + noise)[is_call][order_c] - (mid + noise)[~is_call][order_p]
        slope, intercept = np.polyfit(K[is_call][order_c], y, 1)
        ols.append(-intercept / slope - F_REF)
    elapsed = time.perf_counter() - t0
    ratio = np.std(ols) / np.std(wls)
    print(f"\nweighted vs OLS: sd {np.std(wls):.4f} vs {np.std(ols):.4f}, ratio {ratio:.3f}, {elapsed:.1f} s")
    assert ratio > 1.2
    assert abs(np.mean(wls)) < 3 * np.std(wls) / np.sqrt(len(wls)) * 2  # unbiased within 6 se


def _stale_slice():
    sl = make_slice(F_REF, D_REF, T0)
    df = sl.df.copy()
    stale = (df["right"] == "P") & df["strike"].isin([3800.0, 3825.0, 3850.0])
    df.loc[stale, ["bid", "ask"]] += 3.0
    return Slice(sl.symbol, sl.quote_date, sl.root, sl.expiry, sl.T, df)


def test_stale_far_puts_huber_recovers():
    sl = _stale_slice()
    plain = fit_forward(sl, window=None, huber=False)
    assert abs(plain.forward - F_REF) > 0.03  # the stale quotes tilt an unrobust all-pair line
    robust = fit_forward(sl, window=None, huber=True)
    assert abs(robust.forward - F_REF) < 1e-3
    assert any(n.startswith("huber: 3 pairs") for n in robust.notes)


def test_stale_far_puts_window_recovers():
    sl = _stale_slice()
    fit = fit_forward(sl, window=0.04, huber=False)
    assert abs(fit.forward - F_REF) < 1e-3
    assert abs(fit.discount - D_REF) <= 1e-9


def test_american_chain_modes():
    sl = make_slice(F_REF, D_REF, T0, exercise="american")
    with pytest.raises(ValueError, match="American chain needs a rate"):
        fit_forward(sl)
    with pytest.raises(ValueError, match="american"):
        fit_forward(sl, rate=R0, mode="parity_line")
    fit = fit_forward(sl, rate=R0)
    assert fit.mode == "fixed_discount"
    assert abs(fit.forward / F_REF - 1.0) <= 1e-4
    assert abs(fit.discount - D_REF) <= 1e-12
    forced = fit_forward(sl, mode="parity_line", force=True)
    assert forced.mode == "parity_line" and abs(forced.forward - F_REF) <= 1e-9
    with pytest.raises(ValueError, match="needs a rate"):
        fit_forward(sl, mode="fixed_discount")
    with pytest.raises(ValueError, match="mode"):
        fit_forward(sl, rate=R0, mode="guess")


def test_rate_error_moves_forward_by_under_2bp_at_30dte():
    T = 30.0 / 365.0
    F = S0 * np.exp((R0 - Q0) * T)
    sl = make_slice(F, np.exp(-R0 * T), T, exercise="american")
    base = fit_forward(sl, rate=R0).forward
    assert abs(base / F - 1.0) <= 1e-6
    for dr in (-0.005, 0.005):
        moved = fit_forward(sl, rate=R0 + dr).forward
        assert abs(moved / F - 1.0) < 2e-4, f"rate {R0 + dr}: {moved / F - 1.0:.2e}"


def test_parity_line_rejects_discount_outside_band_and_thin_chains():
    sl = make_slice(F_REF, D_REF, T0)
    thin = Slice(sl.symbol, sl.quote_date, sl.root, sl.expiry, sl.T, sl.df[sl.df["strike"] >= 5100.0].reset_index(drop=True))
    with pytest.raises(ValueError, match="min_pairs"):
        fit_forward(thin)
    # a chain whose line implies D = 1.05: scale y = C - P by 1.05 / D at every strike
    df = sl.df.copy()
    pairs = sl.pairs()
    bump = (1.05 / D_REF - 1.0) * D_REF * (F_REF - pairs["strike"].to_numpy())
    for K, b in zip(pairs["strike"], bump, strict=True):
        m = (df["strike"] == K) & (df["right"] == "C")
        df.loc[m, ["bid", "ask"]] += b
    bad = Slice(sl.symbol, sl.quote_date, sl.root, sl.expiry, sl.T, df)
    with pytest.raises(ValueError, match="discount"):
        fit_forward(bad)


def test_one_sided_pairs_are_dropped_and_counted():
    sl = make_slice(F_REF, D_REF, T0)
    df = sl.df.copy()
    df.loc[(df["right"] == "P") & (df["strike"] == 3800.0), "bid"] = 0.0
    sl2 = Slice(sl.symbol, sl.quote_date, sl.root, sl.expiry, sl.T, df)
    fit = fit_forward(sl2, window=None)
    assert fit.n_pairs == len(STRIKES_REF) - 1
    assert any("two-sided 55" in n for n in fit.notes)


def test_locked_quote_gets_tick_floor_not_infinite_weight():
    sl = make_slice(F_REF, D_REF, T0)
    df = sl.df.copy()
    m = df["strike"] == 4525.0
    df.loc[m, "bid"] = df.loc[m, "ask"]  # locked, zero width on both legs
    sl2 = Slice(sl.symbol, sl.quote_date, sl.root, sl.expiry, sl.T, df)
    fit = fit_forward(sl2)
    assert np.isfinite(fit.forward) and abs(fit.forward - F_REF) <= 1e-6


def test_vix_like_chain_ignores_spot():
    """VIX options settle on the future: quoted spot 15.46, parity forward 16.95 (mztrading 2026-08-10, 9 DTE)."""
    F, D, T = 16.95, 0.9993, 9.0 / 365.0
    strikes = np.arange(10.0, 30.0 + 1e-9, 0.5)
    sl = make_slice(F, D, T, strikes=strikes, smile=lambda k: 0.9 + 0.8 * k, hw_fn=lambda m, k: np.full_like(m, 0.05),
                    symbol="VIX")
    assert not hasattr(sl, "spot") and "spot" not in sl.df.columns  # nothing to read
    fit = fit_forward(sl, window=0.15)
    assert abs(fit.forward - F) <= 1e-9
    assert abs(fit.discount - D) <= 1e-10
    assert abs(fit.forward - 15.46) > 1.4
    # the default 4 % window holds too few 0.5-wide strikes at F = 16.95: it widens and says so
    fit_default = fit_forward(sl)
    assert abs(fit_default.forward - F) <= 1e-9
    assert any(n.startswith("window widened") for n in fit_default.notes)


def _two_expiry_chain(exercise="european"):
    q = date(2026, 9, 18)
    frames = []
    T = {}
    for expiry in ("2026-10-16", "2026-11-20"):
        chain_T = Chain("SPY", q, quote_frame(1.0, 1.0, 1.0, [1.0], lambda k: 0.2 + 0 * k, hw_flat, expiry)).T[0]
        T[expiry] = float(chain_T)
    F1 = 450.0 * np.exp((R0 - Q0) * T["2026-10-16"])
    F2 = F1 - 1.40  # an ex-dividend date between the two expiries
    for expiry, F in (("2026-10-16", F1), ("2026-11-20", F2)):
        D = np.exp(-R0 * T[expiry])
        frames.append(quote_frame(F, D, T[expiry], np.arange(380.0, 520.0 + 1e-9, 2.5), smile_ref, hw_flat, expiry,
                                  root="SPY"))
    return Chain("SPY", q, pd.concat(frames, ignore_index=True), exercise=exercise), F1, F2


def test_forward_table_drop_vs_previous_shows_the_dividend_jump():
    chain, F1, F2 = _two_expiry_chain(exercise="american")
    tbl = forward_table(chain, rate=R0)
    assert list(tbl.columns) == ["root", "expiry", "T", "forward", "discount", "implied_rate", "n_pairs",
                                 "residual_rms", "mode", "drop_vs_previous", "notes"]
    assert len(tbl) == 2
    assert np.isnan(tbl["drop_vs_previous"].iloc[0])
    assert abs(tbl["drop_vs_previous"].iloc[1] - (-1.40)) <= 1e-6
    assert abs(tbl["forward"].iloc[0] - F1) <= 1e-6 and abs(tbl["forward"].iloc[1] - F2) <= 1e-6
    assert (tbl["mode"] == "fixed_discount").all()
    assert np.allclose(tbl["implied_rate"], R0)


def test_forward_table_records_a_failed_slice_instead_of_raising():
    chain, F1, F2 = _two_expiry_chain(exercise="american")
    tbl = forward_table(chain)  # no rate on an American chain: every slice fails, the table still comes back
    assert len(tbl) == 2
    assert (tbl["mode"] == "failed").all() and tbl["forward"].isna().all()
    assert tbl["notes"].str.contains("needs a rate").all()


def test_forward_table_european_parity_line():
    chain, F1, F2 = _two_expiry_chain(exercise="european")
    tbl = forward_table(chain)
    assert (tbl["mode"] == "parity_line").all()
    assert abs(tbl["forward"].iloc[1] - F2) <= 1e-6
    assert np.allclose(tbl["discount"], np.exp(-R0 * tbl["T"]), atol=1e-9)


def test_tree_priced_american_chain_bias_is_the_reported_one():
    """Theory report `pcp_american_single_name`: AAPL-like S = 200, r = 5 %, q = 0.5 %, T = 0.25, K = 150..250,
    iv = 0.28 - 0.5 k + k^2, CRR n = 400 BBS American prices. Near-ATM fixed_discount is biased -11 bp by the
    put early-exercise premium (report: -11.6 bp at |k| < 0.08); a forced all-pair parity line is -33 bp with
    D = 1.011 (report: -33.6 bp, 1.011). That is why `auto` pins the discount on American chains; the
    implied-q refinement that removes the residual bias is v0.2 (PLAN.md)."""
    trees = pytest.importorskip("pricers.trees")
    S, r, q, T = 200.0, 0.05, 0.005, 0.25
    F, D = S * np.exp((r - q) * T), np.exp(-r * T)
    strikes = np.arange(150.0, 250.0 + 1e-9, 2.5)
    rows = []
    for right in ("C", "P"):
        mids = np.array([trees.binomial(S, K, T, 0.28 - 0.5 * np.log(K / F) + np.log(K / F) ** 2, right, n=400, r=r,
                                        q=q, exercise="american", bbs=True).price for K in strikes])
        rows.append(pd.DataFrame({"expiry": pd.Timestamp("2026-12-18"), "strike": strikes, "right": right,
                                  "bid": mids - 0.05, "ask": mids + 0.05, "bid_size": np.nan, "ask_size": np.nan,
                                  "volume": np.nan, "open_interest": np.nan, "root": "AAPL", "settlement": "PM"}))
    df = pd.concat(rows, ignore_index=True)[list(COLUMNS)]
    sl = Slice("AAPL", date(2026, 9, 18), "AAPL", pd.Timestamp("2026-12-18"), T, df, "american")
    pinned = fit_forward(sl, rate=r)
    err_bp = (pinned.forward / F - 1.0) * 1e4
    assert -15.0 < err_bp < -5.0, err_bp
    assert abs(pinned.discount - D) <= 1e-12
    free = fit_forward(sl, mode="parity_line", force=True, window=None)
    assert free.discount > 1.0 and abs(free.discount - 1.011) < 0.003
    assert (free.forward / F - 1.0) * 1e4 < -25.0


def test_fixed_discount_rejects_a_forward_far_from_the_atm_strike():
    """Mids that do not satisfy parity (every put shifted +$500) drive the pinned-discount estimator to F = -401 on
    a chain whose strikes are 80-120; that is not a forward and must be a ValueError (forward_table: a 'failed' row),
    not a fitted row that implied_vols rejects later."""
    from volsurf.forward import FORWARD_BAND, fit_forward, forward_table

    chain, F1, F2 = _two_expiry_chain(exercise="american")
    df = chain.df.copy()
    df.loc[df["right"] == "P", ["bid", "ask"]] += 500.0
    bad = Chain("SPY", chain.quote_date, df, exercise="american")
    sl = next(bad.slices())
    with pytest.raises(ValueError, match=f"within {FORWARD_BAND:g} of the ATM strike"):
        fit_forward(sl, rate=0.05)
    tbl = forward_table(bad, rate=0.05)
    assert (tbl["mode"] == "failed").all() and tbl["notes"].str.contains("ATM strike").all()
    assert fit_forward(next(chain.slices()), rate=0.05).forward == pytest.approx(F1, rel=1e-4)  # the clean chain fits

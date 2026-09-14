"""arb.py: the three checks on known-bad and known-good slices, the repair, and the Heston COS oracle (pricers)."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from volsurf import black
from volsurf.arb import (
    ArbitrageReport,
    Evidence,
    asymptotes,
    butterfly,
    calendar,
    check_arbitrage,
    quote_level,
    repair_jw,
)
from volsurf.svi import SVIParams, SVISlice, fit_svi, from_ssvi, to_jw
from volsurf.synth import SYNTH_QUOTE_DATE, SYNTH_SURFACE, synth_strikes, synthetic_chain

VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
MARKET_LIKE = SVIParams(0.0163, 0.3455, -0.8988, 0.0373, 0.1164)
GOOD = SVIParams(0.02, 0.4, -0.6, 0.05, 0.2)


def _heston_like_ssvi(lam=0.3, rho=-0.7, theta=20.0) -> SVISlice:
    """GJ2014 Example 4.1 phi(theta) = (1/(lambda theta))(1 - (1 - e^{-lambda theta})/(lambda theta))."""
    x = lam * theta
    phi = (1.0 / x) * (1.0 - (1.0 - math.exp(-x)) / x)
    return SVISlice(from_ssvi(theta, rho, phi), 1.0)


# ---------------------------------------------------------------- butterfly


def test_vogt_is_flagged_with_the_paper_minimum():
    """Oracle svi_g_vogt: g_min = -0.0328635735 at k = 0.87926255 (tol 1e-6 / 1e-5), Lee bound and w_min both pass."""
    s = SVISlice(VOGT, 1.0)
    assert s.params.b * (1 + abs(s.params.rho)) < 2 and s.min_total_variance() > 0
    r = butterfly(s, -1.5, 1.5)
    assert not r.ok
    assert r.worst_g == pytest.approx(-0.0328636, abs=1e-6)
    assert r.worst_k == pytest.approx(0.879263, abs=1e-5)
    assert r.n_inside > 0 and r.n_outside == 0
    assert r.evidence == Evidence.numerical_scan
    assert r.asymptote_left > 0 and r.asymptote_right > 0
    assert any("inside" in n for n in r.notes)


def test_market_like_slice_is_flagged_inside_a_normal_range():
    """Oracle svi_g_market_like: (0.0163, 0.3455, -0.8988, 0.0373, 0.1164), T = 1: g_min = -0.02003 at k = -0.276."""
    r = butterfly(SVISlice(MARKET_LIKE, 1.0), -0.5, 0.5)
    assert not r.ok and r.n_inside > 0
    assert r.worst_g == pytest.approx(-0.02003, abs=1e-4)
    assert r.worst_k == pytest.approx(-0.276, abs=1e-2)


def test_good_slice_passes_and_reports_the_range():
    r = butterfly(SVISlice(GOOD, 0.25), -0.4, 0.3)
    assert r.ok and r.n_inside == 0 and r.n_outside == 0 and r.worst_g > 0
    sigma_atm = math.sqrt(float(SVISlice(GOOD, 0.25).w(0.0)) / 0.25)
    margin = max(0.5, 2 * sigma_atm * math.sqrt(0.25))
    assert r.k_range_checked == pytest.approx((-0.4 - margin, 0.3 + margin))
    assert r.notes == ()


def test_raw_svi_asymptote_formula_matches_g_at_large_k():
    """lim g = (4 - b^2 (1 +/- rho)^2)/16, i.e. GJ Lemma 4.2's (16 - (theta phi)^2 (1 +/- rho)^2)/64 with b = theta phi/2."""
    for p in (VOGT, GOOD, MARKET_LIKE, from_ssvi(0.5, -0.4, 3.0)):
        s = SVISlice(p, 1.0)
        a_left, a_right = asymptotes(s)
        assert float(s.g(-1e7)) == pytest.approx(a_left, abs=1e-6)
        assert float(s.g(1e7)) == pytest.approx(a_right, abs=1e-6)
    theta, rho = 20.0, -0.7
    s = _heston_like_ssvi(0.3, rho, theta)
    phi = s.params.b * 2 / theta
    assert asymptotes(s)[0] == pytest.approx((16 - (theta * phi) ** 2 * (1 - rho) ** 2) / 64, rel=1e-12)


def test_heston_like_ssvi_passes_the_grid_but_fails_at_the_asymptote():
    """Oracle ssvi_heston_lambda_bound: lambda = 0.3 < (1+|rho|)/4: g > 0 on |k| <= 3 but the left asymptote is -0.0988."""
    s = _heston_like_ssvi()
    r = butterfly(s, -3.0, 3.0, margin=0.0)
    assert r.n_inside == 0 and r.n_outside == 0 and r.worst_g > 0
    assert not r.ok
    assert r.evidence == Evidence.analytic
    assert r.asymptote_left == pytest.approx(-0.0988, abs=1e-3)
    assert any("asymptote" in n for n in r.notes)
    assert _heston_like_ssvi(lam=0.5).wing_slopes()[0] < 2  # lambda = 0.5 passes Remark 4.5
    assert butterfly(_heston_like_ssvi(lam=0.5), -3.0, 3.0, margin=0.0).ok


def test_extrapolation_violation_is_counted_outside():
    """Vogt's violation lies on k in [0.6424, 1.2569]: a quoted range [-0.5, 0.5] sees it only outside."""
    r = butterfly(SVISlice(VOGT, 1.0), -0.5, 0.5, margin=1.0)
    assert not r.ok and r.n_inside == 0 and r.n_outside > 0
    assert any("extrapolation" in n for n in r.notes)


def test_butterfly_input_validation():
    with pytest.raises(ValueError):
        butterfly(SVISlice(GOOD, 0.25), 0.3, -0.4)


# ---------------------------------------------------------------- calendar


def test_calendar_crossing_pair_is_flagged_with_the_k():
    s1 = SVISlice(SVIParams(0.02, 0.4, -0.6, 0.05, 0.2), 0.25)
    s2 = SVISlice(SVIParams(0.02, 0.3, -0.6, 0.05, 0.2), 0.5)  # flatter wings: crosses s1 away from the ATM
    k = np.linspace(-1, 1, 401)
    gap = s2.w(k) - s1.w(k)
    r = calendar([(0.5, s2), (0.25, s1)], k_grid=k)
    assert not r.ok and r.crossings == int(np.sum(gap < -1e-10))
    assert r.worst_gap == pytest.approx(gap.min()) and r.worst_k == pytest.approx(k[np.argmin(gap)])
    assert r.pair == (0.25, 0.5) and r.k_range_checked == (-1.0, 1.0)


def test_calendar_monotone_pair_passes():
    s1 = SVISlice(SVIParams(0.02, 0.4, -0.6, 0.05, 0.2), 0.25)
    s2 = SVISlice(SVIParams(0.03, 0.4, -0.6, 0.05, 0.2), 0.5)
    r = calendar([(0.25, s1), (0.5, s2)])
    assert r.ok and r.crossings == 0 and r.worst_gap == pytest.approx(0.01)


def test_calendar_single_slice_and_validation():
    s1 = SVISlice(GOOD, 0.25)
    assert calendar([(0.25, s1)]).ok
    with pytest.raises(ValueError):
        calendar([(0.25, s1), (0.25, SVISlice(GOOD, 0.25))])
    with pytest.raises(ValueError):
        calendar([(0.25, s1), (0.5, SVISlice(GOOD, 0.4))])


# ---------------------------------------------------------------- quote level


def _brent_iv(price, F, K, T, right, D):
    f = lambda s: float(black.price(F, K, T, s, right, D)) - price  # noqa: E731
    return brentq(f, 1e-6, 5.0, xtol=1e-14)


def test_on_model_synthetic_quotes_have_zero_within_band_violations():
    """Convex Black prices from one slice: 0 raw-mid and 0 within-band violations at noise 0; with in-band noise the raw
    count can be > 0 while the within-band count stays 0 (the band contains the convex model prices)."""
    strikes = [synth_strikes(e.forward, T, 0.2, n=61) for e, T in zip(SYNTH_SURFACE, (0.05, 0.15, 0.4, 1.0), strict=True)]
    chain = synthetic_chain("SYN", SYNTH_QUOTE_DATE, SYNTH_SURFACE, strikes)
    for sl, e in zip(chain.slices(), SYNTH_SURFACE, strict=True):
        r = quote_level(sl.pairs(), e.forward, e.discount, sl.T)
        assert r.butterfly_raw_mid == 0 and r.butterfly_within_band == 0 and r.calendar_raw is None
        assert r.n_triplets == 2 * (len(sl.pairs()) - 2)
    noisy = synthetic_chain("SYN", SYNTH_QUOTE_DATE, SYNTH_SURFACE, strikes, noise_in_band=0.9, seed=7)
    raw = sum(quote_level(sl.pairs(), 100.0, 1.0, sl.T).butterfly_raw_mid for sl in noisy.slices())
    band = sum(quote_level(sl.pairs(), 100.0, 1.0, sl.T).butterfly_within_band for sl in noisy.slices())
    assert raw > 0 and band == 0


def test_quote_level_counts_a_concave_triplet():
    pairs = pd.DataFrame({"strike": [90.0, 100.0, 110.0], "call_bid": [12.0, 8.0, 1.0], "call_ask": [12.2, 8.2, 1.2],
                          "put_bid": [1.0, 3.0, 9.0], "put_ask": [1.2, 3.2, 9.2]})
    r = quote_level(pairs, 100.0, 1.0, 0.5)
    assert r.n_triplets == 2 and r.butterfly_raw_mid == 1 and r.butterfly_within_band == 1
    with pytest.raises(ValueError):
        quote_level(pairs, 0.0, 1.0, 0.5)
    assert quote_level(pairs.iloc[:2], 100.0, 1.0, 0.5).n_triplets == 0


# ---------------------------------------------------------------- aggregate + repair


def test_check_arbitrage_aggregates():
    s1 = SVISlice(SVIParams(0.02, 0.4, -0.6, 0.05, 0.2), 0.25)
    s2 = SVISlice(SVIParams(0.03, 0.4, -0.6, 0.05, 0.2), 0.5)
    rep = check_arbitrage([(0.25, s1, -0.4, 0.3), (0.5, s2, -0.5, 0.4)])
    assert isinstance(rep, ArbitrageReport) and rep.ok and len(rep.butterfly) == 2 and rep.calendar.ok
    bad = check_arbitrage([(1.0, SVISlice(VOGT, 1.0), -1.5, 1.5)])
    assert not bad.ok and any("single slice" in n for n in bad.notes) and any("g < 0" in n for n in bad.notes)


def test_repair_jw_is_the_gj_closed_form_and_removes_the_arbitrage():
    """GJ2014 section 5.1 / Example 5.1: c' = p + 2 psi = 0.3493158, vtilde' = 0.01548182 make Vogt butterfly-free.
    The paper's 'optimal' (c*, vtilde*) = (0.8564763, 0.0116249) comes from a penalised price-distance search; it is
    checked here only to be arbitrage-free too (min g = +0.0081), not produced by the closed form."""
    fixed = repair_jw(VOGT, 1.0)
    v, psi, p, c, vt = to_jw(fixed, 1.0)
    v0, psi0, p0, _, _ = to_jw(VOGT, 1.0)
    assert (v, psi, p) == pytest.approx((v0, psi0, p0), rel=1e-12)
    assert c == pytest.approx(0.3493158, abs=1e-7) and vt == pytest.approx(0.01548182, abs=1e-7)
    assert butterfly(SVISlice(fixed, 1.0), -3.0, 3.0).ok
    from volsurf.svi import from_jw

    optimal = SVISlice(from_jw(v0, psi0, p0, 0.8564763, 0.0116249, 1.0), 1.0)
    r = butterfly(optimal, -3.0, 3.0)
    assert r.ok and r.worst_g == pytest.approx(0.0081, abs=5e-4)


# ---------------------------------------------------------------- Heston COS oracle


@pytest.mark.parametrize("N", [1024])
def test_heston_cos_surface_is_free_of_static_arbitrage(N):
    """pricers Heston COS (v0=.04, kappa=1.5, theta=.05, sigma_v=.6, rho=-.7), S=100, r=.02, q=.01, T in {0.1, 0.25,
    0.5, 1, 2}, k in [-1.0, 0.6]: 0 within-band quote-level violations on the exact prices and 0 calendar crossings
    between the fitted SVI slices on |k| <= 4 sigma_atm sqrt(T). Skipped when pricers is not installed."""
    heston = pytest.importorskip("pricers.heston")
    hp = heston.HestonParams(v0=0.04, kappa=1.5, theta=0.05, sigma_v=0.6, rho=-0.7)
    S, r, q = 100.0, 0.02, 0.01
    fits = []
    for T in (0.1, 0.25, 0.5, 1.0, 2.0):
        F, D = S * math.exp((r - q) * T), math.exp(-r * T)
        k = np.linspace(-1.0, 0.6, 33)
        K = F * np.exp(k)
        calls = heston.price_strikes(S, K, T, "C", hp, r, q, N=N)
        puts = heston.price_strikes(S, K, T, "P", hp, r, q, N=N)
        pairs = pd.DataFrame({"strike": K, "call_bid": calls, "call_ask": calls, "put_bid": puts, "put_ask": puts})
        ql = quote_level(pairs, F, D, T)
        assert ql.butterfly_within_band == 0 and ql.butterfly_raw_mid == 0
        otm = np.where(k < 0, puts, calls)
        keep = otm > 1e-8 * F  # deep wings price at 0 to COS precision: no Black inverse (the iv module returns NaN there)
        iv = np.array([_brent_iv(p, F, Ki, T, "P" if ki < 0 else "C", D)
                       for p, Ki, ki in zip(otm[keep], K[keep], k[keep], strict=True)])
        fit = fit_svi(k[keep], iv**2 * T, T)
        assert fit.rmse_vol < 0.5  # SVI is not Heston; a coarse bar so the test is about the arbitrage checks
        sigma_atm = float(SVISlice(fit.params, T).iv(0.0))
        half = 4 * sigma_atm * math.sqrt(T)
        fits.append((T, SVISlice(fit.params, T), -half, half))
    kmax = max(h for _, _, _, h in fits)
    cal = calendar([(T, s) for T, s, _, _ in fits], k_grid=np.linspace(-kmax, kmax, 801))
    assert cal.ok and cal.crossings == 0
    rep = check_arbitrage(fits, k_grid=np.linspace(-kmax, kmax, 801))  # butterfly per fitted slice + calendar
    assert rep.ok and all(b.n_inside == 0 and b.n_outside == 0 and b.worst_g > 0 for b in rep.butterfly)
    assert all(b.asymptote_left > 0 and b.asymptote_right > 0 for b in rep.butterfly)

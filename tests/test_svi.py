"""svi.py: curve identities, JW/SSVI maps, the Lee bound, and the fitter's recovery / speed / weighting contract."""

import math
import time

import numpy as np
import pytest

from volsurf.svi import (
    LEE_BOUND,
    SVIFit,
    SVIParams,
    SVISlice,
    fit_svi,
    fit_svi_multistart,
    from_jw,
    from_ssvi,
    to_jw,
)

VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)  # GJ2014 Example 3.1, eq 3.8, t = 1
TRUE = SVIParams(0.02, 0.4, -0.6, 0.05, 0.2)
K25 = np.linspace(-0.4, 0.3, 25)


def _p(params: SVIParams) -> np.ndarray:
    return np.array(params.as_tuple())


# ---------------------------------------------------------------- curve identities


def test_derivatives_match_finite_differences():
    s = SVISlice(TRUE, 0.25)
    k = np.linspace(-0.8, 0.8, 33)
    h = 1e-5
    fd1 = (s.w(k + h) - s.w(k - h)) / (2 * h)
    fd2 = (s.w(k + h) - 2 * s.w(k) + s.w(k - h)) / h**2
    assert np.max(np.abs(s.dw(k) - fd1)) < 1e-9
    assert np.max(np.abs(s.d2w(k) - fd2)) < 1e-5


def test_min_total_variance_is_the_grid_minimum_and_wing_slopes_are_the_limits():
    s = SVISlice(TRUE, 0.25)
    k = np.linspace(-5, 5, 200001)
    assert abs(s.w(k).min() - s.min_total_variance()) < 1e-9
    left, right = s.wing_slopes()
    assert abs(float(s.dw(-1e6)) + left) < 1e-9 and abs(float(s.dw(1e6)) - right) < 1e-9  # w falls on the left
    assert left == pytest.approx(0.4 * 1.6) and right == pytest.approx(0.4 * 0.4)


def test_g_formula_matches_the_paper_expression_and_iv_is_sqrt_w_over_T():
    s = SVISlice(VOGT, 1.0)
    k = np.linspace(-1.5, 1.5, 301)
    w, wp, wpp = s.w(k), s.dw(k), s.d2w(k)
    g_ref = (1 - k * wp / (2 * w)) ** 2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2  # GJ2014 eq 2.1
    assert np.max(np.abs(s.g(k) - g_ref)) < 1e-14
    assert np.max(np.abs(s.iv(k) - np.sqrt(w / 1.0))) < 1e-15


def test_params_validation():
    with pytest.raises(ValueError):
        SVIParams(0.0, -0.1, 0.0, 0.0, 0.1)
    with pytest.raises(ValueError):
        SVIParams(0.0, 0.1, 1.0, 0.0, 0.1)
    with pytest.raises(ValueError):
        SVIParams(0.0, 0.1, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        SVISlice(TRUE, 0.0)


def test_lee_bound_is_two_in_total_variance():
    """Lee (2004): the asymptotic slope of TOTAL implied variance is <= 2 (GJ2014 Remark 4.3; Martini-Mingone 2021),
    hence b(1+|rho|) <= 2. The 4 printed by Gatheral 2004 / Zeliade 2009 is the same bound with a dropped factor:
    a slope-4 wing would imply a density with an infinite first moment."""
    assert LEE_BOUND == 2.0
    # a slope-2 wing sits exactly at g -> 0 in the limit; slope 4 would give g -> (4 - 16)/16 < 0 at the asymptote
    b = 2.0
    s = SVISlice(SVIParams(0.1, b, 0.0, 0.0, 0.2), 1.0)
    assert float(s.g(1e6)) == pytest.approx(0.0, abs=1e-5)
    s4 = SVISlice(SVIParams(0.1, 4.0, 0.0, 0.0, 0.2), 1.0)
    assert float(s4.g(1e6)) < -0.7


# ---------------------------------------------------------------- parameter maps


def test_jw_of_vogt_matches_gj_example_5_1_and_round_trips():
    """GJ2014 Example 5.1: (v, psi, p, c, vtilde) = (0.01742625, -0.1752111, 0.6997381, 1.316798, 0.0116249)."""
    jw = to_jw(VOGT, 1.0)
    ref = (0.0174263, -0.1752111, 0.6997381, 1.3167982, 0.0116249)
    assert np.max(np.abs(np.array(jw) - np.array(ref))) < 1e-7
    back = from_jw(*jw, 1.0)
    assert np.max(np.abs(_p(back) - _p(VOGT))) < 1e-12


@pytest.mark.parametrize("params,T", [(TRUE, 0.25), (SVIParams(0.01, 0.2, 0.3, -0.1, 0.3), 2.0),
                                      (SVIParams(0.004, 0.05, -0.85, 0.0, 0.05), 0.08)])
def test_jw_round_trip_generic(params, T):
    back = from_jw(*to_jw(params, T), T)
    assert np.max(np.abs(_p(back) - _p(params))) < 1e-12


def test_jw_round_trip_m_zero_branch():
    p = SVIParams(0.01, 0.3, -0.4, 0.0, 0.15)
    back = from_jw(*to_jw(p, 0.5), 0.5)
    assert np.max(np.abs(_p(back) - _p(p))) < 1e-10


def test_jw_meaning_atm_variance_and_skew():
    v, psi, p, c, vt = to_jw(TRUE, 0.25)
    s = SVISlice(TRUE, 0.25)
    assert v == pytest.approx(float(s.w(0.0)) / 0.25, rel=1e-14)
    h = 1e-6
    fd_skew = (math.sqrt(float(s.w(h))) - math.sqrt(float(s.w(-h)))) / (2 * h)  # d sqrt(w)/dk = sqrt(T) dsigma/dk
    assert psi == pytest.approx(fd_skew, abs=1e-8)
    assert vt == pytest.approx(s.min_total_variance() / 0.25, rel=1e-14)


def test_from_ssvi_reproduces_the_ssvi_formula():
    """(theta, rho, phi) = (0.01, -0.7, 7.9603): raw (0.00255, 0.0398015, -0.7, 0.0879364, 0.0897131); |dw| <= 1e-15."""
    theta, rho, phi = 0.01, -0.7, 7.9603
    p = from_ssvi(theta, rho, phi)
    assert np.allclose(_p(p), [0.00255, 0.0398015, -0.7, 0.0879364, 0.0897131], atol=1e-7)
    k = np.linspace(-2, 2, 801)
    w_ssvi = theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))
    assert np.max(np.abs(SVISlice(p, 1.0).w(k) - w_ssvi)) <= 1e-15


# ---------------------------------------------------------------- fitting


def test_exact_recovery_from_a_synthetic_slice():
    """Oracle svi_recovery_zeliade: true (0.02, 0.4, -0.6, 0.05, 0.2), T = 0.25, 25 strikes k in linspace(-0.4, 0.3, 25):
    max |dparam| <= 1e-8 (research achieved 1.55e-13)."""
    w = SVISlice(TRUE, 0.25).w(K25)
    fit = fit_svi(K25, w, 0.25)
    assert isinstance(fit, SVIFit)
    assert np.max(np.abs(_p(fit.params) - _p(TRUE))) <= 1e-8
    assert fit.objective < 1e-20 and fit.rmse_vol < 1e-8 and fit.n == 25 and fit.converged
    assert fit.wing_slopes == pytest.approx((0.4 * 1.6, 0.4 * 0.4), abs=1e-8)


def test_vogt_fits_from_its_own_w_with_negative_a():
    """Vogt's a = -0.0410 < 0: a fitter with Zeliade's 0 <= a box could not return it (oracle zeliade_abox_defect)."""
    k = np.linspace(-1.5, 1.5, 60)
    fit = fit_svi(k, SVISlice(VOGT, 1.0).w(k), 1.0)
    assert fit.params.a < 0
    assert np.max(np.abs(_p(fit.params) - _p(VOGT))) <= 1e-8


def test_fit_respects_constraints_when_the_data_want_to_break_them():
    """Data from a slope-3 wing (b = 3, rho = 0): the fit must stop at b(1+|rho|) = 2 and w_min >= 0."""
    k = np.linspace(-1.0, 1.0, 41)
    w = SVISlice(SVIParams(0.05, 3.0, 0.0, 0.0, 0.3), 1.0).w(k)
    fit = fit_svi(k, w, 1.0)
    b, rho = fit.params.b, fit.params.rho
    assert b * (1 + abs(rho)) <= LEE_BOUND + 1e-7
    assert SVISlice(fit.params, 1.0).min_total_variance() >= -1e-9
    assert "slsqp" in fit.method


def _noisy(n=150, seed=0, noise=0.001):
    rng = np.random.default_rng(seed)
    k = np.linspace(-0.4, 0.3, n)
    iv = SVISlice(TRUE, 0.25).iv(k) + rng.normal(0.0, noise, n)
    return k, iv**2 * 0.25


def test_grid_and_multistart_reach_the_same_objective():
    """Oracle svi_grid_init_speed: the grid initialiser + one polish lands on the 36-start SLSQP optimum (|dobj| <= 1e-9)."""
    k, w = _noisy()
    f1 = fit_svi(k, w, 0.25)
    f2 = fit_svi_multistart(k, w, 0.25)
    assert abs(f1.objective - f2.objective) <= 1e-9
    assert f2.starts_tried == 36 and f2.method == "multistart-slsqp"
    assert np.max(np.abs(_p(f1.params) - _p(f2.params))) < 1e-4


def test_fit_is_fast_on_150_quotes():
    k, w = _noisy()
    t0 = time.perf_counter()
    fit = fit_svi(k, w, 0.25)
    dt = time.perf_counter() - t0
    assert dt < 2.0 and fit.wall_time < 2.0  # 0.02 s on the reference laptop; 2 s for slow CI


def test_weights_enter_as_weight_times_squared_residual():
    """Weight 2 on one quote must equal duplicating that quote (sum w r^2); (w r)^2 would count it four times."""
    k, w = _noisy(n=60, seed=3, noise=0.002)
    i = 30
    ww = np.ones_like(w)
    ww[i] = 2.0
    f_weighted = fit_svi(k, w, 0.25, weights=ww)
    k_dup = np.append(k, k[i])
    w_dup = np.append(w, w[i])
    f_dup = fit_svi(k_dup, w_dup, 0.25)
    assert np.max(np.abs(_p(f_weighted.params) - _p(f_dup.params))) < 1e-7
    assert f_weighted.objective == pytest.approx(f_dup.objective, rel=1e-8)
    ww4 = ww.copy()
    ww4[i] = 4.0
    f_four = fit_svi(k, w, 0.25, weights=ww4)
    d2 = np.max(np.abs(_p(f_weighted.params) - _p(fit_svi(k, w, 0.25).params)))
    d4 = np.max(np.abs(_p(f_four.params) - _p(fit_svi(k, w, 0.25).params)))
    assert d4 > d2 > 0  # the fit moves with the weight, monotonically


def test_stationarity_of_the_weighted_objective():
    k, w = _noisy(n=80, seed=5)
    ww = np.linspace(0.5, 2.0, 80)
    fit = fit_svi(k, w, 0.25, weights=ww)
    s = SVISlice(fit.params, 0.25)
    r = s.w(k) - w
    grad_a = 2 * np.sum(ww * r)  # d objective / d a
    assert abs(grad_a) < 1e-9 * max(1.0, np.sum(ww * r * r)) + 1e-12


def test_rmse_is_in_vol_points():
    k, w = _noisy(n=40, seed=1, noise=0.001)
    fit = fit_svi(k, w, 0.25)
    iv_fit = SVISlice(fit.params, 0.25).iv(k)
    rmse_vp = 100 * math.sqrt(np.mean((iv_fit - np.sqrt(w / 0.25)) ** 2))
    assert fit.rmse_vol == pytest.approx(rmse_vp, rel=1e-10)
    assert 0.05 < fit.rmse_vol < 0.15  # 0.1 vp noise in, ~0.1 vp residual out
    assert fit.max_abs_vol >= fit.rmse_vol


def test_polish_false_returns_the_grid_point():
    k, w = _noisy(n=50, seed=2)
    fit = fit_svi(k, w, 0.25, polish=False)
    assert fit.method == "grid" and fit.starts_tried == 0
    assert fit.objective >= fit_svi(k, w, 0.25).objective


def test_input_validation():
    with pytest.raises(ValueError):
        fit_svi([0, 0.1, 0.2], [0.01, 0.01, 0.01], 0.5)
    with pytest.raises(ValueError):
        fit_svi(K25, SVISlice(TRUE, 0.25).w(K25), 0.0)
    with pytest.raises(ValueError):
        fit_svi(K25, SVISlice(TRUE, 0.25).w(K25), 0.25, weights=np.ones(3))

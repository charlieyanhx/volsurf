"""black.py: Black-76 price and Greeks against finite differences, the sigma = 0 / T <= 0 edges, NaN propagation, and
the `right` encodings.

Oracles: every Greek is the central finite difference of `price` (F = 100, K = 95, T = 0.7, sigma = 0.3, D = 0.97:
gamma 0.014601, theta/day -0.018001, vanna -0.096334, volga 2.65862; delta_spot 0.525 = 0.5 * 105 / 100 for a forward
delta of 0.5 with F = 105, S = 100), to 1e-6 relative; put-call parity C - P = D (F - K) to 1e-12.
"""

import numpy as np
import pytest

from volsurf import black

F, K, T, SIG, D = 100.0, 95.0, 0.7, 0.3, 0.97


def _fd(fn, x, h):
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


def test_greeks_match_finite_differences_of_price():
    g = black.greeks(F, K, T, SIG, "C", D)
    p = black.price(F, K, T, SIG, "C", D)
    assert g.price == pytest.approx(p)
    assert g.delta == pytest.approx(_fd(lambda f: black.price(f, K, T, SIG, "C", D), F, 1e-4), rel=1e-7)
    assert g.gamma == pytest.approx((black.price(F + 1e-3, K, T, SIG, "C", D) - 2 * p + black.price(F - 1e-3, K, T, SIG, "C", D)) / 1e-6, rel=1e-5)
    assert g.gamma == pytest.approx(0.014601, abs=2e-6)
    assert g.vega == pytest.approx(_fd(lambda s: black.price(F, K, T, s, "C", D), SIG, 1e-5), rel=1e-7)
    assert g.vega == pytest.approx(black.vega(F, K, T, SIG, D))
    # theta per calendar day holding F and D fixed: d price / d T in years, divided by 365, with the sign flipped
    assert g.theta == pytest.approx(-_fd(lambda t: black.price(F, K, t, SIG, "C", D), T, 1e-5) / black.YEAR, rel=1e-7)
    assert g.theta == pytest.approx(-0.0180013, abs=1e-6)
    assert g.vanna == pytest.approx(_fd(lambda f: black.vega(f, K, T, SIG, D), F, 1e-4), rel=1e-6)
    assert g.vanna == pytest.approx(-0.096334, abs=2e-6)
    assert g.volga == pytest.approx(_fd(lambda s: black.vega(F, K, T, s, D), SIG, 1e-5), rel=1e-6)
    assert g.volga == pytest.approx(2.65862, abs=2e-5)
    gp = black.greeks(F, K, T, SIG, "P", D)
    assert gp.delta == pytest.approx(g.delta - D) and gp.gamma == pytest.approx(g.gamma) and gp.vega == pytest.approx(g.vega)
    assert g.price - gp.price == pytest.approx(D * (F - K), abs=1e-12)  # parity
    assert black.delta_spot(0.5, 105.0, 100.0) == pytest.approx(0.525)


def test_sigma_zero_prices_intrinsic_and_expired_raises():
    assert black.price(F, K, T, 0.0, "C", D) == pytest.approx(D * (F - K))
    assert black.price(F, K, T, 0.0, "P", D) == 0.0 and black.vega(F, K, T, 0.0) == 0.0
    with pytest.raises(ValueError):
        black.price(F, K, 0.0, SIG)
    with pytest.raises(ValueError):
        black.price(F, K, T, -0.1)
    with pytest.raises(ValueError):
        black.greeks(F, K, T, 0.0)


def test_nan_vol_propagates_never_prices_at_intrinsic():
    """The inverter answers NaN below its information floor and an SVI slice answers NaN where w < 0; repricing that
    must stay NaN (a `sigma > 0` test would silently turn it into intrinsic dollars / a zero vega)."""
    assert np.isnan(black.price(100.0, 90.0, 1.0, np.nan, "C"))
    assert np.isnan(black.price(100.0, 110.0, 1.0, np.nan, "C"))
    assert np.isnan(black.vega(100.0, 100.0, 1.0, np.nan))
    g = black.greeks(100.0, 100.0, 1.0, np.array([0.2, np.nan]))
    assert np.isfinite(g.price[0]) and np.isnan(g.price[1]) and np.isnan(g.delta[1]) and np.isnan(g.vega[1])
    mixed = black.price(100.0, [90.0, 110.0, 100.0], 1.0, [0.2, np.nan, 0.0], "C")
    assert np.isfinite(mixed[0]) and np.isnan(mixed[1]) and mixed[2] == 0.0


def test_right_encodings_agree():
    want = black.price(F, [90.0, 90.0], 1.0, 0.2, ["C", "P"])
    assert want[0] > want[1]
    np.testing.assert_allclose(black.price(F, [90.0, 90.0], 1.0, 0.2, np.array([True, False])), want)
    np.testing.assert_allclose(black.price(F, [90.0, 90.0], 1.0, 0.2, np.array([True, False], dtype=object)), want)
    np.testing.assert_allclose(black.price(F, [90.0, 90.0], 1.0, 0.2, np.array(["c", "p"], dtype=object)), want)

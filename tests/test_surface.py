"""surface.py: the whole chain path on the reference synthetic surface, skew by true Black delta, skips with reasons,
the two calendar grids, coverage buckets, and the American-chain rate requirement.

Oracles (docs/PLAN.md): on the on-model reference chain every SVI parameter is recovered to 1e-8 through the whole
path (forward -> IVs -> selection -> fit), F to 1e-9 and D to 1e-12, 100 % coverage under every weighting, 0 within-band
violations, 0 calendar crossings; `skew(expiry, 0.25)` returns strikes at which black.greeks delta (D = 1) is exactly
+0.25 (call) and -0.25 (put) to 1e-10; with in-band noise the model price still lies inside every band (100 %
coverage) and the within-band count stays 0 while the raw-mid count is > 0.
"""

import numpy as np
import pandas as pd
import pytest

from volsurf import black
from volsurf.report import reference_chain
from volsurf.surface import K_BUCKETS, Surface, coverage_by_k, fit_slice, fit_surface
from volsurf.synth import SYNTH_QUOTE_DATE, SYNTH_SURFACE, synth_strikes, synthetic_chain


@pytest.fixture(scope="module")
def exact() -> Surface:
    return fit_surface(reference_chain(0.0))


def test_exact_chain_recovers_every_expiry_through_the_whole_path(exact):
    assert len(exact.fits) == 4 and exact.skipped == ()
    for f, e in zip(exact.fits, SYNTH_SURFACE, strict=True):
        assert np.max(np.abs(np.array(f.params.as_tuple()) - np.array(e.params.as_tuple()))) <= 1e-8
        assert abs(f.forward.forward - e.forward) <= 1e-9
        assert abs(f.forward.discount - e.discount) <= 1e-12
        assert f.forward.mode == "parity_line"
        assert f.coverage_pct == 100.0 and f.inside.all()
        assert f.svi.rmse_vol < 1e-8 and f.butterfly.ok
        assert f.quote_level.butterfly_within_band == 0 and f.quote_level.butterfly_raw_mid == 0
        assert f.n == int(f.mask.sum()) == len(f.weights) == len(f.inside)
    assert exact.calendar.ok and exact.calendar.crossings == 0
    assert exact.discount_source == "discount fitted on the parity line"


def test_every_weighting_gives_full_coverage_on_the_on_model_chain():
    chain = reference_chain(0.0)
    for weighting in ("spread", "vega", "unit"):
        s = fit_surface(chain, weighting=weighting)
        assert len(s.fits) == 4 and all(f.coverage_pct == 100.0 for f in s.fits), weighting
        assert all(f.weighting == weighting for f in s.fits)
    with pytest.raises(ValueError, match="weighting"):
        fit_surface(chain, weighting="volume")


def test_skew_strikes_have_black_delta_exactly_plus_minus_a_quarter(exact):
    for f in exact.fits:
        sk = exact.skew(f.expiry, 0.25)
        F = f.forward.forward
        d_call = black.greeks(F, sk["strike_call"], f.T, sk["iv_call"], "C", 1.0).delta
        d_put = black.greeks(F, sk["strike_put"], f.T, sk["iv_put"], "P", 1.0).delta
        assert abs(float(d_call) - 0.25) <= 1e-10 and abs(float(d_put) + 0.25) <= 1e-10
        assert sk["skew"] == pytest.approx(sk["iv_put"] - sk["iv_call"])
        assert sk["skew"] > 0 and sk["k_put"] < 0 < sk["k_call"]  # negative-skew surface
        assert float(exact.implied_vol(f.expiry, sk["strike_put"])) == pytest.approx(sk["iv_put"], abs=1e-12)
    with pytest.raises(ValueError, match="delta"):
        exact.skew(exact.fits[0].expiry, 0.6)


def test_term_structure_and_queries(exact):
    ts = exact.term_structure()
    assert list(ts["expiry"]) == [e for _, e in exact.expiries()]
    assert (np.diff(ts["T"]) > 0).all() and (np.diff(ts["atm_vol"]) > 0).all()  # SYNTH_SURFACE: rising term structure
    assert (ts["skew_25d"] > 0).all() and (ts["coverage_pct"] == 100.0).all()
    e = exact.fits[1].expiry
    assert exact.atm_vol(e) == pytest.approx(float(exact.fits[1].iv(0.0)))
    assert exact.atm_vol(str(e.date())) == exact.atm_vol(e)
    assert exact.implied_vol(e, exact.fits[1].forward.forward) == pytest.approx(exact.atm_vol(e))
    with pytest.raises(KeyError):
        exact.fit(pd.Timestamp("2030-01-01"))


def test_report_dict_keys_and_coverage_buckets(exact):
    r = exact.report()
    for key in ("expiries_fitted", "expiries_skipped", "skipped", "quotes_in", "quotes_selected", "quotes_dropped",
                "rmse_vp_median", "rmse_vp_worst", "coverage_pct_median", "coverage_pct_worst", "coverage_by_k",
                "slices_g_neg_inside", "slices_g_neg_outside", "within_band_violations", "raw_mid_violations",
                "n_triplets", "calendar_crossings", "calendar_worst_gap", "forward_residual_rms_median",
                "forward_residual_rms_worst", "discount_source", "wall_time"):
        assert key in r, key
    assert r["expiries_fitted"] == 4 and r["expiries_skipped"] == 0
    assert r["quotes_in"] == len(reference_chain(0.0).df)
    assert r["quotes_selected"] == sum(f.n for f in exact.fits)
    assert r["quotes_in"] - r["quotes_selected"] == sum(r["quotes_dropped"].values())  # the ledgers conserve rows
    buckets = coverage_by_k(exact.fits)
    assert list(buckets["bucket"]) == list(r["coverage_by_k"])
    assert int(buckets["n"].sum()) == r["quotes_selected"] and (buckets["coverage_pct"] == 100.0).all()
    assert len(K_BUCKETS) == 4 and coverage_by_k([])["n"].sum() == 0


def test_noisy_chain_keeps_the_model_inside_every_band_and_within_band_count_zero():
    s = fit_surface(reference_chain(0.5))
    assert len(s.fits) == 4
    r = s.report()
    assert r["coverage_pct_worst"] == 100.0 and r["within_band_violations"] == 0 and r["raw_mid_violations"] > 0
    assert 0.0 < r["rmse_vp_median"] < 0.2
    assert s.calendar.ok and s.calendar.k_range_checked is not None  # checked on the quoted k-range only
    lo, hi = s.calendar.k_range_checked
    assert lo >= min(f.k_range[0] for f in s.fits) and hi <= max(f.k_range[1] for f in s.fits)
    wide = fit_surface(reference_chain(0.5), calendar_k_grid=np.linspace(-1.0, 1.0, 401))
    assert wide.calendar.k_range_checked == (-1.0, 1.0)


def test_skipped_slices_carry_their_reason():
    strikes = [synth_strikes(e.forward, T, 0.2, n=81, width=4.0, step=0.5)
               for e, T in zip(SYNTH_SURFACE, (0.05, 0.15, 0.4, 1.0), strict=True)]
    strikes[2] = strikes[2][::9]  # too few strikes on the third expiry
    chain = synthetic_chain("SYN", SYNTH_QUOTE_DATE, SYNTH_SURFACE, strikes,
                            half_spread=lambda p, k: np.maximum(0.02, 0.01 * np.asarray(p)))
    s = fit_surface(chain, min_quotes=8)
    assert len(s.fits) == 3 and len(s.skipped) == 1
    label, reason = s.skipped[0]
    assert "2026-11-08" in label and ("min_quotes" in reason or "min_pairs" in reason)
    assert s.report()["expiries_skipped"] == 1
    with pytest.raises(ValueError):
        fit_slice(next(sl for sl in chain.slices() if sl.expiry == pd.Timestamp("2026-11-08")), min_quotes=8)


def test_american_chain_needs_a_rate_and_then_pins_the_discount():
    strikes = [synth_strikes(e.forward, T, 0.2, n=81, width=4.0, step=0.5)
               for e, T in zip(SYNTH_SURFACE, (0.05, 0.15, 0.4, 1.0), strict=True)]
    chain = synthetic_chain("SPY", SYNTH_QUOTE_DATE, SYNTH_SURFACE, strikes, exercise="american",
                            half_spread=lambda p, k: np.maximum(0.02, 0.01 * np.asarray(p)))
    s = fit_surface(chain)
    assert len(s.fits) == 0 and len(s.skipped) == 4 and all("rate" in r for _, r in s.skipped)
    assert s.discount_source == "no slice fitted"
    s = fit_surface(chain, rate=0.045)
    assert len(s.fits) == 4 and all(f.forward.mode == "fixed_discount" for f in s.fits)
    assert s.discount_source.startswith("discount pinned from rate 0.045")
    for f, e in zip(s.fits, SYNTH_SURFACE, strict=True):
        assert abs(f.forward.forward / e.forward - 1.0) < 1e-4  # Black-priced chain: within 1 bp
        assert abs(f.forward.discount - np.exp(-0.045 * f.T)) < 1e-12


def test_selection_kwargs_reach_weights_select(exact):
    chain = reference_chain(0.0)
    s = fit_surface(chain, k_max=0.1)
    assert all(abs(f.selected()["k"]).max() <= 0.1 for f in s.fits)
    assert all(any(rule.startswith("|k| <=") for rule, _, _ in f.ledger.steps) for f in s.fits)
    assert sum(f.n for f in s.fits) < sum(f.n for f in exact.fits)

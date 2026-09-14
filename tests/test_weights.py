"""weights.py: the selection ledger conserves rows, every rule fires where it should, and the weight schemes behave."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from volsurf import black
from volsurf.quotes import Ledger
from volsurf.svi import SVIParams, SVISlice, fit_svi
from volsurf.weights import select, spread_weights, unit_weights, vega_weights

TRUE = SVIParams(0.02, 0.4, -0.6, 0.05, 0.2)
T, F, D = 0.25, 100.0, 0.99


def _brent_iv(price, K, right):
    f = lambda s: float(black.price(F, K, T, s, right, D)) - price  # noqa: E731
    lo, hi = f(1e-6), f(5.0)
    if not np.isfinite(price) or lo * hi > 0:
        return np.nan
    return brentq(f, 1e-6, 5.0, xtol=1e-14)


def _on_model_frame(half_spread=0.02, n=31) -> pd.DataFrame:
    """The shape `iv.implied_vols` returns: OTM quotes both sides, bands symmetric in price, IVs by a local Brent."""
    k = np.linspace(-0.3, 0.3, n)
    K = F * np.exp(k)
    right = np.where(k < 0, "P", "C")
    sigma = SVISlice(TRUE, T).iv(k)
    mid = black.price(F, K, T, sigma, right, D)
    bid, ask = mid - half_spread, mid + half_spread
    rows = []
    for i in range(n):
        rows.append({"strike": K[i], "right": right[i], "k": k[i], "bid": bid[i], "ask": ask[i],
                     "iv_bid": _brent_iv(bid[i], K[i], right[i]), "iv_mid": _brent_iv(mid[i], K[i], right[i]),
                     "iv_ask": _brent_iv(ask[i], K[i], right[i])})
    return pd.DataFrame(rows)


def test_on_model_frame_passes_every_rule_and_the_ledger_conserves_rows():
    df = _on_model_frame()
    mask, ledger = select(df, T, k_max=0.35)
    assert isinstance(ledger, Ledger)
    assert mask.all() and ledger.total_in == len(df) == ledger.total_out
    steps = ledger.steps
    assert len(steps) == 6  # two-sided, min bid, finite iv, OTM, rel spread, |k|
    assert all(steps[i][2] == steps[i + 1][1] for i in range(len(steps) - 1))
    assert ledger.dropped() == {}


def test_every_rule_fires_in_order_and_counts_add_up():
    df = _on_model_frame()
    df.loc[0, "bid"] = 0.0                       # one-sided
    df.loc[1, ["bid", "ask"]] = [0.01, 0.10]     # below 2 ticks
    df.loc[2, "iv_mid"] = np.nan                 # no Black inverse
    df.loc[3, "right"] = "C"                     # an ITM call at k < 0
    df.loc[4, ["bid", "ask"]] = [1.0, 1.5]       # relative spread 0.4
    df.loc[25, "k"] = 0.9                        # a call pushed beyond k_max
    mask, ledger = select(df, T, k_max=0.5)
    assert (~mask[:5]).all() and not mask[25] and mask[5:25].all() and mask[26:].all()
    names = [s[0] for s in ledger.steps]
    assert names[0].startswith("two-sided") and "OTM" in names[3] and names[5].startswith("|k|")
    assert [n_in - n_out for _, n_in, n_out in ledger.steps] == [1, 1, 1, 1, 1, 1]
    assert ledger.total_out == len(df) - 6
    assert all(ledger.steps[i][2] == ledger.steps[i + 1][1] for i in range(5))


def test_otm_only_off_and_no_k_max_skip_those_rules():
    df = _on_model_frame()
    df.loc[3, "right"] = "C"
    mask, ledger = select(df, T, otm_only=False)
    assert mask.all() and len(ledger.steps) == 4


def test_select_validation():
    df = _on_model_frame()
    with pytest.raises(ValueError):
        select(df.drop(columns=["iv_ask"]), T)
    with pytest.raises(ValueError):
        select(df, 0.0)
    with pytest.raises(ValueError):
        select(df, T, tick=0.0)


def test_spread_weights_monotone_decreasing_in_spread_and_floored():
    iv_bid = np.full(6, 0.20)
    iv_ask = 0.20 + np.array([0.0, 0.00005, 0.001, 0.01, 0.05, 0.2])
    w = spread_weights(iv_bid, iv_ask)
    assert np.all(np.diff(w) <= 0)
    assert w[0] == w[1] == 1.0 / 1e-4**2  # both under the floor
    assert w[3] == pytest.approx(1.0 / 0.01**2)
    assert np.isnan(spread_weights([np.nan], [0.2]))[0]
    with pytest.raises(ValueError):
        spread_weights(iv_bid, iv_ask, floor=0.0)


def test_vega_and_unit_weights():
    K = np.array([80.0, 100.0, 120.0])
    w = vega_weights(F, K, T, 0.2)
    assert np.allclose(w, black.vega(F, K, T, 0.2))
    assert w[1] > w[0] and w[1] > w[2]  # ATM has the most vega
    assert np.array_equal(unit_weights(4), np.ones(4))


def test_every_weighting_gives_full_coverage_on_an_on_model_frame():
    """On-model quotes: the fitted slice sits inside every bid/ask band under unit, spread and vega weights."""
    df = _on_model_frame()
    mask, _ = select(df, T)
    k, w = df["k"].to_numpy(), df["iv_mid"].to_numpy() ** 2 * T
    K = df["strike"].to_numpy()
    for weights in (unit_weights(len(df)), spread_weights(df["iv_bid"], df["iv_ask"]),
                    vega_weights(F, K, T, df["iv_mid"].to_numpy())):
        fit = fit_svi(k[mask], w[mask], T, weights=np.asarray(weights)[mask])
        iv_fit = SVISlice(fit.params, T).iv(k)
        price_fit = black.price(F, K, T, iv_fit, df["right"].to_numpy().astype(str), D)  # object dtype != "U" in black
        inside = (price_fit >= df["bid"].to_numpy() - 1e-12) & (price_fit <= df["ask"].to_numpy() + 1e-12)
        assert inside.mean() == 1.0

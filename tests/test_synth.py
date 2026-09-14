"""synth.py: the chain validates, parity is exact on the mids at noise 0, SYNTH_SURFACE has the documented properties,
and the Cboe-shaped JSON carries every key of the real feed."""

import json
from datetime import date

import numpy as np
import pytest

from volsurf import black
from volsurf.arb import butterfly, calendar
from volsurf.quotes import COLUMNS, Chain, time_to_expiry
from volsurf.svi import SVIParams, SVISlice
from volsurf.synth import (
    SYNTH_QUOTE_DATE,
    SYNTH_SURFACE,
    SynthExpiry,
    default_half_spread,
    occ_symbol,
    synth_strikes,
    synthetic_cboe_json,
    synthetic_chain,
)

NOMINAL_T = (0.05, 0.15, 0.4, 1.0)
CBOE_OPTION_KEYS = {"option", "bid", "bid_size", "ask", "ask_size", "iv", "open_interest", "volume", "delta", "gamma",
                    "vega", "theta", "rho", "theo", "change", "open", "high", "low", "tick", "last_trade_price",
                    "last_trade_time", "percent_change", "prev_day_close"}


def _strikes():
    return [synth_strikes(e.forward, T, 0.2, n=41) for e, T in zip(SYNTH_SURFACE, NOMINAL_T, strict=True)]


def _chain(**kw) -> Chain:
    return synthetic_chain("SYN", SYNTH_QUOTE_DATE, SYNTH_SURFACE, _strikes(), **kw)


def test_chain_validates_and_has_one_slice_per_expiry():
    chain = _chain()
    assert isinstance(chain, Chain) and list(chain.df.columns) == list(COLUMNS)
    slices = list(chain.slices())
    assert len(slices) == len(SYNTH_SURFACE) == 4
    for sl, e, T in zip(slices, SYNTH_SURFACE, NOMINAL_T, strict=True):
        assert sl.expiry.date() == e.expiry and abs(sl.T - T) < 0.002
        assert (sl.df["ask"] >= sl.df["bid"]).all() and (sl.df["bid"] >= 0).all()
    assert chain.source.startswith("synthetic seed=0")


def test_parity_holds_exactly_on_the_mids_at_noise_zero():
    chain = _chain()
    for sl, e in zip(chain.slices(), SYNTH_SURFACE, strict=True):
        p = sl.pairs()
        resid = (p["call_mid"] - p["put_mid"]) - e.discount * (e.forward - p["strike"])
        assert np.max(np.abs(resid)) < 1e-9
        assert len(p) == len(sl.df) // 2


def test_mids_are_the_black_prices_and_bands_are_the_half_spread():
    chain = _chain()
    sl = list(chain.slices())[2]
    e = SYNTH_SURFACE[2]
    K = sl.df["strike"].to_numpy()
    k = np.log(K / e.forward)
    sigma = SVISlice(e.params, sl.T).iv(k)
    price = black.price(e.forward, K, sl.T, sigma, sl.df["right"].to_numpy().astype(str), e.discount)
    mid = 0.5 * (sl.df["bid"] + sl.df["ask"]).to_numpy()
    assert np.max(np.abs(mid - price)) < 1e-12
    h = np.minimum(default_half_spread(price, k), price)
    assert np.max(np.abs(0.5 * (sl.df["ask"] - sl.df["bid"]).to_numpy() - h)) < 1e-12


def test_noise_stays_inside_the_band_and_is_seeded():
    a = _chain(noise_in_band=0.5, seed=3)
    b = _chain(noise_in_band=0.5, seed=3)
    c = _chain(noise_in_band=0.5, seed=4)
    assert a.df.equals(b.df) and not a.df.equals(c.df)
    clean = _chain()
    price = 0.5 * (clean.df["bid"] + clean.df["ask"]).to_numpy()
    assert np.all(a.df["bid"].to_numpy() <= price + 1e-12) and np.all(a.df["ask"].to_numpy() >= price - 1e-12)
    with pytest.raises(ValueError):
        _chain(noise_in_band=1.5)


def test_explicit_T_one_strike_array_and_options():
    e = SynthExpiry(date(2026, 9, 18), 0.25, SVIParams(0.02, 0.4, -0.6, 0.05, 0.2), 100.0, 0.99)
    chain = synthetic_chain("SPY", date(2026, 6, 15), [e], np.arange(80.0, 121.0, 5.0), exercise="american",
                            root="SPYW", settlement="AM", spot_offset=-0.01, seed=1, multiplier=10.0)
    assert chain.exercise == "american" and chain.multiplier == 10.0
    assert set(chain.df["root"]) == {"SPYW"} and set(chain.df["settlement"]) == {"AM"}
    assert "spot=99.004983" in chain.source
    sl = next(iter(chain.slices()))
    assert sl.T == pytest.approx(time_to_expiry(date(2026, 6, 15), "16:15", [np.datetime64(date(2026, 9, 18))], ["AM"])[0])
    with pytest.raises(ValueError):
        synthetic_chain("X", date(2026, 6, 15), [], np.arange(80.0, 121.0, 5.0))
    with pytest.raises(ValueError):
        synthetic_chain("X", date(2026, 6, 15), [e], np.array([-1.0, 100.0]))


def test_synth_surface_is_market_like_calendar_monotone_and_butterfly_free():
    """The documented properties of SYNTH_SURFACE: negative skew, w non-decreasing in T on [-1.5, 1.5], g > 0 on
    [-1, 1] with min g 0.268/0.282/0.307/0.348, wing asymptotes ~0.25, F = 100 e^{0.04 T}, D = e^{-0.045 T}."""
    chain = _chain()
    Ts = [sl.T for sl in chain.slices()]
    slices = [SVISlice(e.params, T) for e, T in zip(SYNTH_SURFACE, Ts, strict=True)]
    for s, e, T, g_min in zip(slices, SYNTH_SURFACE, Ts, (0.268, 0.282, 0.307, 0.348), strict=True):
        assert float(s.iv(-0.2)) > float(s.iv(0.0)) > float(s.iv(0.2))  # negative skew
        r = butterfly(s, -1.0, 1.0, margin=0.0)
        assert r.ok and r.worst_g == pytest.approx(g_min, abs=1e-3)
        assert r.asymptote_left == pytest.approx(0.25, abs=2e-3) and r.asymptote_right == pytest.approx(0.25, abs=2e-3)
        assert e.forward == pytest.approx(100.0 * np.exp(0.04 * T), rel=1e-4)
        assert e.discount == pytest.approx(np.exp(-0.045 * T), rel=1e-4)
    cal = calendar(list(zip(Ts, slices, strict=True)), k_grid=np.linspace(-1.5, 1.5, 3001))
    assert cal.ok and cal.crossings == 0 and cal.worst_gap > 1e-3
    atm = [float(s.iv(0.0)) for s in slices]
    assert all(np.diff(atm) > 0)  # rising term structure


def test_occ_symbol_format():
    assert occ_symbol("SPX", date(2026, 9, 18), "C", 200.0) == "SPX260918C00200000"
    assert occ_symbol("VIX", date(2026, 9, 16), "c", 10.0) == "VIX260916C00010000"
    assert occ_symbol("SPY", "2026-12-18", "P", 452.5) == "SPY261218P00452500"


def test_cboe_json_has_every_key_and_serialises():
    chain = _chain()
    j = synthetic_cboe_json(chain)
    assert set(j) == {"timestamp", "symbol", "data"}
    data = j["data"]
    for key in ("symbol", "current_price", "prev_day_close", "last_trade_time", "options"):
        assert key in data
    rows = data["options"]
    assert len(rows) == len(chain.df)
    assert all(set(r) == CBOE_OPTION_KEYS for r in rows)
    assert rows[0]["last_trade_time"] == "2026-06-15T16:14:59"
    assert rows[0]["option"] == occ_symbol("SYN", chain.df["expiry"].iloc[0], chain.df["right"].iloc[0],
                                            chain.df["strike"].iloc[0])
    assert data["current_price"] == pytest.approx(SYNTH_SURFACE[0].forward)
    text = json.dumps(j)
    assert json.loads(text)["data"]["options"][5]["bid"] == rows[5]["bid"]
    bids = np.array([r["bid"] for r in rows])
    assert np.array_equal(bids, chain.df["bid"].to_numpy())

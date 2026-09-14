"""quotes.py: the ACT/365 time rule pinned to the clock, the one-hour expiry drop with its ledger line, the schema
checks (`expiry` dtype, finite quotes, crossed quotes, duplicate keys, two roots on one date), and the ledger.

Oracles (docs/DESIGN.md, Conventions): from a 16:15 quote, a same-day PM expiry is T = -15 min, a next-morning AM
expiry is 17 h 15 min, PM minus AM on one date is 6.5 h, a PM expiry 30 days out is 30/365 - 15 min; a slice with
T <= one hour is dropped by `Chain.slices()` and counted once in `Chain.ledger`.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from volsurf.quotes import HOUR, Chain, Ledger, time_to_expiry
from volsurf.synth import SYNTH_QUOTE_DATE, SYNTH_SURFACE, SynthExpiry, synth_strikes, synthetic_chain

MIN = 1.0 / (365.0 * 24.0 * 60.0)
QD = date(2026, 6, 15)  # a Monday


def _T(expiry: str, settlement: str, quote_time: str = "16:15") -> float:
    return float(time_to_expiry(QD, quote_time, [np.datetime64(expiry)], [settlement])[0])


def test_time_rule_pinned_to_the_clock():
    assert _T("2026-06-15", "PM") == pytest.approx(-15 * MIN, abs=1e-15)
    assert _T("2026-06-16", "AM") == pytest.approx(17.25 / (365.0 * 24.0), abs=1e-15)
    assert _T("2026-06-19", "PM") - _T("2026-06-19", "AM") == pytest.approx(6.5 / (365.0 * 24.0), abs=1e-15)
    assert _T("2026-07-15", "PM") == pytest.approx(30.0 / 365.0 - 15 * MIN, abs=1e-15)
    assert _T("2026-06-15", "AM", "09:00") == pytest.approx(30 * MIN, abs=1e-15)  # a 09:00 quote of a 09:30 settlement
    assert HOUR == pytest.approx(60 * MIN)


def _two_expiry_chain(**kw) -> Chain:
    same_day = SynthExpiry(QD, 1.0 / 365.0, SYNTH_SURFACE[0].params, 100.0, 1.0)  # priced at T = 1 day, listed today
    strikes = synth_strikes(100.0, 0.05, 0.2, n=21)
    return synthetic_chain("SYN", QD, [same_day, SYNTH_SURFACE[1]], strikes, **kw)


def test_slices_drop_the_expired_slice_and_count_it_once():
    chain = _two_expiry_chain()
    T = chain.T
    assert (T[chain.df["expiry"] == pd.Timestamp(QD)] < 0).all()
    got = list(chain.slices())
    assert [s.expiry.date() for s in got] == [SYNTH_SURFACE[1].expiry]
    assert chain.ledger.steps == [(f"T > {HOUR:.2e} y (expired slices dropped)", len(chain.df), len(got[0].df))]
    list(chain.slices())  # slicing again must not duplicate the step
    assert len(chain.ledger.steps) == 1
    assert sum(len(s.df) for s in chain.slices()) == chain.ledger.total_out


def test_expiry_must_be_datetime64():
    chain = _two_expiry_chain()
    for bad in ([d.date() for d in chain.df["expiry"]], chain.df["expiry"].dt.strftime("%Y-%m-%d")):
        df = chain.df.copy()
        df["expiry"] = bad
        with pytest.raises(ValueError, match="datetime64"):
            Chain("SYN", QD, df)
    df = chain.df.copy()
    df["expiry"] = df["expiry"].astype("datetime64[s]")
    assert len(list(Chain("SYN", QD, df).slices())) == 1


def test_non_finite_and_crossed_quotes_and_duplicates_are_rejected():
    chain = _two_expiry_chain()
    for col in ("strike", "bid", "ask"):
        df = chain.df.copy()
        df.loc[0, col] = np.nan
        with pytest.raises(ValueError, match="finite"):
            Chain("SYN", QD, df)
    df = chain.df.copy()
    df.loc[0, "ask"] = df.loc[0, "bid"] - 0.01
    with pytest.raises(ValueError, match="crossed"):
        Chain("SYN", QD, df)
    with pytest.raises(ValueError, match="duplicate"):
        Chain("SYN", QD, pd.concat([chain.df, chain.df.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="missing columns"):
        Chain("SYN", QD, chain.df.drop(columns=["root"]))


def test_two_roots_on_one_date_are_two_slices():
    strikes = synth_strikes(100.0, 0.15, 0.2, n=21)
    a = synthetic_chain("SPX", SYNTH_QUOTE_DATE, SYNTH_SURFACE[1:2], strikes, root="SPX")
    b = synthetic_chain("SPX", SYNTH_QUOTE_DATE, SYNTH_SURFACE[1:2], strikes, root="SPXW")
    chain = Chain("SPX", SYNTH_QUOTE_DATE, pd.concat([a.df, b.df], ignore_index=True))
    got = list(chain.slices())
    assert [(s.root, s.expiry.date()) for s in got] == [("SPX", SYNTH_SURFACE[1].expiry), ("SPXW", SYNTH_SURFACE[1].expiry)]
    assert got[0].T == got[1].T and len(got[0].pairs()) == len(strikes)
    assert chain.coverage()["roots"] == ["SPX", "SPXW"] and chain.coverage()["expiries"] == 2


def test_ledger_counts():
    led = Ledger()
    assert str(led) == "no filters applied" and led.total_in == 0
    led.record("a", 10, 7)
    led.record("b", 7, 7)
    led.record("c", 7, 2)
    assert led.total_in == 10 and led.total_out == 2 and led.dropped() == {"a": 3, "c": 5}
    assert led.lines() == ["a: 10 -> 7", "b: 7 -> 7", "c: 7 -> 2"]

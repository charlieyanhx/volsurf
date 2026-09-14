"""io.py: OCC parsing, the settlement/exercise rules, and the three readers on synthetic files.

Oracles (docs/PLAN.md): a synthetic Cboe-shaped JSON round-trips through `read_cboe_json`; the mztrading date rule
(quote_date = file date - 1 business day, asserted against max last_trade_time) on a synthetic frame with an expired
row; rows with expiry < quote_date are dropped and counted; the last row wins on a duplicate key; crossed quotes are
dropped and counted; the source's iv column is never consumed (a poisoned iv changes nothing).
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from volsurf import io
from volsurf.quotes import COLUMNS
from volsurf.synth import (
    SYNTH_QUOTE_DATE,
    SYNTH_SURFACE,
    occ_symbol,
    synth_strikes,
    synthetic_cboe_json,
    synthetic_chain,
)

pq = pytest.importorskip("pyarrow.parquet")


def _chain(symbol="SYN", root=None, exercise="european", settlement="PM"):
    strikes = [synth_strikes(e.forward, T, 0.2, n=21) for e, T in zip(SYNTH_SURFACE, (0.05, 0.15, 0.4, 1.0), strict=True)]
    return synthetic_chain(symbol, SYNTH_QUOTE_DATE, SYNTH_SURFACE, strikes, root=root, exercise=exercise,
                           settlement=settlement)


# ---------------------------------------------------------------- OCC and the rules


def test_parse_occ_examples():
    assert io.parse_occ("SPX260918C00200000") == ("SPX", date(2026, 9, 18), "C", 200.0)
    assert io.parse_occ("SPXW260821P07550000") == ("SPXW", date(2026, 8, 21), "P", 7550.0)
    assert io.parse_occ("VIX260916C00010000") == ("VIX", date(2026, 9, 16), "C", 10.0)
    assert io.parse_occ("AAPL251219P00232500") == ("AAPL", date(2025, 12, 19), "P", 232.5)
    for bad in ("SPX", "spx260918C00200000", "SPX260918X00200000", "SPX26091C00200000", ""):
        with pytest.raises(ValueError):
            io.parse_occ(bad)


def test_parse_occ_round_trips_synth_symbol():
    s = occ_symbol("XSP", date(2027, 1, 15), "P", 512.5)
    assert io.parse_occ(s) == ("XSP", date(2027, 1, 15), "P", 512.5)


def test_settlement_rule_third_friday_am_roots():
    assert io.is_third_friday(date(2026, 9, 18)) and not io.is_third_friday(date(2026, 9, 25))
    assert io.settlement_for("SPX", date(2026, 9, 18)) == "AM"
    assert io.settlement_for("SPXW", date(2026, 9, 18)) == "PM"  # weekly root, even on the third Friday
    assert io.settlement_for("SPX", date(2026, 9, 16)) == "PM"  # Wednesday SPX is PM-settled
    assert io.settlement_for("SPY", date(2026, 9, 18)) == "PM"
    assert io.settlement_for("VIX", date(2026, 9, 16)) == "AM"
    for root in io.AM_SETTLED_ROOTS:
        assert io.settlement_for(root, date(2026, 12, 18)) == "AM"


def test_exercise_rule():
    assert io.exercise_for("_SPX") == "european" and io.exercise_for("_XSP") == "european"
    assert io.exercise_for("_VIX") == "european" and io.exercise_for("VIX") == "european"
    assert io.exercise_for("SPY") == "american" and io.exercise_for("AAPL") == "american"


# ---------------------------------------------------------------- Cboe JSON


def test_cboe_json_round_trips_from_dict_string_and_file(tmp_path):
    chain = _chain()
    payload = synthetic_cboe_json(chain)
    back = io.read_cboe_json(payload)
    assert back.quote_date == SYNTH_QUOTE_DATE and back.exercise == "european" and back.symbol == "SYN"
    assert list(back.df.columns) == list(COLUMNS)
    ref = chain.df.sort_values(["expiry", "root", "strike", "right"]).reset_index(drop=True)
    for c in ("expiry", "strike", "right", "bid", "ask", "root", "settlement"):
        assert (back.df[c].to_numpy() == ref[c].to_numpy()).all(), c
    assert back.ledger.total_in == len(chain.df) and back.ledger.total_out == len(chain.df)
    p = tmp_path / "syn.json"
    p.write_text(json.dumps(payload))
    from_file = io.read_cboe_json(p)
    from_str = io.read_cboe_json(json.dumps(payload))
    assert from_file.df.equals(back.df) and from_str.df.equals(back.df)


def test_cboe_json_quote_date_is_max_last_trade_time_and_expired_rows_are_counted():
    chain = _chain()
    payload = synthetic_cboe_json(chain)
    # move one stale trade earlier and the quote date forward past the first expiry
    payload["data"]["options"][0]["last_trade_time"] = "2026-01-05T10:00:00"
    for row in payload["data"]["options"]:
        if row["last_trade_time"].startswith("2026-06-15"):
            row["last_trade_time"] = "2026-07-06T16:14:59"
    back = io.read_cboe_json(payload)
    assert back.quote_date == date(2026, 7, 6)
    n_first = int((chain.df["expiry"] == pd.Timestamp("2026-07-03")).sum())
    assert back.ledger.dropped()["expiry >= quote_date"] == n_first
    assert len(back.df) == len(chain.df) - n_first
    explicit = io.read_cboe_json(payload, quote_date=SYNTH_QUOTE_DATE)
    assert len(explicit.df) == len(chain.df)


def test_cboe_json_source_iv_and_greeks_are_never_read():
    chain = _chain()
    payload = synthetic_cboe_json(chain)
    for row in payload["data"]["options"]:
        row["iv"], row["delta"], row["theo"] = 99.0, 7.0, -1.0
    poisoned = io.read_cboe_json(payload)
    assert poisoned.df.equals(io.read_cboe_json(synthetic_cboe_json(chain)).df)


def test_cboe_json_crossed_and_duplicate_rows_and_exercise_inference():
    chain = _chain()
    payload = synthetic_cboe_json(chain)
    rows = payload["data"]["options"]
    rows[3]["ask"] = rows[3]["bid"] - 0.01  # crossed
    dup = dict(rows[5])
    dup["bid"], dup["ask"] = 1.23, 1.25
    rows.append(dup)  # duplicate key, last row wins
    back = io.read_cboe_json(payload)
    assert back.ledger.dropped()["not crossed (ask >= bid)"] == 1
    assert back.ledger.dropped()["last row per (root, expiry, strike, right)"] == 1
    root, exp, right, K = io.parse_occ(dup["option"])
    hit = back.df[(back.df["expiry"] == pd.Timestamp(exp)) & (back.df["strike"] == K) & (back.df["right"] == right)]
    assert len(hit) == 1 and hit["bid"].iloc[0] == 1.23 and hit["ask"].iloc[0] == 1.25
    payload["data"]["security_type"] = "stock"
    assert io.read_cboe_json(payload).exercise == "american"
    assert io.read_cboe_json(payload, exercise="european").exercise == "european"


# ---------------------------------------------------------------- philippdubach parquet


def _philippdubach_frame(chain, day, type_style="word"):
    df = chain.df
    typ = np.where(df["right"] == "C", "call", "put") if type_style == "word" else df["right"].to_numpy()
    return pd.DataFrame({
        "contract_id": [occ_symbol(r.root, r.expiry, r.right, r.strike) for r in df.itertuples()],
        "symbol": chain.symbol, "expiration": df["expiry"].to_numpy(), "strike": df["strike"].to_numpy(),
        "type": typ, "last": 0.0, "mark": 0.5 * (df["bid"] + df["ask"]).to_numpy(), "bid": df["bid"].to_numpy(),
        "bid_size": 20, "ask": df["ask"].to_numpy(), "ask_size": 20, "volume": 0, "open_interest": 100,
        "date": np.datetime64(day, "ns"), "implied_volatility": np.float32(0.01488), "delta": np.float32(1.0),
    })


@pytest.mark.parametrize("type_style", ["word", "letter"])
def test_philippdubach_reader_filters_the_day_and_sets_root_pm_american(tmp_path, type_style):
    chain = _chain("SPY", exercise="american")
    other_day = date(2026, 6, 12)
    frame = pd.concat([_philippdubach_frame(chain, SYNTH_QUOTE_DATE, type_style),
                       _philippdubach_frame(chain, other_day, type_style).iloc[:10]], ignore_index=True)
    p = tmp_path / "SPY_options.parquet"
    frame.to_parquet(p, index=False)
    back = io.read_philippdubach(p, "SPY", SYNTH_QUOTE_DATE)
    assert back.exercise == "american" and back.quote_date == SYNTH_QUOTE_DATE
    assert set(back.df["root"]) == {"SPY"} and set(back.df["settlement"]) == {"PM"}
    assert len(back.df) == len(chain.df) and back.ledger.total_in == len(chain.df)
    ref = chain.df.sort_values(["expiry", "root", "strike", "right"]).reset_index(drop=True)
    assert np.allclose(back.df["bid"], ref["bid"]) and (back.df["right"].to_numpy() == ref["right"].to_numpy()).all()
    with pytest.raises(ValueError, match="no rows"):
        io.read_philippdubach(p, "QQQ", SYNTH_QUOTE_DATE)


def test_philippdubach_expired_rows_are_dropped_and_counted(tmp_path):
    chain = _chain("SPY", exercise="american")
    day = date(2026, 7, 6)  # after the first synthetic expiry (2026-07-03)
    p = tmp_path / "SPY_options.parquet"
    _philippdubach_frame(chain, day).to_parquet(p, index=False)
    back = io.read_philippdubach(p, "SPY", day)
    n_first = int((chain.df["expiry"] == pd.Timestamp("2026-07-03")).sum())
    assert back.ledger.dropped()["expiry >= quote_date"] == n_first
    assert back.quote_date == day and len(back.df) == len(chain.df) - n_first


# ---------------------------------------------------------------- mztrading parquet


def _mztrading_frame(chain, symbol, file_day, ltt_day, extra_expired=0):
    payload = synthetic_cboe_json(chain)
    rows = pd.DataFrame(payload["data"]["options"])
    rows["last_trade_time"] = pd.Timestamp(ltt_day).strftime("%Y-%m-%dT16:14:59")
    rows.loc[rows.index[::7], "last_trade_time"] = None  # untraded contracts carry no time
    rows["timestamp"] = f"{file_day} 10:01:12"
    rows["symbol"] = symbol
    if extra_expired:
        ex = rows.iloc[:extra_expired].copy()
        ex["option"] = [occ_symbol(io.parse_occ(o)[0], pd.Timestamp(ltt_day) - pd.Timedelta(days=1), r, k)
                        for o, r, k in zip(ex["option"], ["C"] * extra_expired, range(10, 10 + extra_expired), strict=True)]
        rows = pd.concat([rows, ex], ignore_index=True)
    return rows


def test_mztrading_date_rule_expired_rows_and_roots(tmp_path):
    chain = _chain("SPX", root="SPX")
    file_day = date(2026, 6, 16)  # Tuesday: quote date is Monday 2026-06-15
    frame = _mztrading_frame(chain, "_SPX", file_day, SYNTH_QUOTE_DATE, extra_expired=4)
    # relabel the second expiry's contracts as the weekly root so the chain has SPX and SPXW
    second = frame["option"].str.contains("260809")
    frame.loc[second, "option"] = "SPXW" + frame.loc[second, "option"].str[3:]
    p = tmp_path / f"day_{file_day}.parquet"
    frame.to_parquet(p, index=False)
    back = io.read_mztrading(p, "_SPX")
    assert back.quote_date == SYNTH_QUOTE_DATE and back.symbol == "SPX" and back.exercise == "european"
    assert set(back.df["root"]) == {"SPX", "SPXW"}
    assert back.ledger.dropped()["expiry >= quote_date"] == 4
    assert len(back.df) == len(chain.df)
    assert back.ledger.total_in == len(chain.df) + 4


def test_mztrading_raises_when_the_date_rule_disagrees_with_last_trade_time(tmp_path):
    chain = _chain("SPX", root="SPX")
    file_day = date(2026, 6, 16)
    frame = _mztrading_frame(chain, "_SPX", file_day, date(2026, 6, 12))  # trades from the Friday: a holiday snap
    p = tmp_path / f"day_{file_day}.parquet"
    frame.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="date rule"):
        io.read_mztrading(p, "_SPX")
    back = io.read_mztrading(p, "_SPX", quote_date=date(2026, 6, 12))
    assert back.quote_date == date(2026, 6, 12)
    with pytest.raises(ValueError, match="no rows"):
        io.read_mztrading(p, "SPY")
    unnamed = tmp_path / "snap.parquet"
    frame.to_parquet(unnamed, index=False)
    with pytest.raises(ValueError, match="file date"):
        io.read_mztrading(unnamed, "_SPX")


def test_mztrading_am_settlement_and_single_name_exercise(tmp_path):
    from volsurf.svi import SVIParams
    from volsurf.synth import SynthExpiry

    third_friday = date(2026, 9, 18)
    exp = SynthExpiry(third_friday, None, SVIParams(0.02, 0.4, -0.6, 0.05, 0.2), 100.0, 0.99)
    strikes = np.arange(80.0, 121.0, 2.0)
    file_day = date(2026, 6, 16)
    for symbol, root, want_settle, want_ex in (("_SPX", "SPX", "AM", "european"), ("AAPL", "AAPL", "PM", "american"),
                                               ("_VIX", "VIX", "AM", "european")):
        chain = synthetic_chain(root, SYNTH_QUOTE_DATE, [exp], strikes, root=root)
        p = tmp_path / f"day_{file_day}.parquet"
        _mztrading_frame(chain, symbol, file_day, SYNTH_QUOTE_DATE).to_parquet(p, index=False)
        back = io.read_mztrading(p, symbol)
        assert set(back.df["settlement"]) == {want_settle} and back.exercise == want_ex, symbol


def test_xsp_is_pm_settled_on_every_expiry():
    """Cboe relisted Mini-SPX (XSP) with PM settlement in November 2013; a third-Friday XSP is NOT an AM monthly and
    an AM T would be 6.5 h short (0.9 % of T at 30 DTE)."""
    assert "XSP" not in io.AM_SETTLED_ROOTS
    assert io.settlement_for("XSP", date(2026, 9, 18)) == "PM" and io.settlement_for("XSP", date(2026, 9, 16)) == "PM"
    assert io.settlement_for("NDX", date(2026, 9, 18)) == "AM" and io.settlement_for("NDXP", date(2026, 9, 18)) == "PM"


def test_cboe_json_null_and_blank_quotes_are_dropped_and_counted():
    """A null bid or a blank ask in the payload is NaN after coercion; it must be dropped with a ledger line, never
    reach `Chain` (which now rejects NaN quotes), and the identity rows in - rows out holds."""
    chain = _chain()
    payload = synthetic_cboe_json(chain)
    rows = payload["data"]["options"]
    rows[3]["bid"], rows[5]["ask"], rows[7]["bid"] = None, "", "n/a"
    back = io.read_cboe_json(payload)
    assert back.ledger.dropped()["finite bid/ask, strike > 0"] == 3
    assert len(back.df) == len(rows) - 3 == back.ledger.total_out
    assert np.isfinite(back.df[["bid", "ask", "strike"]].to_numpy()).all()
    assert back.coverage()["two_sided"] + back.coverage()["zero_bid"] == back.coverage()["quotes"]

"""Readers: three option-chain sources into `quotes.Chain`, with every row that does not reach the chain counted.

Sources (none of them is redistributed with the repo; see README "Data and privacy"):
  read_cboe_json(path)            Cboe's delayed-quote JSON, saved by hand from the quote-table page
                                  ({timestamp, symbol, data: {..., options: [rows]}}; `synth.synthetic_cboe_json`
                                  renders the same shape for the tests)
  read_philippdubach(path, symbol, day)   the philippdubach parquet mirror (one file per symbol, all days; columns
                                  contract_id, symbol, expiration, strike, type ('call'/'put' or 'C'/'P'), bid, ask,
                                  bid_size, ask_size, volume, open_interest, date, ...)
  read_mztrading(path, symbol)    one mztrading day file (Cboe JSON fields as parquet columns plus timestamp, symbol)

Rules shared by all three, in this order, each recorded in `Chain.ledger` as (rule, rows in, rows out):
  1. OCC symbol parsed -> root, expiry, right, strike (rows whose symbol does not parse are dropped and counted)
  2. rows with expiry < quote_date dropped (an end-of-day file still lists the contracts that expired that day
     or the day before; a same-day expiry is kept and then dropped by `Chain.slices()` when T <= one hour)
  3. crossed quotes (ask < bid) dropped: `Chain` rejects them and a real file has a handful (mztrading _SPX 2026-08-10: 5)
  4. duplicates on (root, expiry, strike, right): the LAST row wins (a file that carries two snaps of one contract)

Quote date: read_cboe_json takes the date of the latest `last_trade_time` over the option rows (the payload's
`timestamp` is the download time, off-hours it is the next morning); read_mztrading takes the file's date minus one
business day and REQUIRES it to equal the date of the latest last_trade_time (the files are the prior session's close
labelled D+1: file 2026-08-11 has max last_trade_time 2026-08-10T16:14:59 and 350 SPY rows that expired 08-10);
read_philippdubach takes the requested day.

Settlement / exercise: settlement is "AM" for the SPX, NDX, RUT, DJX and XSP roots on a third Friday (the monthlies,
which settle at the 09:30 open) and for every VIX expiry (Wednesday open settlement), "PM" otherwise; exercise is
"european" for underscore-prefixed index symbols and VIX, "american" otherwise (ETFs, single names).

Never consumed: the sources' iv / delta / gamma / vega / theta / rho / theo columns. volsurf inverts bid, mid and ask
itself against a parity forward; the file's IV is against an unknown forward, rate and clock (philippdubach 2008-10
IVs are floored at 0.0149) and would make every "fit error in vol points" incomparable.

Units: strikes and prices in dollars per share/unit as quoted; sizes, volume and open interest as floats (NaN when
the source has none); the multiplier is 100 for every source here.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .quotes import COLUMNS, Chain, Ledger

__all__ = [
    "AM_SETTLED_ROOTS",
    "parse_occ",
    "is_third_friday",
    "settlement_for",
    "exercise_for",
    "read_cboe_json",
    "read_philippdubach",
    "read_mztrading",
]

AM_SETTLED_ROOTS = ("SPX", "NDX", "RUT", "DJX", "XSP")
_OCC = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")
_MZ_FILE = re.compile(r"day_(\d{4}-\d{2}-\d{2})")


def parse_occ(symbol: str) -> tuple[str, date, str, float]:
    """'SPXW260821C00200000' -> ('SPXW', date(2026, 8, 21), 'C', 200.0). Raises ValueError when it does not parse."""
    m = _OCC.match(str(symbol).strip())
    if m is None:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    root, ymd, right, strike = m.groups()
    return root, datetime.strptime(ymd, "%y%m%d").date(), right, int(strike) / 1000.0


def is_third_friday(d) -> bool:
    d = pd.Timestamp(d)
    return d.weekday() == 4 and 15 <= d.day <= 21


def settlement_for(root: str, expiry) -> str:
    """"AM" for the index monthlies (third Friday of the AM roots) and for VIX; "PM" otherwise."""
    if root == "VIX":
        return "AM"
    if root in AM_SETTLED_ROOTS and is_third_friday(expiry):
        return "AM"
    return "PM"


def exercise_for(symbol: str) -> str:
    """"european" for underscore-prefixed index symbols ('_SPX', '_VIX', '_XSP') and VIX, "american" otherwise."""
    s = str(symbol)
    return "european" if s.startswith("_") or s.lstrip("_").upper() == "VIX" else "american"


def _parse_occ_column(occ: pd.Series, ledger: Ledger) -> pd.DataFrame:
    """Vectorised OCC parse; unparseable rows dropped and counted."""
    parts = occ.astype(str).str.strip().str.extract(_OCC.pattern)
    ok = parts.notna().all(axis=1).to_numpy()
    ledger.record("OCC symbol parsed", len(occ), int(ok.sum()))
    parts = parts[ok]
    out = pd.DataFrame({
        "root": parts[0].to_numpy(dtype=str),
        "expiry": pd.to_datetime(parts[1], format="%y%m%d").to_numpy().astype("datetime64[ns]"),
        "right": parts[2].to_numpy(dtype=str),
        "strike": parts[3].astype(np.int64).to_numpy() / 1000.0,
    }, index=occ.index[ok])
    return out


def _finish(df: pd.DataFrame, quote_date: date, ledger: Ledger) -> pd.DataFrame:
    """Rules 2-4 (expired, crossed, duplicates) on a frame that already has the `COLUMNS`; returns the chain frame."""
    n = len(df)
    df = df[df["expiry"] >= np.datetime64(quote_date, "ns")]
    ledger.record("expiry >= quote_date", n, len(df))
    n = len(df)
    df = df[~(df["ask"] < df["bid"])]
    ledger.record("not crossed (ask >= bid)", n, len(df))
    n = len(df)
    df = df.drop_duplicates(["root", "expiry", "strike", "right"], keep="last")
    ledger.record("last row per (root, expiry, strike, right)", n, len(df))
    df = df.sort_values(["expiry", "root", "strike", "right"]).reset_index(drop=True)
    return df[list(COLUMNS)]


def _num(series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _latest_trade_date(ltt) -> date | None:
    ts = pd.to_datetime(pd.Series(ltt), errors="coerce")
    return None if ts.notna().sum() == 0 else ts.max().date()


# ------------------------------------------------------------------ Cboe JSON


def _load_payload(path) -> dict:
    if isinstance(path, dict):
        return path
    if isinstance(path, str) and path.lstrip().startswith("{"):
        return json.loads(path)
    return json.loads(Path(path).read_text())


def read_cboe_json(path, exercise: str | None = None, quote_date: date | None = None) -> Chain:
    """A Cboe delayed-quote payload (file path, JSON string, or the loaded dict). `exercise` defaults to european when
    data.security_type is 'index' or the symbol starts with '_', american otherwise. quote_date defaults to the date
    of the latest option last_trade_time."""
    payload = _load_payload(path)
    data = payload["data"]
    symbol = str(payload.get("symbol", data.get("symbol", "")))
    rows = pd.DataFrame(data["options"])
    if rows.empty:
        raise ValueError("the payload has no options")
    if quote_date is None:
        quote_date = _latest_trade_date(rows.get("last_trade_time"))
        if quote_date is None:
            raise ValueError("no parseable last_trade_time in the payload: pass quote_date")
    if exercise is None:
        exercise = "european" if (data.get("security_type") == "index" or symbol.startswith("_")) else "american"
    ledger = Ledger()
    occ = _parse_occ_column(rows["option"], ledger)
    df = pd.DataFrame({
        "expiry": occ["expiry"], "strike": occ["strike"], "right": occ["right"],
        "bid": _num(rows.loc[occ.index, "bid"]), "ask": _num(rows.loc[occ.index, "ask"]),
        "bid_size": _num(rows.loc[occ.index, "bid_size"]), "ask_size": _num(rows.loc[occ.index, "ask_size"]),
        "volume": _num(rows.loc[occ.index, "volume"]), "open_interest": _num(rows.loc[occ.index, "open_interest"]),
        "root": occ["root"], "settlement": [settlement_for(r, e) for r, e in zip(occ["root"], occ["expiry"], strict=True)],
    })
    df = _finish(df, quote_date, ledger)
    src = f"cboe json {Path(str(path)).name if not isinstance(path, dict) else 'dict'} {symbol}"
    return Chain(symbol.lstrip("_"), quote_date, df, exercise=exercise, source=src, ledger=ledger)


# ------------------------------------------------------------------ philippdubach parquet


def _right_from_type(t: pd.Series) -> np.ndarray:
    s = t.astype(str).str.strip().str.upper().str[0].to_numpy()
    if not set(np.unique(s)) <= {"C", "P"}:
        raise ValueError("type column must be call/put or C/P")
    return s


def read_philippdubach(path, symbol: str, day, exercise: str = "american") -> Chain:
    """One (symbol, day) out of a philippdubach-style parquet, read with a pyarrow row filter on `date` (and `symbol`
    when the file has that column). root = symbol, settlement PM (ETFs and single names), exercise american."""
    import pyarrow.parquet as pq

    day = pd.Timestamp(day).normalize()
    schema = pq.ParquetFile(path).schema_arrow.names
    filters = [("date", "==", day)]
    if "symbol" in schema:
        filters.append(("symbol", "==", symbol))
    rows = pq.read_table(path, filters=filters).to_pandas()
    if rows.empty:
        raise ValueError(f"no rows for {symbol} on {day.date()} in {path}")
    ledger = Ledger()
    ledger.record("rows for the day", len(rows), len(rows))
    n = len(rows)
    df = pd.DataFrame({
        "expiry": pd.to_datetime(rows["expiration"]).to_numpy().astype("datetime64[ns]"),
        "strike": _num(rows["strike"]), "right": _right_from_type(rows["type"]),
        "bid": _num(rows["bid"]), "ask": _num(rows["ask"]),
        "bid_size": _num(rows["bid_size"]) if "bid_size" in rows else np.nan,
        "ask_size": _num(rows["ask_size"]) if "ask_size" in rows else np.nan,
        "volume": _num(rows["volume"]) if "volume" in rows else np.nan,
        "open_interest": _num(rows["open_interest"]) if "open_interest" in rows else np.nan,
        "root": symbol, "settlement": "PM",
    })
    df = df[np.isfinite(df["bid"]) & np.isfinite(df["ask"]) & (df["strike"] > 0)]
    ledger.record("finite bid/ask, strike > 0", n, len(df))
    df = _finish(df, day.date(), ledger)
    return Chain(symbol, day.date(), df, exercise=exercise, source=f"philippdubach {Path(str(path)).name} {day.date()}",
                 ledger=ledger)


# ------------------------------------------------------------------ mztrading parquet


def _file_date(path) -> date:
    m = _MZ_FILE.search(Path(str(path)).name)
    if m is None:
        raise ValueError(f"cannot read the file date from {path!r} (expected day_YYYY-MM-DD.parquet)")
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def read_mztrading(path, symbol: str, quote_date: date | None = None) -> Chain:
    """One symbol out of an mztrading day file. quote_date = file date - 1 business day (or the argument), and it
    must equal the date of the latest last_trade_time in the rows, else ValueError (a holiday or a mislabelled file:
    pass quote_date explicitly after checking). Roots come from the OCC symbol (_SPX splits into SPX and SPXW)."""
    import pyarrow.parquet as pq

    rows = pq.read_table(path, filters=[("symbol", "==", symbol)]).to_pandas()
    if rows.empty:
        raise ValueError(f"no rows for {symbol!r} in {path}")
    if quote_date is None:
        quote_date = (pd.Timestamp(_file_date(path)) - pd.offsets.BDay(1)).date()
    seen = _latest_trade_date(rows["last_trade_time"])
    if seen is None or seen != quote_date:
        raise ValueError(f"mztrading date rule: quote_date {quote_date} (file date - 1 business day) but the latest "
                         f"last_trade_time is on {seen}; pass quote_date if the file is a holiday snap")
    ledger = Ledger()
    occ = _parse_occ_column(rows["option"], ledger)
    df = pd.DataFrame({
        "expiry": occ["expiry"], "strike": occ["strike"], "right": occ["right"],
        "bid": _num(rows.loc[occ.index, "bid"]), "ask": _num(rows.loc[occ.index, "ask"]),
        "bid_size": _num(rows.loc[occ.index, "bid_size"]), "ask_size": _num(rows.loc[occ.index, "ask_size"]),
        "volume": _num(rows.loc[occ.index, "volume"]), "open_interest": _num(rows.loc[occ.index, "open_interest"]),
        "root": occ["root"], "settlement": [settlement_for(r, e) for r, e in zip(occ["root"], occ["expiry"], strict=True)],
    })
    n = len(df)
    df = df[np.isfinite(df["bid"]) & np.isfinite(df["ask"]) & (df["strike"] > 0)]
    ledger.record("finite bid/ask, strike > 0", n, len(df))
    df = _finish(df, quote_date, ledger)
    return Chain(symbol.lstrip("_"), quote_date, df, exercise=exercise_for(symbol),
                 source=f"mztrading {Path(str(path)).name} {symbol}", ledger=ledger)

"""Option-chain schema: one `Chain` per (symbol, quote date), quotes in a DataFrame with fixed columns,
a stated time-to-expiry rule, and a filter ledger so every row that leaves the fit is counted.

Columns of `Chain.df` (one row per quote; nothing derived is stored, so nothing can disagree):
  expiry        datetime64[ns]  settlement date
  strike        float
  right         "C" | "P"
  bid, ask      float, dollars per share/unit (bid >= 0, ask >= bid; crossed quotes are rejected)
  bid_size, ask_size, volume, open_interest   float, NaN when the source has none
  root          str, the listing root ("SPX" and "SPXW" are different slices on the same date)
  settlement    "AM" | "PM" — AM-settled options (SPX monthlies) expire at the 09:30 open

Time to expiry (the rule every T in volsurf uses):
  T = (settlement instant − quote instant) / 365 days, ACT/365 with the intraday fraction;
  quote instant = quote_date at `quote_time` ET (default 16:15, the US option close);
  settlement instant = expiry at 16:00 ET for PM, 09:30 ET for AM.
  A slice with T <= 1/(365·24) (one hour) is expired for fitting purposes and is dropped, counted.

`exercise` is "european" (SPX, XSP, VIX — parity holds, the discount is fittable) or "american"
(SPY, QQQ, IWM, single names — parity fails by the early-exercise premium, the discount must come
from a rate). `forward.py` refuses to fit a free discount on an American chain unless forced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterator

import numpy as np
import pandas as pd

__all__ = ["COLUMNS", "HOUR", "Ledger", "Chain", "Slice", "time_to_expiry"]

COLUMNS = ("expiry", "strike", "right", "bid", "ask", "bid_size", "ask_size", "volume", "open_interest", "root", "settlement")
HOUR = 1.0 / (365.0 * 24.0)
_SETTLE_TIME = {"PM": timedelta(hours=16), "AM": timedelta(hours=9, minutes=30)}


@dataclass
class Ledger:
    """Rows in → rows out per named rule, in order. `total_in` is the first count seen."""

    steps: list[tuple[str, int, int]] = field(default_factory=list)

    def record(self, rule: str, n_in: int, n_out: int) -> None:
        self.steps.append((rule, int(n_in), int(n_out)))

    @property
    def total_in(self) -> int:
        return self.steps[0][1] if self.steps else 0

    @property
    def total_out(self) -> int:
        return self.steps[-1][2] if self.steps else 0

    def dropped(self) -> dict[str, int]:
        return {rule: n_in - n_out for rule, n_in, n_out in self.steps if n_in != n_out}

    def lines(self) -> list[str]:
        return [f"{rule}: {n_in} -> {n_out}" for rule, n_in, n_out in self.steps]

    def __str__(self) -> str:
        return "; ".join(self.lines()) if self.steps else "no filters applied"


def time_to_expiry(quote_date: date, quote_time: str, expiry, settlement) -> np.ndarray:
    """ACT/365 years from the quote instant to the settlement instant (vectorised over expiry/settlement)."""
    hh, mm = (int(x) for x in quote_time.split(":"))
    q = datetime(quote_date.year, quote_date.month, quote_date.day, hh, mm)
    exp = pd.to_datetime(np.asarray(expiry)).to_numpy().astype("datetime64[ns]")
    settle = np.asarray(settlement).astype(str)
    offs = np.where(settle == "AM", np.timedelta64(int(9.5 * 3600), "s"), np.timedelta64(16 * 3600, "s"))
    inst = exp + offs
    return (inst - np.datetime64(q)) / np.timedelta64(1, "D") / 365.0


@dataclass(frozen=True)
class Chain:
    """All quotes of one symbol on one quote date."""

    symbol: str
    quote_date: date
    df: pd.DataFrame
    exercise: str = "european"
    quote_time: str = "16:15"
    multiplier: float = 100.0
    source: str = ""
    ledger: Ledger = field(default_factory=Ledger, compare=False)

    def __post_init__(self):
        missing = [c for c in COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"chain is missing columns {missing}")
        if self.exercise not in ("european", "american"):
            raise ValueError("exercise must be 'european' or 'american'")
        df = self.df
        if not set(df["right"].unique()) <= {"C", "P"}:
            raise ValueError("right must be 'C' or 'P'")
        if not set(df["settlement"].unique()) <= {"AM", "PM"}:
            raise ValueError("settlement must be 'AM' or 'PM'")
        if (df["strike"] <= 0).any():
            raise ValueError("strikes must be positive")
        if (df["bid"] < 0).any() or (df["ask"] < df["bid"]).any():
            raise ValueError("bids must be >= 0 and asks >= bids (crossed quotes are rejected, not repaired)")
        key = df[["root", "expiry", "strike", "right"]]
        if key.duplicated().any():
            n = int(key.duplicated().sum())
            raise ValueError(f"{n} duplicate quotes on (root, expiry, strike, right); dedupe in the reader")

    @property
    def T(self) -> np.ndarray:
        return time_to_expiry(self.quote_date, self.quote_time, self.df["expiry"], self.df["settlement"])

    def expiries(self) -> list[tuple[str, pd.Timestamp]]:
        """(root, expiry) keys in expiry order, then root."""
        keys = self.df[["root", "expiry"]].drop_duplicates()
        keys = keys.sort_values(["expiry", "root"])
        return [(str(r), pd.Timestamp(e)) for r, e in keys.itertuples(index=False)]

    def slices(self, min_T: float = HOUR) -> Iterator[Slice]:
        """One Slice per (root, expiry) with T > min_T; expired ones are counted in the ledger."""
        T = self.T
        n_in = len(self.df)
        keep = T > min_T
        self.ledger.record(f"T > {min_T:.2e} y (expired slices dropped)", n_in, int(keep.sum()))
        df = self.df[keep]
        Tk = T[keep]
        for root, exp in self.expiries():
            m = (df["root"] == root).to_numpy() & (df["expiry"] == exp).to_numpy()
            if not m.any():
                continue
            yield Slice(self.symbol, self.quote_date, root, exp, float(Tk[m][0]), df[m].reset_index(drop=True),
                        self.exercise, self.multiplier)

    def coverage(self) -> dict:
        df = self.df
        two_sided = (df["bid"] > 0) & (df["ask"] > 0)
        return {"symbol": self.symbol, "quote_date": str(self.quote_date), "quotes": int(len(df)),
                "expiries": int(df[["root", "expiry"]].drop_duplicates().shape[0]),
                "roots": sorted(df["root"].unique().tolist()), "two_sided": int(two_sided.sum()),
                "zero_bid": int((df["bid"] <= 0).sum()), "strike_min": float(df["strike"].min()),
                "strike_max": float(df["strike"].max()), "exercise": self.exercise, "source": self.source}


@dataclass(frozen=True)
class Slice:
    """The quotes of one (root, expiry): what a forward fit and an SVI fit consume."""

    symbol: str
    quote_date: date
    root: str
    expiry: pd.Timestamp
    T: float
    df: pd.DataFrame
    exercise: str = "european"
    multiplier: float = 100.0

    @property
    def strikes(self) -> np.ndarray:
        return np.sort(self.df["strike"].unique())

    def side(self, right: str) -> pd.DataFrame:
        return self.df[self.df["right"] == right].sort_values("strike").reset_index(drop=True)

    def pairs(self) -> pd.DataFrame:
        """Strikes quoted on both sides: columns strike, call_bid, call_ask, put_bid, put_ask (+ mids)."""
        c = self.side("C")[["strike", "bid", "ask"]].rename(columns={"bid": "call_bid", "ask": "call_ask"})
        p = self.side("P")[["strike", "bid", "ask"]].rename(columns={"bid": "put_bid", "ask": "put_ask"})
        out = c.merge(p, on="strike", how="inner").sort_values("strike").reset_index(drop=True)
        out["call_mid"] = 0.5 * (out["call_bid"] + out["call_ask"])
        out["put_mid"] = 0.5 * (out["put_bid"] + out["put_ask"])
        return out

    def label(self) -> str:
        return f"{self.symbol} {self.root} {self.expiry:%Y-%m-%d} (T={self.T:.4f})"

"""Chain sources for the tool layer: the bundled golden chain, or a public endpoint.

Both produce a `volsurf.quotes.Chain`, so everything downstream is the parent
package's real pipeline — parity forwards, the Black inverter, raw-SVI fits,
arbitrage checks — not a second implementation.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from volsurf.quotes import COLUMNS, Chain

DATA_DIR = Path(__file__).with_name("data")

# The bundled chain models XSP, whose monthlies are PM-settled; volsurf's
# `settlement_for` says the same, so the generator and the reader agree on T.
BUNDLED_UNDERLYING = "XSP"
BUNDLED_SETTLEMENT = "PM"


class ChainSourceError(RuntimeError):
    """A chain could not be obtained. The message says why."""


class ChainSource(ABC):
    @abstractmethod
    def fetch(self, underlying: str) -> Chain: ...

    @abstractmethod
    def available(self) -> list[str]: ...


def _frame(rows: list[dict], root: str, settlement: str) -> pd.DataFrame:
    """Rows of {expiry, strike, is_call, bid, ask} → the canonical volsurf frame."""
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "expiry": pd.to_datetime(df["expiry"]).to_numpy().astype("datetime64[ns]"),
        "strike": df["strike"].astype(float).to_numpy(),
        "right": np.where(df["is_call"].to_numpy(dtype=bool), "C", "P"),
        "bid": df["bid"].astype(float).to_numpy(),
        "ask": df["ask"].astype(float).to_numpy(),
        "bid_size": np.nan, "ask_size": np.nan, "volume": np.nan, "open_interest": np.nan,
        "root": root, "settlement": settlement,
    })
    out = out[np.isfinite(out["bid"]) & np.isfinite(out["ask"]) & (out["strike"] > 0)]
    out = out[~(out["ask"] < out["bid"])]
    out = out.drop_duplicates(["root", "expiry", "strike", "right"], keep="last")
    return out.sort_values(["expiry", "root", "strike", "right"]).reset_index(drop=True)[list(COLUMNS)]


class BundledSource(ChainSource):
    """The golden chain shipped with the package.

    Generated from known SVI parameters by data/make_golden_chain.py, so every
    expected tool output is checkable against the generator (see evals/golden.py).
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else DATA_DIR / "golden_chain.json"

    def available(self) -> list[str]:
        return [BUNDLED_UNDERLYING]

    def fetch(self, underlying: str) -> Chain:
        if underlying.upper() != BUNDLED_UNDERLYING:
            raise ChainSourceError(
                f"the bundled source carries {BUNDLED_UNDERLYING} only, not {underlying!r}; "
                f"pass live=True for another underlying")
        with open(self.path) as f:
            raw = json.load(f)
        as_of = datetime.fromisoformat(raw["as_of"].replace("Z", "+00:00"))
        rows = [{"expiry": exp, **q} for exp, block in raw["expiries"].items() for q in block["quotes"]]
        return Chain(symbol=raw["underlying"], quote_date=as_of.date(), df=_frame(rows, raw["underlying"], BUNDLED_SETTLEMENT),
                     exercise="european", quote_time=as_of.strftime("%H:%M"), multiplier=100.0, source="bundled")


class YFinanceSource(ChainSource):
    """A public endpoint. Rate-limited and unofficial; failures are reported, not retried."""

    def __init__(self, max_expiries: int = 6):
        try:
            import yfinance  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ChainSourceError('yfinance is not installed: pip install "volsurf[live]"') from e
        self.max_expiries = max_expiries

    def available(self) -> list[str]:
        return ["any yfinance ticker with listed options"]

    def fetch(self, underlying: str) -> Chain:
        import yfinance as yf

        try:
            tk = yf.Ticker(underlying)
            expiries = list(tk.options)[: self.max_expiries]
            if not expiries:
                raise ChainSourceError(f"yfinance returned no option expiries for {underlying!r}")
            rows: list[dict] = []
            for exp in expiries:
                oc = tk.option_chain(exp)
                for frame, is_call in ((oc.calls, True), (oc.puts, False)):
                    for r in frame.itertuples(index=False):
                        rows.append({"expiry": exp, "strike": float(r.strike), "is_call": is_call,
                                     "bid": float(r.bid), "ask": float(r.ask)})
        except ChainSourceError:
            raise
        except Exception as e:  # network, parse, rate limit — all reported the same way
            raise ChainSourceError(f"live fetch for {underlying!r} failed: {e}") from e
        # yfinance expiry rows are for the listed date; settlement is inferred by volsurf's own rule per row.
        from volsurf.io import settlement_for
        df = _frame(rows, underlying.upper(), "PM")
        df["settlement"] = [settlement_for(underlying.upper(), e) for e in df["expiry"]]
        return Chain(symbol=underlying.upper(), quote_date=date.today(), df=df, exercise="american",
                     quote_time="16:00", multiplier=100.0, source="yfinance")


def load_chain(underlying: str = BUNDLED_UNDERLYING, live: bool = False) -> Chain:
    return (YFinanceSource() if live else BundledSource()).fetch(underlying)

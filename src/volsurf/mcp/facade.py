"""The shape the tools speak, over a real `volsurf.surface.Surface`.

`volsurf` keys slices by (root, expiry Timestamp), returns pandas frames and
typed reports. Tools want string expiries, plain dicts and JSON-safe numbers.
This module is that translation and nothing more: every number comes from the
parent package, and any method here that computes anything is a bug.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from volsurf import black
from volsurf.arb import ArbitrageReport
from volsurf.quotes import Chain
from volsurf.surface import SliceFit, Surface, fit_surface
from volsurf.svi import SVIParams

_ISO = "%Y-%m-%d"


def _f(x) -> float:
    """A JSON-safe float. NaN and ±inf are not JSON; the tools want None there."""
    x = float(x)
    return x if math.isfinite(x) else None  # type: ignore[return-value]


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime(_ISO)


@dataclass(frozen=True)
class SliceView:
    """One fitted expiry, in the vocabulary the tools use."""
    expiry: str
    t: float
    forward: float
    discount: float
    n_points: int
    rmse_vol: float
    params: SVIParams
    _fit: SliceFit

    def implied_vol(self, k) -> np.ndarray:
        return self._fit.iv(k)


@dataclass(frozen=True)
class GreeksView:
    """Black-76 Greeks at one strike. vega per 1.00 vol, theta per year, as volsurf reports them."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    vanna: float
    volga: float

    def per_vol_point(self) -> float:
        return self.vega / 100.0

    def per_day_theta(self) -> float:
        return self.theta / 365.0


class ArbView:
    """`volsurf.arb.ArbitrageReport`, summarised the way a tool result quotes it."""

    def __init__(self, report: ArbitrageReport):
        self._r = report
        bad = [b for b in report.butterfly if not b.ok]
        self.ok = bool(report.ok)
        self.butterfly_violations = len(bad)
        worst = min(report.butterfly, key=lambda b: b.worst_g, default=None)
        self.worst_g = _f(worst.worst_g) if worst else None
        self.worst_g_k = _f(worst.worst_k) if worst else None
        self.calendar_violations = int(report.calendar.crossings)
        self.worst_calendar_gap = _f(report.calendar.worst_gap)
        self.notes = tuple(report.notes) + tuple(n for b in report.butterfly for n in b.notes) + tuple(report.calendar.notes)

    def summary(self) -> str:
        if self.ok:
            return "no static arbitrage: g(k) >= 0 on every checked range and total variance non-decreasing in maturity"
        parts = []
        if self.butterfly_violations:
            parts.append(f"butterfly: {self.butterfly_violations} slice(s) with g(k) < 0 (worst g={self.worst_g:.3g} at k={self.worst_g_k:.3g})")
        if self.calendar_violations:
            parts.append(f"calendar: {self.calendar_violations} crossing(s), worst gap {self.worst_calendar_gap:.3g}")
        return "; ".join(parts) or "arbitrage report not ok, see notes"


class ToolSurface:
    """A fitted surface with dict-keyed slices and string expiries."""

    def __init__(self, surface: Surface, as_of: str):
        self._s = surface
        self.underlying = surface.symbol
        self.as_of = as_of
        self.fits: dict[str, SliceView] = {}
        for f in surface.fits:
            self.fits[_iso(f.expiry)] = SliceView(
                expiry=_iso(f.expiry), t=float(f.T), forward=float(f.forward.forward),
                discount=float(f.forward.discount), n_points=int(f.n),
                rmse_vol=float(f.svi.rmse_vol), params=f.params, _fit=f)
        self.forwards = {e: v.forward for e, v in self.fits.items()}
        self.discounts = {e: v.discount for e, v in self.fits.items()}
        self._arb: ArbView | None = None

    def expiries(self) -> list[str]:
        return sorted(self.fits)

    def _view(self, expiry: str) -> SliceView:
        try:
            return self.fits[expiry]
        except KeyError:
            raise KeyError(f"no slice for {expiry!r}; have {self.expiries()}") from None

    def implied_vol(self, expiry: str, strike: float) -> float:
        v = self._view(expiry)
        return _f(v.implied_vol(math.log(float(strike) / v.forward)))

    def term_structure(self) -> list[dict]:
        rows = []
        for e in self.expiries():
            v = self.fits[e]
            rows.append({"expiry": e, "t": v.t, "forward": v.forward, "discount": v.discount,
                         "atm_vol": _f(v.implied_vol(0.0)), "quotes": v.n_points, "rmse_vol": v.rmse_vol})
        return rows

    def skew(self, expiry: str, delta: float = 0.25) -> dict:
        """Put-minus-call vol at symmetric log-moneyness, a delta-free proxy.

        Reported against log-moneyness rather than true delta so it does not
        depend on a root-find; `delta` is used only to pick the offset.
        """
        v = self._view(expiry)
        k = abs(math.log(delta / (1 - delta))) * 0.25
        put, call = _f(v.implied_vol(-k)), _f(v.implied_vol(k))
        return {"expiry": expiry, "k_offset": k, "put_wing": put, "call_wing": call, "skew": put - call}

    def greeks_at(self, expiry: str, strike: float, is_call: bool = True) -> GreeksView:
        v = self._view(expiry)
        K = float(strike)
        sigma = float(v.implied_vol(math.log(K / v.forward)))
        g = black.greeks(v.forward, K, v.t, sigma, right="C" if is_call else "P", D=v.discount)
        return GreeksView(**{n: _f(getattr(g, n)) for n in ("price", "delta", "gamma", "vega", "theta", "vanna", "volga")})

    def arbitrage(self) -> ArbView:
        if self._arb is None:
            self._arb = ArbView(self._s.arbitrage())
        return self._arb

    def report(self) -> dict:
        arb = self.arbitrage()
        return {
            "underlying": self.underlying,
            "as_of": self.as_of,
            "expiries": len(self.fits),
            "skipped": [{"expiry": _iso(e), "reason": r} for e, r in self._s.skipped],
            "quotes_used": sum(v.n_points for v in self.fits.values()),
            "worst_rmse_vol": max((v.rmse_vol for v in self.fits.values()), default=0.0),
            "arbitrage_free": arb.ok,
            "arbitrage": arb.summary(),
        }


def chain_coverage(chain: Chain, as_of: str) -> dict:
    """The line every result should carry: what went in."""
    T = chain.T
    return {
        "underlying": chain.symbol,
        "as_of": as_of,
        "expiries": len(chain.expiries()),
        "quotes": int(len(chain.df)),
        "t_range": (_f(T.min()), _f(T.max())) if len(T) else (None, None),
    }


def chain_as_of(chain: Chain) -> str:
    return f"{chain.quote_date.isoformat()}T{chain.quote_time}:00Z"


def build(chain: Chain, max_relative_spread: float | None = None) -> ToolSurface:
    """Fit with volsurf's defaults; `max_relative_spread` narrows the quote filter when given."""
    kw = {}
    if max_relative_spread is not None:
        kw["max_rel_spread"] = float(max_relative_spread)
    return ToolSurface(fit_surface(chain, **kw), as_of=chain_as_of(chain))

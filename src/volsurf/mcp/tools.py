"""The analytics exposed as tools. Plain functions, JSON-shaped returns.

Kept free of any MCP import so they can be tested and evaluated directly; the
server module is a thin registration layer over these. Every tool returns a
dict, and every dict carries enough provenance (underlying, as_of, counts) that
a result can be quoted with what produced it.
"""
from __future__ import annotations

from typing import Dict, Optional

from volsurf.quotes import Chain

from .facade import ToolSurface, build, chain_as_of, chain_coverage
from .sources import BundledSource, ChainSourceError, YFinanceSource

__all__ = [
    "fetch_chain", "calibrate_surface", "surface_term_structure",
    "surface_skew", "option_greeks", "arbitrage_check", "clear_cache", "TOOLS",
]

_CHAIN_CACHE: Dict[str, Chain] = {}
_SURFACE_CACHE: Dict[str, ToolSurface] = {}


def _get_chain(underlying: str, live: bool = False) -> Chain:
    key = f"{'live' if live else 'bundled'}:{underlying.upper()}"
    if key not in _CHAIN_CACHE:
        source = YFinanceSource() if live else BundledSource()
        _CHAIN_CACHE[key] = source.fetch(underlying)
    return _CHAIN_CACHE[key]


def _surface(underlying: str, live: bool = False,
             max_relative_spread: Optional[float] = None) -> ToolSurface:
    """Fit once per (underlying, source, filter) and reuse.

    Calibration is the expensive step — a multistart per slice. Without this a
    host calling six tools recalibrates the same surface six times, and each
    call returns a separately-fitted surface, so two answers in one conversation
    need not agree.
    """
    key = f"{'live' if live else 'bundled'}:{underlying.upper()}:{max_relative_spread}"
    if key not in _SURFACE_CACHE:
        _SURFACE_CACHE[key] = build(_get_chain(underlying, live), max_relative_spread=max_relative_spread)
    return _SURFACE_CACHE[key]


def clear_cache() -> None:
    """Drop cached chains and surfaces. Live data changes; bundled data does not."""
    _CHAIN_CACHE.clear()
    _SURFACE_CACHE.clear()


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def fetch_chain(underlying: str = "XSP", live: bool = False) -> Dict:
    """Load an option chain and report what it contains.

    Args:
        underlying: ticker. The bundled source carries XSP only.
        live: fetch from a public endpoint instead of the bundled chain. Public
            endpoints rate-limit, so this can fail; the error says so.
    """
    try:
        chain = _get_chain(underlying, live)
    except ChainSourceError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, **chain_coverage(chain, chain_as_of(chain)),
            "expiry_list": [e.strftime("%Y-%m-%d") for _, e in chain.expiries()],
            "source": "live" if live else "bundled"}


def calibrate_surface(underlying: str = "XSP", live: bool = False,
                      max_relative_spread: Optional[float] = None) -> Dict:
    """Fit one arbitrage-checked SVI slice per expiry and report the fit quality.

    The returned `arbitrage_free` flag is a *verified* result: butterfly is
    checked as Gatheral's g(k) >= 0 on the quoted range plus the wing
    asymptotes, and calendar as total variance non-decreasing in maturity. It
    is not an assumption of the fit.
    """
    try:
        surf = _surface(underlying, live, max_relative_spread)
    except (ChainSourceError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, **surf.report(),
            "slices": [{"expiry": e,
                        "t": surf.fits[e].t,
                        "forward": surf.forwards[e],
                        "discount": surf.discounts[e],
                        "quotes": surf.fits[e].n_points,
                        "rmse_vol": surf.fits[e].rmse_vol,
                        "params": dict(zip("a b rho m sigma".split(),
                                           surf.fits[e].params.as_tuple(), strict=True))}
                       for e in surf.expiries()]}


def surface_term_structure(underlying: str = "XSP", live: bool = False) -> Dict:
    """At-the-money volatility by maturity."""
    try:
        surf = _surface(underlying, live)
    except (ChainSourceError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "underlying": surf.underlying, "as_of": surf.as_of,
            "term_structure": surf.term_structure()}


def surface_skew(underlying: str = "XSP", expiry: Optional[str] = None,
                 live: bool = False) -> Dict:
    """Put-minus-call wing volatility at symmetric log-moneyness."""
    try:
        surf = _surface(underlying, live)
    except (ChainSourceError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    expiries = [expiry] if expiry else surf.expiries()
    missing = [e for e in expiries if e not in surf.fits]
    if missing:
        return {"ok": False, "error": f"no slice for {missing}; have {surf.expiries()}"}
    return {"ok": True, "underlying": surf.underlying,
            "skew": [surf.skew(e) for e in expiries]}


def option_greeks(underlying: str = "XSP", expiry: Optional[str] = None,
                  strike: Optional[float] = None, is_call: bool = True,
                  live: bool = False) -> Dict:
    """Greeks at a strike, using the surface's fitted volatility.

    vega is per 1.00 of vol and theta per year; `vega_per_point` and
    `theta_per_day` are supplied so the desk convention is unambiguous.
    """
    if expiry is None or strike is None:
        return {"ok": False, "error": "expiry and strike are both required"}
    try:
        surf = _surface(underlying, live)
        g = surf.greeks_at(expiry, float(strike), is_call=is_call)
    except (ChainSourceError, ValueError, KeyError) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "underlying": surf.underlying, "expiry": expiry,
            "strike": float(strike), "is_call": is_call,
            "forward": surf.forwards[expiry],
            "discount": surf.discounts[expiry],
            "implied_vol": surf.implied_vol(expiry, float(strike)),
            "price": g.price, "delta": g.delta, "gamma": g.gamma,
            "vega": g.vega, "vega_per_point": g.per_vol_point(),
            "theta": g.theta, "theta_per_day": g.per_day_theta(),
            "vanna": g.vanna, "volga": g.volga}


def arbitrage_check(underlying: str = "XSP", live: bool = False) -> Dict:
    """Butterfly and calendar violations across the fitted surface."""
    try:
        surf = _surface(underlying, live)
    except (ChainSourceError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    arb = surf.arbitrage()
    return {"ok": True, "underlying": surf.underlying,
            "arbitrage_free": arb.ok,
            "butterfly_violations": arb.butterfly_violations,
            "worst_g": arb.worst_g, "worst_g_k": arb.worst_g_k,
            "calendar_violations": arb.calendar_violations,
            "worst_calendar_gap": arb.worst_calendar_gap,
            "summary": arb.summary(), "notes": list(arb.notes)}


TOOLS = {
    "fetch_chain": fetch_chain,
    "calibrate_surface": calibrate_surface,
    "surface_term_structure": surface_term_structure,
    "surface_skew": surface_skew,
    "option_greeks": option_greeks,
    "arbitrage_check": arbitrage_check,
}

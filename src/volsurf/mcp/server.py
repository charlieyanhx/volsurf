"""MCP server exposing the options-surface tools.

A thin registration layer over `surfacemcp.tools`. The analytics live there,
free of any MCP import, so they can be tested and evaluated without a client;
this module only describes them to a host.

Run:
    python -m surfacemcp.server            # stdio transport

Claude Desktop / Claude Code config:
    {"mcpServers": {"options-surface": {"command": "python",
                                        "args": ["-m", "surfacemcp.server"]}}}
"""
from __future__ import annotations

from typing import Optional

from . import tools

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise SystemExit('the MCP server needs the mcp SDK: pip install "volsurf[mcp]"') from e

mcp = FastMCP("volsurf")


@mcp.tool()
def fetch_chain(underlying: str = "XSP", live: bool = False) -> dict:
    """Load an option chain and report its coverage: expiries, quote count, date range.

    The bundled chain (live=False) is deterministic and offline. live=True uses a
    public endpoint, which rate-limits and may fail.
    """
    return tools.fetch_chain(underlying, live)


@mcp.tool()
def calibrate_surface(underlying: str = "XSP", live: bool = False,
                      max_relative_spread: Optional[float] = None) -> dict:
    """Fit an arbitrage-checked SVI slice per expiry; returns parameters and fit error.

    One slice per maturity. `arbitrage_free` is verified after fitting - butterfly
    via Gatheral's g(k) >= 0 on a dense grid, calendar via total variance
    non-decreasing in maturity - not assumed by the optimiser.
    """
    return tools.calibrate_surface(underlying, live, max_relative_spread)


@mcp.tool()
def surface_term_structure(underlying: str = "XSP", live: bool = False) -> dict:
    """At-the-money implied volatility by maturity, with the forward at each expiry."""
    return tools.surface_term_structure(underlying, live)


@mcp.tool()
def surface_skew(underlying: str = "XSP", expiry: Optional[str] = None,
                 live: bool = False) -> dict:
    """Put-minus-call wing volatility at symmetric log-moneyness, per expiry."""
    return tools.surface_skew(underlying, expiry, live)


@mcp.tool()
def option_greeks(underlying: str, expiry: str, strike: float,
                  is_call: bool = True, live: bool = False) -> dict:
    """Price and Greeks at a strike, using the surface's own fitted volatility.

    vega is per 1.00 of volatility and theta per year; vega_per_point and
    theta_per_day are also returned so the convention is unambiguous.
    """
    return tools.option_greeks(underlying, expiry, strike, is_call, live)


@mcp.tool()
def arbitrage_check(underlying: str = "XSP", live: bool = False) -> dict:
    """Butterfly and calendar arbitrage violations across the fitted surface."""
    return tools.arbitrage_check(underlying, live)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

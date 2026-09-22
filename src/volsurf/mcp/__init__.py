"""volsurf.mcp — the surface as tools an agent can call.

A thin adapter over `volsurf`: six plain functions in `tools`, a FastMCP
registration in `server`, and an eval harness whose expected values are derived
from the parameters the bundled chain was generated from. No pricing, fitting or
arbitrage logic lives here; it all comes from the parent package.

    pip install "volsurf[mcp]"        # the MCP server
    pip install "volsurf[live]"       # yfinance chains instead of the bundled one
    volsurf-mcp                       # stdio server
    python -m volsurf.mcp.evals       # the 30 evals
"""
from .tools import TOOLS, clear_cache

__all__ = ["TOOLS", "clear_cache"]

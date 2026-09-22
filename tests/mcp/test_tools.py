"""The tool layer, as an MCP host would call it."""
import pytest

from volsurf.mcp.tools import TOOLS


def test_all_six_tools_are_registered():
    assert set(TOOLS) == {"fetch_chain", "calibrate_surface", "surface_term_structure",
                          "surface_skew", "option_greeks", "arbitrage_check"}


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_every_tool_returns_a_dict_with_ok(name):
    """A host has to be able to branch on the result without guessing its shape."""
    fn = TOOLS[name]
    out = fn("XSP", "2027-01-15", 420.0) if name == "option_greeks" else fn("XSP")
    assert isinstance(out, dict) and "ok" in out


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_no_tool_raises_on_an_unknown_underlying(name):
    """Errors come back as data. A tool that raises takes the whole session with it."""
    fn = TOOLS[name]
    out = fn("NOSUCH", "2027-01-15", 420.0) if name == "option_greeks" else fn("NOSUCH")
    assert out["ok"] is False and out["error"]


def test_calibrate_reports_arbitrage_as_a_verified_result():
    out = TOOLS["calibrate_surface"]()
    assert out["arbitrage_free"] is True
    assert "no static arbitrage" in out["arbitrage"]


def test_greeks_price_matches_put_call_parity_with_the_discount():
    """Regression: the surface dropped its discount factor and returned an
    undiscounted price, so C - P read F - K instead of D(F - K) and the tool's
    price did not match the quote it was calibrated from."""
    c = TOOLS["option_greeks"]("XSP", "2027-01-15", 420.0, True)
    p = TOOLS["option_greeks"]("XSP", "2027-01-15", 420.0, False)
    d = c["discount"]
    assert d < 1.0, "the golden chain is discounted; a discount of 1 means it was dropped"
    assert c["price"] - p["price"] == pytest.approx(d * (c["forward"] - 420.0), abs=1e-8)


def test_term_structure_carries_the_discount_per_expiry():
    rows = TOOLS["surface_term_structure"]()["term_structure"]
    assert all(0 < r["discount"] < 1 for r in rows)
    assert [r["discount"] for r in rows] == sorted((r["discount"] for r in rows), reverse=True)


def test_skew_is_positive_for_a_negative_rho_surface():
    for row in TOOLS["surface_skew"]()["skew"]:
        assert row["put_wing"] > row["call_wing"]


def test_missing_required_arguments_are_refused():
    out = TOOLS["option_greeks"]("XSP")
    assert out["ok"] is False and "required" in out["error"]


def test_unknown_expiry_lists_what_is_available():
    out = TOOLS["option_greeks"]("XSP", "1999-01-01", 400.0)
    assert out["ok"] is False and "2027-01-15" in out["error"]


def test_repeated_calls_return_the_same_surface():
    """Regression: the surface was refitted on every tool call, so two answers in
    one conversation came from two separate calibrations and need not agree."""
    a = TOOLS["calibrate_surface"]()
    b = TOOLS["calibrate_surface"]()
    assert a["slices"][0]["params"] == b["slices"][0]["params"]
    vol_a = TOOLS["option_greeks"]("XSP", "2027-01-15", 420.0)["implied_vol"]
    vol_b = TOOLS["surface_term_structure"]()["term_structure"]
    atm = [r for r in vol_b if r["expiry"] == "2027-01-15"][0]["atm_vol"]
    assert vol_a == TOOLS["option_greeks"]("XSP", "2027-01-15", 420.0)["implied_vol"]
    assert atm > 0


def test_cache_can_be_cleared():
    from volsurf.mcp.tools import clear_cache
    TOOLS["calibrate_surface"]()
    clear_cache()
    assert TOOLS["calibrate_surface"]()["ok"] is True

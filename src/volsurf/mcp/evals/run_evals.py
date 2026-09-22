"""Eval harness: run every tool against the golden chain, score the outputs.

    python evals/run_evals.py            # human-readable
    python evals/run_evals.py --json     # machine-readable, non-zero exit on failure

Scored against values derived from the generating parameters, not from a
recorded run of this code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ..tools import TOOLS
from .golden import (  # noqa: E402
    AS_OF,
    EXPIRIES,
    N_QUOTES,
    TOL_ATM_VOL,
    TOL_FORWARD,
    TOL_VOL,
    TRUTH,
    UNDERLYING,
    expected_forward,
    expected_vol,
)

RESULTS = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append({"eval": name, "pass": bool(passed), "detail": detail})


def run() -> None:
    # --- fetch_chain ------------------------------------------------------
    r = TOOLS["fetch_chain"]()
    check("chain.loads", r["ok"], r.get("error", ""))
    check("chain.underlying", r.get("underlying") == UNDERLYING, str(r.get("underlying")))
    check("chain.as_of", r.get("as_of") == AS_OF, str(r.get("as_of")))
    check("chain.quote_count", r.get("quotes") == N_QUOTES,
          f"{r.get('quotes')} vs {N_QUOTES}")
    check("chain.expiries", r.get("expiry_list") == EXPIRIES, str(r.get("expiry_list")))

    # --- calibrate_surface ------------------------------------------------
    s = TOOLS["calibrate_surface"]()
    check("surface.fits", s["ok"], s.get("error", ""))
    check("surface.all_expiries", len(s.get("slices", [])) == len(EXPIRIES),
          f"{len(s.get('slices', []))} of {len(EXPIRIES)}")
    check("surface.arbitrage_free", s.get("arbitrage_free") is True, s.get("arbitrage", ""))

    worst_fwd = 0.0
    for sl in s.get("slices", []):
        worst_fwd = max(worst_fwd, abs(sl["forward"] - expected_forward(sl["expiry"])))
    check("surface.forward_recovery", worst_fwd < TOL_FORWARD,
          f"worst |dF| = {worst_fwd:.4f} (tol {TOL_FORWARD})")
    check("surface.fit_quality", s.get("worst_rmse_vol", 1) < TOL_VOL,
          f"worst rmse = {s.get('worst_rmse_vol', float('nan')):.5f} (tol {TOL_VOL})")

    # --- term structure ---------------------------------------------------
    ts = TOOLS["surface_term_structure"]()
    check("term.ok", ts["ok"], ts.get("error", ""))
    rows = ts.get("term_structure", [])
    check("term.monotone_in_t", [r["t"] for r in rows] == sorted(r["t"] for r in rows))
    worst_atm = 0.0
    for row in rows:
        worst_atm = max(worst_atm,
                        abs(row["atm_vol"] - expected_vol(row["expiry"],
                                                          expected_forward(row["expiry"]))))
    check("term.atm_vol_accuracy", worst_atm < TOL_ATM_VOL,
          f"worst |dvol| = {worst_atm:.5f} (tol {TOL_ATM_VOL})")

    # --- skew -------------------------------------------------------------
    sk = TOOLS["surface_skew"]()
    check("skew.ok", sk["ok"], sk.get("error", ""))
    check("skew.put_over_call", all(r["skew"] > 0 for r in sk.get("skew", [])),
          "generated with rho < 0, so puts must be bid over calls")

    # --- greeks -----------------------------------------------------------
    g = TOOLS["option_greeks"]("XSP", "2027-01-15", 420.0, True)
    check("greeks.ok", g["ok"], g.get("error", ""))
    check("greeks.vol_matches_surface",
          abs(g["implied_vol"] - expected_vol("2027-01-15", 420.0)) < TOL_VOL,
          f"{g.get('implied_vol')} vs {expected_vol('2027-01-15', 420.0)}")
    check("greeks.call_delta_in_range", 0 < g["delta"] < 1, str(g.get("delta")))
    check("greeks.gamma_positive", g["gamma"] > 0, str(g.get("gamma")))
    check("greeks.vega_positive", g["vega"] > 0, str(g.get("vega")))
    check("greeks.theta_negative", g["theta"] < 0, str(g.get("theta")))
    check("greeks.vega_convention", abs(g["vega_per_point"] - g["vega"] / 100) < 1e-12)

    p = TOOLS["option_greeks"]("XSP", "2027-01-15", 420.0, False)
    check("greeks.put_delta_negative", p["delta"] < 0, str(p.get("delta")))
    check("greeks.put_call_parity",
          abs((g["price"] - p["price"]) - (g["forward"] - 420.0) * TRUTH["2027-01-15"][1]) < 1e-6,
          "C - P must equal D(F - K)")

    # --- arbitrage --------------------------------------------------------
    a = TOOLS["arbitrage_check"]()
    check("arb.ok", a["ok"], a.get("error", ""))
    check("arb.no_butterfly", a["butterfly_violations"] == 0, str(a.get("worst_g")))
    check("arb.no_calendar", a["calendar_violations"] == 0, str(a.get("worst_calendar_gap")))

    # --- failure paths ----------------------------------------------------
    bad = TOOLS["option_greeks"]("XSP", "1999-01-01", 400.0)
    check("errors.unknown_expiry_refused", bad["ok"] is False and "no slice" in bad["error"])
    miss = TOOLS["option_greeks"]("XSP")
    check("errors.missing_args_refused", miss["ok"] is False and "required" in miss["error"])
    other = TOOLS["fetch_chain"]("AAPL")
    check("errors.unknown_underlying_refused", other["ok"] is False and "bundled" in other["error"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    run()
    n_pass = sum(r["pass"] for r in RESULTS)
    n = len(RESULTS)

    if args.json:
        print(json.dumps({"passed": n_pass, "total": n, "results": RESULTS}, indent=1))
    else:
        for r in RESULTS:
            mark = "PASS" if r["pass"] else "FAIL"
            line = f"  [{mark}] {r['eval']}"
            if r["detail"] and not r["pass"]:
                line += f"  <- {r['detail']}"
            elif r["detail"]:
                line += f"  ({r['detail']})"
            print(line)
        print(f"\n{n_pass}/{n} evals passed")
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    sys.exit(main())

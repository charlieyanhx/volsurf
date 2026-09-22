"""Generate the bundled golden chain. Deterministic; run to regenerate.

Every input the chain is built from is stated here, and the evals in
../evals/golden.py re-derive their expectations from the same statements. The
one thing that is *not* typed by hand is time-to-expiry: it comes from
`volsurf.quotes.time_to_expiry` on the as_of instant and the expiry's
settlement, so the generator and the reader can never disagree about T.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np

from volsurf import black
from volsurf.quotes import time_to_expiry
from volsurf.svi import SVIParams, SVISlice

UNDERLYING = "XSP"
QUOTE_DATE = date(2026, 9, 3)
QUOTE_TIME = "16:15"
SETTLEMENT = "PM"           # XSP monthlies are PM-settled; matches volsurf.io.settlement_for
AS_OF = f"{QUOTE_DATE.isoformat()}T{QUOTE_TIME}:00Z"
SPOT = 412.75

# A plausible equity-index surface: downward skew, term structure in contango.
# Discount and SVI parameters per expiry; T is derived, never typed.
TRUTH = {
    "2026-10-16": dict(D=0.9953, p=SVIParams(0.00206, 0.0186, -0.62, 0.021, 0.049)),
    "2026-11-20": dict(D=0.9895, p=SVIParams(0.00512, 0.0331, -0.58, 0.028, 0.071)),
    "2027-01-15": dict(D=0.9819, p=SVIParams(0.00921, 0.0475, -0.54, 0.034, 0.093)),
    "2027-03-19": dict(D=0.9736, p=SVIParams(0.01418, 0.0620, -0.50, 0.041, 0.114)),
}


def t_for(expiry: str) -> float:
    return float(time_to_expiry(QUOTE_DATE, QUOTE_TIME, [expiry], [SETTLEMENT])[0])


def main() -> None:
    out = {"underlying": UNDERLYING, "as_of": AS_OF, "spot": SPOT, "expiries": {}}
    for exp, cfg in TRUTH.items():
        t, D, params = t_for(exp), cfg["D"], cfg["p"]
        F = SPOT / D                       # forward implied by the discount
        sl = SVISlice(params, t)
        strikes = [float(k) for k in np.arange(340, 486, 5)]
        quotes = []
        for K in strikes:
            k = math.log(K / F)
            iv = float(sl.iv(k))
            for is_call in (True, False):
                mid = float(black.price(F, K, t, iv, "C" if is_call else "P", D))
                if mid < 0.02:
                    continue
                # spread widens in the wings and with maturity, as real chains do
                rel = 0.012 + 0.05 * abs(k) + 0.004 * t
                half = max(mid * rel, 0.01) / 2
                quotes.append({"strike": K, "is_call": is_call,
                               "bid": round(max(mid - half, 0.01), 4),
                               "ask": round(mid + half, 4)})
        out["expiries"][exp] = {"quotes": quotes}
        print(f"  {exp}  t={t:.4f}  F={F:.4f}  quotes={len(quotes)}")

    path = Path(__file__).with_name("golden_chain.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}: {sum(len(v['quotes']) for v in out['expiries'].values())} quotes")


if __name__ == "__main__":
    main()

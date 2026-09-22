"""Expected tool outputs, derived from the generator rather than recorded.

Every figure here comes from the SVI parameters, discounts and dates the bundled
chain was built from (../data/make_golden_chain.py), not from a previous run of
this code. That distinction is the whole point: a golden set recorded from your
own output freezes your bugs as the specification, and passes forever.

Time-to-expiry is re-derived through `volsurf.quotes.time_to_expiry` exactly as
the generator did, so TRUTH carries no hand-typed T.
"""
from __future__ import annotations

import math

from volsurf.svi import SVISlice

from ..data.make_golden_chain import AS_OF, SPOT, UNDERLYING, t_for
from ..data.make_golden_chain import TRUTH as _GEN

# (t, discount, SVI parameters) the chain was generated from.
TRUTH = {exp: (t_for(exp), cfg["D"], cfg["p"]) for exp, cfg in _GEN.items()}

EXPIRIES = list(TRUTH)
N_QUOTES = 240

# Tolerances. Quotes carry a spread, so the mid is not the generating price and
# a perfect recovery is not expected; these are the bars a correct pipeline clears.
TOL_FORWARD = 0.05        # dollars, on a ~415 forward
TOL_VOL = 2e-3            # 0.2 vol points
TOL_ATM_VOL = 3e-3

__all__ = ["AS_OF", "EXPIRIES", "N_QUOTES", "SPOT", "TOL_ATM_VOL", "TOL_FORWARD", "TOL_VOL", "TRUTH",
           "UNDERLYING", "expected_forward", "expected_vol"]


def expected_forward(expiry: str) -> float:
    return SPOT / TRUTH[expiry][1]


def expected_vol(expiry: str, strike: float) -> float:
    t, _, params = TRUTH[expiry]
    return float(SVISlice(params, t).iv(math.log(strike / expected_forward(expiry))))

# volsurf

[![ci](https://github.com/charlieyanhx/volsurf/actions/workflows/ci.yml/badge.svg)](https://github.com/charlieyanhx/volsurf/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Implied-volatility surfaces from end-of-day option chains: the forward and discount of every
expiry read out of put-call parity (never from spot, which is 15 minutes stale against the
option close and can sit 1.5 % away), a vectorised Black inverter with an explicit information
floor, one raw-SVI slice per expiry fitted in total variance with Zeliade's (m, sigma) reduction
as the initialiser and only the two conditions that are actually necessary (w_min >= 0 and Lee's
b(1+|rho|) <= 2, in total-variance units), and static-arbitrage checks on the fitted
parametrisation that are counted and located, never repaired: Gatheral-Jacquier's g(k) on the
quoted k-range plus the analytic wing asymptotes, calendar monotonicity between adjacent slices,
and a model-free within-bid/ask convexity count on the raw quotes. Fit quality is reported in vol
points and as the share of selected quotes whose model price lies inside its own bid/ask. The one
design rule: every number is computed against the expiry's parity forward, and every check
reports where it failed (inside or outside the quoted range, on which pair, at which k) instead of
adjusting the surface until it passes.

## Run it

```bash
pip install -e ".[dev]"
pytest -q          # 139 tests: ~15 s on a laptop (IV inverter, Black, quotes, forward, SVI, arbitrage, weights, synth, readers, surface, CLI); 4 oracle tests skip without the pricers sibling
volsurf report     # regenerates the synthetic tables below in place; run twice, the second run is a no-op
volsurf fit data/private/spy.json --source cboe --rate 0.043   # one surface: term structure, skips, arbitrage notes
volsurf report --data day_2026-08-11.parquet --source mztrading --symbol _SPX --header   # private row
```

`--rate` (continuous, per year) is required on American chains (SPY, QQQ, IWM, single names): the discount
is pinned from the rate there because it is not identifiable from parity (see Design rules), and without it
no slice could be fitted, so `volsurf fit` / `volsurf report --data` exit 2 instead of printing an empty surface.

## What it computes

| module | what | the invariant it keeps |
|---|---|---|
| `quotes` | `Chain` / `Slice` / `Ledger`: the quote schema, the ACT/365 time rule (16:15 quote to 16:00 PM / 09:30 AM settlement), one slice per (root, expiry) | crossed quotes, NaN quotes, a non-datetime `expiry` column and duplicate keys raise; every dropped row is counted |
| `black` | Black-76 on the forward: price, vega, Greeks per 1.00 vol and per calendar day | price = D · Black(F, K, T, sigma); a NaN vol prices as NaN, never as intrinsic |
| `iv` | `implied_vol` (Corrado-Miller start, bracketed Halley in s = sigma sqrt(T)), `implied_vols` per slice | NaN below intrinsic, above the cap, or when time value < 1e-10 F; never a guess |
| `forward` | `fit_forward`: parity line (F, D) for European chains, discount pinned from a rate for American ones; spread weights, strike window, Huber pass | C − P = D (F − K) on the mids; D outside (0.5, 1.02] is rejected |
| `svi` | `SVISlice` (w, w', w'', g), SVI-JW and SSVI maps, `fit_svi` (41 × 41 grid + constrained polish, projected onto the constraint set), `fit_svi_multistart` (36-start SLSQP reference) | constraints w_min >= 0 and b(1+abs(rho)) <= 2 only; weights multiply the squared residual; `converged` and `at_bound` carried on every fit |
| `arb` | `butterfly` (g >= 0 on the quoted range + margin, asymptotes analytic), `calendar`, `quote_level` (raw-mid and within-band convexity), `repair_jw` (opt-in; raises outside GJ Thm 4.2's conditions) | violations counted inside vs outside the quoted range; the fitter never calls the repair |
| `weights` | `select` (two-sided, bid >= 2 ticks, finite iv, OTM only, relative spread, abs(k)) with a ledger; spread / vega / unit weights | the ledger conserves rows |
| `synth` | `synthetic_chain` from known SVI slices (Black prices, half-spread rule, in-band noise), `SYNTH_SURFACE`, `synthetic_cboe_json` | parity exact on the mids at noise 0; the model price stays inside every band |
| `surface` | `fit_surface`: forward → IVs → selection → weights → SVI → butterfly / quote-level / coverage per slice, calendar across consecutive distinct maturities; `Surface.skew` by true Black delta; `report()` | skipped slices carry their reason and their rows; quotes_in − selected = Σ dropped; coverage = model price inside [bid, ask] |
| `io` | `read_cboe_json`, `read_philippdubach`, `read_mztrading`, `parse_occ`; the quote-date rules; AM/PM settlement by root | source IV / Greeks columns are never read |
| `report`, `cli` | the README blocks below from fixed seeds; the private real-chain row; `volsurf fit` | nothing inside a regenerated block depends on the machine |

## Validation

Everything in this section is produced by `volsurf report` from synthetic chains with known answers
(the script regenerates the blocks in place; CI diffs the result). The pass/fail cells compare a
measured maximum with the bar pre-registered in `docs/PLAN.md`; the measured maxima print to stdout.

**Parameter recovery.** An exact slice through `fit_svi`, then `SYNTH_SURFACE`'s four expiries through
the whole chain path (Black prices → bid/ask → parity forward → IVs → selection → fit).

<!-- volsurf:begin:recovery -->
| slice | T | true (a, b, rho, m, sigma) | max abs. param error <= 1e-8 | forward error <= 1e-9 | discount error <= 1e-12 |
|---|---:|---|---|---|---|
| exact w(k), 25 strikes k in [-0.4, 0.3] | 0.25 | (0.02, 0.4, -0.6, 0.05, 0.2) | yes | n/a | n/a |
| reference chain, 2026-07-03, 24 quotes selected | 0.0493 | (0.00038, 0.01550, -0.7, 0.03396, 0.03465) | yes | yes | yes |
| reference chain, 2026-08-09, 36 quotes selected | 0.1507 | (0.00129, 0.02834, -0.7, 0.06231, 0.06357) | yes | yes | yes |
| reference chain, 2026-11-08, 39 quotes selected | 0.4000 | (0.00398, 0.04958, -0.7, 0.11016, 0.11239) | yes | yes | yes |
| reference chain, 2027-06-15, 32 quotes selected | 1.0000 | (0.01159, 0.08342, -0.7, 0.19077, 0.19462) | yes | yes | yes |
<!-- volsurf:end:recovery -->

**IV round trip** on 100,000 seeded (k, sigma, T) points at F = 100, priced as the out-of-the-money
option, bucketed by time value over F (the rows partition the sample: the bucket counts plus the floor
count sum to 100,000). Above the floor the inverter is conditioning-limited, not tolerance-limited; at the
floor it returns NaN rather than a number the price cannot support.

<!-- volsurf:begin:ivroundtrip -->
| time value / F | n | bar on abs(iv - sigma) | met | max abs(price(iv) - price) <= 1e-10 F | NaN returned |
|---|---:|---|---|---|---:|
| >= 1e-04 | 91949 | 1e-13 | yes | yes | 0 |
| [1e-06, 1e-04) | 3386 | 1e-11 | yes | yes | 0 |
| [1e-08, 1e-06) | 1551 | 1e-09 | yes | yes | 0 |
| [1e-10, 1e-08) | 818 | 1e-08 | yes | yes | 0 |
| < 1e-10 (information floor) | 2296 | NaN | yes | n/a | 2296 |
<!-- volsurf:end:ivroundtrip -->

**Fitter timing, Zeliade grid vs 36-start.** On a 151-quote slice with 0.1 vp of Gaussian noise the
grid-initialised fitter and the 36-start SLSQP reference reach the same optimum:

<!-- volsurf:begin:fitters -->
| fitter | polishes / starts | abs. objective difference <= 1e-12 | max abs. param difference <= 1e-6 | RMSE (vp) |
|---|---:|---|---|---:|
| `fit_svi` (41 x 41 Zeliade grid + constrained polish) | 3 | yes | yes | 0.11 |
| `fit_svi_multistart` (36-start SLSQP reference) | 36 | same | same | 0.11 |
<!-- volsurf:end:fitters -->

Wall times are not in a regenerated block (they depend on the machine). Printed by `volsurf report`
on an Apple M1 laptop (Darwin arm64, Python 3.12, numpy 2.5.3), best of 5: `fit_svi` 0.020 s,
`fit_svi_multistart` 0.145 s.

**Known-bad slices flagged with g_min and k.** Lee's bound and w_min > 0 both hold on the first two,
yet g < 0 inside a normal range; the Heston-like SSVI slice passes every grid point on |k| <= 3 and is
caught only by the analytic wing asymptote (evidence `analytic`). Its b(1+|rho|) = 2.36 > 2 is exactly
what the fitter's Lee constraint excludes, so on a fitted slice the asymptotes are >= 0 by construction and
`analytic` evidence can only fire on a slice handed to `butterfly` directly (an SSVI map, external
parameters). The exact slice is the control.

<!-- volsurf:begin:knownbad -->
| slice | b(1+abs(rho)) <= 2 | w_min > 0 | quoted range | g < 0 inside | g < 0 outside | g_min | at k | asymptotes left / right | evidence | flagged |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| Vogt (GJ 2014 Ex. 3.1) | yes (0.174) | yes | [-1.5, 1.5] | 307 | 0 | -0.0329 | +0.88 | +0.2495 / +0.2481 | numerical_scan | YES |
| market-like | yes (0.656) | yes | [-0.5, 0.5] | 183 | 0 | -0.0200 | -0.27 | +0.2231 / +0.2499 | numerical_scan | YES |
| Heston-like SSVI, lambda 0.3 | no (2.362) | yes | [-3, 3] | 0 | 0 | +0.1372 | -11.94 | -0.0988 / +0.2391 | analytic | YES |
| exact slice (control) | yes (0.640) | yes | [-0.4, 0.3] | 0 | 0 | +0.1250 | -0.64 | +0.2244 / +0.2484 | numerical_scan | no |
<!-- volsurf:end:knownbad -->

**Calendar-crossing pair.** Total variance must be non-decreasing in T at every k; a flatter-winged
later slice crosses the earlier one away from the money and is flagged with the k and the pair.

<!-- volsurf:begin:calendar -->
| pair | grid | crossings | worst gap w(T2) - w(T1) | at k | flagged |
|---|---|---:|---:|---:|---|
| crossing pair: b 0.4 -> 0.3 at T 0.25 -> 0.5 (flatter wings) | [-1, 1] x 401 | 401 | -0.1699 | -1.0000 | YES |
| monotone pair: a 0.02 -> 0.03 at T 0.25 -> 0.5 (control) | [-1, 1] x 401 | 0 | +0.0100 | -0.6800 | no |
<!-- volsurf:end:calendar -->

**Reference surface, quote level.** `SYNTH_SURFACE` on 0.5-wide strikes (half-spread
max(0.02, 0.01 price + 0.02 |k|)), fitted through `fit_surface` at noise 0 (mids on the model) and
noise 0.5 (mids jittered by up to half the half-spread, seed 1). Within-band violations are the
model-free ask/bid/ask convexity count on the raw quotes; raw-mid violations on the noisy chain show
why that count, not the raw one, is the evidence. "g < 0 outside" is the extrapolated wing of a slice
whose quotes span |k| < 0.1: reported, not repaired. "polish converged" is the solver's own verdict
(least_squares status > 0 or SLSQP success at its 1e-15 tolerances); "no" on the noisy 18-day slice
means its evaluation budget ran out on a flat objective whose wings its 23 selected quotes do not identify, not
that the curve is wrong (the fitters block above shows the same fitter at the 36-start optimum).

<!-- volsurf:begin:reference -->
| noise | expiry | T | quotes in the slice | selected | RMSE (vp) | max abs. (vp) | inside bid/ask | g < 0 inside | g < 0 outside | within-band violations / triplets | raw-mid violations | polish converged |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 0.0 | 2026-07-03 | 0.0493 | 144 | 24 | 0.00 | 0.00 | 100.0 % | 0 | no | 0 / 140 | 0 | yes |
| 0.0 | 2026-08-09 | 0.1507 | 162 | 36 | 0.00 | 0.00 | 100.0 % | 0 | no | 0 / 158 | 0 | yes |
| 0.0 | 2026-11-08 | 0.4000 | 162 | 39 | 0.00 | 0.00 | 100.0 % | 0 | no | 0 / 158 | 0 | yes |
| 0.0 | 2027-06-15 | 1.0000 | 162 | 32 | 0.00 | 0.00 | 100.0 % | 0 | no | 0 / 158 | 0 | yes |
| 0.5 | 2026-07-03 | 0.0493 | 144 | 23 | 0.07 | 0.20 | 100.0 % | 0 | yes | 0 / 140 | 54 | no |
| 0.5 | 2026-08-09 | 0.1507 | 162 | 35 | 0.06 | 0.13 | 100.0 % | 0 | no | 0 / 158 | 62 | yes |
| 0.5 | 2026-11-08 | 0.4000 | 162 | 39 | 0.05 | 0.14 | 100.0 % | 0 | no | 0 / 158 | 54 | yes |
| 0.5 | 2027-06-15 | 1.0000 | 162 | 32 | 0.03 | 0.07 | 100.0 % | 0 | no | 0 / 158 | 55 | yes |

noise 0.0: 4 of 4 expiries fitted, 0 calendar crossings on the quoted k-range; noise 0.5: 4 of 4 expiries fitted, 0 calendar crossings on the quoted k-range.
<!-- volsurf:end:reference -->

Coverage by |k| bucket, selected quotes pooled over the four expiries:

<!-- volsurf:begin:coverage -->
| noise | abs(k) bucket | quotes | inside bid/ask | coverage |
|---:|---|---:|---:|---:|
| 0.0 | [0.00, 0.05) | 42 | 42 | 100.0 % |
| 0.0 | [0.05, 0.15) | 50 | 50 | 100.0 % |
| 0.0 | [0.15, 0.35) | 39 | 39 | 100.0 % |
| 0.0 | [0.35, inf) | 0 | 0 | no quotes selected (|k| <= 0.35 by default) |
| 0.5 | [0.00, 0.05) | 42 | 42 | 100.0 % |
| 0.5 | [0.05, 0.15) | 49 | 49 | 100.0 % |
| 0.5 | [0.15, 0.35) | 38 | 38 | 100.0 % |
| 0.5 | [0.35, inf) | 0 | 0 | no quotes selected (|k| <= 0.35 by default) |
<!-- volsurf:end:coverage -->

### Real chains (private)

The numbers below are pasted from `volsurf report --data ... --source ... --symbol ... [--day ...]
[--rate ...] --header` runs on the author's machine (Apple M1, Darwin arm64, Python 3.12, numpy 2.5.3)
with the package defaults (spread weights, bid >= 2 ticks, |k| <= 0.35, min_quotes 8); the files are not
in the repo (see Data and privacy). "quotes in" is the number of rows the reader read; coverage is the
share of selected quotes whose model price lies inside its bid/ask; RMSE is in vol points on the selected
quotes; g < 0 counts are slices, inside / outside their quoted k-range; "polish not converged / at bound"
counts slices whose polish stopped on its evaluation budget and slices with a parameter or constraint on
its boundary (SPY 2020-03-16, the linear-in-vol day, pins 30 of 31); calendar crossings are grid points on
the k-range both adjacent slices quote; the rate passed on each American chain is in its discount-source cell.

| symbol | quote date | source | exercise | expiries fitted / skipped | quotes in -> selected | RMSE vp median / worst | inside bid/ask % median / worst | coverage by abs(k) [0,.05) / [.05,.15) / [.15,.35) / [.35,inf) | slices g<0 inside / outside | polish not converged / at bound | within-band violations / triplets | calendar crossings (quoted range) | forward residual rms median / worst | discount source | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPX | 2026-08-10 | mztrading day_2026-08-11.parquet _SPX | european | 58 / 2 | 31254 -> 11042 | 0.044 / 0.181 | 89.4 / 39.3 | 80.1 / 81.5 / 78.1 / - | 0 / 15 | 3 / 2 | 1 / 30202 | 0 | 0.0735 / 1.8487 | discount fitted on the parity line | 2.9 |
| SPY | 2023-11-15 | philippdubach SPY_options.parquet 2023-11-15 | american | 28 / 0 | 7576 -> 2076 | 0.105 / 1.066 | 79.8 / 32.1 | 70.4 / 79.9 / 79.4 / - | 3 / 10 | 8 / 4 | 0 / 7184 | 210 | 0.4199 / 1.7315 | discount pinned from rate 0.053 (fixed_discount) | 1.9 |
| SPY | 2021-06-15 | philippdubach SPY_options.parquet 2021-06-15 | american | 35 / 0 | 10226 -> 3435 | 0.182 / 0.411 | 51.6 / 7.1 | 39.2 / 54.5 / 60.2 / - | 0 / 20 | 13 / 8 | 0 / 10086 | 60 | 0.0551 / 0.4493 | discount pinned from rate 0.0005 (fixed_discount) | 3.1 |
| SPY | 2025-04-08 | philippdubach SPY_options.parquet 2025-04-08 | american | 33 / 0 | 12220 -> 4044 | 0.191 / 1.340 | 77.9 / 23.7 | 77.3 / 79.1 / 77.9 / - | 0 / 19 | 9 / 18 | 0 / 11626 | 341 | 0.1750 / 1.2685 | discount pinned from rate 0.043 (fixed_discount) | 4.0 |
| SPY | 2020-03-16 | philippdubach SPY_options.parquet 2020-03-16 | american | 31 / 5 | 11300 -> 2443 | 2.288 / 35.522 | 19.8 / 0.0 | 32.1 / 32.3 / 32.1 / - | 0 / 1 | 13 / 30 | 8 / 10002 | 958 | 0.0527 / 1.0010 | discount pinned from rate 0.003 (fixed_discount) | 2.2 |
| QQQ | 2023-11-15 | philippdubach QQQ_options.parquet 2023-11-15 | american | 28 / 0 | 6920 -> 2043 | 0.124 / 0.305 | 84.5 / 35.7 | 62.0 / 84.0 / 93.4 / - | 0 / 12 | 18 / 7 | 0 / 6554 | 44 | 0.2054 / 1.3493 | discount pinned from rate 0.053 (fixed_discount) | 3.0 |
| IWM | 2023-11-15 | philippdubach IWM_options.parquet 2023-11-15 | american | 25 / 2 | 4222 -> 1240 | 0.127 / 0.367 | 87.5 / 60.2 | 77.1 / 83.2 / 86.4 / - | 0 / 4 | 8 / 6 | 0 / 3808 | 42 | 0.1360 / 1.2484 | discount pinned from rate 0.053 (fixed_discount) | 1.8 |
| SPY | 2026-08-10 | mztrading day_2026-08-11.parquet SPY | american | 34 / 1 | 14664 -> 4448 | 0.084 / 0.244 | 54.1 / 10.3 | 40.6 / 54.5 / 55.0 / - | 0 / 12 | 6 / 3 | 73 / 14118 | 0 | 0.4783 / 3.9734 | discount pinned from rate 0.0427 (fixed_discount) | 2.0 |
| AAPL | 2026-08-10 | mztrading day_2026-08-11.parquet AAPL | american | 22 / 2 | 3652 -> 508 | 0.123 / 0.421 | 100.0 / 75.0 | 92.4 / 97.1 / 98.4 / - | 0 / 7 | 16 / 6 | 0 / 3268 | 0 | 0.2703 / 1.4480 | discount pinned from rate 0.0427 (fixed_discount) | 2.5 |
| NVDA | 2026-08-10 | mztrading day_2026-08-11.parquet NVDA | american | 23 / 0 | 3922 -> 658 | 0.287 / 0.682 | 100.0 / 20.0 | 77.6 / 77.2 / 88.3 / - | 0 / 1 | 14 / 6 | 69 / 3704 | 0 | 0.1006 / 0.5474 | discount pinned from rate 0.0427 (fixed_discount) | 2.5 |
| XOM | 2026-08-10 | mztrading day_2026-08-11.parquet XOM | american | 11 / 5 | 1264 -> 165 | 0.147 / 0.347 | 100.0 / 100.0 | 100.0 / 100.0 / 100.0 / - | 0 / 2 | 3 / 4 | 0 / 742 | 0 | 0.3595 / 0.6562 | discount pinned from rate 0.0427 (fixed_discount) | 0.8 |

Read it as a desk would. SPX (European, 58 expiries) is the clean case: the parity line fits the
discount itself (implied rates 3.9–4.6 % across 30 expiries, SPX and SPXW on one date agreeing to 0.2
index points), 0.044 vp median RMSE, 89 % of selected quotes inside their bid/ask, no calendar crossing on
the quoted ranges. The American rows carry a pinned discount and the early-exercise caveat, and are worse in
proportion to how low vol was: the calm 2021 day fits worst of the normal days (0.18 vp, 52 % inside)
because a low-vol SPY smile has a steep, convex-in-vol put wing that a raw-SVI hyperbola cannot follow
— the best fit on the full quoted range (both fitters agree) has ρ ≈ +0.9, i.e. wrong-signed wings; that is
why the selection stops at |k| ≤ 0.35 (without the cap: 0.28 vp, 23 % inside, 338 calendar grid crossings,
nearly all wing extrapolation). 2020-03-16 is the documented failure: smiles linear in vol, the polish pins
30 of 31 slices on a bound and the worst slice is 35 vp off. The within-band quote-level counts (0 on every
philippdubach day, 73 and 69 on the delayed mztrading SPY/NVDA snaps) are the honest "arbitrage in the
quotes" number; the raw-mid counts the fitted-surface diagnostics also print are 15–30 % on every day and mean
nothing.

## As tools an agent can call

`volsurf.mcp` exposes the fitted surface as six plain functions — `fetch_chain`, `calibrate_surface`,
`surface_term_structure`, `surface_skew`, `option_greeks`, `arbitrage_check` — and a FastMCP server over
them. It is an adapter, not a second implementation: every forward, fit, Greek and arbitrage verdict comes
from the modules above. A bundled XSP chain ships so the tools answer with nothing installed; `[live]` adds a
yfinance source.

```bash
pip install "volsurf[mcp]"        # the server
volsurf-mcp                       # stdio; register it with your MCP host
python -m volsurf.mcp.evals       # 30 evals against the bundled chain
```

The evals are the part worth reading. Expected values are *derived* from the SVI parameters, discounts and
dates the bundled chain was generated from (`mcp/data/make_golden_chain.py`), never recorded from a previous
run — a golden set recorded from your own output freezes your bugs as the specification and passes forever.
Time-to-expiry is re-derived through `quotes.time_to_expiry`, so the generator and the reader cannot disagree
about T. On the bundled chain the tools recover every forward to under 1e-4 and every slice to 0.25 vol points
RMSE, and `arbitrage_check` returns a verified `arbitrage_free`, not an assumed one.

`fetch_chain` / `calibrate_surface` are cached per (underlying, source, filter), so six tool calls in one
conversation share one calibration and agree with each other.

## Design rules

Tested:
- The Lee bound is b(1+|rho|) <= 2 in total variance, not 4 (`test_svi`): the fitter's constraint, the
  asymptote formula (4 − b²(1±rho)²)/16 and the SSVI form agree.
- Zeliade's 0 <= a box is not imposed: the fitter recovers Vogt's own arbitrage example, whose a is
  −0.041, from its w(k); the grid initialiser lands on the 36-start optimum (`test_svi`).
- Butterfly violations are reported inside vs outside the quoted range and the wing limit is analytic:
  the Heston-like slice passes |k| <= 3 on a grid and fails at the asymptote (`test_arb`).
- Two arbitrage bases: raw-mid convexity counts and within-band counts are separate numbers, and only
  the within-band count is 0 on an on-model chain with in-band noise (`test_arb`, `test_surface`).
- The parity line refuses American chains and the pinned-discount estimator recovers F within 1 bp on a
  Black-priced American-flagged chain; ±50 bp of rate moves F by < 2 bp at 30 DTE (`test_forward`).
- The inverter returns NaN exactly where time value < 1e-10 F and agrees with `pricers.bs.implied_vol`
  and `lets_be_rational` on the ok region (`test_iv`; the two oracles skip when absent).
- `Surface.skew` finds the strikes whose Black delta is exactly ±0.25 by root-finding (`test_surface`).
- The README blocks regenerate byte-identically (`test_report_cli`); every dropped row is in a ledger and
  quotes_in − quotes_selected = Σ dropped holds through a reader with drops and a skipped slice (`test_surface`).
- Two roots on one maturity (SPX and SPXW, both PM) fit and are each compared with the neighbouring
  maturities; a chain with a non-datetime `expiry` column or a NaN quote is rejected, not silently
  empty; a NaN vol prices as NaN (`test_surface`, `test_quotes`, `test_black`).
- A polished point a few 1e-6 outside w_min >= 0 is projected onto the constraint set and kept
  (`method` "+proj"), and `fit_svi` names what sits on a boundary in `at_bound` (`test_svi`).
- The T rule is pinned to the clock: same-day PM −15 min, next-morning AM 17 h 15 min, PM − AM 6.5 h, and
  the one-hour drop is counted once (`test_quotes`).

By construction (not a test):
- No number in this repo reads spot, a source IV column, or an external forward.
- Nothing is repaired: `repair_jw` exists, is tested, and is never called by the fitter or the surface.
- A slice that cannot be fitted is a skip with a reason, never a silent absence.

## What is where

```text
src/volsurf/
  quotes.py     Chain / Slice / Ledger, the T rule, coverage line
  black.py      Black-76 on the forward: price, vega, Greeks, delta_spot
  iv.py         implied_vol (Halley in s with bracket), implied_vols, from_spot
  forward.py    fit_forward (parity_line / fixed_discount, window, Huber), forward_table
  svi.py        SVIParams / SVISlice / g(k), JW and SSVI maps, fit_svi, fit_svi_multistart
  arb.py        butterfly, calendar, quote_level, check_arbitrage, repair_jw
  weights.py    select (with ledger), spread / vega / unit weights
  synth.py      synthetic_chain, SYNTH_SURFACE, synthetic_cboe_json
  surface.py    fit_slice / fit_surface, SliceFit / Surface, skew, term_structure, report
  io.py         parse_occ, read_cboe_json, read_philippdubach, read_mztrading
  report.py     README blocks, private row, timing
  cli.py        volsurf report / fit
  mcp/          the surface as tools: sources (bundled XSP chain, yfinance), facade (Surface -> tool
                vocabulary), tools (the six functions), server (FastMCP), data/ (golden chain + generator),
                evals/ (30 derived-not-recorded evals; python -m volsurf.mcp.evals)
tests/          one file per module (report and cli share test_report_cli.py); tests/mcp/ for the tool
                contract and the eval harness; synthetic data only
docs/PLAN.md    the contract (v0.1 scope, oracles, what changed from plan v1 and why)
docs/DESIGN.md  the design rules and the findings behind them
CHANGELOG.md    what each version added
```

## Roadmap

- v0.2: SSVI (power law with gamma <= 1/2, Heston-like with lambda >= (1+|rho|)/4; GJ Thm 4.1 / 4.2 as tests) and
  eSSVI with Pasquazzi's calendar conditions; a Greeks module (sticky-strike, sticky-moneyness, minimum-variance
  delta); smile and total-variance plots; the American implied-dividend refinement (Brent on q against a binomial
  tree via `pricers.trees`); time interpolation between slices. (The MCP layer that was planned as `options-surface-mcp` v0.2 landed here instead, as `volsurf.mcp`.)
- v0.3: local volatility (Dupire in total variance with the denominator ≡ g(k) as a tested identity; negative local
  variance reported, never clipped); events (straddle expected move, event variance between two expiries); VIX
  single-slice fits with `forward_source=parity|futures` and the calendar check marked not applicable; PyPI.

## Data and privacy

No real option chain is committed and the test suite is synthetic. The philippdubach SPY/QQQ/IWM
parquet mirror carries an MIT tag only second-hand (a release note of a downstream mirror); the
upstream repository and CDN vanished in 2026 and its README said "educational and research purposes",
so nothing from it is redistributed. mztrading day files have no licence and are Cboe delayed data.
Cboe's quote-table terms prohibit auto-extraction, so there is no `fetch` command:
`volsurf.io.read_cboe_json` parses a file the user saved by hand from the quote-table page after
manual ticker entry. Private files live under `data/private/` (gitignored, with
`docs/validation-private.md`); the real-chains table above is filled by pasting the rows that
`volsurf report --data` prints, with the source, day and machine.

## Companion repos

[tcakit](https://github.com/charlieyanhx/tcakit) — transaction cost analysis and market impact ·
[quant-research-agent](https://github.com/charlieyanhx/quant-research-agent) — a backtest review agent
and the evals that measure it ·
[deskboard](https://github.com/charlieyanhx/deskboard) — real-time options risk / P&L dashboard ·
[pricers](https://github.com/charlieyanhx/pricers) — option pricers; the IV and Heston COS oracles here ·
[riskkit](https://github.com/charlieyanhx/riskkit) — portfolio risk with the limit-down and empty-book states first-class ·
[quotesim](https://github.com/charlieyanhx/quotesim) — market-making simulator whose fair surface this
repo will provide ·
[tickq](https://github.com/charlieyanhx/tickq) — DuckDB market-data SQL: partitioned Parquet lake, ASOF
joins with the tie rule stated, quality checks with recall and precision ·
[lobcore](https://github.com/charlieyanhx/lobcore) — bounded-array limit order book in Rust with a
reference-book differential test, ITCH 5.0 replay and PyO3 bindings ·
[exhibitkit](https://github.com/charlieyanhx/exhibitkit) — sell-side research documents from Markdown, exhibits
with mandatory source lines ·
[claimkeeper](https://github.com/charlieyanhx/claimkeeper) — a ledger that scores a note's falsifiable claims
right or wrong once their dates arrive.

MIT © Hanxiong (Charlie) Yan

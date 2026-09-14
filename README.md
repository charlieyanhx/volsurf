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
pytest -q          # 118 tests: ~18 s on a laptop (IV inverter, forward, SVI, arbitrage, weights, synth, readers, surface, CLI)
volsurf report     # regenerates the synthetic tables below in place; run twice, the second run is a no-op
volsurf fit data/private/spy.json --source cboe          # one surface: term structure, skips, arbitrage notes
volsurf report --data day_2026-08-11.parquet --source mztrading --symbol _SPX --header   # private row
```

`volsurf report --data ... --rate 0.053` is required on American chains (SPY, QQQ, IWM, single names): the
discount is pinned from the rate there because it is not identifiable from parity (see Design rules).

## What it computes

| module | what | the invariant it keeps |
|---|---|---|
| `quotes` | `Chain` / `Slice` / `Ledger`: the quote schema, the ACT/365 time rule (16:15 quote to 16:00 PM / 09:30 AM settlement), one slice per (root, expiry) | crossed quotes and duplicate keys raise; every dropped row is counted |
| `black` | Black-76 on the forward: price, vega, Greeks per 1.00 vol and per calendar day | price = D · Black(F, K, T, sigma) |
| `iv` | `implied_vol` (Corrado-Miller start, bracketed Halley in s = sigma sqrt(T)), `implied_vols` per slice | NaN below intrinsic, above the cap, or when time value < 1e-10 F; never a guess |
| `forward` | `fit_forward`: parity line (F, D) for European chains, discount pinned from a rate for American ones; spread weights, strike window, Huber pass | C − P = D (F − K) on the mids; D outside (0.5, 1.02] is rejected |
| `svi` | `SVISlice` (w, w', w'', g), SVI-JW and SSVI maps, `fit_svi` (41 × 41 grid + constrained polish), `fit_svi_multistart` (36-start SLSQP reference) | constraints w_min >= 0 and b(1+abs(rho)) <= 2 only; weights multiply the squared residual |
| `arb` | `butterfly` (g >= 0 on the quoted range + margin, asymptotes analytic), `calendar`, `quote_level` (raw-mid and within-band convexity), `repair_jw` (opt-in) | violations counted inside vs outside the quoted range; the fitter never calls the repair |
| `weights` | `select` (two-sided, bid >= 2 ticks, finite iv, OTM only, relative spread, abs(k)) with a ledger; spread / vega / unit weights | the ledger conserves rows |
| `synth` | `synthetic_chain` from known SVI slices (Black prices, half-spread rule, in-band noise), `SYNTH_SURFACE`, `synthetic_cboe_json` | parity exact on the mids at noise 0; the model price stays inside every band |
| `surface` | `fit_surface`: forward → IVs → selection → weights → SVI → butterfly / quote-level / coverage per slice, calendar across slices; `Surface.skew` by true Black delta; `report()` | skipped slices carry their reason; coverage = model price inside [bid, ask] |
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
| reference chain, 2026-11-08, 45 quotes selected | 0.4000 | (0.00398, 0.04958, -0.7, 0.11016, 0.11239) | yes | yes | yes |
| reference chain, 2027-06-15, 55 quotes selected | 1.0000 | (0.01159, 0.08342, -0.7, 0.19077, 0.19462) | yes | yes | yes |
<!-- volsurf:end:recovery -->

**IV round trip** on 100,000 seeded (k, sigma, T) points at F = 100, priced as the out-of-the-money
option, bucketed by time value over F. Above the floor the inverter is conditioning-limited, not
tolerance-limited; at the floor it returns NaN rather than a number the price cannot support.

<!-- volsurf:begin:ivroundtrip -->
| time value / F | n | bar on abs(iv - sigma) | met | max abs(price(iv) - price) <= 1e-10 F | NaN returned |
|---|---:|---|---|---|---:|
| >= 1e-04 | 91949 | 1e-13 | yes | yes | 0 |
| [1e-06, 1e-04) | 3386 | 1e-11 | yes | yes | 0 |
| [1e-08, 1e-06) | 1551 | 1e-09 | yes | yes | 0 |
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
caught only by the analytic wing asymptote (evidence `analytic`). The exact slice is the control.

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
whose quotes span |k| < 0.1: reported, not repaired.

<!-- volsurf:begin:reference -->
| noise | expiry | T | quotes in the slice | selected | RMSE (vp) | max abs. (vp) | inside bid/ask | g < 0 inside | g < 0 outside | within-band violations / triplets | raw-mid violations |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.0 | 2026-07-03 | 0.0493 | 144 | 24 | 0.00 | 0.00 | 100.0 % | 0 | no | 0 / 140 | 0 |
| 0.0 | 2026-08-09 | 0.1507 | 162 | 36 | 0.00 | 0.00 | 100.0 % | 0 | no | 0 / 158 | 0 |
| 0.0 | 2026-11-08 | 0.4000 | 162 | 45 | 0.00 | 0.00 | 100.0 % | 0 | no | 0 / 158 | 0 |
| 0.0 | 2027-06-15 | 1.0000 | 162 | 55 | 0.00 | 0.00 | 100.0 % | 0 | no | 0 / 158 | 0 |
| 0.5 | 2026-07-03 | 0.0493 | 144 | 23 | 0.07 | 0.20 | 100.0 % | 0 | yes | 0 / 140 | 54 |
| 0.5 | 2026-08-09 | 0.1507 | 162 | 35 | 0.06 | 0.13 | 100.0 % | 0 | no | 0 / 158 | 62 |
| 0.5 | 2026-11-08 | 0.4000 | 162 | 45 | 0.06 | 0.16 | 100.0 % | 0 | no | 0 / 158 | 54 |
| 0.5 | 2027-06-15 | 1.0000 | 162 | 54 | 0.07 | 0.28 | 100.0 % | 0 | no | 0 / 158 | 55 |

noise 0.0: 4 of 4 expiries fitted, 0 calendar crossings on the quoted k-range; noise 0.5: 4 of 4 expiries fitted, 0 calendar crossings on the quoted k-range.
<!-- volsurf:end:reference -->

Coverage by |k| bucket, selected quotes pooled over the four expiries:

<!-- volsurf:begin:coverage -->
| noise | abs(k) bucket | quotes | inside bid/ask | coverage |
|---:|---|---:|---:|---:|
| 0.0 | [0.00, 0.05) | 42 | 42 | 100.0 % |
| 0.0 | [0.05, 0.15) | 50 | 50 | 100.0 % |
| 0.0 | [0.15, 0.35) | 39 | 39 | 100.0 % |
| 0.0 | [0.35, inf) | 29 | 29 | 100.0 % |
| 0.5 | [0.00, 0.05) | 42 | 42 | 100.0 % |
| 0.5 | [0.05, 0.15) | 49 | 49 | 100.0 % |
| 0.5 | [0.15, 0.35) | 38 | 38 | 100.0 % |
| 0.5 | [0.35, inf) | 28 | 28 | 100.0 % |
<!-- volsurf:end:coverage -->

### Real chains (private)

The numbers below are pasted from `volsurf report --data ... --source ... --symbol ... [--day ...]
[--rate ...] --header` runs on the author's machine; the files are not in the repo (see Data and
privacy). Coverage is the share of selected quotes whose model price lies inside its bid/ask;
RMSE is in vol points on the selected quotes; g < 0 counts are slices, inside / outside their quoted
k-range; calendar crossings are grid points on the k-range both adjacent slices quote.

| symbol | quote date | source | exercise | expiries fitted / skipped | quotes in -> selected | RMSE vp median / worst | inside bid/ask % median / worst | coverage by abs(k) [0,.05) / [.05,.15) / [.15,.35) / [.35,inf) | slices g<0 inside / outside | within-band violations / triplets | calendar crossings (quoted range) | forward residual rms median / worst | discount source | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

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
- The README blocks regenerate byte-identically (`test_report_cli`); every dropped row is in a ledger.

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
tests/          one file per module; synthetic data only
docs/PLAN.md    the contract (v0.1 scope, oracles, what changed from plan v1 and why)
docs/DESIGN.md  the design rules and the findings behind them
CHANGELOG.md    what each version added
```

## Roadmap

- v0.2: SSVI (power law with gamma <= 1/2, Heston-like with lambda >= (1+|rho|)/4; GJ Thm 4.1 / 4.2 as tests) and
  eSSVI with Pasquazzi's calendar conditions; a Greeks module (sticky-strike, sticky-moneyness, minimum-variance
  delta); smile and total-variance plots; the American implied-dividend refinement (Brent on q against a binomial
  tree via `pricers.trees`); time interpolation between slices; `options-surface-mcp` v0.2 depending on volsurf.
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
[options-surface-mcp](https://github.com/charlieyanhx/options-surface-mcp) — the MCP server that will
re-export volsurf.

MIT © Hanxiong (Charlie) Yan

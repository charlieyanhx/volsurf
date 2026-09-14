# volsurf — design

## One rule

Every number is computed against the expiry's own parity forward, and every static-arbitrage
check reports where it failed instead of adjusting the surface until it passes. Concretely:
`forward.fit_forward` reads F (and, on European chains, D) out of C − P = D (F − K) on the
quoted mids; `iv.implied_vols` inverts bid, mid and ask against that (F, D); `svi.fit_svi`
fits total variance w(k), k = ln(K/F), under the two conditions that are actually necessary;
`arb.butterfly`, `arb.calendar` and `arb.quote_level` count and locate the violations
(inside or outside the quoted k-range, on which pair, at which k, on which basis); nothing in
the fit path calls `arb.repair_jw`. The tape of decisions is the `SliceFit`: forward fit,
per-quote IVs, the selection mask with its ledger, the weights, the SVI fit, the butterfly
report, the quote-level report and the coverage number, so a reader can see which quotes a
number was computed on. What is not read anywhere: spot, a source IV column, an external
forward.

## Conventions

- **Time.** T is ACT/365 from the quote instant (quote date at 16:15 ET, the US option close)
  to the settlement instant: expiry at 16:00 ET for PM-settled contracts, 09:30 ET for
  AM-settled ones (SPX, NDX, RUT, DJX and XSP monthlies on a third Friday; every VIX expiry).
  The intraday fraction is kept: a same-day PM expiry has T = −15 min < 0 and is expired; a
  next-morning AM expiry has T = 17 h 15 min. A slice with T ≤ one hour is dropped and counted.
  `quotes.time_to_expiry` is the single implementation; every T in the package comes from it.
- **Units.** Vols per 1.00 (0.20 = 20 %); fit errors in vol points (vp = 100 × vol); w = σ²T;
  k = ln(K/F); prices in dollars per share/unit as quoted; forward residuals in price units;
  coverage in percent; the discount D = e^{−rT} with r continuous per year.
- **Signs.** g < 0, w(T₂) − w(T₁) < 0 and a negative second divided difference are the
  violations; `worst_*` is the most negative value seen even when positive. Skew is put minus
  call IV at ±0.25 Black delta (positive for the usual equity smile).
- **Ledgers.** Every rule that removes a row records (rule, rows in, rows out), in the reader
  (`Chain.ledger`) and in the selection (`SliceFit.ledger`); the surface report sums them and
  the tests assert quotes_in − quotes_selected = Σ dropped.

## The Lee bound is 2, not 4

Lee's moment formula bounds the asymptotic slope of *total* implied variance by 2, so raw SVI
needs b(1 + |ρ|) ≤ 2 (Gatheral–Jacquier 2014 Remark 4.3; Martini–Mingone 2021). Gatheral 2004
and Zeliade 2009 print 4 because they bound T · d(σ²)/dk with σ² the annualised variance and
then drop the factor; the author's own earlier scripts and `options-surface-mcp`'s docstring
carry the 4 (that code's SLSQP constraint uses 2 — the docstring is wrong, the code is
right). Shipping 4 lets wings twice as steep as any martingale measure allows through the
fit. `svi.LEE_BOUND = 2.0` is pinned by a test, and the same number appears as the sign of the
wing asymptotes of g: for raw SVI g(k) → (4 − b²(1 ± ρ)²)/16 as k → ±∞, which is ≥ 0 iff
b(1 ± ρ) ≤ 2 per wing (with b = θφ/2 this is GJ Lemma 4.2's (16 − (θφ)²(1 ± ρ)²)/64). The
Lee bound is necessary, not sufficient: Vogt's slice (GJ Example 3.1) has b(1 + |ρ|) = 0.174
and w_min > 0 yet g_min = −0.0329 at k = 0.879; a market-like negative-skew slice
(0.0163, 0.3455, −0.8988, 0.0373, 0.1164) has b(1 + |ρ|) = 0.656 and g_min = −0.0200 at
k = −0.276, inside a normal strike range. Both are in the README's known-bad table.

## Zeliade's box is an initialiser, not a constraint

Zeliade's quasi-explicit calibration (De Marco–Martini 2009) changes variables to
y = (k − m)/σ so that w = a + d·y + c·√(y² + 1) is linear in (a, d, c) for fixed (m, σ); the
outer problem is two-dimensional. Two findings decide how it is used here:

1. **The box 0 ≤ a is not a no-arbitrage condition.** Only w_min = a + bσ√(1 − ρ²) ≥ 0 is.
   Vogt's arbitrage example has a = −0.041, and on SPY 2023-11-15 30 DTE the box degrades the
   fit 18×: RMSE 1.657 vp and 1.3 % of quotes inside bid/ask with a ≥ 0, versus 0.093 vp and
   59.6 % with a free and w_min ≥ 0 (the fitted a is −0.012). The fitter constrains
   w_min ≥ 0 and b(1 + |ρ|) ≤ 2 only, and `test_vogt_fits_from_its_own_w_with_negative_a`
   pins that a negative a is recoverable.
2. **The full Zeliade solver is slow when the boundary is active** (50–160 s per slice with a
   constrained inner solve inside Nelder–Mead when σ → 0 or a sits on the boundary), but a
   41 × 41 grid over (m ∈ [k_min, k_max], σ geometric on [0.005, 1]) with the unconstrained
   linear inner solve, followed by one constrained polish of the best few grid points, reaches
   the same optimum as a 36-start five-parameter SLSQP search (objective difference ≤ 1e-12
   on the README's 151-quote noisy slice) in about a tenth of the time. `fit_svi` is that;
   `fit_svi_multistart` is the reference kept for the bench.

On an exact synthetic slice the reduced problem recovers the parameters to 1e-8 (measured
below 1e-12 by `volsurf report`); in the research pass behind this plan a 36-start
five-parameter search alone reached only ~1e-5 on the same slice — the reduction is better
conditioned, not just faster.

## Two arbitrage bases, and where a violation is

`arb.butterfly` evaluates g on a dense grid over [k_lo − margin, k_hi + margin] with
(k_lo, k_hi) the range the fit actually used, and counts violations *inside* the quoted range
and *outside* it separately; then it adds the two analytic asymptotes. The asymptotes are not
decoration: a Heston-like SSVI slice with λ = 0.3, ρ = −0.7, θ = 20 passes every grid point
on |k| ≤ 3 (g ≥ +0.14) and fails at the left asymptote (−0.0988), so the grid alone would
certify a slice that admits a butterfly at large |k|. The report carries `evidence`
(`analytic` when the asymptote decided, `numerical_scan` otherwise). A violation only outside
the quoted range is extrapolation arbitrage: it is reported as such (the README's "g < 0
outside" column, the `slices_g_neg_outside` count), never repaired, and a user who needs the
wing can call `repair_jw` (GJ §5.1's closed form) explicitly.

`arb.quote_level` is model-free and is the second basis. On real chains raw-mid convexity
violations are 15–29 % of triplets and mean nothing (the mids are not a price anyone can
trade); the within-band count — the most convexity-favourable prices inside the bands, ask /
bid / ask on the outer / middle / outer strikes — is ~0 on SPY (0/7,777; 1/9,913; 0/11,355 on
the private days). The README reports the within-band count and the fitted-surface count as
separate numbers, and the reference-surface table shows the raw-mid count exploding under
in-band noise while the within-band count stays 0.

`arb.calendar` checks w(T₂) ≥ w(T₁) at every grid k for adjacent slices. `fit_surface` runs
it on the k-range both slices quote (the intersection of their selected ranges); the wide
[−1, 1] version is `Surface.arbitrage()`. A crossing at a k neither slice has a quote at is,
again, extrapolation, and the two are not mixed.

## The forward and the discount on American underlyings

Free (F, D) parity lines on SPY return D > 1 — implied r of −0.8 % to −54 % — at every tenor
and every strike window when r ≈ 5 %, because ITM American puts carry early-exercise premium
and the line's slope absorbs it; SPX (European) on the same days is clean (r 3.4–6.8 % across
windows, F identical to the cent). The pair (F, D) is therefore not identifiable from parity on
an American chain. Rule: `exercise = "american"` → `fixed_discount` (D = e^{−rT} from a
supplied rate, F = Σw(C − P + DK)/(DΣw)), and `fit_forward(mode="auto")` raises without a
rate; `parity_line` refuses American chains unless `force=True`. The residual bias of
`fixed_discount` on a dividend-paying single name is about −11 bp of F (a synthetic
AAPL-like chain priced on a binomial tree; the test pins it between −15 and −5 bp); the
implied-q refinement that removes it (Brent on q so that the near-ATM American-tree IV of
calls and puts agree) is v0.2. The rate sensitivity is small where it matters: ±50 bp moves F
by < 2 bp at 30 DTE.

Why never spot: US option marks are 16:15 ET and the stock close is 16:00. On SPY 2020-03-16
the parity forward for every expiry sits +1.5 % above the 16:00 close; using S·e^{(r−q)T}
would have put every IV on that day against the wrong forward. VIX options settle on the
future, not the index. So F comes from the quotes, with spread weights 1/(hw_C² + hw_P²)
(half-widths floored at one tick), a near-ATM window recentred once on the first estimate and
widened only while it holds too few pairs, and a Huber pass (c = 3 on the MAD scale) for the
stale far quote the window lets through.

## The inverter

`iv.implied_vol` normalises by the forward, converts to the out-of-the-money option by parity,
starts from Corrado–Miller and takes Halley steps in s = σ√T inside a bracket with bisection
fallback. Two rules that a first draft got wrong and the tests now pin: the convergence test
runs *before* the bracket test (otherwise a converged point is bisected away), and there is
no absolute |model − p| < ε test (it "converges" deep-OTM options at the wrong root); the
stopping rule is |Δs| ≤ 1e-14·s or |model − p| ≤ 1e-15·p. Time value below 1e-10·F returns
NaN: the map is numerically flat there (∂price/∂σ ≈ 1e-5·F) and a number would be a guess. On
the ok region the inverter agrees with `pricers.bs.implied_vol` and `lets_be_rational` to
1e-9 (both optional oracles).

## Data policy

No real option chain is committed and the whole test suite is synthetic (`synth.py` builds
chains from known SVI slices through Black prices, with a half-spread rule and optional
in-band noise). The reasons are licences, not taste: the philippdubach SPY/QQQ/IWM parquet
mirror carries an MIT tag only second-hand (a downstream release note) while its own README
said "educational and research purposes", and the upstream repository and CDN vanished in
2026; mztrading day files have no licence and are Cboe delayed data; Cboe's quote-table
terms prohibit auto-extraction, so there is no `fetch` command — `io.read_cboe_json` parses a
file the user saved by hand after manual ticker entry. Real-data numbers are produced
privately by `volsurf report --data` and pasted into the README's real-chains table with the
source, day and machine; the files live under `data/private/` (gitignored). The readers never
consume a source's IV or Greeks columns (philippdubach's 2008-10 IVs are floored at 0.0149,
and every file's IV is against an unknown forward, rate and clock).

Reader date rules that real files forced: a Cboe payload's `timestamp` is the download time
(off-hours it is the next morning), so the quote date is the date of the latest
`last_trade_time`; mztrading files are the prior session's close labelled D+1, so the quote
date is the file date minus one business day and the reader raises when that disagrees with
the latest trade time (a holiday snap); rows whose expiry precedes the quote date are dropped
and counted; a crossed quote is dropped and counted; on a duplicate (root, expiry, strike,
right) the last row wins.

## What is not done (v0.1)

- No SSVI / eSSVI fits (`svi.from_ssvi` only maps an SSVI slice to raw parameters). The
  conditions are recorded for v0.2: the power law φ = η/(θ^γ(1 + θ)^{1−γ}) is arbitrage-free
  under η(1 + |ρ|) ≤ 2 only for γ ≤ ½ (γ = 0.7 gives g_min = −1.26 at short maturities);
  Heston-like φ needs λ ≥ (1 + |ρ|)/4; eSSVI calendar needs Pasquazzi 2023 Prop 4.14, not
  Corbetta's wing-only inequality, which passes crossing slices.
- No implied-dividend refinement on American chains (the −11 bp bias above stands).
- No Greeks module, no plots, no time interpolation between slices, no local volatility
  (Dupire's denominator ≡ g(k) is recorded as the identity to test in v0.3), no event
  extraction, no VIX-specific forward source. The SVI shape itself fails on days whose smile
  is linear in vol (SPY 2020-03-16: RMSE 1.1–2.0 vp, σ → 0); that is documented as expected,
  not patched.
- `pandas` DataFrames are the chain container; nothing is optimised beyond the vectorised
  inverter and the grid fitter (a 150-quote slice fits in ~0.02 s, a four-expiry chain in
  ~0.1 s on the author's laptop).

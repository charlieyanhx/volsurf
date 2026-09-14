# volsurf — plan v2 (2026-09-12)

Plan v1 (the user's) asked for a multi-underlying arbitrage-free surface library validated on
public CBOE quotes, and a companion market-making simulator. Five independent research passes and
a cross-check changed the plan in these ways; the reasons are numbers, not taste.

## What changed and why

1. **No real chain is committed.** The only redistributable-looking source (philippdubach SPY/QQQ/IWM
   mirror) has an MIT tag that exists only second-hand in a release note; the upstream repo and CDN
   vanished in 2026 and its README said "educational and research purposes". mztrading day files carry
   no licence and are Cboe delayed data. Cboe's quote-table terms prohibit auto-extraction, so no
   `fetch` command either. Policy, copied from riskkit's CME file: **tests and CI run on synthetic chains
   generated from known parameters; real-data tables are produced privately by `volsurf report --data`
   and pasted into the README with the source, day and machine; the files stay gitignored.**
   `volsurf.io.read_cboe_json(path)` parses a manually saved quote-table download.
2. **The forward is fitted with the discount pinned on American underlyings.** Free (F, D) parity lines
   on SPY return D > 1 (implied r of −0.8 % to −54 %) at every tenor and window when r ≈ 5 %, because
   ITM American puts carry early-exercise premium. SPX (European) is clean (r 3.4–6.8 % across windows,
   F identical to the cent). Rule: `exercise="american"` → `fixed_discount` from a supplied rate,
   spread-weighted near-ATM WLS with a Huber pass; `exercise="european"` → free (F, D) line allowed.
3. **The Lee bound is b(1+|ρ|) ≤ 2 in total variance**, not 4 (Gatheral–Jacquier 2014 Remark 4.3;
   Martini–Mingone 2021). The author's own prior scripts and surfacemcp's docstring say 4; a test pins 2.
4. **Zeliade's 0 ≤ a box is not a no-arbitrage condition** and degrades real SPY fits 4–5× (SPY 2023-11-15
   30 DTE: RMSE 0.44 vp with a ≥ 0 vs 0.093 vp with a free under the research selection — unit weights,
   |k| ≤ 0.35, bid ≥ 5 ticks — and 0.58 vs 0.137 vp under the package defaults; the 18× / 1.657 vp of an
   earlier draft came from a different optimiser and does not reproduce; Vogt's own arbitrage example has
   a < 0). The fitter constrains `w_min = a + bσ√(1−ρ²) ≥ 0` and `b(1+|ρ|) ≤ 2` only. Zeliade's (m, σ)
   reduction is the initialiser: 41×41 grid with an unconstrained 3-parameter linear inner solve, then one
   constrained polish, projected onto the constraint set — the same optimum as 36-start SLSQP on the
   README's 151-quote slice (objective difference ≤ 1e-12), 0.020 s vs 0.145 s there (the README's measured
   numbers; the research pass's 0.14 s / 1.2–1.9 s were a slower prototype).
5. **Butterfly checks are on a declared range plus the wing asymptotes.** A grid on [k_min−m, k_max+m]
   misses wing arbitrage (Heston-like SSVI with λ = 0.3 passes |k| ≤ 3 and fails at the asymptote);
   raw SVI's asymptotic g sign is b(1±ρ) < 2. Violations are labelled inside vs outside the quoted
   range; counted, never repaired (GJ §5.1's repair is an explicit opt-in function).
6. **Two arbitrage bases are reported.** Raw-mid butterfly violations on SPY are 15–29 % of triplets and
   mean nothing; within-bid/ask violations are ~0 (0/7,777; 1/9,913; 0/11,355). The README reports the
   quote-level within-band count and the fitted-surface count separately.
7. **VIX, SSVI/eSSVI, Greeks conventions, local vol, events, plots move to v0.2/v0.3** with the exact
   conditions recorded (SSVI power law needs γ ≤ ½; eSSVI calendar needs Pasquazzi 2023 Prop 4.14, not
   Corbetta's wing-only inequality, which passes crossing slices; Dupire's denominator ≡ g(k)).
8. **The simulator is a separate numpy-only repo (`quotesim`; `mmsim` is taken on PyPI)** with a
   `FairSurface` protocol; a volsurf adapter comes in its v0.2. Its plan is in that repo.
9. **pricers 0.2.1 first**: `bs.implied_vol_vec` could overwrite an already-converged root with the
   bracket midpoint (50 of 20,000 rounded-price puts wrong by up to 0.43 vol). Fixed and pushed; it is
   now a usable oracle.

## v0.1 scope (this release)

| module | contract |
|---|---|
| `quotes.py` | `Chain(symbol, quote_date, df, exercise, quote_time="16:15", multiplier, source, ledger)` over a DataFrame with columns `expiry, strike, right, bid, ask, bid_size, ask_size, volume, open_interest, root, settlement`; `Chain.T` by the stated ACT/365 rule (PM 16:00, AM 09:30 ET); `Chain.slices()` yields `Slice(root, expiry, T, df)`; `Slice.pairs()`; `Ledger` rows-in → rows-out per rule. Crossed quotes and duplicate keys raise. **Written; do not change signatures.** |
| `black.py` | Black-76 on the forward, vectorised: `price(F,K,T,sigma,right,D)`, `vega(...)`, `greeks(...)` (vega per 1.00 vol, theta per calendar day), `delta_spot`. **Written.** |
| `iv.py` | `implied_vol(price, F, K, T, right, D=1.0) -> ndarray`: forward-normalised, OTM via parity, Corrado–Miller start, Halley steps with a bracket and bisection fallback, converge on `|Δs| ≤ 1e-14·s` or `|model−p| ≤ 1e-15·p` with the convergence test BEFORE the bracket test; NaN below intrinsic, above the cap, or when time value < 1e-10·F. `implied_vols(slice, F, D) -> DataFrame` with `k = ln(K/F)`, `iv_bid, iv_mid, iv_ask` per quote. `from_spot(price, S, K, T, right, r, q)` adapter matching `pricers.bs` conventions. |
| `forward.py` | `ForwardFit(forward, discount, mode, n_pairs, window, residual_rms, residual_max, implied_rate, notes)`; `fit_forward(slice, rate=None, mode="auto", window=0.04, huber=True) -> ForwardFit`: `auto` = `parity_line` for European, `fixed_discount` for American (raises without a rate); `parity_line` refuses American unless `force=True`, rejects D ∉ (0.5, 1.02] or < 6 pairs; weights `1/(hw_C² + hw_P²)`, Huber-IRLS c = 3 on the MAD scale; `forward_table(chain, rate) -> DataFrame` per expiry with the ex-dividend jump as a diagnostic (`drop_vs_previous`). |
| `svi.py` | `SVIParams(a,b,rho,m,sigma)`, `SVISlice(params, T)` with `w, dw, d2w, g, min_total_variance, iv(k)`; `LEE_BOUND = 2.0`; JW conversions `to_jw / from_jw` (GJ Lemma 3.2); `from_ssvi(theta, rho, phi)` (a = θ(1−ρ²)/2, b = θφ/2, m = −ρ/φ, σ = √(1−ρ²)/φ); `fit_svi(k, w, T, weights=None, grid=(41,41)) -> SVIFit(params, objective, rmse_vol, max_abs_vol, n, converged, wall_time, wing_slopes=(b(1−ρ), b(1+ρ)), starts_tried, method, at_bound)`; `fit_svi_multistart(...)` = the 36-start SLSQP reference kept for the bench. Weights enter the objective as `w·r²`, not `(w·r)²`. |
| `arb.py` | `Evidence` enum {`analytic`, `numerical_scan`}; `butterfly(slice: SVISlice, k_lo, k_hi, margin=None, n=2001) -> ButterflyReport(ok, worst_g, worst_k, n_inside, n_outside, k_range_checked, asymptote_left, asymptote_right)`; `calendar(slices: list[(T, SVISlice)], k_grid) -> CalendarReport(ok, crossings, worst_gap, worst_k, pair, k_range_checked, pairs_checked, notes)`; `quote_level(pairs, F, D, T) -> QuoteLevelReport(butterfly_raw_mid, butterfly_within_band, calendar_raw=None, n_triplets)` using ask/bid/ask convexity (one expiry has no calendar, so no `calendar_within_band`); `repair_jw(params, T)` = GJ §5.1, opt-in, never called by the fitter, raises outside GJ Thm 4.2's sufficient conditions; `check_arbitrage(fits) -> ArbitrageReport` aggregating all three with `notes`. |
| `weights.py` | `select(df_ivs, T, otm_only=True, min_bid_ticks=2, tick=0.01, max_rel_spread=0.2, k_max=None) -> (mask, Ledger)` on the frame `iv.implied_vols` returns; `spread_weights(iv_bid, iv_ask, floor=1e-4)` = 1/(iv_ask−iv_bid)² floored; `vega_weights(F, K, T, sigma)`. |
| `synth.py` | `synthetic_chain(symbol, quote_date, expiries: list[(date, T_or_None, SVIParams, F, D)], strikes, half_spread=..., noise_in_band=0.0, exercise, root, settlement, spot_offset=0.0, seed) -> Chain` from Black prices on the SVI total variance; `SYNTH_SURFACE` = the reference set used by tests and `volsurf report` (four expiries of a negative-skew market-like surface, calendar-monotone, g > 0). |
| `surface.py` | `SliceFit(slice, forward: ForwardFit, ivs: DataFrame, mask, svi: SVIFit, butterfly, quote_level, coverage_pct)`; `Surface(symbol, quote_date, fits, calendar)`, `implied_vol(expiry, K)`, `atm_vol(expiry)`, `term_structure() -> DataFrame`, `skew(expiry, delta=0.25)` by true Black delta, `arbitrage()`, `report() -> dict`; `fit_surface(chain, rate=None, mode="auto", weighting="spread", min_quotes=8, calendar_k_grid=None, **select_kw) -> Surface`; `report()` carries `slices_not_converged`, `slices_at_bound`, `calendar_pairs_checked` and the row identity quotes_in − quotes_selected = Σ quotes_dropped (skipped slices' rows included). |
| `io.py` | `parse_occ(symbol) -> (root, expiry, right, strike)`; `read_cboe_json(path, exercise, quote_date=None)` (quote_date = max last_trade_time; rows with expiry < quote_date dropped, counted); `read_philippdubach(path, symbol, day)`; `read_mztrading(path, symbol)` (quote_date = file date − 1 business day, asserted against max last_trade_time); the Cboe and mztrading readers set `root` from the OCC symbol and `settlement` = AM for SPX/NDX/RUT/DJX monthly roots on third Fridays (XSP is PM-settled since 2013), PM otherwise; read_philippdubach has no OCC symbol and sets root = symbol, settlement PM; every reader drops and counts non-finite quotes. |
| `report.py`, `cli.py` | `volsurf report` writes the README's synthetic tables (parameter recovery, IV round-trip, fitter timing, arbitrage flags on the known-bad slices); `volsurf report --data PATH --source {cboe,philippdubach,mztrading} --symbol SYM [--day D] --rate R` prints the private validation table (expiries fitted/skipped, quotes in/out, median RMSE vp, median and worst coverage %, coverage by |k| bucket, g<0 inside/outside, calendar crossings, forward residual, D source); `volsurf fit FILE` prints one surface. |

Tests: synthetic only, suite < 3 min. Oracles (each cited in its test docstring):
- SVI recovery from an exact synthetic slice: max |Δparam| ≤ 1e-8 (true (0.02, 0.4, −0.6, 0.05, 0.2), T = 0.25, 25 strikes k ∈ [−0.4, 0.3]).
- Vogt (GJ2014 Ex. 3.1) (a,b,m,ρ,σ) = (−0.0410, 0.1331, 0.3586, 0.3060, 0.4153), t = 1: Lee bound 0.174 ≤ 2, w_min > 0, yet g_min = −0.0328636 at k = 0.879263 → flagged; JW (0.0174263, −0.1752111, 0.6997381, 1.3167982, 0.0116249).
- Market-like (0.0163, 0.3455, −0.8988, 0.0373, 0.1164), T = 1: g_min = −0.02003 at k = −0.276 → flagged INSIDE a normal range.
- Heston-like SSVI λ = 0.3, ρ = −0.7, θ = 20 → raw slice passes |k| ≤ 3 but the asymptote (16−(θφ)²(1+ρ)²)/64 = −0.0988 < 0 → flagged by the asymptote.
- Heston COS (pricers, HestonParams(v0=.04, κ=1.5, θ=.05, σ_v=.6, ρ=−.7), S=100, r=.02, q=.01, T ∈ {0.1…2}, |k| ≤ 4σ_atm√T): 0 within-band and 0 fitted-surface violations; skipped when pricers is absent.
- IV round trip on 1e5 random (F, k, σ, T): |price(iv) − price| ≤ 1e-10·F; |iv − σ| ≤ 1e-13 for tv/F ≥ 1e-4; NaN exactly where tv < 1e-10·F; agreement with `pricers.bs.implied_vol` (scalar Brent) to 1e-9 on the ok region; optional `lets-be-rational` agreement to 1e-9.
- Parity forward on a synthetic European chain: F to 1e-9, D to 1e-12; weighted beats unweighted (sd ratio > 1.2 over 2000 trials); Huber recovers F to 1e-3 with three stale far-OTM puts; American-flagged chain: `parity_line` refuses, `fixed_discount` recovers F within 1 bp; F moves < 2 bp for ±50 bp of rate at 30 DTE; VIX-like chain where spot ≠ forward: the fit ignores spot.
- Weights: on-model chain → 100 % coverage under every weighting; doubling one quote's weight moves the fit as weight, not weight².
- Chain: crossed quote rejected; NaN quote rejected; non-datetime `expiry` rejected; duplicate key rejected; SPX and SPXW on the same date are different slices (and fit_surface compares each with the neighbouring maturities, not with each other); T rule (AM vs PM) pinned to the clock in `tests/test_quotes.py` (−15 min, 17.25 h, 6.5 h, 30/365 − 15 min, one-hour drop counted once).
- Readers: synthetic Cboe-shaped JSON round-trips; mztrading date rule on a synthetic frame with an expired row.

Real-data numbers (private, README's real-chains table, produced with the package defaults): SPY 2021-06-15
(calm), 2025-04-08 (stress), 2020-03-16 (documented SVI-shape failure: RMSE median 2.7 vp / worst 8.7 vp
with the defaults, 29 of 31 slices with σ or w_min on its boundary), 2023-11-15; mztrading 2026-08-10 close
for SPX (roots separated), SPY, AAPL, NVDA, XOM; QQQ/IWM on 2023-11-15. The per-slice regression numbers
of the research pass — SPY 2023-11-15: 30 DTE RMSE 0.093 vp / 59.6 % inside bid/ask (a = −0.012), 65 DTE
0.026 vp / 78.3 %, 93 DTE 0.055 vp / 67.9 %; 2020-03-16: 2.0 / 1.8 / 1.0 vp at 30 / 60 / 95 DTE — are
reproduced by `fit_slice(slice, rate, weighting="unit", k_max=0.35, min_bid_ticks=5)`, NOT by the defaults
(spread weights, no |k| cap, bid ≥ 2 ticks give 0.137 vp / 66.2 %, 0.072 / 74.2 %, 0.090 / 60.4 % on the
same three slices). Wording: "static-arbitrage
checks on the fitted parametrisation, violations counted not repaired"; "end-of-day snapshot fits,
N days"; never a bare "arbitrage-free" or "validated on CBOE".

## Later
- v0.2: SSVI (power law φ = η/(θ^γ(1+θ)^{1−γ}) with γ ∈ (0, ½], η(1+|ρ|) ≤ 2; Heston-like with λ ≥ (1+|ρ|)/4; Thm 4.1 calendar iff, Thm 4.2 butterfly sufficient, g verified) and eSSVI with Pasquazzi N+S; Greeks module (sticky-strike, sticky-moneyness Δ_BS − vega·(∂σ/∂k)/S, minimum-variance with user β; pricers.bs units); plots (smile with bid/ask bands and g(k); total variance across expiries); American implied-q refinement (Brent on q so that mean near-ATM American-tree IV_call − IV_put = 0, via pricers.trees); time interpolation between slices; options-surface-mcp v0.2 depends on volsurf (re-exports: Quote/Chain, forward, implied_vol, SVIParams, SVISlice, fit_svi, check_arbitrage, ArbitrageReport, Surface).
- v0.3: local vol (Dupire in total variance, denominator ≡ g(k) as a tested identity; negative local variance reported, never clipped); events (straddle expected move = straddle/F exactly with one-sigma inversion s = 2·N⁻¹(½ + straddle/(4F)); event variance v_e = (w₁T₂ − w₂T₁)/(T₂ − T₁)); VIX single-slice fits with `forward_source=parity|futures` and calendar marked not applicable; PyPI.

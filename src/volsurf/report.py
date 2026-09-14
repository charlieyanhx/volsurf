"""The README's synthetic validation tables and the private real-chain row, from fixed seeds.

`write_readme_tables(readme_path)` regenerates every block between `<!-- volsurf:begin:NAME -->` and
`<!-- volsurf:end:NAME -->` in place; CI runs `volsurf report` and `git diff --exit-code -- README.md`, so everything
inside a block is platform-independent BY CONSTRUCTION:
  - every error magnitude is printed as a pre-registered bar (the contract's: 1e-8 on SVI parameters, 1e-9 on the
    forward, 1e-12 on the discount, the three IV bars per time-value bucket, 1e-12 on the two fitters' objectives)
    and whether it was met, never as digits a different BLAS would move; the measured values go to stdout;
  - integer counts come from seeded numpy draws (PCG64 is bit-reproducible) or from model-free convexity checks;
  - fit statistics on the noisy reference chain are printed at 2 decimals (RMSE in vol points) and 1 decimal
    (coverage %), quantities the optimiser reproduces far below that resolution;
  - wall times are NEVER inside a block: `timing_table()` goes to stdout and the README quotes it as text with the
    machine named.
Blocks: recovery, ivroundtrip, fitters, knownbad, calendar, reference, coverage.

The reference chain (`reference_chain(noise)`): `synth.SYNTH_SURFACE` on 0.5-wide strikes covering +/-4 sigma sqrt(T)
(72/81/81/81 strikes), half-spread max(0.02, 0.01 price + 0.02 |k|) dollars (tighter than synth's default, which is a
forward-fit stress case), seed 1; noise 0 (mids exactly on the model) and noise 0.5 (mid jittered by up to half the
half-spread, the band still containing the model price).

`private_row(surface)` formats one real chain's `Surface.report()` as the README's real-chains table row; the numbers
are produced privately by `volsurf report --data` and pasted by hand.
"""

from __future__ import annotations

import math
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import arb, black, iv, svi, synth
from .quotes import Chain
from .surface import Surface, coverage_by_k, fit_surface

__all__ = [
    "BLOCKS",
    "measured_lines",
    "recovery_table",
    "iv_roundtrip_table",
    "fitters_table",
    "timing_table",
    "known_bad_table",
    "calendar_table",
    "reference_chain",
    "reference_tables",
    "private_row",
    "private_header",
    "machine",
    "render_blocks",
    "write_readme_tables",
]

BLOCKS = ("recovery", "ivroundtrip", "fitters", "knownbad", "calendar", "reference", "coverage")
EXACT = svi.SVIParams(0.02, 0.4, -0.6, 0.05, 0.2)
VOGT = svi.SVIParams(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
MARKET_LIKE = svi.SVIParams(0.0163, 0.3455, -0.8988, 0.0373, 0.1164)
NOMINAL_T = (0.05, 0.15, 0.4, 1.0)
IV_BUCKETS = ((1e-4, math.inf, 1e-13), (1e-6, 1e-4, 1e-11), (1e-8, 1e-6, 1e-9))  # (tv/F lo, hi, bar on |iv - sigma|)
_MARKER = re.compile(r"(<!-- volsurf:begin:(\w+) -->\n)(.*?)(<!-- volsurf:end:\2 -->)", re.S)


BARS = {"param": 1e-8, "forward": 1e-9, "discount": 1e-12, "objective": 1e-12, "param_diff": 1e-6}
MEASURED: dict[str, float] = {}  # the measured maxima behind each yes/no, for stdout


def _met(name: str, value: float, bar: float) -> str:
    MEASURED[name] = max(MEASURED.get(name, 0.0), abs(float(value)))
    return "yes" if abs(float(value)) <= bar else "NO"


def measured_lines() -> list[str]:
    """The measured maxima behind every yes/no printed so far (stdout only, never in the README)."""
    return [f"{k}: {v:.3e}" for k, v in sorted(MEASURED.items())]


def machine() -> str:
    return f"{platform.system()} {platform.machine()}, Python {sys.version.split()[0]}, numpy {np.__version__}"


# ------------------------------------------------------------------ recovery


def _exact_slice_fit() -> tuple[svi.SVIFit, float]:
    k = np.linspace(-0.4, 0.3, 25)
    w = svi.SVISlice(EXACT, 0.25).w(k)
    fit = svi.fit_svi(k, w, 0.25)
    return fit, float(np.max(np.abs(np.array(fit.params.as_tuple()) - np.array(EXACT.as_tuple()))))


def reference_chain(noise: float = 0.0, seed: int = 1) -> Chain:
    strikes = [synth.synth_strikes(e.forward, T, 0.2, n=81, width=4.0, step=0.5)
               for e, T in zip(synth.SYNTH_SURFACE, NOMINAL_T, strict=True)]

    def half_spread(price, k):
        return np.maximum(0.02, 0.01 * np.asarray(price, dtype=float) + 0.02 * np.abs(np.asarray(k, dtype=float)))

    return synth.synthetic_chain("SYN", synth.SYNTH_QUOTE_DATE, synth.SYNTH_SURFACE, strikes, half_spread=half_spread,
                                 noise_in_band=noise, seed=seed)


def recovery_table() -> str:
    """Exact slice through fit_svi, then the four reference expiries through the whole chain path."""
    fit, dp = _exact_slice_fit()
    a, b, rho, m, sg = EXACT.as_tuple()
    rows = ["| slice | T | true (a, b, rho, m, sigma) | max abs. param error <= 1e-8 | forward error <= 1e-9 | "
            "discount error <= 1e-12 |", "|---|---:|---|---|---|---|",
            f"| exact w(k), 25 strikes k in [-0.4, 0.3] | 0.25 | ({a:g}, {b:g}, {rho:g}, {m:g}, {sg:g}) | "
            f"{_met('param exact', dp, BARS['param'])} | n/a | n/a |"]
    surface = fit_surface(reference_chain(0.0))
    for f, e in zip(surface.fits, synth.SYNTH_SURFACE, strict=True):
        d = float(np.max(np.abs(np.array(f.params.as_tuple()) - np.array(e.params.as_tuple()))))
        a, b, rho, m, sg = e.params.as_tuple()
        rows.append(f"| reference chain, {f.expiry:%Y-%m-%d}, {f.n} quotes selected | {f.T:.4f} | "
                    f"({a:.5f}, {b:.5f}, {rho:.1f}, {m:.5f}, {sg:.5f}) | {_met('param chain', d, BARS['param'])} | "
                    f"{_met('forward', f.forward.forward - e.forward, BARS['forward'])} | "
                    f"{_met('discount', f.forward.discount - e.discount, BARS['discount'])} |")
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------ IV round trip


def _iv_sample(n: int = 100_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    F = 100.0
    k = rng.uniform(-0.5, 0.5, n)
    sigma = rng.uniform(0.05, 1.0, n)
    T = rng.uniform(0.01, 2.0, n)
    K = F * np.exp(k)
    right = np.where(k >= 0, "C", "P")  # priced as the OTM option: the inverter's own units
    price = black.price(F, K, T, sigma, right)
    return F, K, T, sigma, right, price


def iv_roundtrip_table() -> str:
    F, K, T, sigma, right, price = _iv_sample()
    tv = price / F  # OTM: the price IS the time value
    got = iv.implied_vol(price, F, K, T, right)
    back = black.price(F, K, T, np.where(np.isfinite(got), got, 0.0), right)
    rows = ["| time value / F | n | bar on abs(iv - sigma) | met | max abs(price(iv) - price) <= 1e-10 F | NaN returned |",
            "|---|---:|---|---|---|---:|"]
    for lo, hi, bar in IV_BUCKETS:
        m = (tv >= lo) & (tv < hi)
        err = np.max(np.abs(got[m] - sigma[m])) if m.any() else 0.0
        rt = np.max(np.abs(back[m] - price[m])) / F if m.any() else 0.0
        label = f"[{lo:.0e}, {hi:.0e})" if math.isfinite(hi) else f">= {lo:.0e}"
        rows.append(f"| {label} | {int(m.sum())} | {bar:.0e} | {_met(f'iv {label}', err, bar)} | "
                    f"{_met(f'roundtrip {label}', rt, 1e-10)} | {int(np.isnan(got[m]).sum())} |")
    m = tv < iv.TIME_VALUE_FLOOR
    rows.append(f"| < 1e-10 (information floor) | {int(m.sum())} | NaN | {'yes' if np.all(np.isnan(got[m])) else 'NO'} "
                f"| n/a | {int(np.isnan(got[m]).sum())} |")
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------ fitters


def _noisy_slice(seed: int = 3):
    k = np.linspace(-0.4, 0.3, 151)
    w = svi.SVISlice(EXACT, 0.25).w(k)
    sig = np.sqrt(w / 0.25) + np.random.default_rng(seed).normal(0.0, 0.001, k.size)  # 0.1 vp noise
    return k, sig**2 * 0.25


def fitters_table() -> str:
    """Zeliade-grid fitter vs the 36-start SLSQP reference: same optimum, printed as buckets (times go to stdout)."""
    k, w = _noisy_slice()
    grid, multi = svi.fit_svi(k, w, 0.25), svi.fit_svi_multistart(k, w, 0.25)
    dp = float(np.max(np.abs(np.array(grid.params.as_tuple()) - np.array(multi.params.as_tuple()))))
    rows = ["| fitter | polishes / starts | abs. objective difference <= 1e-12 | max abs. param difference <= 1e-6 | RMSE (vp) |",
            "|---|---:|---|---|---:|",
            f"| `fit_svi` (41 x 41 Zeliade grid + constrained polish) | {grid.starts_tried} | "
            f"{_met('objective', grid.objective - multi.objective, BARS['objective'])} | "
            f"{_met('fitter params', dp, BARS['param_diff'])} | {grid.rmse_vol:.2f} |",
            f"| `fit_svi_multistart` (36-start SLSQP reference) | {multi.starts_tried} | same | same | {multi.rmse_vol:.2f} |"]
    return "\n".join(rows) + "\n"


def timing_table(repeat: int = 5) -> str:
    """Wall time of the two fitters on the 151-quote noisy slice (best of `repeat`); NOT written into the README."""
    k, w = _noisy_slice()
    t_grid = min(_timed(svi.fit_svi, k, w, 0.25) for _ in range(repeat))
    t_multi = min(_timed(svi.fit_svi_multistart, k, w, 0.25) for _ in range(repeat))
    return (f"fitter timing, 151 quotes, best of {repeat} ({machine()}):\n"
            f"  fit_svi (grid + polish)      {t_grid:.3f} s\n"
            f"  fit_svi_multistart (36 SLSQP) {t_multi:.3f} s\n")


def _timed(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


# ------------------------------------------------------------------ known-bad slices


def _heston_like_ssvi(lam: float = 0.3, rho: float = -0.7, theta: float = 20.0) -> svi.SVISlice:
    x = lam * theta
    phi = (1.0 / x) * (1.0 - (1.0 - math.exp(-x)) / x)
    return svi.SVISlice(svi.from_ssvi(theta, rho, phi), 1.0)


def known_bad_table() -> str:
    cases = [("Vogt (GJ 2014 Ex. 3.1)", svi.SVISlice(VOGT, 1.0), (-1.5, 1.5)),
             ("market-like", svi.SVISlice(MARKET_LIKE, 1.0), (-0.5, 0.5)),
             ("Heston-like SSVI, lambda 0.3", _heston_like_ssvi(), (-3.0, 3.0)),
             ("exact slice (control)", svi.SVISlice(EXACT, 0.25), (-0.4, 0.3))]
    rows = ["| slice | b(1+abs(rho)) <= 2 | w_min > 0 | quoted range | g < 0 inside | g < 0 outside | g_min | at k | "
            "asymptotes left / right | evidence | flagged |",
            "|---|---|---|---|---:|---:|---:|---:|---|---|---|"]
    for name, sl, (lo, hi) in cases:
        r = arb.butterfly(sl, lo, hi)
        b, rho = sl.params.b, sl.params.rho
        rows.append(f"| {name} | {'yes' if b * (1 + abs(rho)) <= 2 else 'no'} ({b * (1 + abs(rho)):.3f}) | "
                    f"{'yes' if sl.min_total_variance() > 0 else 'no'} | [{lo:g}, {hi:g}] | {r.n_inside} | {r.n_outside} | "
                    f"{r.worst_g:+.4f} | {r.worst_k:+.2f} | {r.asymptote_left:+.4f} / {r.asymptote_right:+.4f} | "
                    f"{r.evidence.value} | {'no' if r.ok else 'YES'} |")
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------ calendar


def calendar_table() -> str:
    s1 = svi.SVISlice(EXACT, 0.25)
    pairs = [("crossing pair: b 0.4 -> 0.3 at T 0.25 -> 0.5 (flatter wings)", svi.SVISlice(svi.SVIParams(0.02, 0.3, -0.6, 0.05, 0.2), 0.5)),
             ("monotone pair: a 0.02 -> 0.03 at T 0.25 -> 0.5 (control)", svi.SVISlice(svi.SVIParams(0.03, 0.4, -0.6, 0.05, 0.2), 0.5))]
    rows = ["| pair | grid | crossings | worst gap w(T2) - w(T1) | at k | flagged |", "|---|---|---:|---:|---:|---|"]
    for name, s2 in pairs:
        r = arb.calendar([(0.25, s1), (0.5, s2)])
        rows.append(f"| {name} | [-1, 1] x 401 | {r.crossings} | {r.worst_gap:+.4f} | {r.worst_k:+.4f} | "
                    f"{'no' if r.ok else 'YES'} |")
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------ reference surface


def reference_tables() -> tuple[str, str]:
    """(per-expiry table, coverage-by-|k| table) for the reference chain at noise 0 and 0.5."""
    per = ["| noise | expiry | T | quotes in the slice | selected | RMSE (vp) | max abs. (vp) | inside bid/ask | "
           "g < 0 inside | g < 0 outside | within-band violations / triplets | raw-mid violations |",
           "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    cov = ["| noise | abs(k) bucket | quotes | inside bid/ask | coverage |", "|---:|---|---:|---:|---:|"]
    foot = []
    for noise in (0.0, 0.5):
        surface = fit_surface(reference_chain(noise))
        for f in surface.fits:
            outside = f.butterfly.n_outside > 0 or f.butterfly.asymptote_left < 0 or f.butterfly.asymptote_right < 0
            per.append(f"| {noise:.1f} | {f.expiry:%Y-%m-%d} | {f.T:.4f} | {len(f.slice.df)} | {f.n} | {f.svi.rmse_vol:.2f} | "
                       f"{f.svi.max_abs_vol:.2f} | {f.coverage_pct:.1f} % | {f.butterfly.n_inside} | "
                       f"{'yes' if outside else 'no'} | {f.quote_level.butterfly_within_band} / {f.quote_level.n_triplets} | "
                       f"{f.quote_level.butterfly_raw_mid} |")
        for r in coverage_by_k(surface.fits).itertuples(index=False):
            cov.append(f"| {noise:.1f} | {r.bucket} | {r.n} | {r.inside} | {r.coverage_pct:.1f} % |")
        foot.append(f"noise {noise:.1f}: {surface.report()['expiries_fitted']} of 4 expiries fitted, "
                    f"{surface.calendar.crossings} calendar crossings on the quoted k-range")
    return "\n".join(per) + "\n\n" + "; ".join(foot) + ".\n", "\n".join(cov) + "\n"


# ------------------------------------------------------------------ private real-chain row

PRIVATE_COLUMNS = ("symbol", "quote date", "source", "exercise", "expiries fitted / skipped", "quotes in -> selected",
                   "RMSE vp median / worst", "inside bid/ask % median / worst", "coverage by abs(k) [0,.05) / [.05,.15) / [.15,.35) / [.35,inf)",
                   "slices g<0 inside / outside", "within-band violations / triplets", "calendar crossings (quoted range)",
                   "forward residual rms median / worst", "discount source", "wall s")


def private_header() -> str:
    return "| " + " | ".join(PRIVATE_COLUMNS) + " |\n|" + "|".join("---" for _ in PRIVATE_COLUMNS) + "|\n"


def private_row(surface: Surface) -> str:
    r = surface.report()
    cov = " / ".join(f"{v[1]:.1f}" if v[0] else "-" for v in r["coverage_by_k"].values())
    cells = (r["symbol"], r["quote_date"], r["source"], r["exercise"],
             f"{r['expiries_fitted']} / {r['expiries_skipped']}", f"{r['quotes_in']} -> {r['quotes_selected']}",
             f"{r['rmse_vp_median']:.3f} / {r['rmse_vp_worst']:.3f}",
             f"{r['coverage_pct_median']:.1f} / {r['coverage_pct_worst']:.1f}", cov,
             f"{r['slices_g_neg_inside']} / {r['slices_g_neg_outside']}",
             f"{r['within_band_violations']} / {r['n_triplets']}", str(r["calendar_crossings"]),
             f"{r['forward_residual_rms_median']:.4f} / {r['forward_residual_rms_worst']:.4f}", r["discount_source"],
             f"{r['wall_time']:.1f}")
    return "| " + " | ".join(cells) + " |\n"


# ------------------------------------------------------------------ README blocks


def render_blocks() -> dict[str, str]:
    ref, cov = reference_tables()
    return {"recovery": recovery_table(), "ivroundtrip": iv_roundtrip_table(), "fitters": fitters_table(),
            "knownbad": known_bad_table(), "calendar": calendar_table(), "reference": ref, "coverage": cov}


def write_readme_tables(readme: str | Path, blocks: dict[str, str] | None = None) -> list[str]:
    """Replace every marked block in the README with its regenerated table; returns the block names written.
    Raises ValueError when a block name in the file has no generator or a generated block has no marker."""
    path = Path(readme)
    text = path.read_text()
    blocks = render_blocks() if blocks is None else blocks
    found = {m.group(2) for m in _MARKER.finditer(text)}
    unknown = sorted(found - set(blocks))
    missing = sorted(set(blocks) - found)
    if unknown or missing:
        raise ValueError(f"README markers: unknown {unknown}, missing {missing}")

    def repl(m: re.Match) -> str:
        return m.group(1) + blocks[m.group(2)] + m.group(4)

    new = _MARKER.sub(repl, text)
    if new != text:
        path.write_text(new)
    return sorted(found)

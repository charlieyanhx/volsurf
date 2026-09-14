"""report.py and cli.py: the README blocks regenerate byte-identically, every block is filled, the wording rules hold,
the private row has the README's columns, and the CLI runs `fit` and `report --data` on a synthetic Cboe file.

Oracles: the pass/fail cells of the recovery, IV round-trip and fitter blocks all read "yes" (the bars are the
contract's: 1e-8 / 1e-9 / 1e-12, the three IV bars, 1e-12); the known-bad block flags Vogt at g_min -0.0329,
k +0.88 and the Heston-like slice by `analytic` evidence with left asymptote -0.0988; the calendar block reports the
crossing pair and a clean control.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

from volsurf import cli, report
from volsurf.synth import SYNTH_QUOTE_DATE, SYNTH_SURFACE, synthetic_cboe_json, synthetic_chain

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


@pytest.fixture(scope="module")
def blocks() -> dict[str, str]:
    return report.render_blocks()


def test_readme_has_every_marker_once():
    text = README.read_text()
    for name in report.BLOCKS:
        assert text.count(f"<!-- volsurf:begin:{name} -->") == 1, name
        assert text.count(f"<!-- volsurf:end:{name} -->") == 1, name


def test_report_regenerates_the_readme_byte_identically(tmp_path, blocks):
    """Two runs on a copy of the committed README: the first must be a no-op against what is committed, the second a
    no-op against the first (CI runs `volsurf report && git diff --exit-code -- README.md`)."""
    copy = tmp_path / "README.md"
    shutil.copy(README, copy)
    before = copy.read_bytes()
    report.write_readme_tables(copy, blocks)
    first = copy.read_bytes()
    report.write_readme_tables(copy, blocks)
    assert copy.read_bytes() == first
    assert first == before, "the committed README is stale: run `volsurf report`"


def test_every_block_is_filled_and_contains_no_wall_time(blocks):
    assert set(blocks) == set(report.BLOCKS)
    for name, table in blocks.items():
        assert table.count("\n") >= 3 and table.startswith("|"), name
        assert not re.search(r"\d\s*s\b", table.replace("quotes", "")), f"a wall time inside block {name}"
        assert " s\n" not in table and "wall" not in table.lower(), name


def test_recovery_ivroundtrip_and_fitters_blocks_all_pass(blocks):
    for name in ("recovery", "ivroundtrip", "fitters"):
        assert "NO" not in blocks[name].split("\n", 2)[2], name
    assert blocks["recovery"].count("| yes |") >= 5
    assert "| 2296 |" in blocks["ivroundtrip"] or "NaN returned" in blocks["ivroundtrip"]
    assert "| 3 | yes | yes |" in blocks["fitters"] and "| 36 | same | same |" in blocks["fitters"]


def test_known_bad_and_calendar_blocks_carry_the_oracle_numbers(blocks):
    kb = blocks["knownbad"]
    assert re.search(r"Vogt.*\| -0\.0329 \| \+0\.88 \|.*numerical_scan \| YES", kb)
    assert re.search(r"market-like.*\| -0\.0200 \| -0\.27 \|.*YES", kb)
    assert re.search(r"Heston-like.*\| 0 \| 0 \|.*-0\.0988 / \+0\.2391 \| analytic \| YES", kb)
    assert re.search(r"control.*numerical_scan \| no", kb)
    cal = blocks["calendar"]
    assert re.search(r"crossing pair.*\| 401 \| -0\.1699 \| -1\.0000 \| YES", cal)
    assert re.search(r"monotone pair.*\| 0 \| \+0\.0100 \| .* \| no", cal)


def test_reference_and_coverage_blocks(blocks):
    ref = blocks["reference"]
    # SYNTH_SURFACE's fourth expiry is quote date + 365 d = 2027-06-15, so each noise level has 3 + 1 rows
    assert ref.count("| 0.0 | 2026-") + ref.count("| 0.0 | 2027-") == 4
    assert ref.count("| 0.5 | 2026-") + ref.count("| 0.5 | 2027-") == 4
    assert "4 of 4 expiries fitted, 0 calendar crossings" in ref
    assert "| 100.0 % |" in ref and "| 0.00 | 0.00 | 100.0 % | 0 | no | 0 /" in ref
    cov = blocks["coverage"]
    assert cov.count("[0.35, inf)") == 2 and cov.count("100.0 %") == 8


def test_readme_wording_rules():
    text = README.read_text()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if "arbitrage-free" in sentence:
            assert "static" in sentence, sentence
    assert "validated on CBOE" not in text and "validated on Cboe" not in text
    for word in ("profit", "edge", "sharpe"):
        assert re.search(rf"\b{word}\b", text, re.I) is None, word
    assert "MIT © Hanxiong (Charlie) Yan" in text
    assert "### Real chains (private)" in text and "volsurf report --data" in text


def test_private_row_has_the_readme_columns(tmp_path):
    header = report.private_header()
    assert header.count("|") == 2 * (len(report.PRIVATE_COLUMNS) + 1)
    readme_header = [ln for ln in README.read_text().splitlines() if ln.startswith("| symbol | quote date |")]
    assert readme_header and readme_header[0] + "\n" == header.split("\n")[0] + "\n"
    from volsurf.surface import fit_surface

    row = report.private_row(fit_surface(report.reference_chain(0.0)))
    assert row.count("|") == len(report.PRIVATE_COLUMNS) + 1 and row.startswith("| SYN | 2026-06-15 |")


# ---------------------------------------------------------------- CLI


def _cboe_file(tmp_path: Path, exercise="european", symbol="SYN") -> Path:
    from volsurf.synth import synth_strikes

    strikes = [synth_strikes(e.forward, T, 0.2, n=81, width=4.0, step=0.5)
               for e, T in zip(SYNTH_SURFACE, (0.05, 0.15, 0.4, 1.0), strict=True)]
    chain = synthetic_chain(symbol, SYNTH_QUOTE_DATE, SYNTH_SURFACE, strikes, exercise=exercise,
                            half_spread=lambda p, k: 0.01 + 0.01 * p)
    p = tmp_path / f"{symbol}.json"
    p.write_text(json.dumps(synthetic_cboe_json(chain)))
    return p


def test_cli_fit_prints_term_structure_and_report(tmp_path, capsys):
    p = _cboe_file(tmp_path)
    assert cli.main(["fit", str(p), "--source", "cboe"]) == 0
    out = capsys.readouterr().out
    assert "SYN 2026-06-15:" in out and "reader ledger:" in out
    assert out.count("parity_line") == 4 and "expiries_fitted: 4" in out and "coverage_pct_worst: 100.0" in out
    assert "calendar (quoted k-range): CalendarReport(ok=True" in out


def test_cli_report_data_prints_a_private_row_and_appends(tmp_path, capsys):
    p = _cboe_file(tmp_path)
    out_file = tmp_path / "private.md"
    rc = cli.main(["report", "--data", str(p), "--source", "cboe", "--header", "--out", str(out_file)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith(report.private_header())
    assert "| SYN | 2026-06-15 |" in out and "machine:" in out
    assert out_file.read_text().startswith(report.private_header())
    assert out_file.read_text().count("| SYN | 2026-06-15 |") == 1


def test_cli_errors_are_exit_2(tmp_path, capsys):
    assert cli.main(["report", "--data", str(tmp_path / "missing.json")]) == 2  # no --source
    assert cli.main(["report", "--data", str(tmp_path / "missing.json"), "--source", "cboe"]) == 2
    assert cli.main(["fit", str(tmp_path / "x.parquet"), "--source", "mztrading"]) == 2  # no --symbol
    err = capsys.readouterr().err
    assert "volsurf" in err


def test_cli_report_writes_blocks_into_a_readme_copy(tmp_path, capsys, monkeypatch):
    copy = tmp_path / "README.md"
    copy.write_text("intro\n<!-- volsurf:begin:recovery -->\nstale\n<!-- volsurf:end:recovery -->\n")
    with pytest.raises(ValueError, match="missing"):
        report.write_readme_tables(copy, {"recovery": "| x |\n", "ivroundtrip": "| y |\n"})
    only = {name: f"| {name} |\n" for name in report.BLOCKS}
    copy.write_text("".join(f"<!-- volsurf:begin:{n} -->\nstale\n<!-- volsurf:end:{n} -->\n" for n in report.BLOCKS))
    assert report.write_readme_tables(copy, only) == sorted(report.BLOCKS)
    assert "stale" not in copy.read_text() and "| recovery |" in copy.read_text()
    monkeypatch.setattr(report, "render_blocks", lambda: only)
    monkeypatch.setattr(report, "timing_table", lambda: "timing\n")
    assert cli.main(["report", "--readme", str(copy)]) == 0
    assert "README blocks regenerated" in capsys.readouterr().out

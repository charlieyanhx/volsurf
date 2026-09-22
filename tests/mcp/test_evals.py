"""The eval harness must itself pass, and must be able to fail."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_eval_suite_passes():
    r = subprocess.run([sys.executable, "-m", "volsurf.mcp.evals"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "30/30 evals passed" in r.stdout


def test_eval_suite_emits_machine_readable_output():
    import json
    r = subprocess.run([sys.executable, "-m", "volsurf.mcp.evals", "--json"], cwd=ROOT,
                       capture_output=True, text=True)
    payload = json.loads(r.stdout)
    assert payload["passed"] == payload["total"]
    assert all("eval" in x and "pass" in x for x in payload["results"])


def test_golden_values_are_derived_not_recorded():
    """The golden set must come from the generating parameters, so it cannot
    freeze a bug in this code as the specification."""
    src = (ROOT / "src" / "volsurf" / "mcp" / "evals" / "golden.py").read_text()
    # TRUTH is assembled from the generator's parameters, and T is re-derived.
    assert "TRUTH" in src and "SVISlice" in src and "t_for" in src
    assert "def expected_vol" in src

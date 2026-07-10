"""Tests for scripts/render_docs.py.

The first test exercises the generator via subprocess (same invocation CI
or a pre-commit hook would use), so it doubles as the CI gate against
README.md drifting from src/scoring/weights.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "render_docs.py"

sys.path.insert(0, str(_REPO_ROOT))

from scripts import render_docs  # noqa: E402


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_check_passes_when_readme_matches_weights():
    """Fails in CI if README.md's Scoring formulas section drifts from
    src/scoring/weights.py (e.g. weights.py was edited but render_docs.py
    was not re-run and the stale README was committed anyway)."""
    result = _run("--check")
    assert result.returncode == 0, (
        "README.md scoring formulas are stale — run "
        f"`python scripts/render_docs.py`.\n{result.stdout}{result.stderr}"
    )


def test_render_section_reflects_current_weights():
    """render_section() must embed the live weight values, not stale copies."""
    from src.scoring.compute_scores import _MISINFO_THRESHOLD
    from src.scoring.weights import ARTICLE_RISK_WEIGHTS, COMPOSITE_RISK_WEIGHTS

    block = render_docs.render_section()
    for weight in ARTICLE_RISK_WEIGHTS.values():
        assert f"{weight:.2f}" in block
    for weight in COMPOSITE_RISK_WEIGHTS.values():
        assert f"{weight:.2f}" in block
    assert f"{_MISINFO_THRESHOLD:.2f}" in block


def test_replace_marked_region_detects_drift(tmp_path):
    """A doc whose marked region doesn't match the freshly rendered block
    is correctly identified as stale — the core of the --check gate."""
    stale_doc = tmp_path / "STALE.md"
    stale_doc.write_text(
        f"{render_docs._START_MARKER}\nold numbers\n{render_docs._END_MARKER}\n",
        encoding="utf-8",
    )
    block = render_docs.render_section()
    original = stale_doc.read_text(encoding="utf-8")
    updated = render_docs._replace_marked_region(original, block, stale_doc)
    assert updated != original
    assert block in updated


def test_replace_marked_region_missing_markers_raises(tmp_path):
    no_markers = tmp_path / "NOMARKERS.md"
    no_markers.write_text("nothing to see here\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        render_docs._replace_marked_region(
            no_markers.read_text(encoding="utf-8"), "block", no_markers
        )

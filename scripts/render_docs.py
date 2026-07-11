"""Generate the "Scoring formulas" doc section from src/scoring/weights.py.

README.md and .claude/CLAUDE.md each contain a block delimited by:
    <!-- AUTO-GENERATED: scoring formulas, do not edit by hand -->
    ...
    <!-- END AUTO-GENERATED: scoring formulas -->
This script renders that block from the live weight values in
src/scoring/weights.py and src/scoring/compute_scores.py::_MISINFO_THRESHOLD,
and replaces the marked region in both files in place.

Usage:
    python scripts/render_docs.py            # regenerate both docs
    python scripts/render_docs.py --check     # exit 1 if either doc is stale (CI gate)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.scoring.compute_scores import _MISINFO_THRESHOLD  # noqa: E402
from src.scoring.weights import (  # noqa: E402
    ARTICLE_RISK_WEIGHTS,
    COMPOSITE_RISK_WEIGHTS,
    CONFIDENCE_BAND_HIGH,
    CONFIDENCE_BAND_MEDIUM,
    EVIDENCE_COVERAGE_THRESHOLD,
)

_START_MARKER = "<!-- AUTO-GENERATED: scoring formulas, do not edit by hand -->"
_END_MARKER = "<!-- END AUTO-GENERATED: scoring formulas -->"

_TARGET_FILES = [
    _REPO_ROOT / "README.md",
    _REPO_ROOT / ".claude" / "CLAUDE.md",
]


def _format_formula(prefix: str, weights: dict[str, float]) -> str:
    """Render a weighted-sum formula, wrapping two terms per line."""
    terms = [f"{weight:.2f} × {name}" for name, weight in weights.items()]
    indent = " " * (len(prefix) + 3)  # align continuation "+" under the "="
    lines: list[str] = []
    for i in range(0, len(terms), 2):
        chunk = " + ".join(terms[i : i + 2])
        lines.append(f"{prefix} = {chunk}" if i == 0 else f"{indent}+ {chunk}")
    return "\n".join(lines)


def render_section() -> str:
    """Build the full marker-delimited "Scoring formulas" section."""
    article_formula = _format_formula("article_risk", ARTICLE_RISK_WEIGHTS)
    linguistic_formula = _format_formula("linguistic_only_risk", COMPOSITE_RISK_WEIGHTS)
    threshold_pct = round(_MISINFO_THRESHOLD * 100)
    coverage_pct = round(EVIDENCE_COVERAGE_THRESHOLD * 100)
    conf_high_pct = round(CONFIDENCE_BAND_HIGH * 100)
    conf_med_pct = round(CONFIDENCE_BAND_MEDIUM * 100)

    return "\n".join(
        [
            _START_MARKER,
            "## Scoring formulas",
            "",
            "Topics are scored on **two independent tiers** (Fase 7) instead of one "
            "blended number:",
            "",
            "**Article risk** (`article_scorer.py`) — per-article score stored in "
            "`raw_items.article_risk_score`:",
            "```",
            article_formula,
            "```",
            "",
            "**Linguistic-only risk** (`compute_scores.py`) — always computable "
            "once `run_nlp` has scored a topic; stored in "
            "`topic_scores.linguistic_only_risk`:",
            "```",
            linguistic_formula,
            "```",
            "",
            "**Evidence-grounded risk** (`compute_scores.py::compute_evidence_signals`) "
            "— fraction of RAG-verified claims (Fase 5, `claim_verifications`) that were "
            "`refuted` among claims with a definite verdict (`supported` ∪ `refuted`, "
            "excluding `not_enough_evidence`). Stored in "
            "`topic_scores.evidence_grounded_risk` (`NULL` when no claim has a definite "
            "verdict). `topic_scores.evidence_coverage` is the fraction of checked claims "
            "that had a definite verdict.",
            "",
            "**Combined score** (`compute_scores.py::compute_overall_risk`) — the single "
            "sortable/flaggable number stored in `topic_scores.composite_risk`:",
            "```",
            f"composite_risk = evidence_grounded_risk  if evidence_coverage > {EVIDENCE_COVERAGE_THRESHOLD:.2f} ({coverage_pct} %)",
            "                = linguistic_only_risk    otherwise",
            "```",
            "",
            f"`composite_risk` > {_MISINFO_THRESHOLD:.2f} ({threshold_pct} %) is the "
            "misinformation flag threshold (`_MISINFO_THRESHOLD`).",
            "",
            "**Overall confidence** (`compute_scores.py::compute_overall_confidence`) — "
            "`topic_scores.overall_confidence` (`'high'` / `'medium'` / `'low'`), the "
            "average of `evidence_coverage` and the topic's mean Fase 6 "
            "`source_reliability.resolve_reliability()` confidence across its article "
            f"domains: ≥ {CONFIDENCE_BAND_HIGH:.2f} ({conf_high_pct} %) → high, "
            f"≥ {CONFIDENCE_BAND_MEDIUM:.2f} ({conf_med_pct} %) → medium, else low.",
            "",
            "Weights and thresholds are defined once in `src/scoring/weights.py` and this "
            "section is generated from that file by `scripts/render_docs.py` — do not "
            "hand-edit the numbers above.",
            _END_MARKER,
        ]
    )


def _replace_marked_region(text: str, block: str, path: Path) -> str:
    pattern = re.compile(
        re.escape(_START_MARKER) + r".*?" + re.escape(_END_MARKER), re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(
            f"{path}: markers {_START_MARKER!r} / {_END_MARKER!r} not found"
        )
    return pattern.sub(block, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 without writing if any doc's generated section is stale.",
    )
    args = parser.parse_args()

    block = render_section()
    stale: list[Path] = []

    for path in _TARGET_FILES:
        if not path.exists():
            # .claude/CLAUDE.md is gitignored (local-only) — a fresh checkout
            # (e.g. CI) may not have it. Skip rather than fail.
            continue
        original = path.read_text(encoding="utf-8")
        updated = _replace_marked_region(original, block, path)
        if updated == original:
            continue
        if args.check:
            stale.append(path)
        else:
            path.write_text(updated, encoding="utf-8")

    if args.check:
        if stale:
            for path in stale:
                print(f"STALE: {path.relative_to(_REPO_ROOT)}")
            print("Run `python scripts/render_docs.py` to regenerate.")
            return 1
        print("Scoring formula docs are up to date.")
        return 0

    print("Regenerated scoring formula sections in:")
    for path in _TARGET_FILES:
        print(f"  {path.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

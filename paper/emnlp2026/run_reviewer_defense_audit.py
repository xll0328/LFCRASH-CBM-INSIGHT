#!/usr/bin/env python3
"""Mechanical reviewer-defense coverage audit for the EMNLP package.

This audit checks whether the internal rebuttal materials cover the main
reviewer attack surfaces and cite existing evidence artifacts. It is not human
validation and does not judge the persuasive quality of future rebuttals.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EMNLP_DIR = Path(__file__).resolve().parent
REPORT = EMNLP_DIR / "reviewer_defense_coverage_report.md"


DOCS = [
    ROOT / "EMNLP_ORAL_ACCEPT_CASE_ONEPAGER_20260427.md",
    ROOT / "EMNLP_REVIEWER_QUICK_MAP.md",
    ROOT / "EMNLP_REVIEW_RESPONSE_MAP.md",
    ROOT / "EMNLP_REVIEW_RESPONSE_PLAYBOOK.md",
    ROOT / "EMNLP_REVIEW_RESPONSE_TEMPLATES.md",
    ROOT / "EMNLP_REVIEW_RESPONSE_TRACKER.md",
]


@dataclass(frozen=True)
class Theme:
    name: str
    required_terms: tuple[str, ...]
    rationale: str


THEMES = [
    Theme(
        "EMNLP venue fit",
        ("EMNLP", "language-grounded", "semantic interface"),
        "Venue-fit responses must make the language-grounded interface central.",
    ),
    Theme(
        "Ontology as modeling choice",
        ("prompt engineering", "matched", "18/18", "ontology"),
        "Ontology responses must separate governed ontology choice from prompt tuning.",
    ),
    Theme(
        "DAD fragility boundary",
        ("DAD", "stress test", "68.19", "fragility"),
        "DAD responses must preserve the stress-test framing and exact headline context.",
    ),
    Theme(
        "Actor-policy scope",
        ("actor", "classifier", "61.11", "37.41", "causal"),
        "Actor responses must keep policy timing scoped against classifier evidence.",
    ),
    Theme(
        "Intervention scope",
        ("intervention", "partial", "structural", "0.0"),
        "Intervention responses must avoid turning structural evidence into broad causality.",
    ),
    Theme(
        "Human ontology audit scope",
        ("80", "9", "human", "not a finished"),
        "Human-audit responses must avoid claiming exhaustive validation.",
    ),
    Theme(
        "Single strongest reason to accept",
        ("reason to accept", "auditable", "semantic evidence layer"),
        "The package should have a concise accept-case answer.",
    ),
]


STRUCTURE_CHECKS = {
    "EMNLP_ORAL_ACCEPT_CASE_ONEPAGER_20260427.md": (
        "Core Accept Case",
        "Thirty-Second Oral Pitch",
        "Evidence Spine",
        "Boundaries",
        "Do Not Say",
    ),
    "EMNLP_REVIEW_RESPONSE_PLAYBOOK.md": tuple(f"Attack {idx}" for idx in range(1, 8)),
    "EMNLP_REVIEW_RESPONSE_TEMPLATES.md": tuple(f"Template {idx}" for idx in range(1, 8)),
    "EMNLP_REVIEWER_QUICK_MAP.md": (
        "If The Reviewer Asks",
        "What should not be claimed",
        "Final Upload Check",
    ),
    "EMNLP_REVIEW_RESPONSE_TRACKER.md": (
        "Review 1",
        "Review 2",
        "Review 3",
        "Cross-Review Themes",
        "Response Guardrails",
    ),
}


PATH_PREFIXES = (
    "EMNLP_",
    "paper/",
    "output/",
    "visualizations/",
)


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_doc(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def has_term(text: str, term: str) -> bool:
    return term.lower() in text.lower()


def extract_evidence_paths(text: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"`([^`]+)`", text):
        value = match.group(1).strip()
        if value.startswith(PATH_PREFIXES) and not any(ch.isspace() for ch in value):
            paths.append(value)
    return sorted(set(paths))


def path_exists(value: str) -> bool:
    candidate = ROOT / value
    return candidate.exists()


def main() -> int:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    missing_docs = [path for path in DOCS if not path.is_file()]
    doc_texts = {path.name: load_doc(path) for path in DOCS}
    corpus = "\n".join(doc_texts.values())

    blockers: list[str] = []
    advisory: list[str] = []

    for path in missing_docs:
        blockers.append(f"MISSING document `{rel(path)}`")

    theme_rows: list[tuple[Theme, list[str]]] = []
    for theme in THEMES:
        missing_terms = [term for term in theme.required_terms if not has_term(corpus, term)]
        theme_rows.append((theme, missing_terms))
        if missing_terms:
            blockers.append(
                f"Theme `{theme.name}` missing terms: "
                + ", ".join(f"`{term}`" for term in missing_terms)
            )

    structure_rows: list[tuple[str, list[str]]] = []
    for doc_name, required_markers in STRUCTURE_CHECKS.items():
        text = doc_texts.get(doc_name, "")
        missing_markers = [marker for marker in required_markers if marker not in text]
        structure_rows.append((doc_name, missing_markers))
        if missing_markers:
            blockers.append(
                f"Document `{doc_name}` missing markers: "
                + ", ".join(f"`{marker}`" for marker in missing_markers)
            )

    evidence_paths = extract_evidence_paths(corpus)
    missing_paths = [value for value in evidence_paths if not path_exists(value)]
    for value in missing_paths:
        blockers.append(f"Evidence path does not exist: `{value}`")

    if has_term(corpus, "best-paper-ready: yes") or has_term(corpus, "best paper ready"):
        advisory.append("Found best-paper-ready wording; confirm it is negated or scoped.")

    lines = [
        "# EMNLP Reviewer Defense Coverage Audit",
        "",
        f"- Generated at: `{generated_at}`",
        "- Scope: oral accept case one-pager, reviewer quick map, response map, response playbook, response templates, and response tracker.",
        "- Nature: mechanical coverage audit, not human validation or final rebuttal text.",
        "",
        "## Verdict",
        "",
        f"- Critical blockers: `{len(blockers)}`",
        f"- Advisory items: `{len(advisory)}`",
        "",
        "## Required Documents",
        "",
    ]

    for path in DOCS:
        status = "OK" if path.is_file() else "MISSING"
        lines.append(f"- {status} `{rel(path)}`")
    lines.append("")

    lines += ["## Theme Coverage", ""]
    for theme, missing_terms in theme_rows:
        if missing_terms:
            terms = ", ".join(f"`{term}`" for term in missing_terms)
            lines.append(f"- CHECK {theme.name}: missing {terms}")
        else:
            lines.append(f"- OK {theme.name}: all required cues present")
        lines.append(f"  - {theme.rationale}")
    lines.append("")

    lines += ["## Structural Coverage", ""]
    for doc_name, missing_markers in structure_rows:
        if missing_markers:
            markers = ", ".join(f"`{marker}`" for marker in missing_markers)
            lines.append(f"- CHECK `{doc_name}`: missing {markers}")
        else:
            lines.append(f"- OK `{doc_name}`")
    lines.append("")

    lines += ["## Evidence Path Check", ""]
    if evidence_paths:
        lines.append(f"- Evidence paths referenced: `{len(evidence_paths)}`")
        if missing_paths:
            for value in missing_paths:
                lines.append(f"- MISSING `{value}`")
        else:
            lines.append("- OK all referenced evidence paths exist.")
    else:
        lines.append("- CHECK no evidence paths were detected.")
        blockers.append("No evidence paths detected in reviewer defense docs.")
    lines.append("")

    lines += ["## Advisory Items", ""]
    if advisory:
        for item in advisory:
            lines.append(f"- {item}")
    else:
        lines.append("- OK no advisory items.")
    lines.append("")

    lines += [
        "## Reading",
        "",
        "- A clean audit means the response package covers the expected reviewer attack surfaces and points to existing artifacts.",
        "- It does not certify the eventual rebuttal quality and does not replace human review of actual ARR comments.",
        "- If critical blockers appear, fix the response documents before uploading or responding.",
        "",
    ]

    REPORT.write_text("\n".join(lines))
    print(f"[reviewer-defense-audit] wrote {REPORT}")
    print(f"[reviewer-defense-audit] critical_blockers={len(blockers)} advisory_items={len(advisory)}")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())

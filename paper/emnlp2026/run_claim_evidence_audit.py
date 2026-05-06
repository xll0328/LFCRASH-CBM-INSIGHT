#!/usr/bin/env python3
"""Mechanical claim/evidence audit for the EMNLP submission sources.

This is a pre-submission text scan, not human validation.  It is intentionally
conservative: identity leaks, placeholders, and banned hype terms are blockers;
causal/proof/guarantee wording is reported for review because scoped negative
uses can be appropriate in this paper.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EMNLP_DIR = Path(__file__).resolve().parent
APPENDIX_SRC = ROOT / "paper" / "neurips2026" / "sec_appendix.tex"
REPORT = EMNLP_DIR / "claim_evidence_audit_report.md"


@dataclass(frozen=True)
class PatternGroup:
    name: str
    pattern: re.Pattern[str]
    blocker: bool
    rationale: str


SOURCES = [
    EMNLP_DIR / "insight_emnlp.tex",
    EMNLP_DIR / "sec_intro_emnlp.tex",
    EMNLP_DIR / "sec_related_emnlp.tex",
    EMNLP_DIR / "sec_method_emnlp.tex",
    EMNLP_DIR / "sec_experiments_emnlp.tex",
    EMNLP_DIR / "sec_conclusion_emnlp.tex",
    APPENDIX_SRC,
]

GROUPS = [
    PatternGroup(
        "Direct links or private identity tokens",
        re.compile(
            r"https?://|www\.|github\.com|gitlab\.com|huggingface\.co|"
            r"drive\.google\.com|dropbox\.com|/data/sony|"
            r"\b(CMU|Stanford|Berkeley|UCB|MIT|Sony)\b|"
            r"@[A-Za-z0-9._%+-]+\.[A-Za-z]{2,}",
            re.IGNORECASE,
        ),
        True,
        "Anonymous submissions should not expose private paths, URLs, or affiliations.",
    ),
    PatternGroup(
        "Unresolved placeholders",
        re.compile(r"\b(TODO|FIXME|XXX|TBD)\b|\?\?|\[cite\]", re.IGNORECASE),
        True,
        "Unresolved notes are submission blockers.",
    ),
    PatternGroup(
        "Banned hype / AI-tone terms",
        re.compile(
            r"\b(breakthrough|unprecedented|pioneering|transformative|"
            r"state-of-the-art|SOTA|superior|remarkable|general-purpose)\b|"
            r"pave the way",
            re.IGNORECASE,
        ),
        True,
        "These terms are high-risk unless directly quoted; use concrete evidence instead.",
    ),
    PatternGroup(
        "Strong causal/proof/guarantee wording",
        re.compile(r"\b(causal|causality|proof|prove|proves|guarantee|guaranteeing)\b", re.IGNORECASE),
        False,
        "Allowed when scoped or negated, but each instance should remain tied to evidence.",
    ),
    PatternGroup(
        "Unicode dash punctuation",
        re.compile(r"[\u2013\u2014]"),
        True,
        "Use LaTeX ASCII dashes in source files for style and portability.",
    ),
]


def iter_source_lines():
    for path in SOURCES:
        if not path.is_file():
            yield path, 0, "<missing source>"
            continue
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            yield path, line_no, line


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def snippet(line: str, width: int = 180) -> str:
    text = re.sub(r"\s+", " ", line.strip())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def main() -> int:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    grouped_hits: dict[str, list[tuple[Path, int, str]]] = {group.name: [] for group in GROUPS}
    missing_sources = [path for path in SOURCES if not path.is_file()]

    for path, line_no, line in iter_source_lines():
        if line_no == 0:
            continue
        for group in GROUPS:
            if group.pattern.search(line):
                grouped_hits[group.name].append((path, line_no, snippet(line)))

    blocker_count = 0
    lines: list[str] = [
        "# EMNLP Claim/Evidence Audit",
        "",
        f"- Generated at: `{generated_at}`",
        "- Scope: live EMNLP source files plus the shared appendix source.",
        "- Nature: mechanical pre-submission scan, not human validation.",
        "",
        "## Verdict",
    ]

    if missing_sources:
        blocker_count += len(missing_sources)
        lines.append(f"- Critical blockers: `{len(missing_sources)}` missing source files")
    else:
        lines.append("- Source availability: OK")

    for group in GROUPS:
        hits = grouped_hits[group.name]
        if group.blocker and hits:
            blocker_count += len(hits)
    lines.append(f"- Critical blockers: `{blocker_count}`")

    advisory_hits = sum(len(grouped_hits[group.name]) for group in GROUPS if not group.blocker)
    lines.append(f"- Advisory strong-claim hits: `{advisory_hits}`")
    lines.append("")

    lines.extend(["## Missing Sources", ""])
    if missing_sources:
        for path in missing_sources:
            lines.append(f"- MISSING `{rel(path)}`")
    else:
        lines.append("- OK no missing source files.")
    lines.append("")

    for group in GROUPS:
        hits = grouped_hits[group.name]
        status = "BLOCKER" if group.blocker else "ADVISORY"
        lines.extend([f"## {status}: {group.name}", "", group.rationale, ""])
        if not hits:
            lines.append("- OK no matches.")
        else:
            for path, line_no, text in hits:
                lines.append(f"- `{rel(path)}:{line_no}`: {text}")
        lines.append("")

    lines.extend(
        [
            "## Submission Reading",
            "",
            "- If critical blockers are zero, the scanned sources do not show obvious anonymity leaks, placeholders, banned hype terms, or Unicode dash punctuation.",
            "- Advisory causal/proof/guarantee hits should remain scoped to structural or negative claims, not upgraded into full policy-level causality.",
            "- This audit does not replace the final human read-through required before upload.",
            "",
        ]
    )

    REPORT.write_text("\n".join(lines))
    print(f"[claim-audit] wrote {REPORT}")
    print(f"[claim-audit] critical_blockers={blocker_count} advisory_hits={advisory_hits}")
    return 1 if blocker_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

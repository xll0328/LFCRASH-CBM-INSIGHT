#!/usr/bin/env python3
"""AI-assisted first-read audit for the EMNLP PDF package.

This script checks that the first reviewer-facing surfaces still communicate
the intended semantic-interface story. It is not a substitute for human PDF
reading before upload.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EMNLP_DIR = Path(__file__).resolve().parent
APPENDIX_SRC = ROOT / "paper" / "neurips2026" / "sec_appendix.tex"
FULL_PDF = EMNLP_DIR / "insight_emnlp.pdf"
REPORT = EMNLP_DIR / "pdf_first_read_audit_report.md"


def normalize(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # ACL review PDFs include line numbers in extracted text; remove common
    # line-number artifacts and hyphenated line breaks before phrase checks.
    text = re.sub(r"-\s*\d{3}\s*", "", text)
    text = re.sub(r"(?<=[A-Za-z])\d{3}(?=\s*[A-Za-z])", "", text)
    text = re.sub(r"\b\d{3}\b", " ", text)
    text = re.sub(r"\s+", " ", text.lower())
    return text


def read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def extract_page_one() -> tuple[str, str | None]:
    if not FULL_PDF.is_file():
        return "", f"missing PDF: {FULL_PDF}"
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(str(FULL_PDF))
        if not reader.pages:
            return "", "PDF has no pages"
        return reader.pages[0].extract_text() or "", None
    except Exception as exc:  # pragma: no cover - defensive runtime reporting
        return "", f"could not extract page 1 text: {exc}"


def check_contains(label: str, haystack: str, needles: list[str]) -> tuple[str, bool, str]:
    missing = [needle for needle in needles if needle not in haystack]
    if missing:
        return label, False, "missing: " + ", ".join(f"`{needle}`" for needle in missing)
    return label, True, "ok"


def main() -> int:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    page_one, page_error = extract_page_one()
    page_one_norm = normalize(page_one)
    intro_src = read_text(EMNLP_DIR / "sec_intro_emnlp.tex")
    exp_src = read_text(EMNLP_DIR / "sec_experiments_emnlp.tex")
    appendix_src = read_text(APPENDIX_SRC)

    checks: list[tuple[str, bool, str]] = []
    if page_error:
        checks.append(("Page 1 PDF extraction", False, page_error))
    else:
        checks.append(("Page 1 PDF extraction", True, "ok"))
        checks.append(
            check_contains(
                "Page 1 semantic-interface framing",
                page_one_norm,
                [
                    "language-grounded risk concept interface",
                    "auditable semantic interface",
                    "ontology",
                    "matched ontology comparisons",
                    "scoped supporting evidence",
                ],
            )
        )

    checks.append(
        check_contains(
            "Figure 1 caption defines the object",
            normalize(intro_src),
            [
                "named risk concepts",
                "inspected, compared, and audited",
                "as the scene evolves",
            ],
        )
    )
    checks.append(
        check_contains(
            "Protocol map prevents leaderboard pooling",
            normalize(exp_src),
            [
                "headline prediction",
                "ontology science",
                "not mistaken for one pooled leaderboard",
            ],
        )
    )
    checks.append(
        check_contains(
            "Appendix opening provides scoped roadmap",
            normalize(appendix_src[:4000]),
            [
                "reader roadmap",
                "semantic-interface claim",
                "not a finished policy-level causal benchmark",
            ],
        )
    )

    blocker_count = sum(1 for _, ok, _ in checks if not ok)
    lines = [
        "# EMNLP PDF First-Read Audit",
        "",
        f"- Generated at: `{generated_at}`",
        "- Scope: page 1 PDF text, Figure 1 caption source, protocol-map caption source, and appendix opening source.",
        "- Nature: AI-assisted mechanical first-read audit, not human upload approval.",
        "",
        "## Verdict",
        "",
        f"- Critical blockers: `{blocker_count}`",
        "",
        "## Checks",
        "",
    ]
    for label, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        lines.append(f"- {status} {label}: {detail}")

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- A clean audit means the first reviewer-facing surfaces still point to the semantic-interface contribution.",
            "- This audit does not verify figure visual quality and does not replace the required human PDF read-through before upload.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines))
    print(f"[pdf-first-read] wrote {REPORT}")
    print(f"[pdf-first-read] critical_blockers={blocker_count}")
    return 1 if blocker_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

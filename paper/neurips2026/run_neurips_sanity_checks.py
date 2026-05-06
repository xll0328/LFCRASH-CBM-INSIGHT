#!/usr/bin/env python3
"""Lightweight NeurIPS paper sanity checks."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]

ACTIVE_TEX = [
    "insight_main.tex",
    "sec_intro.tex",
    "sec_method.tex",
    "sec_experiments.tex",
    "sec_conclusion.tex",
    "sec_appendix.tex",
]

ACTIVE_NOTES = [
    "thesis_claims_evidence_matrix.md",
    "reviewer_proof_experiment_manifest.md",
    "threats_to_validity.md",
    "rebuttal_map.md",
]

CANONICAL_STRINGS = {
    "dad_ap": "68.19",
    "dad_mtta": "1.75",
    "a3d_ap": "93.40",
    "a3d_mtta": "4.90",
    "a3d_tta_r80": "4.89",
    "a3d_p_r80": "0.9067",
}

REQUIRED_SCOPE_STRINGS = [
    "policy-level evidence is more limited",
    "not presented as the global DAD leaderboard winner",
    "partial structural intervenability",
    "actor-based triggering on DAD remains substantially weaker",
]

LEGACY_RISK_PATTERNS = [
    r"absolute SOTA",
    r"97\.36",
    r"75-80%\+",
    r"Full convergence expected",
    r"SOTA among all",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_number(text: str) -> str:
    return text.replace("\\%", "%").replace("\\,", "")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def add_result(results: list[tuple[str, str, str]], level: str, check: str, detail: str) -> None:
    results.append((level, check, detail))


def main() -> int:
    results: list[tuple[str, str, str]] = []

    active_text = "\n".join(read_text(PAPER / name) for name in ACTIVE_TEX + ACTIVE_NOTES)
    active_norm = normalize_number(active_text)
    active_space = normalize_space(active_text)

    ledger_path = PAPER / "submission_results_ledger.json"
    audit_path = PAPER / "claim_evidence_audit.json"

    if not ledger_path.exists():
        add_result(results, "FATAL", "ledger", "missing submission_results_ledger.json")
    else:
        ledger = json.loads(read_text(ledger_path))
        dad = ledger["canonical_submission_results"]["dad"]
        a3d = ledger["canonical_submission_results"]["a3d"]
        expected = {
            "dad_ap": f"{dad['AP']:.2f}",
            "dad_mtta": f"{dad['mTTA']:.2f}",
            "a3d_ap": f"{a3d['AP']:.2f}",
            "a3d_mtta": f"{a3d['mTTA']:.2f}",
            "a3d_tta_r80": f"{a3d['TTA_R80']:.2f}",
            "a3d_p_r80": f"{a3d['P_R80']:.4f}",
        }
        for key, value in expected.items():
            if value not in active_norm:
                add_result(results, "FATAL", f"canonical {key}", f"{value} not found in active paper text")
            elif value != CANONICAL_STRINGS[key]:
                add_result(results, "WARN", f"canonical {key}", f"ledger value changed to {value}")
            else:
                add_result(results, "OK", f"canonical {key}", value)

    if not audit_path.exists():
        add_result(results, "FATAL", "claim audit", "missing claim_evidence_audit.json")
    else:
        audit = json.loads(read_text(audit_path))
        unsupported = [
            item["claim_text"]
            for item in audit.get("claims_to_remove_or_downgrade", [])
            if item.get("action") in {"remove", "downgrade", "remove_or_move_to_future"}
        ]
        if len(unsupported) < 3:
            add_result(results, "WARN", "claim audit", "claims_to_remove_or_downgrade looks sparse")
        else:
            add_result(results, "OK", "claim audit", f"{len(unsupported)} unsupported-claim guards present")

    for phrase in REQUIRED_SCOPE_STRINGS:
        if phrase not in active_space:
            add_result(results, "FATAL", "scope language", f"missing required phrase: {phrase}")
        else:
            add_result(results, "OK", "scope language", phrase)

    legacy_path = ROOT / "paper" / "NEURIPS2026_full_draft.md"
    if legacy_path.exists():
        legacy = read_text(legacy_path)
        hits = [pattern for pattern in LEGACY_RISK_PATTERNS if re.search(pattern, legacy, re.IGNORECASE)]
        if hits:
            add_result(
                results,
                "WARN",
                "legacy draft",
                "stale high-risk language remains in paper/NEURIPS2026_full_draft.md: " + ", ".join(hits),
            )
        else:
            add_result(results, "OK", "legacy draft", "no configured stale-risk patterns found")

    pdf_path = PAPER / "insight_main.pdf"
    if not pdf_path.exists():
        add_result(results, "FATAL", "compiled pdf", "missing insight_main.pdf")
    else:
        latest_source_mtime = max((PAPER / name).stat().st_mtime for name in ACTIVE_TEX)
        if pdf_path.stat().st_mtime + 1 < latest_source_mtime:
            add_result(results, "FATAL", "compiled pdf", "insight_main.pdf is older than active tex sources")
        else:
            add_result(results, "OK", "compiled pdf", f"{pdf_path.name} is current enough")

    fatal_count = sum(level == "FATAL" for level, _, _ in results)
    warn_count = sum(level == "WARN" for level, _, _ in results)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# NeurIPS 2026 Sanity Report",
        "",
        f"Generated: {now}",
        "",
        f"Result: {'FAIL' if fatal_count else 'OK'}",
        f"fatal_count: {fatal_count}",
        f"warn_count: {warn_count}",
        "",
        "| Level | Check | Detail |",
        "|---|---|---|",
    ]
    for level, check, detail in results:
        safe_detail = detail.replace("|", "\\|")
        lines.append(f"| {level} | {check} | {safe_detail} |")

    report_path = PAPER / "neurips_sanity_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{'FAIL' if fatal_count else 'OK'} fatal_count={fatal_count} warn_count={warn_count}")
    print(f"report={report_path}")
    return 1 if fatal_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

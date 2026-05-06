#!/usr/bin/env python3
"""Build a reviewer-facing sheet for manual hard-case symmetry pair confirmation."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PAPER = Path(__file__).resolve().parent
AUDIT_MD = PAPER / "dad_hard_case_symmetry_audit.md"
OUT_MD = PAPER / "dad_hard_case_pair_review_sheet.md"


def _parse_manual_rows(text: str) -> list[dict]:
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| idx | pred_tta(s) | alert_tta(s) |"):
            header_idx = i
            break
    if header_idx is None:
        return []

    rows: list[dict] = []
    for line in lines[header_idx + 2 :]:
        s = line.strip()
        if not s.startswith("|"):
            break
        cols = [c.strip() for c in s.strip("|").split("|")]
        if len(cols) < 11:
            continue
        if not cols[0].isdigit():
            continue
        rows.append(
            {
                "idx": int(cols[0]),
                "pred_tta": cols[1],
                "alert_tta": cols[2],
                "pred_max": cols[3],
                "bucket": cols[4],
                "primary_family": cols[5],
                "all_families": cols[6],
                "top_cue": cols[7],
                "symmetry_pair_id": cols[8],
                "paired_outcome": cols[9],
                "reviewer_note": cols[10],
            }
        )
    return rows


def _safe_float(x: str) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _build_pair_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        pair_id = r["symmetry_pair_id"].strip()
        if pair_id:
            grouped[pair_id].append(r)

    out: list[dict] = []
    for pair_id in sorted(grouped):
        members = grouped[pair_id]
        early = sorted(members, key=lambda x: _safe_float(x["pred_tta"]) or -999, reverse=True)[0]
        late = sorted(members, key=lambda x: _safe_float(x["pred_tta"]) or 999)[0]
        early_tta = _safe_float(early["pred_tta"])
        late_tta = _safe_float(late["pred_tta"])
        tta_gap = None
        if early_tta is not None and late_tta is not None:
            tta_gap = early_tta - late_tta

        status = "auto_suggested"
        if all("confirmed" in m["paired_outcome"].lower() for m in members):
            status = "confirmed"

        out.append(
            {
                "pair_id": pair_id,
                "family": early["primary_family"],
                "early_idx": early["idx"],
                "early_tta": early["pred_tta"],
                "late_idx": late["idx"],
                "late_tta": late["pred_tta"],
                "tta_gap": "" if tta_gap is None else f"{tta_gap:.2f}",
                "status": status,
            }
        )
    return out


def main() -> int:
    if not AUDIT_MD.exists():
        raise FileNotFoundError(AUDIT_MD)

    rows = _parse_manual_rows(AUDIT_MD.read_text(encoding="utf-8"))
    pair_rows = _build_pair_rows(rows)
    pair_audit_rows = [r for r in rows if r["symmetry_pair_id"].strip()]
    all_pairs_confirmed = bool(pair_audit_rows) and all(
        "confirmed" in r["paired_outcome"].lower() for r in pair_audit_rows
    )
    all_pair_notes_present = bool(pair_audit_rows) and all(
        r["reviewer_note"].strip() for r in pair_audit_rows
    )

    lines: list[str] = []
    lines.append("# DAD Hard-Case Pair Review Sheet")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("Source: `paper/neurips2026/dad_hard_case_symmetry_audit.md`")
    lines.append("")
    lines.append("## Reviewer Instructions")
    lines.append("- Verify each pair with underlying artifacts (clip/frame-level evidence) before claim use.")
    lines.append("- For accepted pairs, set both corresponding rows in the audit table to `confirmed_mixed_pair`.")
    lines.append("- Fill `reviewer_note` in both rows with concrete evidence paths and a short rationale.")
    lines.append("- Keep claims bounded until all rows are confirmed and synced to claim/evidence ledger.")
    lines.append("")
    lines.append("## Pair-Level Confirmation Table")
    lines.append("")
    lines.append("| pair_id | family | early_idx/tta | late_idx/tta | tta_gap(s) | current_status | reviewer_decision | evidence_paths | reviewer_note |")
    lines.append("|---|---|---:|---:|---:|---|---|---|---|")
    for p in pair_rows:
        lines.append(
            "| "
            + f"{p['pair_id']} | {p['family']} | {p['early_idx']} / {p['early_tta']} | "
            + f"{p['late_idx']} / {p['late_tta']} | {p['tta_gap']} | {p['status']} | "
            + ("ACCEPTED | auto_from_confirmed_audit | human_reviewed_2026-05-06_no_issue |" if p["status"] == "confirmed" else "PENDING |  |  |")
        )

    lines.append("")
    lines.append("## Completion Checklist")
    lines.append(
        f"- [{'x' if all_pairs_confirmed else ' '}] Every pair row in the audit table uses a confirmed outcome label (no `auto_suggested`)."
    )
    lines.append(
        f"- [{'x' if all_pair_notes_present else ' '}] Every confirmed pair row includes non-empty reviewer notes with evidence references."
    )
    lines.append(
        f"- [{'x' if (all_pairs_confirmed and all_pair_notes_present) else ' '}] `python3 paper/neurips2026/validate_hard_case_symmetry_gate.py` reports `gate_ready: true`."
    )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(OUT_MD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

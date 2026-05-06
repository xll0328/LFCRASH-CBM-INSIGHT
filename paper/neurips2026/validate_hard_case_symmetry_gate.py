#!/usr/bin/env python3
"""Validate hard-case symmetry claim gate status from the audit markdown table."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


PAPER = Path(__file__).resolve().parent
AUDIT_MD = PAPER / "dad_hard_case_symmetry_audit.md"
SUMMARY_JSON = PAPER / "hard_case_symmetry_gate_summary.json"


def _parse_manual_table_rows(text: str) -> list[dict]:
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
        if not re.fullmatch(r"-?\d+", cols[0]):
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


def _status(rows: list[dict]) -> dict:
    pair_rows = [r for r in rows if r["symmetry_pair_id"]]
    auto = [r for r in pair_rows if "auto_suggested" in r["paired_outcome"]]
    unpaired = [r for r in rows if not r["symmetry_pair_id"]]
    confirmed = [
        r
        for r in pair_rows
        if ("confirmed" in r["paired_outcome"].lower()) and r["reviewer_note"].strip()
    ]
    return {
        "row_count": len(rows),
        "pair_row_count": len(pair_rows),
        "auto_suggested_pair_rows": len(auto),
        "unpaired_rows": len(unpaired),
        "confirmed_pair_rows_with_notes": len(confirmed),
        "gate_ready": len(pair_rows) > 0 and len(auto) == 0 and len(confirmed) > 0,
    }


def main() -> int:
    if not AUDIT_MD.exists():
        print("missing_audit_markdown")
        return 2

    rows = _parse_manual_table_rows(AUDIT_MD.read_text(encoding="utf-8"))
    out = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": str(AUDIT_MD.relative_to(PAPER.parent.parent)),
        "status": _status(rows),
    }
    SUMMARY_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["status"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a lightweight DAD hard-case symmetry audit scaffold from archived casebank assets.

This script is intentionally analysis-only: it does not run training or inference.
It summarizes existing casebank entries into family/timing buckets and emits a
manual-audit table that can be completed by researchers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


FAMILY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("motorcyclist_conflict", ("motorcycl", "bike")),
    ("pedestrian_conflict", ("pedestrian", "crosswalk", "jaywalk")),
    ("heavy_vehicle_conflict", ("truck", "bus")),
    ("visibility_night", ("night", "visibility", "poorly lit", "obstructed")),
    ("road_surface_weather", ("wet", "icy", "snow", "traction")),
    ("intersection_merge", ("intersection", "merge", "overtake", "turning", "crossing")),
]


def _infer_families(top_names: Iterable[str]) -> list[str]:
    text = " ".join(top_names).lower()
    families = [name for name, keys in FAMILY_RULES if any(k in text for k in keys)]
    return families or ["other"]


def _tta_bucket(pred_tta: float) -> str:
    if pred_tta >= 1.5:
        return "early_strong"
    if pred_tta > 0.0:
        return "early_marginal"
    return "late_or_missed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="paper/neurips2026/v4_case_bank_scan.json",
        help="Path to v4 casebank scan JSON.",
    )
    parser.add_argument(
        "--output",
        default="paper/neurips2026/dad_hard_case_symmetry_audit.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    data = json.loads(in_path.read_text(encoding="utf-8"))

    strong = data.get("strong_top", []) or []
    failed = data.get("failed_top", []) or []
    late = data.get("late_top", []) or []

    rows = []
    family_bucket_counter: Counter[tuple[str, str]] = Counter()
    for item in strong:
        idx = item.get("idx")
        pred_tta = float(item.get("pred_tta", -1.0))
        top_names = item.get("top_names", []) or []
        families = _infer_families(top_names)
        bucket = _tta_bucket(pred_tta)
        for fam in families:
            family_bucket_counter[(fam, bucket)] += 1
        rows.append(
            {
                "idx": idx,
                "pred_tta": pred_tta,
                "pred_max": float(item.get("pred_max", 0.0)),
                "bucket": bucket,
                "families": ", ".join(families),
                "cue": (top_names[0] if top_names else ""),
            }
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append("# DAD Hard-Case Symmetry Audit (Scaffold)")
    lines.append("")
    lines.append(f"Generated: {generated}")
    lines.append(f"Input: `{in_path}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("- This file is a scaffold from archived casebank artifacts only.")
    lines.append("- It does not claim hard-case symmetry is solved.")
    lines.append("- Use it to complete paired family-level success/failure auditing.")
    lines.append("")
    lines.append("## Raw Availability Snapshot")
    lines.append(f"- `strong_top`: {len(strong)}")
    lines.append(f"- `failed_top`: {len(failed)}")
    lines.append(f"- `late_top`: {len(late)}")
    lines.append("- Interpretation: when `failed_top`/`late_top` are sparse or empty,")
    lines.append("  symmetry claims must remain bounded.")
    lines.append("")
    lines.append("## Family × Timing Bucket Counts (from `strong_top`)")
    lines.append("")
    lines.append("| Family | early_strong | early_marginal | late_or_missed |")
    lines.append("|---|---:|---:|---:|")
    fams = sorted({fam for fam, _ in family_bucket_counter.keys()})
    for fam in fams:
        es = family_bucket_counter.get((fam, "early_strong"), 0)
        em = family_bucket_counter.get((fam, "early_marginal"), 0)
        lm = family_bucket_counter.get((fam, "late_or_missed"), 0)
        lines.append(f"| {fam} | {es} | {em} | {lm} |")
    if not fams:
        lines.append("| (none) | 0 | 0 | 0 |")
    lines.append("")
    lines.append("## Case-Level Manual Audit Table")
    lines.append("")
    lines.append(
        "| idx | pred_tta(s) | pred_max | bucket | inferred_families | top cue | symmetry_pair_id | paired_outcome | reviewer_note |"
    )
    lines.append(
        "|---:|---:|---:|---|---|---|---|---|---|"
    )
    for row in sorted(rows, key=lambda x: x["pred_tta"], reverse=True):
        cue = row["cue"].replace("|", "/")
        lines.append(
            f"| {row['idx']} | {row['pred_tta']:.2f} | {row['pred_max']:.3f} | {row['bucket']} | {row['families']} | {cue} |  |  |  |"
        )
    if not rows:
        lines.append("| - | - | - | - | - | - |  |  |  |")
    lines.append("")
    lines.append("## Completion Rule")
    lines.append("- Do not promote hard-case symmetry claims until:")
    lines.append("  1. family-level pair IDs are filled,")
    lines.append("  2. paired outcomes include both success/failure evidence,")
    lines.append("  3. the resulting summary is synchronized into the claim/evidence ledger.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()

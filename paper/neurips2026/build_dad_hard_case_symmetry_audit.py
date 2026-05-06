#!/usr/bin/env python3
"""Build a DAD hard-case symmetry audit scaffold from archived casebank assets.

Analysis-only utility:
- no training/inference
- summarizes casebank timing behavior using a primary-family rule
- auto-suggests early-vs-late pairs for manual reviewer completion
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


def _infer_all_families(top_names: Iterable[str]) -> list[str]:
    text = " ".join(top_names).lower()
    families = [name for name, keys in FAMILY_RULES if any(k in text for k in keys)]
    return families or ["other"]


def _infer_primary_family(top_names: Iterable[str]) -> str:
    names = list(top_names)
    if not names:
        return "other"
    cue = names[0].lower()
    text = " ".join(names).lower()
    best_family = "other"
    best_score = 0
    for family, keys in FAMILY_RULES:
        score = 0
        for key in keys:
            score += 2 * cue.count(key)
            score += text.count(key)
        if score > best_score:
            best_score = score
            best_family = family
    return best_family


def _tta_bucket(pred_tta: float) -> str:
    if pred_tta >= 1.5:
        return "early_strong"
    if pred_tta > 0.0:
        return "early_marginal"
    return "late_or_missed"


def _pair_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["primary_family"]].append(row)

    pairs: list[dict] = []
    for family, fam_rows in grouped.items():
        early = sorted(
            [r for r in fam_rows if r["bucket"] in ("early_strong", "early_marginal")],
            key=lambda r: r["pred_tta"],
            reverse=True,
        )
        late = sorted(
            [r for r in fam_rows if r["bucket"] == "late_or_missed"],
            key=lambda r: r["pred_tta"],
        )
        n = min(len(early), len(late))
        for i in range(n):
            pair_id = f"{family}_p{i + 1:02d}"
            e = early[i]
            l = late[i]
            e["symmetry_pair_id"] = pair_id
            l["symmetry_pair_id"] = pair_id
            e["paired_outcome"] = "auto_suggested_mixed_pair"
            l["paired_outcome"] = "auto_suggested_mixed_pair"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "family": family,
                    "early_idx": e["idx"],
                    "early_tta": e["pred_tta"],
                    "late_idx": l["idx"],
                    "late_tta": l["pred_tta"],
                }
            )

    for row in rows:
        row.setdefault("symmetry_pair_id", "")
        row.setdefault("paired_outcome", "unpaired_needs_manual_match")

    return sorted(pairs, key=lambda p: (p["family"], p["pair_id"]))


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

    rows: list[dict] = []
    family_bucket_counter: Counter[tuple[str, str]] = Counter()
    for item in strong:
        idx = item.get("idx")
        pred_tta = float(item.get("pred_tta", -1.0))
        alert_tta = float(item.get("alert_tta", 0.0))
        pred_max = float(item.get("pred_max", 0.0))
        alert_max = float(item.get("alert_max", 0.0))
        top_names = item.get("top_names", []) or []
        all_families = _infer_all_families(top_names)
        primary_family = _infer_primary_family(top_names)
        bucket = _tta_bucket(pred_tta)
        family_bucket_counter[(primary_family, bucket)] += 1
        rows.append(
            {
                "idx": idx,
                "pred_tta": pred_tta,
                "alert_tta": alert_tta,
                "pred_max": pred_max,
                "alert_max": alert_max,
                "bucket": bucket,
                "primary_family": primary_family,
                "all_families": ", ".join(all_families),
                "cue": (top_names[0] if top_names else ""),
            }
        )

    pairs = _pair_rows(rows)
    pair_count_by_family = Counter(p["family"] for p in pairs)
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
    lines.append("- Primary-family assignment is a keyword heuristic for audit organization, not a causal label.")
    lines.append("- Auto-suggested pairs require manual reviewer confirmation.")
    lines.append("")
    lines.append("## Raw Availability Snapshot")
    lines.append(f"- `strong_top`: {len(strong)}")
    lines.append(f"- `failed_top`: {len(failed)}")
    lines.append(f"- `late_top`: {len(late)}")
    lines.append(f"- auto-suggested mixed pairs: {len(pairs)}")
    lines.append("- Interpretation: when `failed_top`/`late_top` are sparse or empty,")
    lines.append("  symmetry claims must remain bounded.")
    lines.append("")
    lines.append("## Primary Family × Timing Bucket Counts (from `strong_top`)")
    lines.append("")
    lines.append("| Primary family | early_strong | early_marginal | late_or_missed | auto_pair_count |")
    lines.append("|---|---:|---:|---:|---:|")
    fams = sorted({fam for fam, _ in family_bucket_counter.keys()})
    for fam in fams:
        es = family_bucket_counter.get((fam, "early_strong"), 0)
        em = family_bucket_counter.get((fam, "early_marginal"), 0)
        lm = family_bucket_counter.get((fam, "late_or_missed"), 0)
        pc = pair_count_by_family.get(fam, 0)
        lines.append(f"| {fam} | {es} | {em} | {lm} | {pc} |")
    if not fams:
        lines.append("| (none) | 0 | 0 | 0 | 0 |")
    lines.append("")
    lines.append("## Auto-Suggested Mixed Pairs (Manual Confirmation Required)")
    lines.append("")
    lines.append("| pair_id | family | early_idx | early_tta(s) | late_idx | late_tta(s) | tta_gap(s) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for p in pairs:
        gap = p["early_tta"] - p["late_tta"]
        lines.append(
            f"| {p['pair_id']} | {p['family']} | {p['early_idx']} | {p['early_tta']:.2f} | {p['late_idx']} | {p['late_tta']:.2f} | {gap:.2f} |"
        )
    if not pairs:
        lines.append("| (none) | - | - | - | - | - | - |")
    lines.append("")
    lines.append("## Case-Level Manual Audit Table")
    lines.append("")
    lines.append(
        "| idx | pred_tta(s) | alert_tta(s) | pred_max | bucket | primary_family | all_families | top cue | symmetry_pair_id | paired_outcome | reviewer_note |"
    )
    lines.append("|---:|---:|---:|---:|---|---|---|---|---|---|---|")
    for row in sorted(rows, key=lambda x: x["pred_tta"], reverse=True):
        cue = row["cue"].replace("|", "/")
        lines.append(
            f"| {row['idx']} | {row['pred_tta']:.2f} | {row['alert_tta']:.2f} | {row['pred_max']:.3f} | {row['bucket']} | {row['primary_family']} | {row['all_families']} | {cue} | {row['symmetry_pair_id']} | {row['paired_outcome']} |  |"
        )
    if not rows:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - |")
    lines.append("")
    lines.append("## Completion Rule")
    lines.append("- Do not promote hard-case symmetry claims until:")
    lines.append("  1. family-level pair IDs are reviewed/edited by a human,")
    lines.append("  2. paired outcomes include both success/failure evidence,")
    lines.append("  3. the resulting summary is synchronized into the claim/evidence ledger.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()

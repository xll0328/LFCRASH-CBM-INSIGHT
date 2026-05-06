#!/usr/bin/env python3
import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "emnlp2026_support"
DAD_DIR = ROOT / "output" / "dad_curriculum"

CANONICAL_TAG = "dad_curriculum_v2"
CLEAN_TAGS = [
    "dad_curriculum_s7_clean",
    "dad_curriculum_s43_clean",
    "dad_curriculum_s123_clean",
]
RECOVERY_TAGS = [
    "dad_curriculum_v2_s314",
    "dad_curriculum_v2_s2718",
    "dad_curriculum_v2_s3407",
]


def discover_recovery_tags():
    known = set(RECOVERY_TAGS)
    discovered = []
    for path in sorted(DAD_DIR.glob("dad_curriculum_v2_s*")):
        if path.is_dir():
            discovered.append(path.name)
            known.add(path.name)
    # Keep the historical paper-facing seeds first, then append any newly launched seeds.
    ordered = [tag for tag in RECOVERY_TAGS if tag in known]
    ordered.extend(tag for tag in discovered if tag not in RECOVERY_TAGS)
    return ordered


def load_json(path: Path):
    return json.loads(path.read_text())


def maybe_result(tag: str):
    path = DAD_DIR / tag / "results.json"
    if not path.exists():
        return None, path
    return load_json(path), path


def tag_status(tag: str):
    result, result_path = maybe_result(tag)
    log_path = DAD_DIR / tag / "train.log"
    if result is not None:
        return {
            "tag": tag,
            "status": "completed",
            "results_path": str(result_path),
            "train_log": str(log_path) if log_path.exists() else None,
            "AP": result["AP"],
            "mTTA": result["mTTA"],
            "TTA_R80": result["TTA_R80"],
            "P_R80": result["P_R80"],
            "epoch": result.get("epoch"),
        }
    if log_path.exists():
        return {
            "tag": tag,
            "status": "started",
            "results_path": None,
            "train_log": str(log_path),
        }
    return {
        "tag": tag,
        "status": "pending",
        "results_path": None,
        "train_log": None,
    }


def metric_summary(values):
    if not values:
        return None
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {"mean": mean(values), "std": pstdev(values), "n": len(values)}


def aggregate(rows):
    completed = [row for row in rows if row["status"] == "completed"]
    if not completed:
        return None
    return {
        "AP": metric_summary([row["AP"] for row in completed]),
        "mTTA": metric_summary([row["mTTA"] for row in completed]),
        "TTA_R80": metric_summary([row["TTA_R80"] for row in completed]),
        "P_R80": metric_summary([row["P_R80"] for row in completed]),
    }


def pct(value):
    return f"{100.0 * value:.2f}%"


def build_block(tags):
    rows = [tag_status(tag) for tag in tags]
    return {
        "expected_tags": tags,
        "num_completed": sum(row["status"] == "completed" for row in rows),
        "num_started": sum(row["status"] == "started" for row in rows),
        "num_pending": sum(row["status"] == "pending" for row in rows),
        "rows": rows,
        "aggregate": aggregate(rows),
    }


def completed_rows(block):
    return [row for row in block["rows"] if row["status"] == "completed"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    canonical, canonical_path = maybe_result(CANONICAL_TAG)
    if canonical is None:
      raise SystemExit(f"missing canonical result: {canonical_path}")

    clean_block = build_block(CLEAN_TAGS)
    recovery_tags = discover_recovery_tags()
    recovery_block = build_block(recovery_tags)

    combined_rows = [
        {
            "tag": CANONICAL_TAG,
            "status": "completed",
            "results_path": str(canonical_path),
            "train_log": str(DAD_DIR / CANONICAL_TAG / "train.log"),
            "AP": canonical["AP"],
            "mTTA": canonical["mTTA"],
            "TTA_R80": canonical["TTA_R80"],
            "P_R80": canonical["P_R80"],
            "epoch": canonical.get("epoch"),
        }
    ]
    combined_rows.extend(completed_rows(clean_block))
    combined_rows.extend(completed_rows(recovery_block))
    combined_block = {
        "num_completed": len(combined_rows),
        "rows": combined_rows,
        "aggregate": aggregate(combined_rows),
    }

    recovery_vs_clean = None
    if clean_block["aggregate"] and recovery_block["aggregate"]:
        recovery_vs_clean = {
            "AP_mean_delta": recovery_block["aggregate"]["AP"]["mean"] - clean_block["aggregate"]["AP"]["mean"],
            "mTTA_mean_delta": recovery_block["aggregate"]["mTTA"]["mean"] - clean_block["aggregate"]["mTTA"]["mean"],
        }

    summary = {
        "canonical": {
            "tag": CANONICAL_TAG,
            "results_path": str(canonical_path),
            "AP": canonical["AP"],
            "mTTA": canonical["mTTA"],
            "TTA_R80": canonical["TTA_R80"],
            "P_R80": canonical["P_R80"],
            "epoch": canonical.get("epoch"),
        },
        "clean_seed_block": clean_block,
        "recovery_block": recovery_block,
        "combined_v2_family": combined_block,
        "recovery_vs_clean": recovery_vs_clean,
    }

    json_path = OUT_DIR / "dad_curriculum_recovery_status.json"
    md_path = OUT_DIR / "dad_curriculum_recovery_status.md"
    json_path.write_text(json.dumps(summary, indent=2))

    lines = [
        "# DAD Curriculum Recovery Status",
        "",
        "## Canonical Line",
        "",
        f"- Canonical tag: `{CANONICAL_TAG}`",
        f"- AP: `{pct(canonical['AP'])}`",
        f"- mTTA: `{canonical['mTTA']:.2f}s`",
        f"- TTA@R80: `{canonical['TTA_R80']:.2f}s`",
        f"- P@R80: `{canonical['P_R80']:.3f}`",
        f"- Best epoch: `{canonical.get('epoch')}`",
        "",
        "## Clean Three-Seed Diagnostic",
        "",
        f"- Completed: `{clean_block['num_completed']}/{len(CLEAN_TAGS)}`",
    ]

    if clean_block["aggregate"]:
        agg = clean_block["aggregate"]
        lines.extend(
            [
                f"- AP mean+-std: `{pct(agg['AP']['mean'])} +- {100.0 * agg['AP']['std']:.2f}`",
                f"- mTTA mean+-std: `{agg['mTTA']['mean']:.2f}s +- {agg['mTTA']['std']:.2f}s`",
            ]
        )

    lines += [
        "",
        "## Recovery Block",
        "",
        f"- Expected tags: `{', '.join(recovery_tags)}`",
        f"- Completed: `{recovery_block['num_completed']}/{len(recovery_tags)}`",
        f"- Started: `{recovery_block['num_started']}`",
        f"- Pending: `{recovery_block['num_pending']}`",
        "",
        "| Tag | Status | AP | mTTA | Epoch |",
        "|---|---|---:|---:|---:|",
    ]

    for row in recovery_block["rows"]:
        ap = pct(row["AP"]) if row["status"] == "completed" else "-"
        mtta = f"{row['mTTA']:.2f}s" if row["status"] == "completed" else "-"
        epoch = str(row["epoch"]) if row["status"] == "completed" else "-"
        lines.append(f"| {row['tag']} | {row['status']} | {ap} | {mtta} | {epoch} |")

    if recovery_block["aggregate"]:
        agg = recovery_block["aggregate"]
        lines += [
            "",
            f"- Recovery AP mean+-std: `{pct(agg['AP']['mean'])} +- {100.0 * agg['AP']['std']:.2f}`",
            f"- Recovery mTTA mean+-std: `{agg['mTTA']['mean']:.2f}s +- {agg['mTTA']['std']:.2f}s`",
        ]

    if combined_block["aggregate"]:
        agg = combined_block["aggregate"]
        lines += [
            "",
            "## Combined Exact-v2 Family",
            "",
            f"- Completed snapshots: `{combined_block['num_completed']}`",
            f"- AP mean+-std: `{pct(agg['AP']['mean'])} +- {100.0 * agg['AP']['std']:.2f}`",
            f"- mTTA mean+-std: `{agg['mTTA']['mean']:.2f}s +- {agg['mTTA']['std']:.2f}s`",
        ]

    if recovery_vs_clean:
        lines += [
            "",
            "## Reading",
            "",
            f"- Recovery minus clean AP mean: `{100.0 * recovery_vs_clean['AP_mean_delta']:.2f}` points",
            f"- Recovery minus clean mTTA mean: `{recovery_vs_clean['mTTA_mean_delta']:.2f}s`",
            "- This block is intended to test whether the canonical `dad_curriculum_v2` recipe generalizes beyond the archived clean-seed diagnostic.",
        ]
    else:
        lines += [
            "",
            "## Reading",
            "",
            "- Recovery metrics will appear here once at least one `dad_curriculum_v2_s*` run writes `results.json`.",
            "- The comparison target is the existing clean three-seed diagnostic, not the weaker low-lambda `insight_journal_*` family.",
        ]

    md_path.write_text("\n".join(lines))
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

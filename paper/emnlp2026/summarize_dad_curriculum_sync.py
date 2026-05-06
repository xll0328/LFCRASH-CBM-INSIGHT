#!/usr/bin/env python3
import json
import re
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "emnlp2026_support"
DAD_DIR = ROOT / "output" / "dad_curriculum"
TAGS = [
    "dad_curriculum_s7_clean",
    "dad_curriculum_s43_clean",
    "dad_curriculum_s123_clean",
]
EXTENDED_TAGS = TAGS + [
    "insight_journal_s314_curriculum",
    "insight_journal_s2718_curriculum",
    "insight_journal_s3407_curriculum",
]
TARGET_EPOCHS = [25, 30]


def parse_log(path: Path):
    rows = []
    current_epoch = None
    for line in path.read_text().splitlines():
        epoch_match = re.search(r"Epoch\s+(\d+)/150", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
        eval_match = re.search(
            r"EVAL \| AP=([0-9.]+) mTTA=([0-9.]+) TTA_R80=([0-9.]+) P_R80=([0-9.]+)",
            line,
        )
        if eval_match and current_epoch in TARGET_EPOCHS:
            rows.append(
                {
                    "epoch": current_epoch,
                    "AP": float(eval_match.group(1)),
                    "mTTA": float(eval_match.group(2)),
                    "TTA_R80": float(eval_match.group(3)),
                    "P_R80": float(eval_match.group(4)),
                }
            )
    # Keep the last logged evaluation per target epoch in case a log contains
    # repeated launches or appended reruns under the same tag.
    dedup = {}
    for row in rows:
        dedup[row["epoch"]] = row
    return [dedup[epoch] for epoch in TARGET_EPOCHS if epoch in dedup]


def summarize(values):
    return {"mean": mean(values), "std": pstdev(values), "n": len(values)}


def build_block(tags):
    per_seed = {}
    by_epoch = {epoch: [] for epoch in TARGET_EPOCHS}
    for tag in tags:
        log_path = DAD_DIR / tag / "train.log"
        rows = parse_log(log_path)
        per_seed[tag] = {"log_path": str(log_path), "rows": rows}
        for row in rows:
            by_epoch[row["epoch"]].append({"tag": tag, **row})

    aggregate = {}
    for epoch, rows in by_epoch.items():
        if not rows:
            continue
        aggregate[str(epoch)] = {
            "AP": summarize([row["AP"] for row in rows]),
            "mTTA": summarize([row["mTTA"] for row in rows]),
            "TTA_R80": summarize([row["TTA_R80"] for row in rows]),
            "P_R80": summarize([row["P_R80"] for row in rows]),
        }
    return {"tags": tags, "per_seed": per_seed, "aggregate": aggregate}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "target_epochs": TARGET_EPOCHS,
        "clean_seed_block": build_block(TAGS),
        "extended_seed_block": build_block(EXTENDED_TAGS),
    }

    json_path = OUT_DIR / "dad_curriculum_sync_summary.json"
    md_path = OUT_DIR / "dad_curriculum_sync_summary.md"
    json_path.write_text(json.dumps(summary, indent=2))

    lines = [
        "# DAD Curriculum Synchronized-Epoch Summary",
        "",
        "- Clean seeds: `7`, `43`, `123`",
        "- Extended same-family seeds: `7`, `43`, `123`, `314`, `2718`, `3407`",
        "- Target epochs: `25`, `30`",
        "",
        "## Clean-Seed Aggregate",
        "",
        "| Epoch | AP mean+-std | mTTA mean+-std | TTA@R80 mean+-std | P@R80 mean+-std |",
        "|---|---:|---:|---:|---:|",
    ]
    for epoch in TARGET_EPOCHS:
        agg = summary["clean_seed_block"]["aggregate"].get(str(epoch))
        if not agg:
            continue
        lines.append(
            "| "
            f"{epoch} | "
            f"{100.0 * agg['AP']['mean']:.2f}% +- {100.0 * agg['AP']['std']:.2f} | "
            f"{agg['mTTA']['mean']:.2f}s +- {agg['mTTA']['std']:.2f}s | "
            f"{agg['TTA_R80']['mean']:.2f}s +- {agg['TTA_R80']['std']:.2f}s | "
            f"{agg['P_R80']['mean']:.3f} +- {agg['P_R80']['std']:.3f} |"
        )

    lines += [
        "",
        "## Extended Same-Family Aggregate",
        "",
        "| Epoch | AP mean+-std | mTTA mean+-std | TTA@R80 mean+-std | P@R80 mean+-std |",
        "|---|---:|---:|---:|---:|",
    ]
    for epoch in TARGET_EPOCHS:
        agg = summary["extended_seed_block"]["aggregate"].get(str(epoch))
        if not agg:
            continue
        lines.append(
            "| "
            f"{epoch} | "
            f"{100.0 * agg['AP']['mean']:.2f}% +- {100.0 * agg['AP']['std']:.2f} | "
            f"{agg['mTTA']['mean']:.2f}s +- {agg['mTTA']['std']:.2f}s | "
            f"{agg['TTA_R80']['mean']:.2f}s +- {agg['TTA_R80']['std']:.2f}s | "
            f"{agg['P_R80']['mean']:.3f} +- {agg['P_R80']['std']:.3f} |"
        )

    lines += [
        "",
        "## Clean-Seed Per Seed",
        "",
        "| Seed tag | Epoch | AP | mTTA | TTA@R80 | P@R80 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tag in TAGS:
        for row in summary["clean_seed_block"]["per_seed"][tag]["rows"]:
            lines.append(
                f"| {tag} | {row['epoch']} | {100.0 * row['AP']:.2f}% | {row['mTTA']:.2f}s | "
                f"{row['TTA_R80']:.2f}s | {row['P_R80']:.3f} |"
            )

    lines += [
        "",
        "## Reading",
        "",
        "- This artifact is intended as a checkpoint-sensitivity diagnostic, not a replacement for the canonical single best DAD line.",
        "- The synchronized-epoch view is materially weaker than the canonical `68.19%` AP line and therefore isolates DAD fragility rather than hiding it.",
        "- The clean three-seed block remains the strictest reviewer-facing diagnostic and should stay the main paper-facing caution signal.",
        "- The extended same-family block is useful side evidence that DAD is not collapsing across every seed, but it should not be used to blur the difference between the canonical line and synchronized-epoch diagnostics.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

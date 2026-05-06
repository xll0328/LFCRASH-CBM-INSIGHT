#!/usr/bin/env python3
import json
from pathlib import Path

from audit_utils import load_json, parse_train_enhanced_log, stats


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "emnlp2026_support"
SEEDS = [42, 123, 3407]
BASE_TAG = "a3d_sota"


def candidate_paths(seed: int):
    if seed == 42:
        return [
            ROOT / "output" / "sota_push" / BASE_TAG / "results.json",
            ROOT / "output" / "sota_push" / f"{BASE_TAG}_s42" / "results.json",
        ]
    return [ROOT / "output" / "sota_push" / f"{BASE_TAG}_s{seed}" / "results.json"]


def find_result(seed: int):
    for path in candidate_paths(seed):
        if path.exists():
            return path
    return None


def find_partial(seed: int):
    if seed == 42:
        candidates = [
            ROOT / "output" / "sota_push" / BASE_TAG / "train.log",
            ROOT / "output" / "sota_push" / f"{BASE_TAG}_s42" / "train.log",
        ]
    else:
        candidates = [ROOT / "output" / "sota_push" / f"{BASE_TAG}_s{seed}" / "train.log"]

    for path in candidates:
        parsed = parse_train_enhanced_log(path)
        if parsed is not None:
            return path, parsed
    return None, None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    num_in_progress = 0
    missing = []
    for seed in SEEDS:
        path = find_result(seed)
        if path is not None:
            data = load_json(path)
            if "AP" not in data:
                raise RuntimeError(f"Result file missing paper metrics: {path}")
            rows.append(
                {
                    "seed": seed,
                    "path": str(path),
                    "AP": data["AP"],
                    "mTTA": data["mTTA"],
                    "TTA_R80": data["TTA_R80"],
                    "P_R80": data["P_R80"],
                    "epoch": data["epoch"],
                    "last_epoch_seen": data.get("epoch"),
                    "tag": data.get("tag", BASE_TAG if seed == 42 else f"{BASE_TAG}_s{seed}"),
                    "status": "completed",
                    "source": "results.json",
                }
            )
            continue

        log_path, partial = find_partial(seed)
        if partial is None:
            missing.append(seed)
            continue

        num_in_progress += 1
        rows.append(
            {
                "seed": seed,
                "path": str(log_path),
                "AP": partial.get("AP"),
                "mTTA": partial.get("mTTA"),
                "TTA_R80": partial.get("TTA_R80"),
                "P_R80": partial.get("P_R80"),
                "epoch": partial.get("epoch"),
                "last_epoch_seen": partial["last_epoch_seen"],
                "tag": BASE_TAG if seed == 42 else f"{BASE_TAG}_s{seed}",
                "status": "artifact-missing" if partial["completed_in_log"] else "in-progress",
                "source": "train.log(best-so-far)" if partial.get("AP") is not None else "train.log(started)",
            }
        )

    completed_rows = [row for row in rows if row["status"] == "completed"]
    rows_with_metrics = [row for row in rows if row["AP"] is not None]
    summary = {
        "dataset": "a3d",
        "base_tag": BASE_TAG,
        "expected_seeds": SEEDS,
        "num_completed": len(completed_rows),
        "num_in_progress": num_in_progress,
        "num_available": len(rows),
        "num_expected": len(SEEDS),
        "missing_seeds": missing,
        "rows": rows,
    }
    if completed_rows:
        summary["aggregate_completed"] = {
            "AP": stats([r["AP"] for r in completed_rows]),
            "mTTA": stats([r["mTTA"] for r in completed_rows]),
            "TTA_R80": stats([r["TTA_R80"] for r in completed_rows]),
            "P_R80": stats([r["P_R80"] for r in completed_rows]),
        }
    if rows_with_metrics:
        summary["aggregate_best_so_far"] = {
            "AP": stats([r["AP"] for r in rows_with_metrics]),
            "mTTA": stats([r["mTTA"] for r in rows_with_metrics]),
            "TTA_R80": stats([r["TTA_R80"] for r in rows_with_metrics]),
            "P_R80": stats([r["P_R80"] for r in rows_with_metrics]),
        }

    json_path = OUT_DIR / "a3d_headline_multiseed_status.json"
    md_path = OUT_DIR / "a3d_headline_multiseed_status.md"
    json_path.write_text(json.dumps(summary, indent=2))

    lines = [
        "# A3D Headline Multi-Seed Status",
        "",
        f"- Base tag: `{BASE_TAG}`",
        f"- Expected seeds: `{', '.join(str(seed) for seed in SEEDS)}`",
        f"- Completed: `{len(completed_rows)}/{len(SEEDS)}`",
    ]
    if summary["num_in_progress"]:
        lines.append(f"- In progress / best-so-far only: `{summary['num_in_progress']}`")
    if "aggregate_completed" in summary:
        agg = summary["aggregate_completed"]
        lines.extend(
            [
                "",
                "## Completed Aggregate",
                "",
                f"- AP: `{100.0 * agg['AP']['mean']:.2f}% +- {100.0 * agg['AP']['std']:.2f}`",
                f"- mTTA: `{agg['mTTA']['mean']:.3f}s +- {agg['mTTA']['std']:.3f}s`",
                f"- TTA@R80: `{agg['TTA_R80']['mean']:.3f}s +- {agg['TTA_R80']['std']:.3f}s`",
                f"- P@R80: `{agg['P_R80']['mean']:.4f} +- {agg['P_R80']['std']:.4f}`",
            ]
        )
    if "aggregate_best_so_far" in summary and summary["num_available"] > summary["num_completed"]:
        agg = summary["aggregate_best_so_far"]
        lines.extend(
            [
                "",
                "## Current Best-So-Far Aggregate",
                "",
                f"- AP: `{100.0 * agg['AP']['mean']:.2f}% +- {100.0 * agg['AP']['std']:.2f}`",
                f"- mTTA: `{agg['mTTA']['mean']:.3f}s +- {agg['mTTA']['std']:.3f}s`",
                f"- TTA@R80: `{agg['TTA_R80']['mean']:.3f}s +- {agg['TTA_R80']['std']:.3f}s`",
                f"- P@R80: `{agg['P_R80']['mean']:.4f} +- {agg['P_R80']['std']:.4f}`",
                "",
                "This block includes best-so-far metrics parsed from `train.log` for seeds that have not written `results.json` yet.",
            ]
        )
    if missing:
        lines.extend(["", f"- Missing seeds: `{', '.join(str(seed) for seed in missing)}`"])

    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Seed | Tag | Status | Source | AP | mTTA | TTA@R80 | P@R80 | Best Epoch | Last Epoch |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        ap = f"{100.0 * row['AP']:.2f}%" if row["AP"] is not None else "--"
        mtta = f"{row['mTTA']:.3f}s" if row["mTTA"] is not None else "--"
        tta_r80 = f"{row['TTA_R80']:.3f}s" if row["TTA_R80"] is not None else "--"
        p_r80 = f"{row['P_R80']:.4f}" if row["P_R80"] is not None else "--"
        best_epoch = row["epoch"] if row["epoch"] is not None else "--"
        last_epoch = row.get("last_epoch_seen")
        last_epoch = last_epoch if last_epoch is not None else "--"
        lines.append(
            f"| {row['seed']} | {row['tag']} | {row['status']} | {row['source']} | "
            f"{ap} | {mtta} | {tta_r80} | {p_r80} | {best_epoch} | {last_epoch} |"
        )

    md_path.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

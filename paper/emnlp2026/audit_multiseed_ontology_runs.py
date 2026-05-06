#!/usr/bin/env python3
import json
from pathlib import Path

from audit_utils import (
    discover_active_train_multi_tags,
    discover_ontology_queue_targets,
    load_json,
    parse_train_multi_log,
    stats,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "emnlp2026_support"
MANIFEST_PATH = ROOT / "output" / "concept_sets" / "neurips2026_controlled_ontology_manifest.json"
SEEDS = [42, 123, 3407]


def find_seed_result(dataset: str, tag: str, seed: int):
    seeded_tag = f"{tag}_s{seed}"
    candidate = ROOT / "output" / f"{dataset}_ac" / seeded_tag / "results.json"
    return candidate if candidate.exists() else None


def find_seed_progress(dataset: str, tag: str, seed: int):
    seeded_tag = f"{tag}_s{seed}"
    log_path = ROOT / "logs" / f"{seeded_tag}.launch.log"
    parsed = parse_train_multi_log(log_path)
    return parsed


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_json(MANIFEST_PATH)
    concept_sets = manifest["concept_sets"]
    recommended = manifest["recommended_controlled_runs"]
    queue_targets = discover_ontology_queue_targets(ROOT)
    active_tags = discover_active_train_multi_tags(ROOT)

    summary_rows = []
    for run in recommended:
        dataset = run["dataset"]
        tag = run["tag"]
        concept_set = run["concept_set"]
        results = []
        in_progress = []
        queued = []
        missing = []
        for seed in SEEDS:
            path = find_seed_result(dataset, tag, seed)
            progress = find_seed_progress(dataset, tag, seed)
            if path is not None:
                data = load_json(path)
                result_row = {
                    "seed": seed,
                    "path": str(path),
                    "AP": data["AP"],
                    "mTTA": data["mTTA"],
                    "TTA_R80": data["TTA_R80"],
                    "P_R80": data["P_R80"],
                    "epoch": data.get("epoch"),
                }
                seeded_tag = f"{tag}_s{seed}"
                is_actively_running = seeded_tag in active_tags
                if progress is not None and not progress.get("completed_in_log", False) and is_actively_running:
                    progress["seed"] = seed
                    progress["path"] = str(path)
                    progress["source"] = "results.json(best-so-far)"
                    for key in ("AP", "mTTA", "TTA_R80", "epoch"):
                        if result_row.get(key) is not None:
                            progress[key] = result_row[key]
                    if result_row.get("P_R80") is not None:
                        progress["P_R80"] = result_row["P_R80"]
                    in_progress.append(progress)
                else:
                    results.append(result_row)
                continue

            if progress is not None:
                progress["seed"] = seed
                in_progress.append(progress)
                continue

            queue_item = queue_targets.get((dataset, tag, seed))
            if queue_item is not None:
                queued.append(queue_item)
                continue

            missing.append(seed)

        row = {
            "dataset": dataset,
            "concept_set": concept_set,
            "concept_count": concept_sets[concept_set]["num_concepts"],
            "base_tag": tag,
            "num_completed": len(results),
            "num_in_progress": len(in_progress),
            "num_queued": len(queued),
            "num_expected": len(SEEDS),
            "missing_seeds": missing,
            "results": results,
            "in_progress": in_progress,
            "queued": queued,
        }
        if results:
            row["aggregate"] = {
                "AP": stats([r["AP"] for r in results]),
                "mTTA": stats([r["mTTA"] for r in results]),
                "TTA_R80": stats([r["TTA_R80"] for r in results]),
                "P_R80": stats([r["P_R80"] for r in results]),
            }
        summary_rows.append(row)

    json_path = OUT_DIR / "multiseed_ontology_status.json"
    md_path = OUT_DIR / "multiseed_ontology_status.md"
    json_path.write_text(json.dumps({"seeds": SEEDS, "rows": summary_rows}, indent=2))

    lines = [
        "# Multi-Seed Ontology Status",
        "",
        f"- Expected seeds: `{', '.join(str(seed) for seed in SEEDS)}`",
        "",
        "| Dataset | Concept set | #Concepts | Seeds done | Running | Queued | AP mean+-std | mTTA mean+-std | Missing |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in summary_rows:
        agg = row.get("aggregate")
        if agg:
            lines.append(
                f"| {row['dataset']} | {row['concept_set']} | {row['concept_count']} | "
                f"{row['num_completed']}/{row['num_expected']} | "
                f"{row['num_in_progress']} | "
                f"{row['num_queued']} | "
                f"{100.0 * agg['AP']['mean']:.2f}% +- {100.0 * agg['AP']['std']:.2f} | "
                f"{agg['mTTA']['mean']:.2f}s +- {agg['mTTA']['std']:.2f}s | "
                f"{', '.join(str(seed) for seed in row['missing_seeds']) or '--'} |"
            )
        else:
            lines.append(
                f"| {row['dataset']} | {row['concept_set']} | {row['concept_count']} | "
                f"0/{row['num_expected']} | {row['num_in_progress']} | {row['num_queued']} | -- | -- | "
                f"{', '.join(str(seed) for seed in row['missing_seeds']) or '--'} |"
            )

    running_rows = [row for row in summary_rows if row["in_progress"]]
    if running_rows:
        lines.extend(["", "## In Progress", ""])
        for row in running_rows:
            for item in row["in_progress"]:
                epoch_note = ""
                if item.get("last_epoch_seen") is not None and item.get("total_epochs") is not None:
                    epoch_note = f" last_epoch={item['last_epoch_seen']}/{item['total_epochs']}"
                elif item.get("last_epoch_seen") is not None:
                    epoch_note = f" last_epoch={item['last_epoch_seen']}"
                best_note = ""
                if item.get("AP") is not None:
                    best_note = f" best_so_far={100.0 * item['AP']:.2f}% AP, {item['mTTA']:.2f}s mTTA"
                lines.append(
                    f"- {row['dataset']} / {row['concept_set']} / seed {item['seed']}:{epoch_note}{best_note}"
                )

    queued_rows = [row for row in summary_rows if row["queued"]]
    if queued_rows:
        lines.extend(["", "## Queued", ""])
        for row in queued_rows:
            for item in row["queued"]:
                script_name = Path(item["queue_script"]).name
                lines.append(
                    f"- {row['dataset']} / {row['concept_set']} / seed {item['seed']}: "
                    f"queued on gpu {item['gpu']} via `{script_name}`"
                )

    md_path.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "emnlp2026_support"
CONCEPT_META = {
    "historical_full": 837,
    "risk_core_v1": 30,
    "perfect_v1": 80,
}
SEEDS = [42, 123, 3407, 7, 11, 2718, 314, 2026]


def stats(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": mean, "std": var ** 0.5}


def load_json(path: Path):
    return json.loads(path.read_text())


def get_running_tags():
    cmd = "ps -eo cmd"
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()

    tags = set()
    for line in proc.stdout.splitlines():
        if "train_multi.py" not in line or "--tag" not in line:
            continue
        parts = line.split("--tag", 1)[1].strip().split()
        if parts:
            tags.add(parts[0])
    return tags


def main():
    running_tags = get_running_tags()
    rows = []
    for concept_set, concept_count in CONCEPT_META.items():
        results = []
        running = []
        pending = []
        for seed in SEEDS:
            tag = f"dad_shared_{concept_set}_s{seed}"
            path = ROOT / "output" / "dad_ac" / tag / "results.json"
            is_running = tag in running_tags

            if is_running:
                running.append(seed)
                continue

            if not path.exists():
                pending.append(seed)
                continue

            data = load_json(path)
            results.append(
                {
                    "seed": seed,
                    "tag": tag,
                    "path": str(path),
                    "AP": data["AP"],
                    "mTTA": data["mTTA"],
                    "TTA_R80": data["TTA_R80"],
                    "P_R80": data["P_R80"],
                    "epoch": data.get("epoch"),
                }
            )

        row = {
            "dataset": "dad",
            "concept_set": concept_set,
            "concept_count": concept_count,
            "num_completed": len(results),
            "num_running": len(running),
            "num_expected": len(SEEDS),
            "running_seeds": running,
            "pending_seeds": pending,
            "results": results,
        }
        if results:
            row["aggregate"] = {
                "AP": stats([r["AP"] for r in results]),
                "mTTA": stats([r["mTTA"] for r in results]),
                "TTA_R80": stats([r["TTA_R80"] for r in results]),
                "P_R80": stats([r["P_R80"] for r in results]),
            }
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "dad_ontology_seed_extension_status.json"
    md_path = OUT_DIR / "dad_ontology_seed_extension_status.md"
    json_path.write_text(json.dumps({"seeds": SEEDS, "rows": rows}, indent=2))

    md_lines = [
        "# DAD Ontology Seed Extension Status",
        "",
        f"- Seeds tracked: `{', '.join(str(s) for s in SEEDS)}`",
        "",
        "| Concept set | #Concepts | Completed | Running | AP mean+-std (completed) | mTTA mean+-std (completed) | Pending |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        agg = row.get("aggregate")
        if agg is None:
            ap = "--"
            mtta = "--"
        else:
            ap = f"{100.0 * agg['AP']['mean']:.2f}% +- {100.0 * agg['AP']['std']:.2f}"
            mtta = f"{agg['mTTA']['mean']:.2f}s +- {agg['mTTA']['std']:.2f}s"
        pending = ", ".join(str(s) for s in row["pending_seeds"]) or "--"
        md_lines.append(
            f"| {row['concept_set']} | {row['concept_count']} | "
            f"{row['num_completed']}/{row['num_expected']} | "
            f"{row['num_running']} | {ap} | {mtta} | {pending} |"
        )

    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

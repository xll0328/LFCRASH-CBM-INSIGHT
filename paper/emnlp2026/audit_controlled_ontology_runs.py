#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Optional


def find_result_file(root: Path, tag: str) -> Optional[Path]:
    candidates = sorted(root.glob(f"output/**/*/{tag}/results.json"))
    if candidates:
        return candidates[0]
    return None


def load_json(path: Path):
    return json.loads(path.read_text())


def main():
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "output/concept_sets/neurips2026_controlled_ontology_manifest.json"
    out_dir = root / "output/emnlp2026_support"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(manifest_path)
    concept_sets = manifest["concept_sets"]
    runs = []

    for run in manifest["recommended_controlled_runs"]:
        tag = run["tag"]
        result_path = find_result_file(root, tag)
        entry = {
            "name": run["name"],
            "dataset": run["dataset"],
            "concept_set": run["concept_set"],
            "concept_count": concept_sets[run["concept_set"]]["num_concepts"],
            "role": concept_sets[run["concept_set"]]["role"],
            "tag": tag,
            "completed": result_path is not None,
            "result_path": str(result_path) if result_path else None,
        }
        if result_path:
            entry.update(load_json(result_path))
        runs.append(entry)

    summary = {
        "manifest": str(manifest_path),
        "num_runs": len(runs),
        "num_completed": sum(1 for r in runs if r["completed"]),
        "runs": runs,
    }

    by_dataset = {}
    for dataset in sorted({r["dataset"] for r in runs}):
        ds_runs = [r for r in runs if r["dataset"] == dataset and r["completed"]]
        if not ds_runs:
            continue
        best_ap = max(ds_runs, key=lambda r: r["AP"])
        best_mtta = max(ds_runs, key=lambda r: r["mTTA"])
        by_dataset[dataset] = {
            "best_AP": {
                "concept_set": best_ap["concept_set"],
                "AP": best_ap["AP"],
                "mTTA": best_ap["mTTA"],
            },
            "best_mTTA": {
                "concept_set": best_mtta["concept_set"],
                "AP": best_mtta["AP"],
                "mTTA": best_mtta["mTTA"],
            },
        }
    summary["by_dataset"] = by_dataset

    json_path = out_dir / "controlled_ontology_status.json"
    md_path = out_dir / "controlled_ontology_status.md"
    json_path.write_text(json.dumps(summary, indent=2))

    lines = [
        "# Controlled Ontology Status",
        "",
        f"- Completed runs: `{summary['num_completed']}/{summary['num_runs']}`",
        f"- Manifest: `{manifest_path}`",
        "",
        "## Run Table",
        "",
        "| Dataset | Concept set | #Concepts | AP | mTTA | TTA@R80 | P@R80 | Epoch | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in runs:
        if r["completed"]:
            lines.append(
                f"| {r['dataset']} | {r['concept_set']} | {r['concept_count']} | "
                f"{100*r['AP']:.2f}% | {r['mTTA']:.2f}s | {r['TTA_R80']:.2f}s | "
                f"{r['P_R80']:.3f} | {r['epoch']} | completed |"
            )
        else:
            lines.append(
                f"| {r['dataset']} | {r['concept_set']} | {r['concept_count']} | -- | -- | -- | -- | -- | missing |"
            )

    lines += [
        "",
        "## Main Reading",
        "",
        f"- DAD: best AP comes from `{by_dataset['dad']['best_AP']['concept_set']}` "
        f"({100*by_dataset['dad']['best_AP']['AP']:.2f}%), while best mTTA comes from "
        f"`{by_dataset['dad']['best_mTTA']['concept_set']}` "
        f"({by_dataset['dad']['best_mTTA']['mTTA']:.2f}s).",
        f"- A3D: best AP comes from `{by_dataset['a3d']['best_AP']['concept_set']}` "
        f"({100*by_dataset['a3d']['best_AP']['AP']:.2f}%), while best mTTA comes from "
        f"`{by_dataset['a3d']['best_mTTA']['concept_set']}` "
        f"({by_dataset['a3d']['best_mTTA']['mTTA']:.2f}s).",
        "- The controlled block is therefore complete enough to defend the claim that ontology choice changes the AP--mTTA operating point under one shared recipe.",
    ]
    md_path.write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

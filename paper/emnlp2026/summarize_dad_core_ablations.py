#!/usr/bin/env python3
import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "emnlp2026_support"
BASE_DIR = ROOT / "output" / "insight_journal_phase2_ablation_20260327_071733"

VARIANTS = ["no_cbm", "no_align", "no_sparse", "no_recon"]


def summarize(vals):
    return {"mean": mean(vals), "std": pstdev(vals), "n": len(vals)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        paths = sorted(BASE_DIR.glob(f"insight_journal_dad_{variant}_r*/results.json"))
        results = []
        for path in paths:
            data = json.loads(path.read_text())
            results.append(
                {
                    "tag": path.parent.name,
                    "AP": data["AP"],
                    "mTTA": data["mTTA"],
                    "TTA_R80": data["TTA_R80"],
                    "P_R80": data["P_R80"],
                }
            )
        if not results:
            continue
        rows.append(
            {
                "variant": variant,
                "results": results,
                "aggregate": {
                    "AP": summarize([row["AP"] for row in results]),
                    "mTTA": summarize([row["mTTA"] for row in results]),
                    "TTA_R80": summarize([row["TTA_R80"] for row in results]),
                    "P_R80": summarize([row["P_R80"] for row in results]),
                },
            }
        )

    report = {"source_dir": str(BASE_DIR), "rows": rows}
    json_path = OUT_DIR / "dad_core_ablation_summary.json"
    md_path = OUT_DIR / "dad_core_ablation_summary.md"
    json_path.write_text(json.dumps(report, indent=2))

    lines = [
        "# DAD Core Ablation Summary",
        "",
        "- Source block: `insight_journal_phase2_ablation_20260327_071733`",
        "- Reviewer-priority variants: `no_cbm`, `no_align`",
        "",
        "| Variant | AP mean+-std | mTTA mean+-std | TTA@R80 mean+-std | P@R80 mean+-std | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        agg = row["aggregate"]
        lines.append(
            f"| {row['variant']} | "
            f"{100.0 * agg['AP']['mean']:.2f}% +- {100.0 * agg['AP']['std']:.2f} | "
            f"{agg['mTTA']['mean']:.2f}s +- {agg['mTTA']['std']:.2f}s | "
            f"{agg['TTA_R80']['mean']:.2f}s +- {agg['TTA_R80']['std']:.2f}s | "
            f"{agg['P_R80']['mean']:.3f} +- {agg['P_R80']['std']:.3f} | "
            f"{agg['AP']['n']} |"
        )

    lines += [
        "",
        "## Reading",
        "",
        "- `no_cbm` and `no_align` are the highest-priority mechanism checks because they most directly test whether the semantic bottleneck and semantic supervision matter.",
        "- These numbers should be read as support analyses within one DAD ablation family, not as replacements for the canonical curriculum headline line.",
        "- If `no_cbm` stays too strong relative to the full model, that weakens any overly simplistic claim that the current DAD gain comes entirely from the semantic bottleneck.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

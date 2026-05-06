#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = ROOT / "output" / "emnlp2026_support"
EVAL_SCRIPT = ROOT / "eval_actor_vs_classifier.py"

CHECKPOINTS = [
    ("dad_curriculum_s7_clean", ROOT / "output" / "dad_curriculum" / "dad_curriculum_s7_clean" / "best_model.pt"),
    ("dad_curriculum_s43_clean", ROOT / "output" / "dad_curriculum" / "dad_curriculum_s43_clean" / "best_model.pt"),
    ("dad_curriculum_s123_clean", ROOT / "output" / "dad_curriculum" / "dad_curriculum_s123_clean" / "best_model.pt"),
    ("insight_journal_s314_curriculum", ROOT / "output" / "dad_curriculum" / "insight_journal_s314_curriculum" / "best_model.pt"),
    ("insight_journal_s2718_curriculum", ROOT / "output" / "dad_curriculum" / "insight_journal_s2718_curriculum" / "best_model.pt"),
    ("insight_journal_s3407_curriculum", ROOT / "output" / "dad_curriculum" / "insight_journal_s3407_curriculum" / "best_model.pt"),
]


def summary(values):
    return {"mean": mean(values), "std": pstdev(values), "n": len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = SUPPORT_DIR / "dad_trigger_compare_extended"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag, checkpoint in CHECKPOINTS:
        output_json = out_dir / f"{tag}.json"
        if args.force or not output_json.exists():
            cmd = [
                args.python,
                str(EVAL_SCRIPT),
                "--checkpoint",
                str(checkpoint),
                "--dataset",
                "dad",
                "--gpu",
                str(args.gpu),
                "--num_workers",
                "0",
                "--output_json",
                str(output_json),
            ]
            print("[run]", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        data = json.loads(output_json.read_text())
        cls = data["classifier_trigger"]
        actor = data["actor_trigger"]
        rows.append(
            {
                "tag": tag,
                "checkpoint": str(checkpoint),
                "epoch": data.get("epoch"),
                "classifier_trigger": cls,
                "actor_trigger": actor,
                "actor_diagnostics": data.get("actor_diagnostics", {}),
                "delta": {
                    "AP": actor["AP"] - cls["AP"],
                    "mTTA": actor["mTTA"] - cls["mTTA"],
                    "TTA_R80": actor["TTA_R80"] - cls["TTA_R80"],
                    "P_R80": actor["P_R80"] - cls["P_R80"],
                },
            }
        )

    classifier_ap = [row["classifier_trigger"]["AP"] for row in rows]
    classifier_mtta = [row["classifier_trigger"]["mTTA"] for row in rows]
    actor_ap = [row["actor_trigger"]["AP"] for row in rows]
    actor_mtta = [row["actor_trigger"]["mTTA"] for row in rows]
    delta_ap = [row["delta"]["AP"] for row in rows]
    delta_mtta = [row["delta"]["mTTA"] for row in rows]
    crossing = [row["actor_diagnostics"]["crossing_rate_at_threshold"] for row in rows]

    report = {
        "gpu": args.gpu,
        "rows": rows,
        "aggregate": {
            "classifier_AP": summary(classifier_ap),
            "classifier_mTTA": summary(classifier_mtta),
            "actor_AP": summary(actor_ap),
            "actor_mTTA": summary(actor_mtta),
            "delta_AP": summary(delta_ap),
            "delta_mTTA": summary(delta_mtta),
            "actor_crossing_rate_at_threshold": summary(crossing),
        },
    }

    json_path = SUPPORT_DIR / "dad_trigger_compare_extended_summary.json"
    md_path = SUPPORT_DIR / "dad_trigger_compare_extended_summary.md"
    json_path.write_text(json.dumps(report, indent=2))

    agg = report["aggregate"]
    lines = [
        "# DAD Trigger-Source Extended Summary",
        "",
        "- Dataset: `dad`",
        "- Checkpoint family: clean + same-curriculum support checkpoints",
        f"- Num checkpoints: `{len(rows)}`",
        "",
        "## Aggregate",
        "",
        f"- Classifier trigger: `{100.0 * agg['classifier_AP']['mean']:.2f}% +- {100.0 * agg['classifier_AP']['std']:.2f}` AP, `{agg['classifier_mTTA']['mean']:.2f}s +- {agg['classifier_mTTA']['std']:.2f}s` mTTA",
        f"- Actor trigger: `{100.0 * agg['actor_AP']['mean']:.2f}% +- {100.0 * agg['actor_AP']['std']:.2f}` AP, `{agg['actor_mTTA']['mean']:.2f}s +- {agg['actor_mTTA']['std']:.2f}s` mTTA",
        f"- Actor minus classifier: `{100.0 * agg['delta_AP']['mean']:.2f}% +- {100.0 * agg['delta_AP']['std']:.2f}` AP, `{agg['delta_mTTA']['mean']:.2f}s +- {agg['delta_mTTA']['std']:.2f}s` mTTA",
        f"- Actor crossing rate at threshold 0.5: `{agg['actor_crossing_rate_at_threshold']['mean']:.3f} +- {agg['actor_crossing_rate_at_threshold']['std']:.3f}`",
        "",
        "## Per Checkpoint",
        "",
        "| Tag | Epoch | Classifier AP | Actor AP | Delta AP | Classifier mTTA | Actor mTTA | Delta mTTA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['tag']} | {row['epoch']} | "
            f"{100.0 * row['classifier_trigger']['AP']:.2f}% | "
            f"{100.0 * row['actor_trigger']['AP']:.2f}% | "
            f"{100.0 * row['delta']['AP']:.2f}% | "
            f"{row['classifier_trigger']['mTTA']:.2f}s | "
            f"{row['actor_trigger']['mTTA']:.2f}s | "
            f"{row['delta']['mTTA']:.2f}s |"
        )

    lines += [
        "",
        "## Reading",
        "",
        "- This block is a timing-related support analysis, not a replacement for the headline classifier-based tables.",
        "- The classifier trajectory remains consistently stronger in AP across the sampled DAD checkpoints.",
        "- If the actor mTTA remains close while AP collapses, that is evidence that current policy timing is not yet mature enough to carry the main paper claim.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "emnlp2026_support"
BASE_DIR = ROOT / "output" / "v3_final"

VARIANTS = ["a3d_full", "a3d_no_cbm", "a3d_no_align", "a3d_no_sparse", "a3d_no_recon"]


def load_json(path: Path):
    return json.loads(path.read_text())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_json(BASE_DIR / "final_summary.json")
    rows = []
    full = summary["a3d_full"]
    for key in VARIANTS:
        data = summary[key]
        rows.append(
            {
                "variant": key.replace("a3d_", ""),
                "source": str(BASE_DIR / key / "results.json"),
                "AP": data["AP"] / 100.0,
                "mTTA": data["mTTA"],
                "TTA_R80": data["TTA_R80"],
                "P_R80": data["P_R80"] / 100.0,
                "best_epoch": data["best_epoch"],
                "delta_vs_full": {
                    "AP": (data["AP"] - full["AP"]) / 100.0,
                    "mTTA": data["mTTA"] - full["mTTA"],
                    "TTA_R80": data["TTA_R80"] - full["TTA_R80"],
                    "P_R80": (data["P_R80"] - full["P_R80"]) / 100.0,
                },
            }
        )

    report = {
        "source_summary": str(BASE_DIR / "final_summary.json"),
        "protocol": "single-run A3D support family from v3_final",
        "rows": rows,
    }

    json_path = OUT_DIR / "a3d_core_ablation_summary.json"
    md_path = OUT_DIR / "a3d_core_ablation_summary.md"
    json_path.write_text(json.dumps(report, indent=2))

    lines = [
        "# A3D Core Ablation Summary",
        "",
        "- Source block: `output/v3_final/final_summary.json`",
        "- Protocol: single-run A3D support family",
        "- Role: support evidence for the semantic-interface mechanism, not a replacement for the headline A3D recipe",
        "",
        "| Variant | AP | mTTA | TTA@R80 | P@R80 | Delta AP vs full | Delta mTTA vs full |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["delta_vs_full"]
        lines.append(
            f"| {row['variant']} | "
            f"{100.0 * row['AP']:.2f}% | "
            f"{row['mTTA']:.3f}s | "
            f"{row['TTA_R80']:.3f}s | "
            f"{100.0 * row['P_R80']:.2f}% | "
            f"{100.0 * delta['AP']:+.2f} | "
            f"{delta['mTTA']:+.3f}s |"
        )

    lines += [
        "",
        "## Reading",
        "",
        "- Removing the concept bottleneck (`no_cbm`) lowers AP relative to the full A3D support line, which is cleaner semantic support than the mixed DAD ablation picture.",
        "- Removing alignment (`no_align`) raises AP but worsens mTTA and TTA@R80, so it changes the accuracy--timing trade-off rather than cleanly dominating the full model.",
        "- The A3D support family therefore suggests that the semantic interface is most convincing when read as part of an operating-point design, not as a guarantee that every auxiliary term monotonically improves every metric.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

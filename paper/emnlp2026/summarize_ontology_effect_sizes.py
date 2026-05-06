#!/usr/bin/env python3
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = ROOT / "output" / "emnlp2026_support"


CONCEPT_LABEL = {
    "historical_full": "Historical full",
    "risk_core_v1": "Risk-core manual",
    "perfect_v1": "Perfect concept set v1",
}


DATASET_LABEL = {"dad": "DAD", "a3d": "A3D"}


def load_json(path: Path):
    return json.loads(path.read_text())


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def fmt_sec(x: float) -> str:
    return f"{x:.2f}s"


def ci_diff(mean_a: float, std_a: float, n_a: int, mean_b: float, std_b: float, n_b: int):
    diff = mean_a - mean_b
    se = math.sqrt((std_a * std_a) / max(n_a, 1) + (std_b * std_b) / max(n_b, 1))
    ci_lo = diff - 1.96 * se
    ci_hi = diff + 1.96 * se
    return {"diff": diff, "se": se, "ci95": [ci_lo, ci_hi]}


def build_rows(multiseed_data):
    by_dataset = {"dad": {}, "a3d": {}}
    for row in multiseed_data["rows"]:
        agg = row.get("aggregate")
        if agg is None:
            continue
        by_dataset[row["dataset"]][row["concept_set"]] = {
            "n": row["num_completed"],
            "AP_mean": agg["AP"]["mean"],
            "AP_std": agg["AP"]["std"],
            "mTTA_mean": agg["mTTA"]["mean"],
            "mTTA_std": agg["mTTA"]["std"],
        }

    comparisons = []
    for dataset in ("dad", "a3d"):
        ds = by_dataset[dataset]
        required = {"historical_full", "risk_core_v1", "perfect_v1"}
        if not required.issubset(set(ds.keys())):
            continue

        pairs = [
            ("risk_core_v1", "historical_full"),
            ("perfect_v1", "historical_full"),
            ("risk_core_v1", "perfect_v1"),
        ]
        for a, b in pairs:
            ra = ds[a]
            rb = ds[b]
            ap = ci_diff(ra["AP_mean"], ra["AP_std"], ra["n"], rb["AP_mean"], rb["AP_std"], rb["n"])
            mtta = ci_diff(
                ra["mTTA_mean"], ra["mTTA_std"], ra["n"], rb["mTTA_mean"], rb["mTTA_std"], rb["n"]
            )
            comparisons.append(
                {
                    "dataset": dataset,
                    "A": a,
                    "B": b,
                    "A_n": ra["n"],
                    "B_n": rb["n"],
                    "AP_diff": ap,
                    "mTTA_diff": mtta,
                }
            )
    return comparisons


def main():
    multiseed = load_json(SUPPORT_DIR / "multiseed_ontology_status.json")
    rows = build_rows(multiseed)

    json_path = SUPPORT_DIR / "ontology_effect_size_summary.json"
    md_path = SUPPORT_DIR / "ontology_effect_size_summary.md"
    tex_path = SUPPORT_DIR / "ontology_effect_size_rows.tex"

    json_path.write_text(json.dumps({"comparisons": rows}, indent=2))

    md_lines = [
        "# Ontology Effect Size Summary (Seed-Aggregated)",
        "",
        "Differences are reported as `A - B` with normal-approx 95% CI.",
        "",
        "| Dataset | Comparison | AP diff | AP 95% CI | mTTA diff | mTTA 95% CI |",
        "|---|---|---:|---|---:|---|",
    ]
    tex_lines = []

    for r in rows:
        ap_diff = 100.0 * r["AP_diff"]["diff"]
        ap_lo = 100.0 * r["AP_diff"]["ci95"][0]
        ap_hi = 100.0 * r["AP_diff"]["ci95"][1]
        mtta_diff = r["mTTA_diff"]["diff"]
        mtta_lo = r["mTTA_diff"]["ci95"][0]
        mtta_hi = r["mTTA_diff"]["ci95"][1]
        cmp_name = f"{CONCEPT_LABEL[r['A']]} - {CONCEPT_LABEL[r['B']]}"
        md_lines.append(
            f"| {DATASET_LABEL[r['dataset']]} | {cmp_name} | "
            f"{ap_diff:+.2f} pp | [{ap_lo:+.2f}, {ap_hi:+.2f}] pp | "
            f"{mtta_diff:+.2f}s | [{mtta_lo:+.2f}, {mtta_hi:+.2f}]s |"
        )
        tex_lines.append(
            f"{DATASET_LABEL[r['dataset']]} & {cmp_name} & "
            f"{ap_diff:+.2f} pp & [{ap_lo:+.2f}, {ap_hi:+.2f}] pp & "
            f"{mtta_diff:+.2f}s & [{mtta_lo:+.2f}, {mtta_hi:+.2f}]s \\\\"
        )

    md_path.write_text("\n".join(md_lines) + "\n")
    tex_path.write_text("\n".join(tex_lines) + "\n")

    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")
    print(f"[wrote] {tex_path}")


if __name__ == "__main__":
    main()

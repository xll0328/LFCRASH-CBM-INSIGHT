#!/usr/bin/env python3
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = ROOT / "output" / "emnlp2026_support"

DATASETS = ["dad", "a3d"]
SEEDS = [42, 123, 3407]
CONDS = [
    ("historical_stratified_30", 30, "Historical-stratified (30)"),
    ("risk_core_v1", 30, "Risk-core manual (30)"),
    ("historical_stratified_80", 80, "Historical-stratified (80)"),
    ("perfect_v1", 80, "Perfect v1 (80)"),
]

DATASET_LABEL = {"dad": "DAD", "a3d": "A3D"}


def load_result(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def get_running_tags():
    try:
        proc = subprocess.run(
            "ps -eo cmd",
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


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def std(xs):
    if len(xs) < 2:
        return 0.0 if xs else None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci95(mu_a, sd_a, n_a, mu_b, sd_b, n_b):
    if min(n_a, n_b) < 2:
        return None
    diff = mu_a - mu_b
    se = math.sqrt((sd_a * sd_a) / n_a + (sd_b * sd_b) / n_b)
    return {
        "diff": diff,
        "ci95": [diff - 1.96 * se, diff + 1.96 * se],
        "se": se,
    }


def collect():
    running_tags = get_running_tags()
    aggregates = {ds: {} for ds in DATASETS}

    for ds in DATASETS:
        for cond, k, label in CONDS:
            rows = []
            running_preview_rows = []
            for seed in SEEDS:
                tag = f"{ds}_sizectrl_{cond}_s{seed}"
                p = ROOT / "output" / f"{ds}_ac" / tag / "results.json"
                if tag in running_tags:
                    rr = load_result(p)
                    if rr is not None:
                        running_preview_rows.append({"seed": seed, **rr})
                    continue
                r = load_result(p)
                if r is not None:
                    rows.append({"seed": seed, **r})

            ap = [float(r["AP"]) for r in rows if isinstance(r.get("AP"), (int, float))]
            mtta = [float(r["mTTA"]) for r in rows if isinstance(r.get("mTTA"), (int, float))]

            aggregates[ds][cond] = {
                "label": label,
                "concept_count": k,
                "n": len(rows),
                "n_running_preview": len(running_preview_rows),
                "total": len(SEEDS),
                "missing_seeds": [s for s in SEEDS if s not in {r["seed"] for r in rows}],
                "AP": {
                    "mean": mean(ap),
                    "std": std(ap),
                },
                "mTTA": {
                    "mean": mean(mtta),
                    "std": std(mtta),
                },
                "rows": rows,
                "running_preview_rows": running_preview_rows,
            }

    return aggregates


def build_comparisons(aggregates):
    comps = []
    for ds in DATASETS:
        # 30-budget source comparison
        a = aggregates[ds]["risk_core_v1"]
        b = aggregates[ds]["historical_stratified_30"]
        ap = ci95(a["AP"]["mean"], a["AP"]["std"], a["n"], b["AP"]["mean"], b["AP"]["std"], b["n"]) if a["AP"]["mean"] is not None and b["AP"]["mean"] is not None else None
        t = ci95(a["mTTA"]["mean"], a["mTTA"]["std"], a["n"], b["mTTA"]["mean"], b["mTTA"]["std"], b["n"]) if a["mTTA"]["mean"] is not None and b["mTTA"]["mean"] is not None else None
        comps.append({
            "dataset": ds,
            "budget": 30,
            "A": "risk_core_v1",
            "B": "historical_stratified_30",
            "A_n": a["n"],
            "B_n": b["n"],
            "AP": ap,
            "mTTA": t,
            "ready": ap is not None and t is not None,
        })

        # 80-budget source comparison
        a = aggregates[ds]["perfect_v1"]
        b = aggregates[ds]["historical_stratified_80"]
        ap = ci95(a["AP"]["mean"], a["AP"]["std"], a["n"], b["AP"]["mean"], b["AP"]["std"], b["n"]) if a["AP"]["mean"] is not None and b["AP"]["mean"] is not None else None
        t = ci95(a["mTTA"]["mean"], a["mTTA"]["std"], a["n"], b["mTTA"]["mean"], b["mTTA"]["std"], b["n"]) if a["mTTA"]["mean"] is not None and b["mTTA"]["mean"] is not None else None
        comps.append({
            "dataset": ds,
            "budget": 80,
            "A": "perfect_v1",
            "B": "historical_stratified_80",
            "A_n": a["n"],
            "B_n": b["n"],
            "AP": ap,
            "mTTA": t,
            "ready": ap is not None and t is not None,
        })

    return comps


def fmt_pct(x):
    return "--" if x is None else f"{100*x:.2f}%"


def fmt_pp(x):
    return "--" if x is None else f"{100*x:+.2f} pp"


def fmt_sec(x):
    return "--" if x is None else f"{x:.2f}s"


def main():
    agg = collect()
    comps = build_comparisons(agg)

    out_json = SUPPORT_DIR / "ontology_size_matched_effects.json"
    out_md = SUPPORT_DIR / "ontology_size_matched_effects.md"
    out_tex = SUPPORT_DIR / "ontology_size_matched_effect_rows.tex"

    out_json.write_text(json.dumps({"aggregates": agg, "comparisons": comps}, indent=2))

    lines = [
        "# Ontology Size-Matched Effect Summary",
        "",
        "This file tracks source-vs-size disentanglement controls (30/80 concept budgets).",
        "All aggregates use completed runs only; running jobs are excluded until they finish.",
        "",
        "## Aggregate Status",
        "",
        "| Dataset | Condition | #Concepts | n/3 completed | running | AP mean±std (completed) | mTTA mean±std (completed) | Missing |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for ds in DATASETS:
        for cond, k, label in CONDS:
            row = agg[ds][cond]
            ap_m = row["AP"]["mean"]
            ap_s = row["AP"]["std"]
            t_m = row["mTTA"]["mean"]
            t_s = row["mTTA"]["std"]
            ap_txt = "--" if ap_m is None else f"{100*ap_m:.2f}% ± {100*ap_s:.2f}%"
            t_txt = "--" if t_m is None else f"{t_m:.2f}s ± {t_s:.2f}s"
            missing = "--" if not row["missing_seeds"] else ", ".join(str(x) for x in row["missing_seeds"])
            lines.append(
                f"| {DATASET_LABEL[ds]} | {label} | {k} | {row['n']}/3 | {row['n_running_preview']} | {ap_txt} | {t_txt} | {missing} |"
            )

    lines += [
        "",
        "## Source Effect Under Fixed Budget",
        "",
        "Comparisons are `A - B` with normal-approx 95% CI; reported when both sides have n>=2.",
        "",
        "| Dataset | Budget | Comparison | AP diff (95% CI) | mTTA diff (95% CI) | Status |",
        "|---|---:|---|---|---|---|",
    ]

    tex_lines = []
    label_map = {
        "risk_core_v1": "Risk-core manual",
        "historical_stratified_30": "Historical-stratified",
        "perfect_v1": "Perfect v1",
        "historical_stratified_80": "Historical-stratified",
    }

    for c in comps:
        cmp_name = f"{label_map[c['A']]} - {label_map[c['B']]}"
        if c["AP"] is None or c["mTTA"] is None:
            lines.append(
                f"| {DATASET_LABEL[c['dataset']]} | {c['budget']} | {cmp_name} | -- | -- | pending ({c['A_n']}/{c['B_n']}) |"
            )
            continue

        ap = c["AP"]
        mt = c["mTTA"]
        ap_txt = f"{100*ap['diff']:+.2f} pp [{100*ap['ci95'][0]:+.2f}, {100*ap['ci95'][1]:+.2f}]"
        mt_txt = f"{mt['diff']:+.2f}s [{mt['ci95'][0]:+.2f}, {mt['ci95'][1]:+.2f}]"
        lines.append(
            f"| {DATASET_LABEL[c['dataset']]} | {c['budget']} | {cmp_name} | {ap_txt} | {mt_txt} | ready |"
        )
        tex_lines.append(
            f"{DATASET_LABEL[c['dataset']]} & {c['budget']} & {cmp_name} & "
            f"{100*ap['diff']:+.2f} pp [{100*ap['ci95'][0]:+.2f}, {100*ap['ci95'][1]:+.2f}] & "
            f"{mt['diff']:+.2f}s [{mt['ci95'][0]:+.2f}, {mt['ci95'][1]:+.2f}] \\" 
        )

    out_md.write_text("\n".join(lines) + "\n")
    out_tex.write_text("\n".join(tex_lines) + "\n")

    print(f"[wrote] {out_json}")
    print(f"[wrote] {out_md}")
    print(f"[wrote] {out_tex}")


if __name__ == "__main__":
    main()

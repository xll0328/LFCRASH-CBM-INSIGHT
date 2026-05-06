#!/usr/bin/env python3
import json
import math
import subprocess
from pathlib import Path

ROOT = Path("/data/sony/LFCRASH/LFCRASH-CBM")
OUT = ROOT / "output" / "emnlp2026_support"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["dad", "a3d"]
CONDS = [
    ("historical_stratified_30", 30),
    ("risk_core_v1", 30),
    ("historical_stratified_80", 80),
    ("perfect_v1", 80),
]
SEEDS = [42, 123, 3407]


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


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main():
    running_tags = get_running_tags()
    rows = []
    status = {}

    for ds in DATASETS:
        status[ds] = {}
        for cond, k in CONDS:
            completed = []
            running = []
            pending = []

            for s in SEEDS:
                tag = f"{ds}_sizectrl_{cond}_s{s}"
                path = ROOT / "output" / f"{ds}_ac" / tag / "results.json"
                is_running = tag in running_tags

                if is_running:
                    running.append(s)
                    continue

                if not path.exists():
                    pending.append(s)
                    continue

                data = read_json(path)
                if data is None:
                    pending.append(s)
                    continue

                completed.append(
                    {
                        "seed": s,
                        "tag": tag,
                        "AP": float(data["AP"]) if isinstance(data.get("AP"), (int, float)) else None,
                        "mTTA": float(data["mTTA"]) if isinstance(data.get("mTTA"), (int, float)) else None,
                        "epoch": data.get("epoch"),
                        "path": str(path),
                    }
                )

            ap_vals = [x["AP"] for x in completed if x["AP"] is not None]
            tta_vals = [x["mTTA"] for x in completed if x["mTTA"] is not None]

            ap_mean = mean(ap_vals)
            ap_std = std(ap_vals)
            tta_mean = mean(tta_vals)
            tta_std = std(tta_vals)

            status[ds][cond] = {
                "concept_count": k,
                "done": len(completed),
                "running": len(running),
                "total": len(SEEDS),
                "ap_mean_percent": None if ap_mean is None else ap_mean * 100.0,
                "ap_std_percent": None if ap_std is None else ap_std * 100.0,
                "mtta_mean_sec": tta_mean,
                "mtta_std_sec": tta_std,
                "running_seeds": running,
                "pending_seeds": pending,
                "completed_rows": completed,
            }

            rows.append((ds, cond, k, len(completed), len(running), len(SEEDS), ap_mean, ap_std, tta_mean, tta_std, pending))

    json_path = OUT / "ontology_size_matched_status.json"
    md_path = OUT / "ontology_size_matched_status.md"
    json_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    lines = [
        "# Ontology Size-Matched Control Status",
        "",
        "- Seeds tracked: `42, 123, 3407`",
        "",
        "| Dataset | Condition | #Concepts | Completed | Running | AP mean±std (completed) | mTTA mean±std (completed) | Pending |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for ds, cond, k, done, running, total, ap_m, ap_s, tta_m, tta_s, pending in rows:
        ap_s_txt = "--" if ap_m is None else f"{ap_m*100.0:.2f}% ± {ap_s*100.0:.2f}%"
        tta_s_txt = "--" if tta_m is None else f"{tta_m:.2f}s ± {tta_s:.2f}s"
        pending_txt = "--" if not pending else ", ".join(str(x) for x in pending)
        lines.append(
            f"| {ds} | {cond} | {k} | {done}/{total} | {running} | {ap_s_txt} | {tta_s_txt} | {pending_txt} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

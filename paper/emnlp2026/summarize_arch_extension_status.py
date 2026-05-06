#!/usr/bin/env python3
"""Summarize architecture-extension support runs (DAD RWKV + A3D h384)."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = ROOT / "output" / "emnlp2026_support"
DAD_DIR = ROOT / "output" / "dad_ac"
A3D_DIR = ROOT / "output" / "a3d_ac"
JSON_OUT = SUPPORT_DIR / "arch_extension_status.json"
MD_OUT = SUPPORT_DIR / "arch_extension_status.md"

DAD_TAGS = [
    "dad_ac_perfect_v1_rwkv_q1",
    "dad_ac_perfect_v1_rwkv_q2",
    "dad_ac_perfect_v1_rwkv_q3",
]
A3D_TAGS = [
    "a3d_ac_perfect_v1_h384_q1",
    "a3d_ac_perfect_v1_h384_q2",
    "a3d_ac_perfect_v1_h384_q3",
]

EPOCH_RE = re.compile(
    r"Ep\s+(?P<epoch>\d+)\s*/\s*(?P<total>\d+).*?loss=(?P<loss>-?\d+\.\d+)"
)
EVAL_RE = re.compile(
    r"EVAL\s+\|\s+AP=(?P<AP>\d+\.\d+)\s+mTTA=(?P<mTTA>\d+\.\d+)\s+TTA[@_]R80=(?P<TTA_R80>\d+\.\d+)"
)
ERROR_PATTERNS = (
    "Traceback (most recent call last):",
    "RuntimeError:",
    "CUDA out of memory",
    "AttributeError:",
    "Killed",
)


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def sec(x: float) -> str:
    return f"{x:.2f}s"


def summarize(vals: list[float]) -> dict | None:
    if not vals:
        return None
    return {
        "mean": mean(vals),
        "std": pstdev(vals),
        "n": len(vals),
    }


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def log_age_seconds(path: Path) -> int | None:
    if not path.exists():
        return None
    return max(0, int(time.time() - path.stat().st_mtime))


def active_pids(tag: str) -> list[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    table: dict[int, tuple[int, str]] = {}
    tagged: dict[int, tuple[int, str]] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        args = parts[2]
        table[pid] = (ppid, args)
        if tag in args and ("train_dad_ac.py" in args or "train_multi.py" in args):
            tagged[pid] = (ppid, args)

    # Keep only top-level trainer processes. DataLoader workers inherit the same
    # command line and should not be counted as independent active runs.
    masters = [pid for pid, (ppid, _) in tagged.items() if ppid not in tagged]
    return sorted(masters)


def latest_log_signals(log_path: Path) -> tuple[dict | None, dict | None, str | None]:
    if not log_path.exists():
        return None, None, None
    latest_epoch = None
    latest_eval = None
    latest_error = None
    for line in log_path.read_text(errors="replace").splitlines():
        em = EPOCH_RE.search(line)
        if em:
            latest_epoch = {
                "epoch": int(em.group("epoch")),
                "total": int(em.group("total")),
                "loss": float(em.group("loss")),
            }
        vm = EVAL_RE.search(line)
        if vm:
            latest_eval = {
                "AP": float(vm.group("AP")),
                "mTTA": float(vm.group("mTTA")),
                "TTA_R80": float(vm.group("TTA_R80")),
            }
        if any(tok in line for tok in ERROR_PATTERNS):
            latest_error = line.strip()
    return latest_epoch, latest_eval, latest_error


def run_status(tag: str, family: str, run_dir: Path) -> dict:
    log_path = run_dir / "nohup.log"
    result_path = run_dir / "results.json"
    results = load_json(result_path)
    pids = active_pids(tag)
    latest_epoch, latest_eval, latest_error = latest_log_signals(log_path)

    if pids and results:
        status = "running_with_result"
    elif pids:
        status = "running"
    elif results:
        status = "completed"
    elif latest_error:
        status = "failed"
    elif log_path.exists():
        status = "started"
    else:
        status = "not_started"

    row = {
        "family": family,
        "tag": tag,
        "status": status,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "result_path": str(result_path),
        "log_age_seconds": log_age_seconds(log_path),
        "active_pids": pids,
        "latest_epoch": latest_epoch,
        "latest_eval": latest_eval,
        "latest_error": latest_error,
        "results": results,
    }
    return row


def aggregate(rows: list[dict]) -> dict | None:
    completed = [r["results"] for r in rows if r.get("results")]
    if not completed:
        return None
    out = {
        "AP": summarize([r["AP"] for r in completed if "AP" in r]),
        "mTTA": summarize([r["mTTA"] for r in completed if "mTTA" in r]),
    }
    if all("TTA_R80" in r for r in completed):
        out["TTA_R80"] = summarize([r["TTA_R80"] for r in completed])
    if all("P_R80" in r for r in completed):
        out["P_R80"] = summarize([r["P_R80"] for r in completed])
    return out


def aggregate_completed_only(rows: list[dict]) -> dict | None:
    return aggregate([r for r in rows if r.get("status") == "completed"])


def build_block(family: str, tags: list[str], root_dir: Path) -> dict:
    rows = [run_status(tag, family=family, run_dir=root_dir / tag) for tag in tags]
    return {
        "family": family,
        "num_expected": len(tags),
        "num_started": sum(r["status"] != "not_started" for r in rows),
        "num_running": sum(r["status"] in {"running", "running_with_result"} for r in rows),
        "num_running_with_result": sum(r["status"] == "running_with_result" for r in rows),
        "num_completed": sum(r["status"] == "completed" for r in rows),
        "num_failed": sum(r["status"] == "failed" for r in rows),
        "rows": rows,
        "current_best_aggregate": aggregate(rows),
        "completed_aggregate": aggregate_completed_only(rows),
    }


def row_md(row: dict) -> str:
    pids = ",".join(str(x) for x in row["active_pids"]) if row["active_pids"] else "--"
    le = row.get("latest_epoch")
    lv = row.get("latest_eval")
    err = row.get("latest_error")
    epoch_txt = f"{le['epoch']}/{le['total']} (loss={le['loss']:.4f})" if le else "--"
    eval_txt = f"AP={pct(lv['AP'])}, mTTA={sec(lv['mTTA'])}" if lv else "--"
    result_txt = "--"
    if row.get("results"):
        result_txt = f"AP={pct(row['results']['AP'])}, mTTA={sec(row['results']['mTTA'])}"
    err_txt = err if err else "--"
    return (
        f"| {row['tag']} | {row['status']} | {pids} | {epoch_txt} | "
        f"{eval_txt} | {result_txt} | {err_txt} |"
    )


def agg_line(name: str, block: dict) -> str:
    agg = block.get("current_best_aggregate")
    if not agg or not agg.get("AP") or not agg.get("mTTA"):
        return f"- {name}: no result-backed aggregate yet."
    ap = agg["AP"]
    mtta = agg["mTTA"]
    return (
        f"- {name}: AP={pct(ap['mean'])}±{100.0 * ap['std']:.2f}pp, "
        f"mTTA={sec(mtta['mean'])}±{mtta['std']:.2f}s (n={ap['n']})."
    )


def main() -> int:
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dad = build_block("DAD RWKV", DAD_TAGS, DAD_DIR)
    a3d = build_block("A3D h384", A3D_TAGS, A3D_DIR)

    report = {
        "generated_at": now,
        "blocks": {
            "dad_rwkv": dad,
            "a3d_h384": a3d,
        },
        "note": (
            "Architecture extensions are support-only evidence and should not "
            "replace canonical headline tables."
        ),
    }
    JSON_OUT.write_text(json.dumps(report, indent=2))

    lines = [
        "# Architecture Extension Status (Support-Only)",
        "",
        f"- Generated at: `{now}`",
        "- Scope: DAD RWKV and A3D h_dim=384 extension runs.",
        "- Role: support evidence only; do not mix with headline claims before completion.",
        "",
        "## Completion Snapshot",
        "",
        f"- DAD RWKV: started `{dad['num_started']}/{dad['num_expected']}`, running `{dad['num_running']}`, completed `{dad['num_completed']}`, failed `{dad['num_failed']}`.",
        f"- A3D h384: started `{a3d['num_started']}/{a3d['num_expected']}`, running `{a3d['num_running']}`, completed `{a3d['num_completed']}`, failed `{a3d['num_failed']}`.",
        "",
        "## Aggregate Snapshot",
        "",
        "- Current best-checkpoint aggregates (rows with `results.json`, including running runs):",
        agg_line("DAD RWKV", dad),
        agg_line("A3D h384", a3d),
        "- Strict completed-only counts:",
        f"  DAD RWKV completed runs: {dad['num_completed']}/{dad['num_expected']}",
        f"  A3D h384 completed runs: {a3d['num_completed']}/{a3d['num_expected']}",
        "",
        "## Per-Run Status",
        "",
        "| Tag | Status | PID(s) | Latest epoch | Latest eval | results.json | Latest error line |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in dad["rows"] + a3d["rows"]:
        lines.append(row_md(row))
    lines.append("")
    lines.append("## Interpretation Guardrail")
    lines.append("")
    lines.append("- Keep these runs in support tier until each 3-run family is complete and aggregated.")
    lines.append("- `running_with_result` means the current best checkpoint is already serialized while training is still active.")
    lines.append("- If a failed run exists, rerun only the missing/failing tag; do not cherry-pick partial best points.")

    MD_OUT.write_text("\n".join(lines) + "\n")
    print(
        "[arch-extension-status] "
        f"dad completed={dad['num_completed']}/{dad['num_expected']} "
        f"a3d completed={a3d['num_completed']}/{a3d['num_expected']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

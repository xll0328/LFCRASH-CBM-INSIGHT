#!/usr/bin/env python3
"""Summarize the predeclared DAD light-regularization mechanism block.

Behavior is overridable by environment variables for staged experiments:
- DAD_LIGHTREG_BLOCK_DIR: custom output directory.
- DAD_LIGHTREG_RUN_TAG_PREFIX: custom tag prefix.
- DAD_LIGHTREG_STATUS_JSON: custom json filename under output/emnlp2026_support.
- DAD_LIGHTREG_STATUS_MD: custom md filename under output/emnlp2026_support.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
import os


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = ROOT / "output" / "emnlp2026_support"
BLOCK_DIR = Path(
    os.environ.get("DAD_LIGHTREG_BLOCK_DIR", str(ROOT / "output" / "dad_mechanism_lightreg_block"))
)
JSON_OUT = SUPPORT_DIR / os.environ.get(
    "DAD_LIGHTREG_STATUS_JSON", "dad_mechanism_lightreg_status.json"
)
MD_OUT = SUPPORT_DIR / os.environ.get(
    "DAD_LIGHTREG_STATUS_MD", "dad_mechanism_lightreg_status.md"
)
RUN_TAG_PREFIX = os.environ.get("DAD_LIGHTREG_RUN_TAG_PREFIX", "insight_journal_dad_lightreg")
RUN_TAGS = [f"{RUN_TAG_PREFIX}_r{i}" for i in range(1, 4)]

EPOCH_RE = re.compile(
    r"Ep\s+(?P<epoch>\d+)\s+\|\s+loss=(?P<loss>\d+\.\d+)\s+"
    r"ce=(?P<ce>\d+\.\d+)\s+aux=(?P<aux>\d+\.\d+)\s+"
    r"align=(?P<align>\d+\.\d+)\s+sparse=(?P<sparse>\d+\.\d+)\s+"
    r"recon=(?P<recon>\d+\.\d+)\s+lr=(?P<lr>[0-9.e+-]+)\s+nan=(?P<nan>\d+)"
)
EVAL_RE = re.compile(
    r"Ep\s+(?P<epoch>\d+)\s+EVAL\s+\|\s+AP=(?P<AP>\d+\.\d+)\s+"
    r"mTTA=(?P<mTTA>\d+\.\d+)s\s+TTA@R80=(?P<TTA_R80>\d+\.\d+)s\s+"
    r"P@R80=(?P<P_R80>\d+\.\d+)"
)


def summarize(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {"mean": mean(values), "std": pstdev(values), "n": len(values)}


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def sec(x: float) -> str:
    return f"{x:.2f}s"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def active_processes(tag: str) -> list[dict]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,etimes=,pcpu=,pmem=,args="],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        if tag not in line or "train.py" not in line:
            continue
        parts = line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        pid, elapsed_s, pcpu, pmem, args = parts
        rows.append(
            {
                "pid": int(pid),
                "elapsed_s": int(elapsed_s),
                "pcpu": float(pcpu),
                "pmem": float(pmem),
                "args": args,
            }
        )
    return rows


def tmux_sessions(tag: str) -> list[str]:
    tag_core = re.sub(r"^insight_journal_", "", tag)
    needle = tag_core
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return sorted(
        session.strip()
        for session in result.stdout.splitlines()
        if needle in session
    )


def log_age_seconds(path: Path) -> int | None:
    if not path.exists():
        return None
    return max(0, int(time.time() - path.stat().st_mtime))


def latest_log_matches(path: Path) -> tuple[dict | None, dict | None]:
    latest_epoch = None
    latest_eval = None
    if not path.exists():
        return latest_epoch, latest_eval
    for line in path.read_text(errors="replace").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            latest_epoch = {
                "epoch": int(epoch_match.group("epoch")),
                "loss": float(epoch_match.group("loss")),
                "ce": float(epoch_match.group("ce")),
                "aux": float(epoch_match.group("aux")),
                "align": float(epoch_match.group("align")),
                "sparse": float(epoch_match.group("sparse")),
                "recon": float(epoch_match.group("recon")),
                "lr": epoch_match.group("lr"),
                "nan": int(epoch_match.group("nan")),
            }
        eval_match = EVAL_RE.search(line)
        if eval_match:
            latest_eval = {
                "epoch": int(eval_match.group("epoch")),
                "AP": float(eval_match.group("AP")),
                "mTTA": float(eval_match.group("mTTA")),
                "TTA_R80": float(eval_match.group("TTA_R80")),
                "P_R80": float(eval_match.group("P_R80")),
            }
    return latest_epoch, latest_eval


def run_status(tag: str) -> dict:
    run_dir = BLOCK_DIR / tag
    log_path = run_dir / "train.log"
    history_path = run_dir / "history.json"
    results_path = run_dir / "results.json"
    latest_epoch, latest_eval = latest_log_matches(log_path)
    results = load_json(results_path)
    processes = active_processes(tag)
    sessions = tmux_sessions(tag)
    status = "not_started"
    if results:
        status = "completed"
    elif processes:
        status = "running"
    elif log_path.exists():
        status = "in_progress"
    return {
        "tag": tag,
        "status": status,
        "run_dir": str(run_dir),
        "train_log": str(log_path),
        "history_json": str(history_path),
        "results_json": str(results_path),
        "log_age_seconds": log_age_seconds(log_path),
        "active_processes": processes,
        "tmux_sessions": sessions,
        "latest_epoch": latest_epoch,
        "latest_eval": latest_eval,
        "results": results,
    }


def aggregate_completed(rows: list[dict]) -> dict | None:
    completed = [row["results"] for row in rows if row.get("results")]
    if not completed:
        return None
    return {
        "AP": summarize([row["AP"] for row in completed]),
        "mTTA": summarize([row["mTTA"] for row in completed]),
        "TTA_R80": summarize([row["TTA_R80"] for row in completed]),
        "P_R80": summarize([row["P_R80"] for row in completed]),
    }


def aggregate_latest_eval(rows: list[dict]) -> dict | None:
    evals = [row["latest_eval"] for row in rows if row.get("latest_eval")]
    if not evals:
        return None
    return {
        "epochs": sorted({row["epoch"] for row in evals}),
        "AP": summarize([row["AP"] for row in evals]),
        "mTTA": summarize([row["mTTA"] for row in evals]),
        "TTA_R80": summarize([row["TTA_R80"] for row in evals]),
        "P_R80": summarize([row["P_R80"] for row in evals]),
    }


def main() -> int:
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [run_status(tag) for tag in RUN_TAGS]
    aggregate = aggregate_completed(rows)
    latest_eval_snapshot = aggregate_latest_eval(rows)
    num_started = sum(1 for row in rows if row["status"] != "not_started")
    num_completed = sum(1 for row in rows if row["status"] == "completed")

    report = {
        "generated_at": generated_at,
        "block_dir": str(BLOCK_DIR),
        "num_started": num_started,
        "num_completed": num_completed,
        "runs": rows,
        "latest_eval_snapshot": latest_eval_snapshot,
        "completed_aggregate": aggregate,
        "interpretation_gate": {
            "strong_success": "3/3 complete, AP>=64.0%, mTTA<=2.30s, AP std<=2.0pp, P@R80 within 0.02 of matched full",
            "useful_tie": "3/3 complete, AP in [62.5%,64.0%), mTTA<=2.35s, no failed-training signature",
            "failure": "AP<62.5%, AP std>2.5pp, mTTA>2.50s, incomplete/corrupted runs, or instability",
        },
    }
    JSON_OUT.write_text(json.dumps(report, indent=2))

    lines = [
        "# DAD Mechanism Light-Regularization Status",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Block directory: `{BLOCK_DIR}`",
        "- Nature: status monitor for a predeclared GPU experiment; not paper evidence until all runs complete and are aggregated.",
        "",
        "## Verdict",
        "",
        f"- Runs started: `{num_started}/3`",
        f"- Runs completed: `{num_completed}/3`",
        "",
        "## Per-Run Status",
        "",
        "| Run | Status | Active PID(s) | tmux | Log age | Latest train epoch | Latest eval | Result |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        latest_epoch = row.get("latest_epoch")
        latest_eval = row.get("latest_eval")
        pids = ", ".join(
            str(proc["pid"]) for proc in row.get("active_processes", [])
        ) or "--"
        sessions = ", ".join(row.get("tmux_sessions", [])) or "--"
        log_age = row.get("log_age_seconds")
        log_age_text = f"{log_age}s" if log_age is not None else "--"
        epoch_text = str(latest_epoch["epoch"]) if latest_epoch else "--"
        if latest_eval:
            eval_text = (
                f"ep{latest_eval['epoch']} AP={pct(latest_eval['AP'])}, "
                f"mTTA={sec(latest_eval['mTTA'])}"
            )
        else:
            eval_text = "--"
        result_text = "present" if row.get("results") else "--"
        lines.append(
            f"| {row['tag']} | {row['status']} | {pids} | {sessions} | {log_age_text} | {epoch_text} | {eval_text} | {result_text} |"
        )
    lines.append("")

    lines += ["## Latest Eval Snapshot", ""]
    if latest_eval_snapshot:
        epoch_text = ", ".join(
            str(epoch) for epoch in latest_eval_snapshot["epochs"]
        )
        lines += [
            "- Non-final monitoring snapshot only; do not use as paper evidence.",
            f"- Eval epochs represented: `{epoch_text}`",
            f"- AP: `{pct(latest_eval_snapshot['AP']['mean'])} +- {100.0 * latest_eval_snapshot['AP']['std']:.2f}` over `n={latest_eval_snapshot['AP']['n']}`",
            f"- mTTA: `{sec(latest_eval_snapshot['mTTA']['mean'])} +- {latest_eval_snapshot['mTTA']['std']:.2f}s`",
            f"- TTA@R80: `{sec(latest_eval_snapshot['TTA_R80']['mean'])} +- {latest_eval_snapshot['TTA_R80']['std']:.2f}s`",
            f"- P@R80: `{latest_eval_snapshot['P_R80']['mean']:.3f} +- {latest_eval_snapshot['P_R80']['std']:.3f}`",
        ]
    else:
        lines.append("- No eval snapshot yet.")
    lines.append("")

    lines += ["## Completed Aggregate", ""]
    if aggregate:
        lines += [
            f"- AP: `{pct(aggregate['AP']['mean'])} +- {100.0 * aggregate['AP']['std']:.2f}` over `n={aggregate['AP']['n']}`",
            f"- mTTA: `{sec(aggregate['mTTA']['mean'])} +- {aggregate['mTTA']['std']:.2f}s`",
            f"- TTA@R80: `{sec(aggregate['TTA_R80']['mean'])} +- {aggregate['TTA_R80']['std']:.2f}s`",
            f"- P@R80: `{aggregate['P_R80']['mean']:.3f} +- {aggregate['P_R80']['std']:.3f}`",
        ]
    else:
        lines.append("- No completed aggregate yet.")
    lines.append("")

    lines += [
        "## Predeclared Interpretation Gate",
        "",
        "- Strong success: `3/3` complete, AP mean at least `64.0%`, mTTA at most `2.30s`, AP std at most `2.0` percentage points, and P@R80 within `0.02` of matched full.",
        "- Useful tie: `3/3` complete, AP mean in `[62.5%, 64.0%)`, mTTA at most `2.35s`, and no failed-training signature.",
        "- Failure: AP mean below `62.5%`, AP std above `2.5` percentage points, mTTA above `2.50s`, incomplete/corrupted runs, or training instability.",
        "",
        "## Reading",
        "",
        "- Do not update paper claims from this block until all three runs complete.",
        "- If the block remains in progress, use it only as an execution status, not as evidence.",
        "- Compare completed aggregate against matched full, `no_cbm`, and `no_align` before changing claim tier.",
        "",
    ]
    MD_OUT.write_text("\n".join(lines))
    print(f"[wrote] {JSON_OUT}")
    print(f"[wrote] {MD_OUT}")
    print(f"[dad-lightreg-status] started={num_started}/3 completed={num_completed}/3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

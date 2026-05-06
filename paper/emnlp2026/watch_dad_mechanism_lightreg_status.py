#!/usr/bin/env python3
"""Bounded watcher for the DAD light-regularization mechanism block."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = ROOT / "output" / "emnlp2026_support"
SUMMARY_SCRIPT = ROOT / "paper" / "emnlp2026" / "summarize_dad_mechanism_lightreg_status.py"
STATUS_JSON = SUPPORT_DIR / os.environ.get(
    "DAD_LIGHTREG_STATUS_JSON", "dad_mechanism_lightreg_status.json"
)
LOG_NAME = os.environ.get("DAD_LIGHTREG_WATCH_LOG", "watch_dad_mechanism_lightreg_status.log")
LOG = SUPPORT_DIR / LOG_NAME


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    with LOG.open("a") as f:
        f.write(line)
    print(line, end="", flush=True)


def run_summary() -> int:
    result = subprocess.run(
        [sys.executable, str(SUMMARY_SCRIPT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        log(f"summary failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.returncode


def load_status() -> dict:
    return json.loads(STATUS_JSON.read_text())


def active_pid_count(status: dict) -> int:
    return sum(
        len(row.get("active_processes") or [])
        for row in status.get("runs", [])
    )


def latest_eval_text(status: dict) -> str:
    snapshot = status.get("latest_eval_snapshot")
    if not snapshot:
        return "latest_eval=none"
    ap = snapshot["AP"]
    mtta = snapshot["mTTA"]
    epochs = ",".join(str(epoch) for epoch in snapshot.get("epochs", []))
    return (
        f"latest_eval_epochs={epochs} "
        f"AP_mean={100.0 * ap['mean']:.2f}% "
        f"mTTA_mean={mtta['mean']:.2f}s "
        f"n={ap['n']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--max-minutes", type=int, default=24 * 60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    started = time.time()
    log(
        "watch_dad_mechanism_lightreg_status.py started "
        f"interval_sec={args.interval_sec} max_minutes={args.max_minutes}"
    )

    while True:
        rc = run_summary()
        if rc != 0:
            return rc

        status = load_status()
        completed = int(status.get("num_completed", 0))
        active = active_pid_count(status)
        log(f"completed={completed}/3 active_pids={active} {latest_eval_text(status)}")

        max_required = int(os.environ.get("DAD_LIGHTREG_REQUIRED_COMPLETED", "3"))
        if completed >= max_required:
            log("all runs completed; watcher exiting")
            return 0
        if active == 0:
            log("no active train processes before completion; watcher exiting with failure")
            return 2
        if args.once:
            return 0
        if time.time() - started >= args.max_minutes * 60:
            log("max watch window reached before completion; watcher exiting")
            return 3
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())

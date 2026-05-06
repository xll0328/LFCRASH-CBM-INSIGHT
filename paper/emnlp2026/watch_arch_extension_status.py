#!/usr/bin/env python3
"""Bounded watcher for architecture-extension support runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = ROOT / "output" / "emnlp2026_support"
SUMMARY_SCRIPT = ROOT / "paper" / "emnlp2026" / "summarize_arch_extension_status.py"
REFRESH_SCRIPT = ROOT / "paper" / "emnlp2026" / "refresh_emnlp_status.py"
STATUS_JSON = SUPPORT_DIR / "arch_extension_status.json"
LOG = SUPPORT_DIR / "watch_arch_extension_status.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    with LOG.open("a") as f:
        f.write(line)
    print(line, end="", flush=True)


def compact_output(text: str, max_lines: int = 3) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) <= max_lines:
        return " | ".join(lines)
    head = " | ".join(lines[:max_lines])
    return f"{head} | ... (+{len(lines) - max_lines} lines)"


def run_script(script: Path, label: str) -> int:
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no stderr/stdout"
        log(f"{label} failed: {detail}")
        return result.returncode

    out = compact_output(result.stdout)
    if out:
        log(f"{label}: {out}")
    return 0


def run_summary() -> int:
    return run_script(SUMMARY_SCRIPT, "summary")


def run_refresh() -> int:
    return run_script(REFRESH_SCRIPT, "refresh")


def snapshot_counts(status: dict) -> tuple[int, int, int, int]:
    dad = status["blocks"]["dad_rwkv"]
    a3d = status["blocks"]["a3d_h384"]
    dad_done = int(dad["num_completed"])
    a3d_done = int(a3d["num_completed"])
    running = int(dad["num_running"]) + int(a3d["num_running"])
    failed = int(dad["num_failed"]) + int(a3d["num_failed"])
    return dad_done, a3d_done, running, failed


def expected_counts(status: dict) -> tuple[int, int]:
    dad = status["blocks"]["dad_rwkv"]
    a3d = status["blocks"]["a3d_h384"]
    return int(dad["num_expected"]), int(a3d["num_expected"])


def full_refresh_or_exit() -> int:
    rc = run_refresh()
    if rc != 0:
        log("refresh failed; watcher exiting")
    return rc


def load_status() -> dict:
    return json.loads(STATUS_JSON.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--max-minutes", type=int, default=24 * 60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--no-refresh-on-transition",
        action="store_true",
        help="Disable full refresh when completion/running counters change.",
    )
    args = parser.parse_args()

    start = time.time()
    log(
        "watch_arch_extension_status.py started "
        f"interval_sec={args.interval_sec} max_minutes={args.max_minutes}"
    )

    last_transition: tuple[int, int, int, int] | None = None
    while True:
        rc = run_summary()
        if rc != 0:
            return rc

        status = load_status()
        dad_total, a3d_total = expected_counts(status)
        dad_done, a3d_done, running, failed = snapshot_counts(status)
        transition = (dad_done, a3d_done, running, failed)

        log(
            f"dad={dad_done}/{dad_total} a3d={a3d_done}/{a3d_total} "
            f"running={running} failed={failed}"
        )

        if last_transition is None:
            last_transition = transition
        elif transition != last_transition:
            log(f"status transition: {last_transition} -> {transition}")
            if not args.no_refresh_on_transition:
                rc = full_refresh_or_exit()
                if rc != 0:
                    return rc
            last_transition = transition

        if dad_done >= dad_total and a3d_done >= a3d_total:
            log("both architecture-extension families completed; running final refresh")
            rc = full_refresh_or_exit()
            if rc != 0:
                return rc
            log("both architecture-extension families completed; watcher exiting")
            return 0
        if running == 0:
            log("no active runs before full completion; running final refresh")
            rc = full_refresh_or_exit()
            if rc != 0:
                return rc
            log("no active runs before full completion; watcher exiting with failure")
            return 2
        if args.once:
            return 0
        if time.time() - start >= args.max_minutes * 60:
            log("max watch window reached before completion; watcher exiting")
            return 3

        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())

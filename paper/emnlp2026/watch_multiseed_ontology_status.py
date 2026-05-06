#!/usr/bin/env python3
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFRESH = ROOT / "paper" / "emnlp2026" / "refresh_emnlp_status.py"
LOG = ROOT / "output" / "emnlp2026_support" / "watch_multiseed_ontology_status.log"
INTERVAL_SEC = 120


def log(message: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    with LOG.open("a") as f:
        f.write(line)
    print(line, end="")


def main():
    log("watch_multiseed_ontology_status.py started")
    while True:
        result = subprocess.run(
            ["python", str(REFRESH)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            log("refreshed EMNLP status audits")
        else:
            log(f"audit failed: {result.stderr.strip() or result.stdout.strip()}")
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()

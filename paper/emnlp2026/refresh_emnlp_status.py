#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = [
    ROOT / "paper" / "emnlp2026" / "audit_multiseed_ontology_runs.py",
    ROOT / "paper" / "emnlp2026" / "audit_a3d_headline_multiseed.py",
    ROOT / "paper" / "emnlp2026" / "visualize_experiment_portfolio.py",
    ROOT / "paper" / "emnlp2026" / "audit_emnlp_oral_readiness.py",
    ROOT / "paper" / "emnlp2026" / "run_top_conference_quality_audit.py",
    ROOT / "paper" / "emnlp2026" / "summarize_dad_curriculum_recovery.py",
    ROOT / "paper" / "emnlp2026" / "summarize_dad_hardening_status.py",
    ROOT / "paper" / "emnlp2026" / "summarize_dad_mechanism_lightreg_status.py",
    ROOT / "paper" / "emnlp2026" / "summarize_arch_extension_status.py",
    ROOT / "paper" / "emnlp2026" / "export_emnlp_status_snippets.py",
]


def main():
    for script in SCRIPTS:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            sys.stderr.write(result.stderr or result.stdout)
            raise SystemExit(result.returncode)
        if result.stdout:
            sys.stdout.write(result.stdout)


if __name__ == "__main__":
    main()

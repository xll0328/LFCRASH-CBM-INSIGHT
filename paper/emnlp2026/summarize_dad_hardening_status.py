#!/usr/bin/env python3
import json
import re
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = ROOT / "output" / "emnlp2026_support"

ORAL_AUDIT_JSON = SUPPORT_DIR / "oral_readiness_audit.json"
DAD_CURRICULUM_SYNC_JSON = SUPPORT_DIR / "dad_curriculum_sync_summary.json"
DAD_CORE_ABLATION_JSON = SUPPORT_DIR / "dad_core_ablation_summary.json"
DAD_TRIGGER_JSON = SUPPORT_DIR / "dad_trigger_compare_extended_summary.json"
MATCHED_FULL_DIR = ROOT / "output" / "dad_full_support_block"
EVAL_RE = re.compile(
    r"Ep\s+(?P<epoch>\d+)\s+EVAL\s+\|\s+AP=(?P<AP>\d+\.\d+)\s+"
    r"mTTA=(?P<mTTA>\d+\.\d+)s\s+TTA@R80=(?P<TTA_R80>\d+\.\d+)s\s+"
    r"P@R80=(?P<P_R80>\d+\.\d+)"
)


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def sec(x: float) -> str:
    return f"{x:.2f}s"


def summarize(values):
    if not values:
        return None
    return {"mean": mean(values), "std": pstdev(values), "n": len(values)}


def load_matched_full_runs():
    rows = []
    if not MATCHED_FULL_DIR.exists():
        return rows
    for path in sorted(MATCHED_FULL_DIR.glob("insight_journal_dad_full_r*/results.json")):
        data = json.loads(path.read_text())
        rows.append(
            {
                "tag": path.parent.name,
                "path": str(path),
                "AP": data["AP"],
                "mTTA": data["mTTA"],
                "TTA_R80": data["TTA_R80"],
                "P_R80": data["P_R80"],
                "best_epoch": data.get("best_epoch"),
            }
        )
    return rows


def load_live_full_runs():
    rows = []
    if not MATCHED_FULL_DIR.exists():
        return rows

    for run_dir in sorted(MATCHED_FULL_DIR.glob("insight_journal_dad_full_r*")):
        if not run_dir.is_dir():
            continue
        if (run_dir / "results.json").exists():
            continue
        log_path = run_dir / "train.log"
        if not log_path.exists():
            continue

        latest_eval = None
        for line in log_path.read_text().splitlines():
            match = EVAL_RE.search(line)
            if not match:
                continue
            latest_eval = {
                "epoch": int(match.group("epoch")),
                "AP": float(match.group("AP")),
                "mTTA": float(match.group("mTTA")),
                "TTA_R80": float(match.group("TTA_R80")),
                "P_R80": float(match.group("P_R80")),
            }

        rows.append(
            {
                "tag": run_dir.name,
                "path": str(run_dir),
                "log_path": str(log_path),
                "latest_eval": latest_eval,
                "has_results_json": False,
            }
        )

    return rows


def compare_variant(full_agg, ablation_rows, variant):
    row = next((r for r in ablation_rows if r["variant"] == variant), None)
    if not row or not full_agg:
        return None
    agg = row["aggregate"]
    return {
        "variant": variant,
        "full_minus_variant": {
            "AP": full_agg["AP"]["mean"] - agg["AP"]["mean"],
            "mTTA": full_agg["mTTA"]["mean"] - agg["mTTA"]["mean"],
            "TTA_R80": full_agg["TTA_R80"]["mean"] - agg["TTA_R80"]["mean"],
            "P_R80": full_agg["P_R80"]["mean"] - agg["P_R80"]["mean"],
        },
        "variant_aggregate": agg,
    }


def aggregate_live_evals(live_rows):
    eval_rows = [row["latest_eval"] for row in live_rows if row.get("latest_eval")]
    if not eval_rows:
        return None

    epochs = {row["epoch"] for row in eval_rows}
    agg = {
        "AP": summarize([row["AP"] for row in eval_rows]),
        "mTTA": summarize([row["mTTA"] for row in eval_rows]),
        "TTA_R80": summarize([row["TTA_R80"] for row in eval_rows]),
        "P_R80": summarize([row["P_R80"] for row in eval_rows]),
        "n": len(eval_rows),
        "epoch_mode": min(epochs) if len(epochs) == 1 else None,
    }
    return agg


def main():
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)

    oral = load_json(ORAL_AUDIT_JSON) or {}
    sync = load_json(DAD_CURRICULUM_SYNC_JSON) or {}
    core = load_json(DAD_CORE_ABLATION_JSON) or {}
    trigger = load_json(DAD_TRIGGER_JSON) or {}
    strengths = oral.get("current_strengths", {})

    full_rows = load_matched_full_runs()
    live_rows = load_live_full_runs()
    full_agg = None
    if full_rows:
        full_agg = {
            "AP": summarize([r["AP"] for r in full_rows]),
            "mTTA": summarize([r["mTTA"] for r in full_rows]),
            "TTA_R80": summarize([r["TTA_R80"] for r in full_rows]),
            "P_R80": summarize([r["P_R80"] for r in full_rows]),
        }
    live_agg = aggregate_live_evals(live_rows)

    ablation_rows = core.get("rows", [])
    cmp_no_cbm = compare_variant(full_agg, ablation_rows, "no_cbm")
    cmp_no_align = compare_variant(full_agg, ablation_rows, "no_align")

    report = {
        "canonical_dad": strengths.get("canonical_dad"),
        "dad_seed_diagnostic": strengths.get("dad_seed_diagnostic"),
        "dad_sync_diagnostic": strengths.get("dad_sync_diagnostic"),
        "dad_trigger_diagnostic": strengths.get("dad_trigger_diagnostic"),
        "dad_core_ablation": core,
        "matched_full_support_block": {
            "path": str(MATCHED_FULL_DIR),
            "num_completed": len(full_rows),
            "rows": full_rows,
            "aggregate": full_agg,
            "in_progress_rows": live_rows,
            "in_progress_latest_eval_aggregate": live_agg,
        },
        "comparisons": {
            "full_vs_no_cbm": cmp_no_cbm,
            "full_vs_no_align": cmp_no_align,
        },
    }

    json_path = SUPPORT_DIR / "dad_hardening_status.json"
    md_path = SUPPORT_DIR / "dad_hardening_status.md"
    json_path.write_text(json.dumps(report, indent=2))

    lines = [
        "# DAD Hardening Status",
        "",
        "This status board tracks the DAD-side evidence that matters most for",
        "the EMNLP oral-to-best-paper push.",
        "",
    ]

    canonical = strengths.get("canonical_dad")
    if canonical:
        lines += [
            "## Canonical Headline",
            "",
            f"- Canonical DAD line: `{pct(canonical['AP'])}` AP, `{sec(canonical['mTTA'])}` mTTA",
            f"- Source tag: `{canonical.get('tag', '--')}` at epoch `{canonical.get('epoch', '--')}`",
            "",
        ]

    seed_diag = strengths.get("dad_seed_diagnostic")
    if seed_diag:
        lines += [
            "## Seed Diagnostics",
            "",
            f"- Clean-seed diagnostic: `n={seed_diag['AP']['n']}`, "
            f"`{pct(seed_diag['AP']['mean'])} +- {100.0 * seed_diag['AP']['std']:.2f}` AP, "
            f"`{sec(seed_diag['mTTA']['mean'])} +- {seed_diag['mTTA']['std']:.2f}s` mTTA",
            "",
        ]

    clean_block = sync.get("clean_seed_block", {}).get("aggregate", {})
    if clean_block:
        epoch25 = clean_block.get("25")
        epoch30 = clean_block.get("30")
        lines += ["## Synchronized Epoch View", ""]
        if epoch25:
            lines.append(
                f"- Epoch 25 aggregate: `{pct(epoch25['AP']['mean'])} +- {100.0 * epoch25['AP']['std']:.2f}` AP, "
                f"`{sec(epoch25['mTTA']['mean'])} +- {epoch25['mTTA']['std']:.2f}s` mTTA"
            )
        if epoch30:
            lines.append(
                f"- Epoch 30 aggregate: `{pct(epoch30['AP']['mean'])} +- {100.0 * epoch30['AP']['std']:.2f}` AP, "
                f"`{sec(epoch30['mTTA']['mean'])} +- {epoch30['mTTA']['std']:.2f}s` mTTA"
            )
        lines.append("")

    trig = strengths.get("dad_trigger_diagnostic", {}).get("aggregate", {})
    if trig:
        lines += [
            "## Trigger-Source Boundary",
            "",
            f"- Classifier trigger aggregate: `{pct(trig['classifier_AP']['mean'])} +- {100.0 * trig['classifier_AP']['std']:.2f}` AP, "
            f"`{sec(trig['classifier_mTTA']['mean'])} +- {trig['classifier_mTTA']['std']:.2f}s` mTTA",
            f"- Actor trigger aggregate: `{pct(trig['actor_AP']['mean'])} +- {100.0 * trig['actor_AP']['std']:.2f}` AP, "
            f"`{sec(trig['actor_mTTA']['mean'])} +- {trig['actor_mTTA']['std']:.2f}s` mTTA",
            f"- Actor minus classifier AP delta: `{100.0 * trig['delta_AP']['mean']:.2f}% +- {100.0 * trig['delta_AP']['std']:.2f}`",
            "",
        ]

    if ablation_rows:
        lines += [
            "## Current Support Ablations",
            "",
            "| Variant | AP mean+-std | mTTA mean+-std | TTA@R80 mean+-std | P@R80 mean+-std | n |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in ablation_rows:
            agg = row["aggregate"]
            lines.append(
                f"| {row['variant']} | "
                f"{pct(agg['AP']['mean'])} +- {100.0 * agg['AP']['std']:.2f} | "
                f"{sec(agg['mTTA']['mean'])} +- {agg['mTTA']['std']:.2f}s | "
                f"{sec(agg['TTA_R80']['mean'])} +- {agg['TTA_R80']['std']:.2f}s | "
                f"{agg['P_R80']['mean']:.3f} +- {agg['P_R80']['std']:.3f} | "
                f"{agg['AP']['n']} |"
            )
        lines.append("")

    lines += [
        "## Matched Full Support Block",
        "",
        f"- Output directory: `{MATCHED_FULL_DIR}`",
        f"- Completed full runs: `{len(full_rows)}/3`",
    ]
    if full_agg:
        lines += [
            f"- Full aggregate: `{pct(full_agg['AP']['mean'])} +- {100.0 * full_agg['AP']['std']:.2f}` AP, "
            f"`{sec(full_agg['mTTA']['mean'])} +- {full_agg['mTTA']['std']:.2f}s` mTTA",
        ]
    else:
        lines += [
            "- No matched `full` support runs are available yet.",
            "- This is the main missing DAD-side evidence block for the current best-paper push.",
        ]
    lines.append("")

    if live_rows:
        lines += [
            "## In-Progress Snapshot",
            "",
            f"- Active in-progress runs detected: `{len(live_rows)}`",
        ]
        if live_agg:
            epoch_text = (
                f"epoch `{live_agg['epoch_mode']}`"
                if live_agg["epoch_mode"] is not None
                else "mixed latest eval epochs"
            )
            lines.append(
                f"- Latest eval aggregate at {epoch_text}: "
                f"`{pct(live_agg['AP']['mean'])} +- {100.0 * live_agg['AP']['std']:.2f}` AP, "
                f"`{sec(live_agg['mTTA']['mean'])} +- {live_agg['mTTA']['std']:.2f}s` mTTA, "
                f"`{sec(live_agg['TTA_R80']['mean'])} +- {live_agg['TTA_R80']['std']:.2f}s` TTA@R80, "
                f"`{live_agg['P_R80']['mean']:.3f} +- {live_agg['P_R80']['std']:.3f}` P@R80"
            )
        lines.append("")
        lines += [
            "| Run | Latest eval | AP | mTTA | TTA@R80 | P@R80 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in live_rows:
            latest_eval = row.get("latest_eval")
            if latest_eval:
                lines.append(
                    f"| {row['tag']} | {latest_eval['epoch']} | {pct(latest_eval['AP'])} | "
                    f"{sec(latest_eval['mTTA'])} | {sec(latest_eval['TTA_R80'])} | {latest_eval['P_R80']:.3f} |"
                )
            else:
                lines.append(f"| {row['tag']} | -- | -- | -- | -- | -- |")
        lines.append("")

    if cmp_no_cbm or cmp_no_align:
        lines += ["## Full-vs-Ablation Comparison", ""]
        if cmp_no_cbm:
            delta = cmp_no_cbm["full_minus_variant"]
            lines.append(
                f"- Full minus `no_cbm`: `{100.0 * delta['AP']:.2f}%` AP, `{delta['mTTA']:.2f}s` mTTA, "
                f"`{delta['TTA_R80']:.2f}s` TTA@R80, `{delta['P_R80']:.3f}` P@R80"
            )
        if cmp_no_align:
            delta = cmp_no_align["full_minus_variant"]
            lines.append(
                f"- Full minus `no_align`: `{100.0 * delta['AP']:.2f}%` AP, `{delta['mTTA']:.2f}s` mTTA, "
                f"`{delta['TTA_R80']:.2f}s` TTA@R80, `{delta['P_R80']:.3f}` P@R80"
            )
        lines.append("")

    lines += [
        "## Reading",
        "",
        "- A3D is still the cleaner flagship result; DAD remains the harder stress test.",
        "- The actor branch remains support evidence rather than a safe headline claim.",
    ]
    if len(full_rows) >= 3:
        lines += [
            "- The matched `full` DAD support block is complete, so the current gap is no longer missing coverage.",
            "- Full-vs-ablation deltas show that DAD mechanism evidence is mixed; this supports the current paper stance that DAD is a stress test rather than the flagship mechanism win.",
            "- The next high-value compute action would be a targeted DAD mechanism-hardening block, not another ontology search or a repeat of the completed full block.",
        ]
    else:
        lines += [
            "- The current DAD support family is asymmetric until the matched `full` block is completed.",
            "- The next high-value compute action is therefore the missing 3-run `full` DAD support block, not more ontology search.",
        ]
    lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

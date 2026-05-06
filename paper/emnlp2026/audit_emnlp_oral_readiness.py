#!/usr/bin/env python3
import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "emnlp2026_support"


def load_json(path: Path):
    return json.loads(path.read_text())


def maybe_load(path: Path):
    return load_json(path) if path.exists() else None


def maybe_load_first(paths):
    for path in paths:
        data = maybe_load(path)
        if data is not None:
            return data, path
    return None, None


def pct(value):
    return f"{100.0 * value:.2f}%"


def metric_summary(values):
    if not values:
        return None
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {"mean": mean(values), "std": pstdev(values), "n": len(values)}


def collect_seed_results(paths):
    rows = []
    for path in sorted(paths):
        data = load_json(path)
        rows.append(
            {
                "path": str(path),
                "tag": data.get("tag", path.parent.name),
                "AP": data["AP"],
                "mTTA": data["mTTA"],
                "epoch": data.get("epoch"),
            }
        )
    return rows


def build_report():
    controlled = maybe_load(OUT_DIR / "controlled_ontology_status.json")
    ontology_multiseed = maybe_load(OUT_DIR / "multiseed_ontology_status.json")
    a3d_headline_multiseed = maybe_load(OUT_DIR / "a3d_headline_multiseed_status.json")
    topm, topm_path = maybe_load_first(
        [
            OUT_DIR / "topm_pseudolabel_sensitivity_dad500.json",
            OUT_DIR / "topm_pseudolabel_sensitivity_dad120.json",
        ]
    )
    verbal, verbal_path = maybe_load_first(
        [
            OUT_DIR / "concept_verbalization_sensitivity_dad500.json",
            OUT_DIR / "concept_verbalization_sensitivity_dad120.json",
        ]
    )
    human_audit, human_audit_path = maybe_load_first(
        [
            OUT_DIR / "human_ontology_audit_summary.json",
        ]
    )
    dad_sync, dad_sync_path = maybe_load_first(
        [
            OUT_DIR / "dad_curriculum_sync_summary.json",
        ]
    )
    dad_trigger, dad_trigger_path = maybe_load_first(
        [
            OUT_DIR / "dad_trigger_compare_extended_summary.json",
        ]
    )
    a3d_ablation, a3d_ablation_path = maybe_load_first(
        [
            OUT_DIR / "a3d_core_ablation_summary.json",
        ]
    )
    dad_seed_rows = collect_seed_results((ROOT / "output" / "dad_curriculum").glob("dad_curriculum_s*_clean/results.json"))
    dad_seed_ap = metric_summary([row["AP"] for row in dad_seed_rows])
    dad_seed_mtta = metric_summary([row["mTTA"] for row in dad_seed_rows])

    canonical_dad = load_json(ROOT / "output" / "dad_curriculum" / "dad_curriculum_v2" / "results.json")
    canonical_a3d = load_json(ROOT / "output" / "sota_push" / "a3d_sota" / "results.json")

    must_do = []
    ontology_multiseed_complete = False
    ontology_multiseed_completed = 0
    ontology_multiseed_running = 0
    ontology_multiseed_queued = 0
    ontology_multiseed_expected = 0
    if ontology_multiseed:
        ontology_multiseed_completed = sum(row["num_completed"] for row in ontology_multiseed["rows"])
        ontology_multiseed_running = sum(row.get("num_in_progress", 0) for row in ontology_multiseed["rows"])
        ontology_multiseed_queued = sum(row.get("num_queued", 0) for row in ontology_multiseed["rows"])
        ontology_multiseed_expected = sum(row["num_expected"] for row in ontology_multiseed["rows"])
        ontology_multiseed_complete = all(
            row["num_completed"] >= row["num_expected"] for row in ontology_multiseed["rows"]
        )
    a3d_headline_complete = (
        a3d_headline_multiseed is not None
        and a3d_headline_multiseed["num_completed"] >= a3d_headline_multiseed["num_expected"]
    )

    if controlled is None:
        must_do.append("Recover the controlled ontology manifest audit before any further claim hardening.")
    elif not ontology_multiseed_complete:
        if ontology_multiseed_completed + ontology_multiseed_running + ontology_multiseed_queued >= ontology_multiseed_expected:
            must_do.append(
                "Let the queued controlled-ontology multi-seed block finish, then refresh the mean/std aggregates."
            )
        else:
            must_do.append(
                "Upgrade the controlled ontology block from one seed per cell to >=3 seeds per cell on DAD and A3D."
            )
    if not a3d_headline_complete:
        must_do.append("Run 3-5 seeds for the A3D headline line and report mean/std for AP and mTTA.")
    if dad_sync is None:
        must_do.append("Refresh DAD curriculum stability with synchronized-epoch reporting and explicit variance.")

    should_do = []
    support_frames = max(
        topm.get("num_frames", 0) if topm else 0,
        verbal.get("num_frames", 0) if verbal else 0,
    )
    if support_frames < 500:
        should_do.append(
            "Expand pseudo-label and concept-verbalization audits beyond the current support-frame budget."
        )
    if human_audit is None:
        should_do.append("Add a lightweight human ontology audit over canonical names, merge choices, and family coverage.")
    if a3d_ablation is None:
        should_do.append("Surface the existing A3D support ablations so the cleanest mechanism evidence is not carried only by DAD.")
    should_do.append("If more compute is spent, prioritize one DAD-side hardening block that directly addresses mechanism fragility rather than more ontology search.")
    should_do.append("Keep the paper's mechanism emphasis on the cleaner A3D operating-point evidence, with DAD positioned as the harder stress test.")

    avoid = [
        "Do not reopen a large architecture rewrite before the seed story is hardened.",
        "Do not spend immediate compute on crash-only leaderboard chasing.",
        "Do not promote the actor branch to a main claim unless a new block materially changes its behavior.",
    ]

    oral_ready = ontology_multiseed_complete and a3d_headline_complete and dad_seed_ap is not None

    report = {
        "generated_at": str(Path(__file__).resolve()),
        "verdict": {
            "arr_ready": True,
            "oral_ready": oral_ready,
            "best_paper_ready": False,
            "summary": (
                "The project is submission-ready for ARR/EMNLP, but the current evidence depth is "
                + (
                    "not yet strong enough for oral-level confidence."
                    if not oral_ready
                    else "now materially stronger on seed-backed evidence, though best-paper-level confidence still requires stronger breadth."
                )
            ),
        },
        "current_strengths": {
            "controlled_ontology_block_complete": controlled["num_completed"] == controlled["num_runs"] if controlled else False,
            "controlled_ontology_multiseed_complete": ontology_multiseed_complete,
            "controlled_ontology_multiseed": ontology_multiseed,
            "canonical_dad": canonical_dad,
            "canonical_a3d": canonical_a3d,
            "a3d_headline_multiseed": a3d_headline_multiseed,
            "dad_seed_diagnostic": {
                "rows": dad_seed_rows,
                "AP": dad_seed_ap,
                "mTTA": dad_seed_mtta,
            },
            "dad_sync_diagnostic": {
                "available": dad_sync is not None,
                "path": str(dad_sync_path) if dad_sync_path else None,
                "clean_seed_block": dad_sync.get("clean_seed_block") if dad_sync else None,
                "extended_seed_block": dad_sync.get("extended_seed_block") if dad_sync else None,
            },
            "dad_trigger_diagnostic": {
                "available": dad_trigger is not None,
                "path": str(dad_trigger_path) if dad_trigger_path else None,
                "aggregate": dad_trigger.get("aggregate") if dad_trigger else None,
            },
            "a3d_ablation_diagnostic": {
                "available": a3d_ablation is not None,
                "path": str(a3d_ablation_path) if a3d_ablation_path else None,
                "rows": a3d_ablation.get("rows") if a3d_ablation else None,
            },
            "language_support_available": topm is not None and verbal is not None,
            "language_support": {
                "topm_path": str(topm_path) if topm_path else None,
                "verbal_path": str(verbal_path) if verbal_path else None,
                "support_frames": support_frames,
                "top3_family_diversity": (
                    topm.get("topm_summary", {}).get("3", {}).get("avg_family_diversity_per_frame")
                    if topm
                    else None
                ),
                "top3_score_mass_vs_top20": (
                    topm.get("topm_summary", {}).get("3", {}).get("avg_relative_score_mass_vs_top20")
                    if topm
                    else None
                ),
                "verbal_mean_text_cosine": (
                    verbal.get("aggregate", {}).get("mean_text_cosine") if verbal else None
                ),
                "verbal_mean_frame_score_correlation": (
                    verbal.get("aggregate", {}).get("mean_frame_score_correlation") if verbal else None
                ),
                "verbal_mean_abs_score_diff": (
                    verbal.get("aggregate", {}).get("mean_abs_score_diff") if verbal else None
                ),
            },
            "human_ontology_audit": {
                "available": human_audit is not None,
                "path": str(human_audit_path) if human_audit_path else None,
                "num_reviewed_concepts": human_audit.get("num_reviewed_concepts") if human_audit else None,
                "num_families": human_audit.get("num_families") if human_audit else None,
                "review_counts": human_audit.get("review_counts") if human_audit else None,
            },
        },
        "must_do": must_do,
        "should_do": should_do,
        "avoid": avoid,
    }
    return report


def write_markdown(report):
    lines = [
        "# EMNLP Oral Readiness Audit",
        "",
        f"- ARR-ready: `{report['verdict']['arr_ready']}`",
        f"- Oral-ready: `{report['verdict']['oral_ready']}`",
        f"- Best-paper-ready: `{report['verdict']['best_paper_ready']}`",
        f"- Summary: {report['verdict']['summary']}",
        "",
        "## Current Evidence",
        "",
        f"- Canonical DAD line: `{pct(report['current_strengths']['canonical_dad']['AP'])}` AP, "
        f"`{report['current_strengths']['canonical_dad']['mTTA']:.2f}s` mTTA",
        f"- Canonical A3D line: `{pct(report['current_strengths']['canonical_a3d']['AP'])}` AP, "
        f"`{report['current_strengths']['canonical_a3d']['mTTA']:.2f}s` mTTA",
        f"- Controlled ontology block complete: `{report['current_strengths']['controlled_ontology_block_complete']}`",
        f"- Controlled ontology multi-seed complete: `{report['current_strengths']['controlled_ontology_multiseed_complete']}`",
    ]

    ontology_multiseed = report["current_strengths"].get("controlled_ontology_multiseed")
    if ontology_multiseed:
        completed = sum(row["num_completed"] for row in ontology_multiseed["rows"])
        running = sum(row.get("num_in_progress", 0) for row in ontology_multiseed["rows"])
        queued = sum(row.get("num_queued", 0) for row in ontology_multiseed["rows"])
        expected = sum(row["num_expected"] for row in ontology_multiseed["rows"])
        suffix = []
        if running:
            suffix.append(f"running={running}")
        if queued:
            suffix.append(f"queued={queued}")
        detail = f" ({'; '.join(suffix)})" if suffix else ""
        lines.append(f"- Controlled ontology multi-seed coverage: `{completed}/{expected}` seeded cells completed{detail}")

    dad_ap = report["current_strengths"]["dad_seed_diagnostic"]["AP"]
    dad_mtta = report["current_strengths"]["dad_seed_diagnostic"]["mTTA"]
    if dad_ap and dad_mtta:
        lines += [
            f"- DAD clean-seed diagnostic: `n={dad_ap['n']}`, "
            f"AP mean/std = `{pct(dad_ap['mean'])} +/- {100.0 * dad_ap['std']:.2f}`, "
            f"mTTA mean/std = `{dad_mtta['mean']:.2f}s +/- {dad_mtta['std']:.2f}s`",
        ]
    dad_sync = report["current_strengths"].get("dad_sync_diagnostic") or {}
    if dad_sync.get("available"):
        clean_block = dad_sync.get("clean_seed_block") or {}
        clean_agg = clean_block.get("aggregate") or {}
        epoch25 = clean_agg.get("25")
        epoch30 = clean_agg.get("30")
        if epoch25:
            lines.append(
                f"- DAD synchronized epoch 25: AP `{pct(epoch25['AP']['mean'])} +/- {100.0 * epoch25['AP']['std']:.2f}`, "
                f"mTTA `{epoch25['mTTA']['mean']:.2f}s +/- {epoch25['mTTA']['std']:.2f}s`"
            )
        if epoch30:
            lines.append(
                f"- DAD synchronized epoch 30: AP `{pct(epoch30['AP']['mean'])} +/- {100.0 * epoch30['AP']['std']:.2f}`, "
                f"mTTA `{epoch30['mTTA']['mean']:.2f}s +/- {epoch30['mTTA']['std']:.2f}s`"
            )
        extended_block = dad_sync.get("extended_seed_block") or {}
        extended_agg = extended_block.get("aggregate") or {}
        extended25 = extended_agg.get("25")
        if extended25:
            lines.append(
                f"- DAD extended same-family epoch 25: AP `{pct(extended25['AP']['mean'])} +/- {100.0 * extended25['AP']['std']:.2f}`, "
                f"mTTA `{extended25['mTTA']['mean']:.2f}s +/- {extended25['mTTA']['std']:.2f}s` over "
                f"`n={extended25['AP']['n']}` seeds"
            )
    a3d_multiseed = report["current_strengths"].get("a3d_headline_multiseed")
    if a3d_multiseed and a3d_multiseed.get("aggregate_completed"):
        agg = a3d_multiseed["aggregate_completed"]
        progress_bits = []
        if a3d_multiseed.get("num_in_progress"):
            progress_bits.append(f"in-progress={a3d_multiseed['num_in_progress']}")
        if a3d_multiseed.get("missing_seeds"):
            progress_bits.append(
                "missing=" + ",".join(str(seed) for seed in a3d_multiseed["missing_seeds"])
            )
        suffix = f" ({'; '.join(progress_bits)})" if progress_bits else ""
        lines += [
            f"- A3D headline seed status: `n={a3d_multiseed['num_completed']}/{a3d_multiseed['num_expected']}`, "
            f"AP mean/std = `{pct(agg['AP']['mean'])} +/- {100.0 * agg['AP']['std']:.2f}`, "
            f"mTTA mean/std = `{agg['mTTA']['mean']:.2f}s +/- {agg['mTTA']['std']:.2f}s`{suffix}",
        ]
        if a3d_multiseed.get("aggregate_best_so_far") and a3d_multiseed.get("num_in_progress"):
            current = a3d_multiseed["aggregate_best_so_far"]
            lines += [
                f"- A3D headline current best-so-far across available seeds: "
                f"`{pct(current['AP']['mean'])}` AP, `{current['mTTA']['mean']:.2f}s` mTTA",
            ]
    lines += [
        f"- Language-side support artifacts present: `{report['current_strengths']['language_support_available']}`",
    ]
    language_support = report["current_strengths"].get("language_support") or {}
    if language_support.get("support_frames"):
        lines.append(
            f"- Language-side support frame budget: `{language_support['support_frames']}` frames"
        )
    if language_support.get("top3_family_diversity") is not None:
        lines.append(
            f"- Top-3 pseudo-label audit: family diversity `{language_support['top3_family_diversity']:.3f}` "
            f"with relative score mass `{language_support['top3_score_mass_vs_top20']:.3f}`"
        )
    if language_support.get("verbal_mean_text_cosine") is not None:
        lines.append(
            f"- Concept verbalization audit: text cosine `{language_support['verbal_mean_text_cosine']:.3f}`, "
            f"frame-score correlation `{language_support['verbal_mean_frame_score_correlation']:.3f}`, "
            f"mean abs diff `{language_support['verbal_mean_abs_score_diff']:.4f}`"
        )
    human_audit = report["current_strengths"].get("human_ontology_audit") or {}
    if human_audit.get("available"):
        lines.append(
            f"- Human ontology audit: `{human_audit['num_reviewed_concepts']}` reviewed concepts across "
            f"`{human_audit['num_families']}` families"
        )
    dad_trigger = report["current_strengths"].get("dad_trigger_diagnostic") or {}
    if dad_trigger.get("available"):
        agg = dad_trigger.get("aggregate") or {}
        if agg.get("classifier_AP") and agg.get("actor_AP"):
            lines.append(
                f"- DAD trigger-source extended audit: classifier `{pct(agg['classifier_AP']['mean'])}` AP vs "
                f"actor `{pct(agg['actor_AP']['mean'])}` AP over `n={agg['classifier_AP']['n']}` checkpoints"
            )
    a3d_ablation = report["current_strengths"].get("a3d_ablation_diagnostic") or {}
    if a3d_ablation.get("available") and a3d_ablation.get("rows"):
        rows = {row["variant"]: row for row in a3d_ablation["rows"]}
        full = rows.get("full")
        no_cbm = rows.get("no_cbm")
        no_align = rows.get("no_align")
        if full and no_cbm:
            lines.append(
                f"- A3D support ablation: removing CBM shifts AP from `{pct(full['AP'])}` to `{pct(no_cbm['AP'])}` "
                f"and mTTA from `{full['mTTA']:.2f}s` to `{no_cbm['mTTA']:.2f}s`"
            )
        if full and no_align:
            lines.append(
                f"- A3D support ablation: removing alignment raises AP to `{pct(no_align['AP'])}` but reduces "
                f"mTTA / TTA@R80 to `{no_align['mTTA']:.2f}s / {no_align['TTA_R80']:.2f}s` "
                f"from `{full['mTTA']:.2f}s / {full['TTA_R80']:.2f}s`"
            )
    lines += [
        "",
        "## Must Do",
        "",
    ]
    if report["must_do"]:
        for item in report["must_do"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None. The current blockers are now strategic rather than missing-support artifacts.")
    lines += [
        "",
        "## Should Do",
        "",
    ]
    for item in report["should_do"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Avoid",
        "",
    ]
    for item in report["avoid"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = OUT_DIR / "oral_readiness_audit.json"
    md_path = OUT_DIR / "oral_readiness_audit.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(write_markdown(report))
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

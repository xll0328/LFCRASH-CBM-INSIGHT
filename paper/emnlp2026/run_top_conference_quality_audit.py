#!/usr/bin/env python3
"""Audit whether the current EMNLP package looks like a top-conference paper."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EMNLP = ROOT / "paper" / "emnlp2026"
APPENDIX = ROOT / "paper" / "neurips2026" / "sec_appendix.tex"
SUPPORT = ROOT / "output" / "emnlp2026_support"
JSON_OUT = SUPPORT / "top_conference_quality_audit.json"
MD_OUT = ROOT / "EMNLP_TOP_CONFERENCE_QUALITY_AUDIT_20260427.md"

MAIN_TEX = [
    EMNLP / "insight_emnlp.tex",
    EMNLP / "sec_intro_emnlp.tex",
    EMNLP / "sec_related_emnlp.tex",
    EMNLP / "sec_method_emnlp.tex",
    EMNLP / "sec_experiments_emnlp.tex",
    EMNLP / "sec_conclusion_emnlp.tex",
]


def read(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite[a-zA-Z*]*\{([^}]+)\}", text):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,\s]+)", text))


def figure_paths(text: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", text)


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def sec(x: float) -> str:
    return f"{x:.2f}s"


def main() -> int:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    main_text = "\n".join(read(path) for path in MAIN_TEX)
    appendix_text = read(APPENDIX)
    all_text = main_text + "\n" + appendix_text
    bib_text = read(EMNLP / "insight.bib")

    bib_key_set = bib_keys(bib_text)
    cited_key_set = citation_keys(all_text)
    main_cited_key_set = citation_keys(main_text)
    missing_bib = sorted(cited_key_set - bib_key_set)
    unused_bib = sorted(bib_key_set - cited_key_set)

    stats = {
        "main_tex_lines": sum(read(path).count("\n") + 1 for path in MAIN_TEX),
        "appendix_lines": appendix_text.count("\n") + 1,
        "bib_entries": len(bib_key_set),
        "citation_commands": count_pattern(all_text, r"\\cite[a-zA-Z*]*\{"),
        "unique_cited_keys_all": len(cited_key_set),
        "unique_cited_keys_main": len(main_cited_key_set),
        "missing_bib_keys": missing_bib,
        "unused_bib_keys": unused_bib,
        "main_figures": count_pattern(main_text, r"\\begin\{figure\*?\}"),
        "appendix_figures": count_pattern(appendix_text, r"\\begin\{figure\*?\}"),
        "main_tables": count_pattern(main_text, r"\\begin\{table\*?\}"),
        "appendix_tables": count_pattern(appendix_text, r"\\begin\{table\*?\}"),
        "main_includegraphics": figure_paths(main_text),
        "appendix_includegraphics_count": len(figure_paths(appendix_text)),
    }

    oral = load_json(SUPPORT / "oral_readiness_audit.json") or {}
    a3d = load_json(SUPPORT / "a3d_headline_multiseed_status.json") or {}
    dad_light = load_json(SUPPORT / "dad_mechanism_lightreg_status.json") or {}
    ontology = load_json(SUPPORT / "multiseed_ontology_status.json") or {}

    dad_started = dad_light.get("num_started")
    dad_completed = dad_light.get("num_completed")

    observations = {
        "arr_ready": oral.get("arr_ready"),
        "oral_ready": oral.get("oral_ready"),
        "best_paper_ready": oral.get("best_paper_ready"),
        "a3d_completed": a3d.get("num_completed") or a3d.get("completed"),
        "dad_lightreg_started": dad_started,
        "dad_lightreg_completed": dad_completed,
        "dad_lightreg_latest_eval_snapshot": dad_light.get("latest_eval_snapshot"),
        "dad_lightreg_completed_aggregate": dad_light.get("completed_aggregate"),
        "ontology_status_available": bool(ontology),
    }

    figure1_bridge_ready = (
        re.search(
            r"risk trajectory,\s+concept activations,\s+and alert\s+marker",
            main_text,
        )
        and "risk trace points to headline classifier evidence" in main_text
        and "timing diagnostics rather than deployment-level policy validation" in main_text
    )
    observations["figure1_bridge_ready"] = figure1_bridge_ready

    if dad_completed == 3:
        best_paper_critical_finding = (
            "The current package is not best-paper-ready: although the DAD "
            "light-regularization block is now complete at 3/3 runs, aggregate "
            "DAD mechanism evidence remains mixed."
        )
        best_paper_critical_fix = (
            "Use the completed 3-run aggregate only as bounded stress evidence; "
            "either keep oral-ready framing or add new independent evidence "
            "before any best-paper claim."
        )
        queue_item_1 = (
            "Treat the completed DAD light-reg 3/3 aggregate as support-only; "
            "do not upgrade DAD claims beyond stress-test framing."
        )
    else:
        best_paper_critical_finding = (
            "The current package is not best-paper-ready: DAD mechanism evidence "
            "remains mixed and the active light-regularization block has not "
            "completed 3/3 runs."
        )
        best_paper_critical_fix = (
            "Wait for the predeclared 3-run aggregate, then either update only "
            "bounded DAD wording or explicitly leave the paper in oral-ready mode."
        )
        queue_item_1 = (
            "Keep the current DAD light-reg block running to completion; do not "
            "update paper claims until 3/3 aggregate exists."
        )

    if figure1_bridge_ready:
        visualization_finding = (
            "The visual story is improving but still not best-paper-level on "
            "first read. The concept-pipeline figure is repaired, the "
            "safety-utility plot exposes three-seed ontology intervals, the "
            "framework figure centers the interface, and Figure 1 now "
            "explicitly maps its risk trace, concept activations, and alert "
            "marker to the protocol-separated evidence blocks."
        )
        visualization_fix = (
            "Preserve the Figure 1 bridge during final read-through; the next "
            "visual/story gap is broader decisiveness, not missing "
            "cross-reference scaffolding."
        )
        queue_item_2 = (
            "Figure 1 bridge is addressed in the current package; preserve the "
            "headline/ontology/timing-support mapping during final PDF "
            "read-through."
        )
    else:
        visualization_finding = (
            "The visual story is improving but still not best-paper-level on "
            "first read. The concept-pipeline figure is repaired, the "
            "safety-utility plot exposes three-seed ontology intervals, the "
            "framework figure centers the interface, and Figure 1 has been "
            "compacted. The remaining gap is a sharper first-glance bridge "
            "from motivated example to quantitative evidence."
        )
        visualization_fix = (
            "Keep Figure 1 compact and strengthen explicit cross-references "
            "from Figure 1 to the headline/support evidence blocks."
        )
        queue_item_2 = (
            "Strengthen explicit cross-references from Figure 1 to headline/"
            "support evidence so first-read intuition lands on quantitative "
            "blocks."
        )

    findings = [
        {
            "severity": "CRITICAL",
            "area": "Best-paper evidence",
            "finding": best_paper_critical_finding,
            "evidence": "EMNLP_ORAL_BEST_PAPER_GAP_LEDGER_20260427.md; output/emnlp2026_support/dad_mechanism_lightreg_status.md",
            "fix": best_paper_critical_fix,
        },
        {
            "severity": "MAJOR",
            "area": "Experiment breadth",
            "finding": "Main predictive evidence is concentrated on DAD and A3D; this can support ARR/oral framing but is thin for a decisive best-paper generality claim.",
            "evidence": "paper/emnlp2026/sec_experiments_emnlp.tex reports DAD and A3D as the main datasets.",
            "fix": "Add a predeclared independent stress setting only as a separate post-submission or revision block; do not claim broad generality from the current two-dataset package.",
        },
        {
            "severity": "MAJOR",
            "area": "Experimental coherence",
            "finding": "The paper relies on multiple recipe families: canonical headline, support protocol, ontology launcher, actor/classifier trigger comparison, and archived intervention assets. The protocol table plus visual evidence map improves readability, but reviewers may still read the package as fragmented unless claim tiers stay explicit.",
            "evidence": "Table protocol_map_main in sec_experiments_emnlp.tex.",
            "fix": "Make every table caption state its evidence tier and keep headline, support, and stress evidence visually separated.",
        },
        {
            "severity": "MAJOR",
            "area": "Visualization",
            "finding": visualization_finding,
            "evidence": "paper/figures/insight_framework.png; paper/figures/insight_fig5_safety_utility.png; paper/figures/insight_fig_concept_pipeline.png; paper/figures/insight_fig6_dad_stress_summary.pdf",
            "fix": visualization_fix,
        },
        {
            "severity": "MAJOR",
            "area": "Related work",
            "finding": f"The bibliography now has {stats['bib_entries']} entries and {stats['unique_cited_keys_main']} unique main-text cited keys, which is much healthier than the earlier draft but still not a best-paper-level map of ontology governance, concept bottlenecks, and safety-critical VLM evaluation.",
            "evidence": "paper/emnlp2026/insight.bib and main LaTeX citation scan.",
            "fix": "Add only targeted missing literature next: ontology/concept governance in NLP, safety-critical VLM evaluation, and human validation protocols.",
        },
        {
            "severity": "MAJOR",
            "area": "Human validation",
            "finding": "The ontology review is useful, but it is still light-touch review over concept entries rather than a human-verified frame-level concept benchmark.",
            "evidence": "output/emnlp2026_support/human_ontology_audit_summary.md; sec_conclusion_emnlp.tex limitations.",
            "fix": "Keep the claim as governed semantic interface; do not imply exhaustive human concept-label validation.",
        },
        {
            "severity": "MINOR",
            "area": "Paper polish",
            "finding": "The paper has many tables relative to narrative figures; several tables are dense and may slow first-read comprehension.",
            "evidence": f"Main body has {stats['main_tables']} tables and {stats['main_figures']} figures.",
            "fix": "Convert one dense table into a visual evidence map or move lower-tier support tables to appendix if page budget allows.",
        },
    ]

    scores = {
        "arr_submission": 93,
        "emnlp_accept": 80,
        "emnlp_oral": 65,
        "best_paper": 38,
    }

    report = {
        "generated_at": generated_at,
        "scores": scores,
        "stats": stats,
        "observations": observations,
        "findings": findings,
    }
    SUPPORT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2))

    lines = [
        "# EMNLP Top-Conference Quality Audit",
        "",
        f"- Generated at: `{generated_at}`",
        "- Scope: current LFCRASH-CBM EMNLP/ARR paper package and support artifacts.",
        "- Standard used: top-conference acceptance, oral, and best-paper competitiveness; not just submission sanity.",
        "",
        "## Bottom Line",
        "",
        "- ARR / regular acceptance package: `credible`.",
        "- Oral: `possible but not locked`; depends on reviewer buy-in to the semantic-interface framing.",
        "- Best paper: `not currently credible`; the gap is research evidence and visual/story decisiveness, not only polish.",
        "",
        "## Internal Scores",
        "",
        "| Target | Score | Reading |",
        "|---|---:|---|",
        f"| ARR submission | {scores['arr_submission']}/100 | technically coherent package |",
        f"| EMNLP accept | {scores['emnlp_accept']}/100 | plausible if claims stay bounded |",
        f"| EMNLP oral | {scores['emnlp_oral']}/100 | needs stronger first-read story and defense |",
        f"| Best paper | {scores['best_paper']}/100 | not enough decisive evidence yet |",
        "",
        "## Measured Package Statistics",
        "",
        f"- BibTeX entries: `{stats['bib_entries']}`",
        f"- Unique cited keys in main text: `{stats['unique_cited_keys_main']}`",
        f"- Citation commands across main+appendix: `{stats['citation_commands']}`",
        f"- Main-body figures: `{stats['main_figures']}`",
        f"- Main-body tables: `{stats['main_tables']}`",
        f"- Appendix figures/tables: `{stats['appendix_figures']}` / `{stats['appendix_tables']}`",
        f"- Missing BibTeX keys: `{len(missing_bib)}`",
        f"- Unused BibTeX keys: `{len(unused_bib)}`",
        "",
        "## Findings",
        "",
        "| Severity | Area | Finding | Required action |",
        "|---|---|---|---|",
    ]
    for item in findings:
        lines.append(
            f"| {item['severity']} | {item['area']} | {item['finding']} | {item['fix']} |"
        )

    snapshot = observations.get("dad_lightreg_latest_eval_snapshot")
    completed_aggregate = observations.get("dad_lightreg_completed_aggregate")
    lines += [
        "",
        "## Active DAD Light-Reg Block",
        "",
        f"- Started: `{observations.get('dad_lightreg_started')}/3`",
        f"- Completed: `{observations.get('dad_lightreg_completed')}/3`",
    ]
    if snapshot:
        lines += [
            "- Latest eval snapshot is monitoring only, not paper evidence.",
            f"- AP mean: `{pct(snapshot['AP']['mean'])}` over `n={snapshot['AP']['n']}`",
            f"- mTTA mean: `{sec(snapshot['mTTA']['mean'])}`",
        ]
    else:
        lines.append("- No latest eval snapshot available.")
    if completed_aggregate:
        lines += [
            "- Completed aggregate is available for support interpretation only.",
            f"- Completed AP mean: `{pct(completed_aggregate['AP']['mean'])}` over `n={completed_aggregate['AP']['n']}`",
            f"- Completed mTTA mean: `{sec(completed_aggregate['mTTA']['mean'])}`",
        ]

    lines += [
        "",
        "## Immediate Fix Queue",
        "",
        f"1. {queue_item_1}",
        f"2. {queue_item_2}",
        "3. Continue targeted related-work hardening around ontology governance, safety-critical VLM evaluation, and human validation protocols.",
        "4. Tighten experiment narration so single-run headline, seed-backed evidence, support ablations, and archived diagnostics cannot be confused.",
        "5. Keep human-validation language bounded unless a real frame-level concept annotation study is added.",
        "",
    ]
    MD_OUT.write_text("\n".join(lines))
    print(f"[wrote] {JSON_OUT}")
    print(f"[wrote] {MD_OUT}")
    print("[top-conference-quality] best_paper_ready=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

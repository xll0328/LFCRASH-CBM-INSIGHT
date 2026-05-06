#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = ROOT / "output" / "emnlp2026_support"
CONCEPT_DIR = ROOT / "output" / "concept_sets"


def load_json(path: Path):
    return json.loads(path.read_text())


def main():
    topm = load_json(SUPPORT_DIR / "topm_pseudolabel_sensitivity_dad500.json")
    verbal = load_json(SUPPORT_DIR / "concept_verbalization_sensitivity_dad500.json")
    human = load_json(SUPPORT_DIR / "human_ontology_audit_summary.json")
    polish = load_json(CONCEPT_DIR / "perfect_concept_set_v1.audit.json")

    top1 = topm["topm_summary"]["1"]
    top3 = topm["topm_summary"]["3"]
    top10 = topm["topm_summary"]["10"]

    summary = {
        "dataset_slice": "DAD-500 frame audit slice",
        "pseudo_label_topm": {
            "top1": {
                "family_diversity_per_frame": top1["avg_family_diversity_per_frame"],
                "relative_mass_vs_top20": top1["avg_relative_score_mass_vs_top20"],
            },
            "top3": {
                "family_diversity_per_frame": top3["avg_family_diversity_per_frame"],
                "relative_mass_vs_top20": top3["avg_relative_score_mass_vs_top20"],
            },
            "top10": {
                "family_diversity_per_frame": top10["avg_family_diversity_per_frame"],
                "relative_mass_vs_top20": top10["avg_relative_score_mass_vs_top20"],
            },
        },
        "paraphrase_stability": verbal["aggregate"],
        "ontology_governance": {
            "source_pool_size": polish["pre_polish_num_concepts"],
            "post_polish_size": polish["post_polish_num_concepts"],
            "reviewed_concepts": human["num_reviewed_concepts"],
            "released_concepts": human["num_concepts"],
            "families": human["num_families"],
            "review_policy": human["review_policy"],
        },
    }

    json_path = SUPPORT_DIR / "semantic_validity_summary.json"
    md_path = SUPPORT_DIR / "semantic_validity_summary.md"
    tex_path = SUPPORT_DIR / "semantic_validity_rows.tex"

    json_path.write_text(json.dumps(summary, indent=2))

    md_lines = [
        "# Semantic Validity Summary",
        "",
        f"- Slice: `{summary['dataset_slice']}`",
        "",
        "## Pseudo-Label Concentration (top-m)",
        "",
        "| Setting | Family diversity/frame | Relative score mass vs top-20 |",
        "|---|---:|---:|",
        (
            f"| top-1 | {top1['avg_family_diversity_per_frame']:.3f} | "
            f"{100.0 * top1['avg_relative_score_mass_vs_top20']:.2f}% |"
        ),
        (
            f"| top-3 (main) | {top3['avg_family_diversity_per_frame']:.3f} | "
            f"{100.0 * top3['avg_relative_score_mass_vs_top20']:.2f}% |"
        ),
        (
            f"| top-10 | {top10['avg_family_diversity_per_frame']:.3f} | "
            f"{100.0 * top10['avg_relative_score_mass_vs_top20']:.2f}% |"
        ),
        "",
        "## Canonical Name Stability",
        "",
        f"- Mean text cosine: `{verbal['aggregate']['mean_text_cosine']:.3f}`",
        f"- Mean frame-score correlation: `{verbal['aggregate']['mean_frame_score_correlation']:.3f}`",
        f"- Mean top-10 frame overlap: `{verbal['aggregate']['mean_top10_frame_overlap']:.3f}`",
        f"- Mean absolute score difference: `{verbal['aggregate']['mean_abs_score_diff']:.4f}`",
        "",
        "## Ontology Governance Coverage",
        "",
        f"- Source pool: `{polish['pre_polish_num_concepts']}` concepts",
        f"- Post-polish pool: `{polish['post_polish_num_concepts']}` concepts",
        f"- Reviewed concepts: `{human['num_reviewed_concepts']}/{human['num_concepts']}`",
        f"- Families covered: `{human['num_families']}`",
        f"- Review mode: `{human['review_policy']}`",
        "",
    ]
    md_path.write_text("\n".join(md_lines))

    tex_lines = [
        "% Auto-generated semantic validity rows",
        (
            f"Pseudo-label concentration (DAD-500, top-1 / top-3 / top-10) & "
            f"{top1['avg_family_diversity_per_frame']:.2f} / "
            f"{top3['avg_family_diversity_per_frame']:.2f} / "
            f"{top10['avg_family_diversity_per_frame']:.2f} families per frame; "
            f"{100.0 * top1['avg_relative_score_mass_vs_top20']:.1f}\\% / "
            f"{100.0 * top3['avg_relative_score_mass_vs_top20']:.1f}\\% / "
            f"{100.0 * top10['avg_relative_score_mass_vs_top20']:.1f}\\% top-20 mass \\\\"
        ),
        (
            f"Canonical paraphrase stability (12 concepts, 2 variants each) & "
            f"text cosine {verbal['aggregate']['mean_text_cosine']:.3f}, "
            f"frame correlation {verbal['aggregate']['mean_frame_score_correlation']:.3f}, "
            f"top-10 overlap {verbal['aggregate']['mean_top10_frame_overlap']:.3f}, "
            f"|$\\Delta$score| {verbal['aggregate']['mean_abs_score_diff']:.4f} \\\\"
        ),
        (
            f"Governance coverage and review traceability & "
            f"{polish['pre_polish_num_concepts']} $\\rightarrow$ {polish['post_polish_num_concepts']} "
            f"post-polish candidates; "
            f"{human['num_reviewed_concepts']}/{human['num_concepts']} released concepts reviewed "
            f"across {human['num_families']} families \\\\"
        ),
    ]
    tex_path.write_text("\n".join(tex_lines) + "\n")

    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")
    print(f"[wrote] {tex_path}")


if __name__ == "__main__":
    main()

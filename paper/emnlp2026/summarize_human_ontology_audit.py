#!/usr/bin/env python3
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONCEPT_DIR = ROOT / "output" / "concept_sets"
SUPPORT_DIR = ROOT / "output" / "emnlp2026_support"

REVIEW_MD = CONCEPT_DIR / "PER_CONCEPT_HUMAN_REVIEW.md"
META_JSON = CONCEPT_DIR / "perfect_concept_set_v1.meta.json"
AUDIT_JSON = CONCEPT_DIR / "perfect_concept_set_v1.audit.json"
MERGE_JSON = CONCEPT_DIR / "perfect_concept_set_v1.merge_examples.json"


def parse_review_markdown(path: Path):
    family = None
    entries = []

    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            family = line[3:].strip()
            i += 1
            continue
        m = re.match(r"^###\s+\d+\.\s+`(.+?)`$", line)
        if m:
            concept = m.group(1)
            decision = None
            reason = None
            j = i + 1
            while j < len(lines):
                cur = lines[j]
                if cur.startswith("### ") or cur.startswith("## "):
                    break
                if cur.startswith("- Decision:"):
                    decision = cur.split(":", 1)[1].strip()
                if cur.startswith("- Reason:"):
                    reason = cur.split(":", 1)[1].strip()
                j += 1
            entries.append(
                {
                    "family": family,
                    "concept": concept,
                    "decision": decision or "unknown",
                    "reason": reason or "",
                }
            )
            i = j
            continue
        i += 1

    return entries


def main():
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)

    meta = json.loads(META_JSON.read_text())
    audit = json.loads(AUDIT_JSON.read_text())
    merges = json.loads(MERGE_JSON.read_text())
    review_entries = parse_review_markdown(REVIEW_MD)

    by_family = defaultdict(list)
    decision_counts = defaultdict(int)
    for entry in review_entries:
        by_family[entry["family"]].append(entry)
        decision_counts[entry["decision"]] += 1

    family_rows = []
    for family, entries in sorted(by_family.items()):
        family_decisions = {
            k: sum(1 for e in entries if e["decision"] == k)
            for k in sorted({e["decision"] for e in entries})
        }
        family_rows.append(
            {
                "family": family,
                "num_concepts": len(entries),
                "decisions": family_decisions,
                "sample_concepts": [e["concept"] for e in entries[:3]],
            }
        )

    merge_rows = []
    for canonical, variants in list(merges.items())[:12]:
        merge_rows.append(
            {
                "canonical": canonical,
                "num_variants": len(variants),
                "sample_variants": variants[:4],
            }
        )

    summary = {
        "ontology_name": meta["name"],
        "num_concepts": meta["num_concepts"],
        "design_goal": meta["design_goal"],
        "review_policy": "very-light wording-level human review with ontology structure preserved",
        "review_counts": dict(sorted(decision_counts.items())),
        "num_reviewed_concepts": len(review_entries),
        "num_families": len(by_family),
        "family_balance_before": audit["pre_family_counts"],
        "family_balance_after": audit["post_family_counts"],
        "main_removed_noise_types": audit["summary"]["main_removed_noise_types"],
        "high_priority_cluster_summary": audit["high_priority_cluster_summary"],
        "family_rows": family_rows,
        "merge_rows": merge_rows,
    }

    json_path = SUPPORT_DIR / "human_ontology_audit_summary.json"
    md_path = SUPPORT_DIR / "human_ontology_audit_summary.md"
    json_path.write_text(json.dumps(summary, indent=2))

    lines = [
        "# Human Ontology Audit Summary",
        "",
        f"- Ontology: `{meta['name']}`",
        f"- Concepts reviewed: `{len(review_entries)}`",
        f"- Families: `{len(by_family)}`",
        f"- Review policy: `{summary['review_policy']}`",
        f"- Design goal: {meta['design_goal']}",
        "",
        "## Headline Reading",
        "",
        f"- The paper-facing ontology compresses `{audit['pre_polish_num_concepts']}` mined concepts into `{audit['post_polish_num_concepts']}` reviewed canonical concepts.",
        f"- The current review document contains `{len(review_entries)}` concept-level decisions across `{len(by_family)}` semantic families.",
        f"- Main noise removed during polishing: {', '.join(audit['summary']['main_removed_noise_types'])}.",
        "",
        "## Family Balance",
        "",
        "| Family | Before | After | Sample canonical concepts |",
        "|---|---:|---:|---|",
    ]

    for family in sorted(audit["post_family_counts"]):
        samples = ", ".join(next((row["sample_concepts"] for row in family_rows if row["family"] == family), [])[:3])
        lines.append(
            f"| {family} | {audit['pre_family_counts'].get(family, 0)} | {audit['post_family_counts'].get(family, 0)} | {samples or '--'} |"
        )

    lines.extend(
        [
            "",
            "## Review Decisions",
            "",
            "| Decision | Count |",
            "|---|---:|",
        ]
    )
    for decision, count in sorted(decision_counts.items()):
        lines.append(f"| {decision} | {count} |")

    lines.extend(["", "## Merge Provenance Examples", "", "| Canonical concept | #Merged variants | Sample source phrases |", "|---|---:|---|"])
    for row in merge_rows:
        samples = ", ".join(row["sample_variants"])
        lines.append(f"| {row['canonical']} | {row['num_variants']} | {samples} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The human audit is light-touch rather than ontology-rebuilding: wording is reviewed, but the ontology structure is preserved.",
            "- The strongest reviewer-facing value is provenance: family balance, canonical naming, and merge history are explicit rather than buried in prompt iteration.",
            "- This asset should be cited as evidence that the paper-facing ontology is governed, not merely discovered.",
            "",
        ]
    )

    md_path.write_text("\n".join(lines))
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")


if __name__ == "__main__":
    main()

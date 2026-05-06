# EMNLP 2026 Oral Push Plan

Date: 2026-04-26

This document is the execution plan for turning the current ARR-ready draft
into a paper with materially stronger oral upside. The goal is not to chase
more single-run headline numbers. The goal is to harden the evidence that the
paper's central claim survives scrutiny.

## Current position

- The paper is already submission-shaped and scientifically coherent for ARR.
- The strongest blocks are now the controlled ontology comparison on DAD/A3D,
  the complete multi-seed ontology extension, and the stable A3D headline
  aggregate.
- The weakest blocks for best-paper ambitions are DAD mechanism clarity and
  actor-policy maturity, not missing ontology coverage or language-side audits.
- Current project posture should be:
  acceptance-first, oral-upside-through-stronger-evidence.

## Must Do

### 1. Multi-seed controlled ontology block -- complete

Status:

- Completed across all six dataset--ontology cells.
- Coverage: `18/18` seeded runs over seeds `42, 123, 3407`.
- Current board: `output/emnlp2026_support/multiseed_ontology_status.md`.

Why it matters:

- It directly strengthens the paper's central claim that ontology choice changes
  the AP-mTTA operating point.
- It is now response-ready evidence rather than an open execution item.

Target:

- DAD and A3D
- concept sets: `historical_full`, `risk_core_v1`, `perfect_v1`
- seeds: at least `3`
- total runs: `18`

Success criterion:

- The same operating-point story still holds in mean performance, not just in
  one cherry-pickable run.

Outcome:

- The block is complete and should be cited in rebuttal materials.
- Do not repeat it unless an artifact is corrupted or a reviewer asks for a
  very specific additional seed.

### 2. A3D headline stability -- complete

Status:

- Completed: `3/3` seeds for the A3D headline recipe.
- Aggregate: `94.16% +/- 0.95` AP and `4.62s +/- 0.42s` mTTA.
- Current board: `output/emnlp2026_support/a3d_headline_multiseed_status.md`.

Why this matters:

- A3D is the cleanest flagship result.
- The main table now has seed-backed stability rather than only a canonical run.

Success criterion:

- The current A3D story survives without collapsing into large variance.

### 3. DAD headline discipline -- complete as diagnosis

Why this matters:

- DAD is fragile, and the paper already admits that.
- For oral-level credibility, the paper must show that this fragility is
  measured and bounded rather than hand-waved away.

Status:

- Canonical DAD remains `68.19%` AP / `1.75s` mTTA.
- Clean-seed, synchronized-epoch, trigger-source, full-vs-ablation, and
  recovery summaries are present.
- Current board: `output/emnlp2026_support/dad_hardening_status.md`.

Success criterion:

- Reviewers can see the DAD weakness clearly and still trust the paper.

## Should Do

### 4. Expand language-side support evidence -- complete for current submission

Current status:

- Pseudo-label sensitivity uses `500` DAD frames.
- Verbalization robustness uses `12` concepts with `2` paraphrases each.
- Human ontology audit covers `80` concepts across `9` families.

Current artifacts:

- `EMNLP_SUPPORT_RESULTS.md`
- `output/emnlp2026_support/topm_pseudolabel_sensitivity_dad500.json`
- `output/emnlp2026_support/concept_verbalization_sensitivity_dad500.json`
- `output/emnlp2026_support/human_ontology_audit_summary.md`

Why:

- This is the most direct way to make the EMNLP fit stronger without changing
  the core method.

### 5. Add a light human ontology audit -- complete

Status:

- The support board records `80` reviewed concepts across `9` families.
- Keep this as semantic-artifact provenance, not as a claim of exhaustive human
  validation.

Why:

- It converts ontology curation from "plausible pipeline" to
  "audited semantic artifact".

### 6. Tighten ablation reliability -- mixed evidence

Status:

- A3D support ablations support the operating-point reading.
- DAD full-vs-ablation evidence is mixed and should remain a stress-test
  diagnosis.

Why:

- Oral reviewers care whether the core mechanism matters, not whether every
  old ablation table has another single-seed line.

## Avoid

- Do not reopen a major architecture rewrite.
- Do not burn compute on crash-only leaderboard vanity.
- Do not over-invest in the actor branch unless a quick pilot shows a real
  gain; right now it is support evidence, not the main acceptance lever.
- Do not mix search runs, support runs, and canonical tables in one story.

## Recommended run order

1. Final human read-through of page 1, main tables, key figures, and appendix opening.
2. Keep review response materials synchronized with the latest support boards.
3. If GPU work is explicitly authorized, run one targeted DAD mechanism-hardening block.
4. Avoid repeating completed ontology, A3D headline, DAD full-support, or DAD-500 language audits.
5. Actor branch only if a quick pilot shows a material behavior change.

## Operational rule

After each block finishes:

- aggregate results immediately
- decide whether the block strengthens or weakens the paper
- update the paper only from aggregated evidence, not from one unusually good
  seed

## Immediate commands

- Readiness audit:
  `python paper/emnlp2026/audit_emnlp_oral_readiness.py`
- Submission sanity:
  `bash paper/emnlp2026/run_submission_sanity_checks.sh`
- Refresh support boards:
  `python paper/emnlp2026/refresh_emnlp_status.py`
- Freeze current package:
  `bash paper/emnlp2026/freeze_arr_submission.sh`

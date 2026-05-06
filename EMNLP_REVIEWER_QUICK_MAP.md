# EMNLP Reviewer Quick Map

Date: 2026-04-29

This is a one-page navigation map for the current INSIGHT submission package.
It is meant for internal response preparation and final pre-upload checks.
For longer response drafts, use `EMNLP_REVIEW_RESPONSE_TEMPLATES.md`.
For short-answer and formal rebuttal variants, use
`EMNLP_REVIEW_RESPONSE_PLAYBOOK.md`.

## Core Thesis

INSIGHT is strongest when read as a semantic-interface paper:
accident anticipation should expose a governed language-grounded risk concept
interface, and ontology construction should be evaluated as a modeling choice
rather than hidden preprocessing.

## If The Reviewer Asks...

### 1. Why is this an EMNLP paper?

Answer:
The central object is a language-grounded semantic interface: canonical concept
names, ontology construction, family balancing, paraphrase robustness, and
semantic auditability are methodological parts of the model.

Evidence paths:
- `paper/emnlp2026/sec_intro_emnlp.tex`
- `paper/emnlp2026/sec_method_emnlp.tex`
- `paper/figures/insight_fig_concept_pipeline.pdf`
- `paper/figures/insight_fig_concept_family_coverage.pdf`
- `output/emnlp2026_support/concept_verbalization_sensitivity_dad500.json`

### 2. Is ontology construction really a modeling choice?

Answer:
Yes. Under matched launchers, changing the ontology changes the AP--mTTA
operating point on both DAD and A3D. The multi-seed controlled ontology block
is complete across `18/18` seeded dataset--ontology cells.

Evidence paths:
- `output/emnlp2026_support/controlled_ontology_status.md`
- `output/emnlp2026_support/multiseed_ontology_status.md`
- `EMNLP_CONTROLLED_ONTOLOGY_STATUS.md`
- `paper/emnlp2026/sec_experiments_emnlp.tex`

### 3. What is the strongest claim-tier evidence?

Answer:
The strongest package is the combination of governed ontology construction,
matched ontology comparisons, language-side audits, and the stable A3D
interpretable operating point. A3D is the clean flagship dataset.

Evidence paths:
- `output/emnlp2026_support/a3d_headline_multiseed_status.md`
- `EMNLP_SUPPORT_RESULTS.md`
- `paper/emnlp2026/sec_experiments_emnlp.tex`

Key numbers:
- A3D headline multi-seed: `94.16% +/- 0.95` AP,
  `4.62s +/- 0.42s` mTTA over `3/3` seeds.
- A3D canonical table line: `93.40%` AP / `4.90s` mTTA.

### 4. What is the honest DAD story?

Answer:
DAD is the harder stress test. The canonical line is competitive and useful,
but the support diagnostics show mechanism fragility. The paper should present
DAD as measured stress evidence, not as the cleanest mechanism proof.

Evidence paths:
- `output/emnlp2026_support/dad_hardening_status.md`
- `EMNLP_RERUN_PLAN.md`
- `paper/emnlp2026/sec_experiments_emnlp.tex`

Key numbers:
- Canonical DAD: `68.19%` AP / `1.75s` mTTA.
- Clean three-seed diagnostic: `62.31% +/- 1.90` AP,
  `2.07s +/- 0.09s` mTTA.
- Recovery block: `63.52% +/- 0.81` AP,
  `2.16s +/- 0.25s` mTTA over `6/6` completed runs.
- Matched full support block: `63.19% +/- 1.21` AP,
  `2.17s +/- 0.05s` mTTA over `3/3` completed runs.

### 5. Are actor-policy or intervention claims overstated?

Answer:
No, as long as the current scoped wording is preserved. Actor-policy timing is
support evidence, not the central claim. Intervention evidence supports partial
structural intervenability, not a completed policy-level causal benchmark.

Evidence paths:
- `EMNLP_INTERVENTION_STATUS.md`
- `paper/neurips2026/sec_appendix.tex`
- `paper/emnlp2026/claim_evidence_audit_report.md`

Key numbers:
- Trigger-source diagnostic: classifier `61.11% +/- 4.58` AP vs actor
  `37.41% +/- 7.35` AP over `n=6` checkpoints.
- Strongest archived hybrid boost: mean alert shift `-2.04` frames, but median
  shift remains `0.0`.

### 6. What should not be claimed?

Do not claim:
- best overall accident anticipation performance;
- a finished human-verified concept benchmark;
- broad policy-level causal timing control;
- that DAD mechanism evidence is as clean as A3D;
- that actor-policy behavior is the main contribution.

Safe formulation:
The paper provides a governed semantic interface whose ontology choice affects
the predictive--timing operating point, with strong A3D evidence, honest DAD
stress diagnostics, and scoped intervention/timing support.

## Final Upload Check

- Latest freeze path should match `EMNLP_STAGE_STATUS.md`.
- `paper/emnlp2026/submission_sanity_report.txt` must report
  `OK fatal_count=0`.
- `paper/emnlp2026/claim_evidence_audit_report.md` must report
  zero critical blockers.
- Final PDF page 1, Figure 1, Table 1, and the appendix opening still require
  human read-through before upload.

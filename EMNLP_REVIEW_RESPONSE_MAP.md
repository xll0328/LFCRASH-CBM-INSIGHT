# EMNLP Review Response Map

This is the compact reviewer-facing evidence map for the current INSIGHT draft.
For a one-page pre-upload navigation sheet, also use
`EMNLP_REVIEWER_QUICK_MAP.md`.
For rebuttal-style draft language, use
`EMNLP_REVIEW_RESPONSE_TEMPLATES.md`.
For short-answer and formal response variants, use
`EMNLP_REVIEW_RESPONSE_PLAYBOOK.md`.

## 1. Why is this an EMNLP paper rather than only a CV/ML paper?

Short answer:

- the paper's central object is a language-grounded semantic interface, not
  only an accident predictor
- ontology construction, canonical naming, family balancing, and semantic
  auditability are treated as methodological components

Best evidence:

- `paper/emnlp2026/sec_intro_emnlp.tex`
- `paper/emnlp2026/sec_method_emnlp.tex`
- `paper/figures/insight_fig_concept_pipeline.pdf`
- `paper/figures/insight_fig_concept_family_coverage.pdf`
- `paper/figures/insight_fig_concept_evolution.pdf`

## 2. Is ontology construction really a modeling choice?

Short answer:

- yes; changing the ontology changes the AP--mTTA operating point under one
  matched training recipe

Best evidence:

- `output/emnlp2026_support/controlled_ontology_status.md`
- `output/emnlp2026_support/multiseed_ontology_status.md`
- `output/concept_sets/neurips2026_controlled_ontology_manifest.json`
- `paper/emnlp2026/sec_experiments_emnlp.tex`

Current factual summary:

- DAD: best AP from `risk_core_v1`, best mTTA from `perfect_v1`
- A3D: best AP from `perfect_v1`, best mTTA from `risk_core_v1`
- multi-seed extension is complete across all six dataset--ontology cells:
  `18/18` seeded runs over seeds `42, 123, 3407`

## 3. What is the strongest evidence in the paper right now?

Short answer:

- ontology pipeline and auditability
- controlled ontology comparison
- strong interpretable operating point on A3D
- competitive canonical DAD line with explicit limitations

Best evidence:

- `EMNLP_CONTROLLED_ONTOLOGY_STATUS.md`
- `EMNLP_SUPPORT_RESULTS.md`
- `output/emnlp2026_support/a3d_headline_multiseed_status.md`
- `output/emnlp2026_support/dad_hardening_status.md`
- `paper/emnlp2026/sec_experiments_emnlp.tex`

Current factual summary:

- A3D headline multi-seed aggregate is complete: `94.16% +/- 0.95` AP,
  `4.62s +/- 0.42s` mTTA over `3/3` seeds
- DAD canonical headline remains `68.19%` AP / `1.75s` mTTA, but DAD support
  families are mixed and should be presented as stress-test evidence

## 4. Are the timing / actor claims overstated?

Short answer:

- no, if we keep the current disciplined wording
- prediction timing is supported; actor-policy timing remains partial

Best evidence:

- `EMNLP_INTERVENTION_STATUS.md`
- `paper/neurips2026/timing_faithfulness_package.md`
- `paper/neurips2026/threats_to_validity.md`

Current factual summary:

- prediction lead time: `39.41` frames over `39` valid DAD cases
- archived actor branch remains flat around `0.498`
- extended trigger-source diagnostic: classifier trigger aggregate is
  `61.11% +/- 4.58` AP, while actor trigger aggregate is
  `37.41% +/- 7.35` AP over `6` checkpoints

## 5. What does the intervention package really support?

Short answer:

- partial structural intervenability
- not a complete policy-level causal benchmark

Best evidence:

- `EMNLP_INTERVENTION_STATUS.md`
- `paper/neurips2026/sec_appendix.tex`

Current factual summary:

- strongest hybrid boost setting: `mean_alert_shift = -2.04` over `64` cases
- median shift remains `0.0`
- null/risk baselines both stay at `0.0` mean shift over `32` cases

## 6. What limitations should we state proactively?

Short answer:

- actor-policy evidence is still limited
- multi-seed stability is now stronger for ontology and A3D headline evidence,
  but DAD mechanism evidence remains mixed
- the matched DAD full support block is complete; its full-vs-ablation deltas
  do not justify a stronger DAD mechanism claim
- DAD remains harsher and less stable than A3D

Best evidence:

- `paper/emnlp2026/sec_conclusion_emnlp.tex`
- `paper/neurips2026/threats_to_validity.md`
- `output/emnlp2026_support/dad_hardening_status.md`

## 7. What should we avoid spending compute on immediately?

Short answer:

- rerunning the six controlled ontology lines from scratch
- repeating the completed matched DAD full support block
- trying to rescue a maximal actor-policy claim

Why:

- the controlled block is already complete enough for the current paper
- the matched DAD full block is complete; the remaining DAD issue is mechanism
  fragility, not missing coverage
- the highest-value work is now presentation discipline, appendix clarity,
  response readiness, and only then a targeted DAD mechanism-hardening block

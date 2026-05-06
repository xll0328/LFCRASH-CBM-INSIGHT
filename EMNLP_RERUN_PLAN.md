# EMNLP Rerun Plan

## Objective

Only rerun experiments that materially strengthen the new EMNLP thesis:
the ontology and semantic interface should be the empirical center of gravity.

Current status as of 2026-04-26: the original ontology refresh, ontology audit,
DAD-500 language-side support analyses, A3D headline multi-seed block, and DAD
full support block are complete enough for the current submission package. This
file should therefore be read as a rerun guardrail, not as a command queue.

## High Priority

### 1. Controlled ontology comparison refresh -- complete

Why:

- this is the cleanest empirical support for the paper's new main claim

Keep fixed:

- dataset split
- optimizer family
- scheduler family
- epoch budget
- batch size
- evaluation cadence

Vary only:

- ontology source

Required outputs:

- AP
- mTTA
- TTA@R80
- P@R80
- concept count
- training recipe id
- seed

Concrete launcher:

```bash
./run_controlled_ontology_experiment.sh dad historical_full 0 dad_shared_historical_full
./run_controlled_ontology_experiment.sh dad risk_core_v1 1 dad_shared_risk_core_v1
./run_controlled_ontology_experiment.sh dad perfect_v1 2 dad_shared_perfect_v1
./run_controlled_ontology_experiment.sh a3d historical_full 3 a3d_shared_historical_full
./run_controlled_ontology_experiment.sh a3d risk_core_v1 4 a3d_shared_risk_core_v1
./run_controlled_ontology_experiment.sh a3d perfect_v1 5 a3d_shared_perfect_v1
```

Protocol note:

- keep this block fully separate from the canonical DAD curriculum headline run
- do not rerun this block unless a stored artifact is corrupted or a reviewer
  requests a specific additional seed; the current multi-seed status is
  `18/18` completed

### 2. Ontology audit export -- complete

Why:

- strengthens the claim that the ontology is governed and reviewable

Desired outputs:

- concept-family coverage summary
- merge provenance summary
- canonical naming examples
- paper-facing concept statistics

Source files already available:

- `output/concept_sets/perfect_concept_set_v1.audit.json`
- `output/concept_sets/perfect_concept_set_v1.family_meta.json`
- `output/concept_sets/perfect_concept_set_v1.merge_examples.json`

### 3. Concept verbalization / pseudo-label sensitivity summary -- complete

Why:

- useful support for NLP-facing readers
- lower cost than large new training runs

Desired outputs:

- concise top-m sensitivity table
- family spread statistics
- failure modes of over-diffuse or over-concentrated supervision

Concrete low-cost commands:

```bash
paper/emnlp2026/run_emnlp_support_analyses.sh
```

Underlying outputs:

- `output/emnlp2026_support/dad500_frame_manifest.json`
- `output/emnlp2026_support/topm_pseudolabel_sensitivity_dad500.json`
- `output/emnlp2026_support/concept_verbalization_sensitivity_dad500.json`

## Medium Priority

### 4. Clean support ablation refresh

Why:

- good to keep, but not the central evidence block

Variants:

- full model
- no CBM
- no CAAC
- no CGTA
- no CRS
- no TSD

### 5. Compact case-study refresh

Why:

- better qualitative support if the main text is rebuilt around semantic evidence

Desired outputs:

- 2 to 3 very readable cases
- concept activations
- concept-family timeline
- classifier trajectory

## Lower Priority

### 6. Actor-policy expansion

Why lower:

- this is no longer the paper's main contribution
- easy to burn time for relatively little narrative gain

Keep only if:

- there is a clean checkpoint line where actor behavior is clearly strong

### 7. Broad DAD leaderboard chasing

Why lower:

- the new paper does not need to win the full DAD leaderboard to be persuasive

## Stop Rules

- Do not rerun large experiments just to rescue the strongest actor claim.
- Do not mix search runs and canonical runs in one headline table.
- Do not invest heavily in figures that mainly defend appendix-only claims.

## Suggested Order

1. Refresh ontology comparison.
2. Export ontology audit summaries.
3. Refresh one clean case-study package.
4. Only then decide whether additional actor/timing reruns are worth it.

## Current Default Execution Order

1. Run `bash paper/emnlp2026/run_submission_sanity_checks.sh` after any paper or support-document edit.
2. Run `python paper/emnlp2026/refresh_emnlp_status.py` after any completed support run.
3. Cut a new freeze with `bash paper/emnlp2026/freeze_arr_submission.sh` after package-facing changes.
4. Keep actor-policy expansion gated on clearly stronger evidence than the current support block.
5. Treat additional GPU work as a targeted DAD mechanism-hardening block only after explicit authorization.

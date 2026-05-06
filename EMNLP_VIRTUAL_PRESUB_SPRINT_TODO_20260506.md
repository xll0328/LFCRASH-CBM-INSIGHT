# EMNLP Virtual Pre-Review Sprint To-Do

Date: 2026-05-06  
Target: turn virtual borderline review into acceptance-level revision package
Action matrix: `EMNLP_VIRTUAL_REVIEW_ACTION_MATRIX_20260506.md`

## Mission

- Close the largest reviewer risks: claim calibration, stability evidence,
  interpretability validation, and ontology reproducibility.
- Keep ARR submission safety gates green while improving oral competitiveness.

## Priority Tracks

### P0 (must): Claim calibration + narrative compression

- [x] Remove over-scaffolded "how to read" prose from Intro/Experiments.
- [x] Keep one concise evidence taxonomy, remove repeated meta guidance.
- [x] Separate "inspectability" vs "faithfulness" wording explicitly in Intro,
  Method, and Conclusion.
- [x] Replace idiosyncratic terms (`paper-facing`, `launcher`) with
  standard phrasing or define once.

Deliverable:
- Revised `sec_intro_emnlp.tex`, `sec_experiments_emnlp.tex`,
  `sec_method_emnlp.tex`, `sec_conclusion_emnlp.tex`.

### P1 (must): Stability-first empirical rewrite

- [x] Add explicit seed-aggregate framing for ontology effect (AP/mTTA with
  mean±std and uncertainty-aware interpretation).
- [x] Add compact effect-size narrative for ontology deltas on DAD and A3D.
- [x] Demote singleton lines to "headline snapshot" and foreground aggregates
  for core claims.

Deliverable:
- Revised ontology result subsection and safety-utility interpretation.

### P2 (must): Concept validity / faithfulness support

- [x] Add a compact "concept validity" table from existing assets:
  pseudo-label concentration, paraphrase stability, review coverage.
- [x] Clarify that current evidence is partial faithfulness support, not full
  causal semantics.

Deliverable:
- New table + tighter interpretability claims in text.

### P3 (must): Ontology protocol reproducibility

- [x] Add procedural criteria for each ontology stage:
  candidate filtering, duplicate clustering, merge decision, family balancing,
  and human-review acceptance rule.
- [x] Anchor each rule to logged artifacts (`human_ontology_audit_summary.*`,
  merge examples, family balance files).

Deliverable:
- Method section protocol details and reproducibility paragraph.

### P4 (should): Related-work and comparison fairness

- [x] Add verified 2025 related works (where bibliographic metadata is
  verifiable now).
- [x] Tighten fairness language around backbone/protocol mismatch.
- [x] Keep benchmark positioning conservative and explicit.

Deliverable:
- Updated `sec_related_emnlp.tex` + `insight.bib`.

### P5 (should): New compute block for DAD ontology stability

- [x] Launch phase-1 DAD ontology seed-extension block (`perfect_v1`, +5 seeds).
- [x] Launch phase-2 extension for `historical_full` and `risk_core_v1` (+5 seeds each, queued workers on GPU2/3).
- [x] Add at least 5 additional seeds per ontology cell.
- [ ] Refresh multiseed audit and integrate into paper after completion.

Suggested seeds:
- `7, 11, 2718, 314, 2026`

Launch command template:
- `bash run_controlled_ontology_experiment.sh dad <concept_set> <gpu> dad_shared_<concept_set>_s<seed> --seed <seed> --num_workers 0`

Stop rule:
- Do not change headline claim tier before all new seeds are aggregated.

### P6 (must): Size-vs-ontology confound isolation

- [x] Add a size-matched control protocol (30/80 concept budgets) so ontology source and concept count are not fully coupled.
- [x] Build deterministic subset rules for historical vocabulary (seeded sampling with family-stratified quotas).
- [x] Launch DAD + A3D pilot runs for size-matched controls (minimum 3 seeds each condition).
- [ ] Add one compact table/paragraph separating “source effect” vs “count effect”.

Deliverable:
- `paper/emnlp2026/run_ontology_size_matched_controls.sh` (`DONE`)
- `paper/emnlp2026/build_historical_size_matched_subsets.py` (`DONE`)
- `paper/emnlp2026/audit_ontology_size_matched_controls.py` (`DONE`)
- updated `sec_experiments_emnlp.tex` section near Table 6.

### P7 (must): Reproducibility hardening in main paper

- [x] Replace high-level ontology prose with executable criteria language (inputs, deterministic parts, review-only parts).
- [x] Add deterministic-vs-judgment boundary sentence block in Method.
- [x] Add concrete artifact pointers for each ontology stage in main text.

Deliverable:
- updated `sec_method_emnlp.tex` with explicit protocol bullets and governance trace links.

### P8 (should): Presentation precision cleanup

- [x] Reformat reward equation block for readability and compactness.
- [x] Resolve naming ambiguity around ontology counts (`611/81/80` vs `837` historical full vocabulary) with one explicit disambiguation sentence.
- [x] Re-check terminology consistency (`training protocol` vs `launcher`, `semantic interface` vs legacy wording).

Deliverable:
- updated `sec_method_emnlp.tex`, `sec_experiments_emnlp.tex`, and sanity-clean compile.

### P9 (should): Comparison set and fairness wording

- [ ] Expand the A3D comparability caveat (backbone/data pipeline mismatch) in result discussion.
- [ ] Add one sentence in conclusion on what would be needed for stronger cross-stack fairness claims.

Deliverable:
- updated `sec_experiments_emnlp.tex`, `sec_conclusion_emnlp.tex`.

## Immediate Execution Order (start now)

1. P0 + P3 manuscript surgery (high impact, low compute risk). `DONE`  
2. P1 + P2 evidence table/prose upgrade from existing support artifacts. `DONE`  
3. P4 related-work updates with verified references only. `DONE`  
4. Start P5 extended DAD seed jobs in background and track status. `STARTED (phase-1 + phase-2 launched, perfect_v1 completed)`  
5. Run P6 size-matched control setup + launch pilot runs. `DONE (queue running)`  
6. Recompile + sanity gate + claim audit + tracker sync. `DONE (latest sanity: 2026-05-06T10:11:32Z)`  

## Gate Criteria Before Next Freeze

- `bash paper/emnlp2026/run_submission_sanity_checks.sh` -> `OK fatal_count=0`
- `paper/emnlp2026/claim_evidence_audit_report.md` keeps `critical_blockers=0`
- New text does not reintroduce policy-level or causal overclaim.
- Revised paper clearly states:
  - what is headline evidence,
  - what is support evidence,
  - what remains unresolved (DAD stability + actor transfer fragility + size/source disentanglement status).

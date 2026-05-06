# NeurIPS Rebuttal Map

## Reviewer attack 1 — "Is this really more than post-hoc explanation?"
### Short answer
Yes, at the architectural level: concepts are part of the state used by the timing policy, not an explanation attached after prediction.

### Evidence to cite
- `paper/neurips2026/sec_method.tex` for concept-augmented state definition
- `paper/neurips2026/thesis_claims_evidence_matrix.md`
- intervention appendix figures for partial structural intervenability

### Honest scope note
Current strongest intervention evidence is backbone-level / case-based, not a fully completed large-scale policy-control proof.

---

## Reviewer attack 2 — "Where is the real gain over black-box methods?"
### Short answer
The paper does not claim the highest overall AP on every benchmark. Its contribution is a strong interpretable operating point plus an intervenable semantic interface.

### Evidence to cite
- main DAD/A3D tables in `sec_experiments.tex`
- safety-utility figure positioning
- efficiency table

### Honest scope note
We explicitly separate interpretable competitiveness from absolute leaderboard leadership.

---

## Reviewer attack 3 — "Is ontology construction just manual engineering?"
### Short answer
No. The repository contains a reproducible discovery, refinement, and polishing pipeline plus a controlled ontology evaluation manifest.

### Evidence to cite
- ontology sections in `sec_method.tex` and `sec_appendix.tex`
- `output/concept_sets/neurips2026_controlled_ontology_manifest.json`
- `controlled_ontology_matrix.md`
- ontology evolution / family coverage / case-study figures

### Honest scope note
The full matched DAD/A3D controlled evaluation block is the next major strengthening experiment.

---

## Reviewer attack 4 — "Is the policy actually doing more than thresholding?"
### Short answer
Architecturally yes; empirically the current paper has stronger evidence for prediction-timing and concept-conditioned design than for a fully archived actor-crossing story.

### Evidence to cite
- CAAC design in method section
- timing-faithfulness package
- prediction-branch timing summaries

### Honest scope note
Do not overclaim actor-level crossing evidence from archived flat-actor outputs.

---

## Reviewer attack 5 — "Are the claims inflated?"
### Short answer
We explicitly audited claims against stable artifacts and downgraded unsupported ones.

### Evidence to cite
- `claim_evidence_audit.json`
- `submission_results_ledger.json`
- revised intro/experiments/conclusion language

---

## Reviewer attack 6 — "Where are the robustness and fairness checks?"
### Short answer
We provide a multi-seed DAD stability diagnostic, protocol separation, and a reviewer-proof experiment manifest.

### Evidence to cite
- DAD three-seed section in `sec_experiments.tex`
- `reviewer_proof_experiment_manifest.md`
- `threats_to_validity.md`

### Honest scope note
Current multi-seed support is strongest for one DAD line; broader seed coverage remains future strengthening.

---

## Reviewer attack 7 — "Did you prove hard-case symmetry on DAD?"
### Short answer
Not globally. We completed reviewer confirmation for the current mixed-pair audit slice and pass the hard-case process gate, but we still treat symmetry as bounded to current audited coverage.

### Evidence to cite
- hard-case scope language in `sec_experiments.tex` / `sec_conclusion.tex`
- `dad_hard_case_symmetry_audit.md` (family buckets + reviewer-confirmed mixed pairs)
- `build_dad_hard_case_symmetry_audit.py` (reproducible audit scaffold)
- `dad_hard_case_pair_review_sheet.md` (pair-level acceptance record)
- `hard_case_symmetry_gate_summary.json` (`gate_ready=true`)
- `reviewer_proof_experiment_manifest.md` hard-case gating checklist

### Honest scope note
Current snapshot is a confirmed audit checkpoint, not a solved-symmetry result. Coverage breadth across families remains the open risk, so claims stay bounded.

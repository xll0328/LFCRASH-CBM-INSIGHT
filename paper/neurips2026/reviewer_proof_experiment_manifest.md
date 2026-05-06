# Reviewer-Proof Final Experiment Manifest

## Purpose
This manifest defines the minimum experiment package required to defend the paper under NeurIPS-style review.

## Block A — Canonical headline runs
These are the runs allowed to support the main DAD/A3D tables.

### DAD canonical line
- Use the curriculum-stabilized DAD line already referenced in `sec_experiments.tex`
- Report: AP, mTTA, TTA@R80, P@R80, parameter count
- Do **not** replace this line with ontology-search or local-search runs

### A3D canonical line
- Use the current paper-facing A3D line already referenced in `insight_main.tex` / `sec_experiments.tex`
- Report: AP, mTTA, TTA@R80, P@R80, parameter count

## Block B — Stability diagnostics
These runs are not headline replacements; they quantify robustness.

### Required now
- Keep existing DAD multi-seed aggregate already described in `sec_experiments.tex`
- Present as stability evidence for one curriculum line only

### Desired next
- Add analogous 3-seed aggregate for the final A3D line
- Add mean/std for ontology-controlled runs if compute permits
- Add a DAD hard-case symmetry audit table (predefined scenario families, paired success/failure counts, fixed checkpoint rule)
- Use `paper/neurips2026/build_dad_hard_case_symmetry_audit.py` to refresh `paper/neurips2026/dad_hard_case_symmetry_audit.md` before each claim-evidence freeze
- Manually validate auto-suggested mixed pairs in `paper/neurips2026/dad_hard_case_symmetry_audit.md` and replace heuristic pair IDs/outcomes with reviewer-confirmed labels before promoting any symmetry claim

## Block C — Controlled ontology block
Use `controlled_ontology_matrix.md` and the shared launcher.

Required table columns:
- dataset
- concept set
- AP
- mTTA
- TTA@R80
- P@R80
- concept count
- seed / protocol ID

Scientific question:
- does ontology quality change the performance–timing–auditability frontier?

## Block D — Intervenability block
Use `intervention_protocol_and_casebank.md`.

Minimum acceptable evidence:
1. one successful amplification case in the main paper
2. one successful suppression or contrast case in appendix
3. one failure case shown honestly
4. quantitative summary from archived intervention outputs with scope caveat

## Block E — Timing-faithfulness block
Use `timing_faithfulness_package.md`.

Supported now:
- prediction crossing / lead-time evidence

Not yet supported as a strong claim:
- actor-policy crossing dominance

## Fairness / validity checklist
Before paper freeze, answer each item explicitly:
1. Are headline runs separated from support runs?
2. Are support runs labeled as controlled blocks rather than headline SOTA claims?
3. Is seed variability disclosed where available?
4. Is compute/hardware described?
5. Are intervention limits stated honestly?
6. Are actor-branch limitations stated honestly where relevant?
7. Are external baselines clearly labeled as interpretable vs non-interpretable?
8. Is DAD hard-case symmetry treated as open unless supported by an explicit family-level audit table?

## Paper usage rule
A result may appear in the main narrative only if it is one of:
- canonical headline result
- controlled support result with fixed protocol
- appendix diagnostic explicitly labeled as such

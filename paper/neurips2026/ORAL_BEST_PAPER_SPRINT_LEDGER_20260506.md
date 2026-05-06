# INSIGHT NeurIPS 2026 Oral / Best Paper Sprint Ledger

Date: 2026-05-06  
Scope: Paper-facing strengthening without new long GPU jobs

## 1) What Is Already Strong (Keep and Defend)

- Semantic-state thesis is coherent and consistent across abstract/intro/method.
- A3D operating-point evidence is strong and clearly reported (`93.40 AP`, `4.90s mTTA`).
- Protocol-separation discipline is explicit (headline vs support vs diagnostic).
- Claim-evidence guardrails are active (`run_neurips_sanity_checks.py`).
- Intervention/timing language is bounded and no longer overclaims full policy proof.

## 2) What Still Blocks Oral/Best-Paper-Level Conviction

- DAD hard-case symmetry is not yet demonstrated at family-pair level.
- Policy-level timing/intervention remains mixed for archived actor branch.
- Current hard-case evidence is scaffolded and auditable, but still needs reviewer-confirmed pair outcomes.

## 3) New Assets Added in This Sprint

- `build_dad_hard_case_symmetry_audit.py`
- `dad_hard_case_symmetry_audit.md`
- Appendix snapshot table `tab:appendix_hard_case_symmetry` in `sec_appendix.tex`
- Rebuttal entry: hard-case symmetry challenge in `rebuttal_map.md`

## 4) Minimal Next Wins (No New GPU Required)

1. Manually finalize 4 suggested mixed pairs in `dad_hard_case_symmetry_audit.md`:
   - fill `reviewer_note`
   - replace `auto_suggested_mixed_pair` with reviewer-confirmed outcomes.
2. Add a short appendix summary paragraph reporting:
   - confirmed pair count
   - family coverage
   - explicit unresolved families.
3. Mirror those outcomes into:
   - `thesis_claims_evidence_matrix.md`
   - `claim_evidence_audit.json`
   - `reviewer_proof_experiment_manifest.md`.

## 5) Evidence-Discipline Rule

Until pair outcomes are reviewer-confirmed, keep hard-case symmetry as:

- **open** in main text,
- **audit checkpoint** in appendix,
- **not settled** in claim ledger.

## 6) Stop Conditions (Carry Forward)

- Any wording implying full policy-level control is solved.
- Any statement that hard-case symmetry is established without pair-level audit completion.
- Any blending of support/search runs into canonical headline claims.

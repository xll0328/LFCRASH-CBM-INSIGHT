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

- DAD hard-case process gate is now complete for the current audited mixed pairs
  (`gate_ready=true`), but coverage is still limited and not a blanket global symmetry proof.
- Policy-level timing/intervention remains mixed for archived actor branch.
- Hard-case evidence is now reviewer-confirmed for the tracked pairs; remaining risk is breadth, not process completion.

## 3) New Assets Added in This Sprint

- `build_dad_hard_case_symmetry_audit.py`
- `dad_hard_case_symmetry_audit.md`
- Appendix snapshot table `tab:appendix_hard_case_symmetry` in `sec_appendix.tex`
- Rebuttal entry: hard-case symmetry challenge in `rebuttal_map.md`

## 4) Minimal Next Wins (No New GPU Required)

1. Add a short appendix summary paragraph reporting:
   - confirmed pair count
   - family coverage
   - explicit unresolved families.
2. Mirror confirmed outcomes into:
   - `thesis_claims_evidence_matrix.md`
   - `claim_evidence_audit.json`
   - `reviewer_proof_experiment_manifest.md`.
3. Add one rebuttal-facing sentence clarifying:
   - gate completion status,
   - remaining coverage limits,
   - why the claim remains bounded.

## 5) Evidence-Discipline Rule

Even with current pair confirmations, keep hard-case symmetry as:

- **open** in main text,
- **audit-backed but bounded** in appendix,
- **not settled** in claim ledger.

## 6) Stop Conditions (Carry Forward)

- Any wording implying full policy-level control is solved.
- Any statement that hard-case symmetry is established without pair-level audit completion.
- Any blending of support/search runs into canonical headline claims.

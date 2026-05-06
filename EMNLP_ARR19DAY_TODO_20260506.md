# EMNLP ARR 19-Day Execution Plan + ToDo

Date: 2026-05-06  
Window: 2026-05-06 to 2026-05-25 (ARR submission deadline)

## Objective Hierarchy

- P0: Submit a technically clean, claim-disciplined ARR package on time.
- P1: Maximize oral probability by reducing reviewer ambiguity.
- P2: Preserve a best-paper escalation lane without destabilizing submission safety.

## Phase Plan (Calendar)

### Phase A (T-19 to T-16, 2026-05-06 to 2026-05-09): Lock Baseline

- [x] Re-run EMNLP compile and sanity after latest appendix edits.
- [x] Refresh top-conference quality audit and oral-readiness audit.
- [x] Refresh oral/best gap ledger to 2026-05-06 snapshot.
- [x] Refresh stage-status doc with the new baseline timestamp.
- [x] Build a short "upload candidate checklist" block in stage docs.

Completion rule:
- `paper/emnlp2026/submission_sanity_report.txt` stays `OK fatal_count=0`.

### Phase B (T-15 to T-9, 2026-05-10 to 2026-05-16): Oral Defense Hardening

- [ ] Do one full human read-through at 100% zoom: page 1, Fig 1, Table 1, appendix opening.
- [ ] Mark all sentences that can be misread as causal/policy-overclaim.
- [ ] Apply only minimal wording edits; re-run sanity immediately.
- [ ] Run a rebuttal dry-run against 5 highest-risk reviewer attacks.
- [ ] Sync quick-map/playbook/templates with any wording edits.

Completion rule:
- Reviewer quick responses can answer "why EMNLP" and "what DAD proves" in 2-3 sentences without adding claims.

### Phase C (T-8 to T-3, 2026-05-17 to 2026-05-22): Final Freeze Candidate

- [ ] Freeze candidate tarball cut (`freeze_arr_submission.sh`) only if human read-through passes.
- [ ] Verify package (`verify_arr_freeze.sh`) and record freeze path.
- [ ] Confirm author/reviewer-registration logistics ownership.
- [ ] Run one last sanity and first-25 PDF freshness check.

Completion rule:
- One identified tarball is both verifier-clean and human-approved for upload logistics.

### Phase D (T-2 to T-0, 2026-05-23 to 2026-05-25): Upload Ops

- [ ] Final no-change sanity gate.
- [ ] Human upload + ARR metadata checks.
- [ ] Commit/venue-lock decisions logged (human).
- [ ] Freeze post-upload rebuttal mode; no story rewrite.

Completion rule:
- ARR upload finished by 2026-05-25 AoE with clean evidence trail.

## P0/P1/P2 Detailed ToDo

### P0 Submission Safety (Must)

- [ ] Keep `run_submission_sanity_checks.sh` green after every manuscript edit.
- [ ] Keep `claim_evidence_audit_report.md` at `critical_blockers=0`.
- [ ] Keep `pdf_first_read_audit_report.md` at `critical_blockers=0`.
- [ ] Keep anonymization metadata empty in PDF checks.

### P1 Oral Lift (Must)

- [ ] Ensure every major table caption declares evidence tier: headline/support/stress.
- [ ] Keep A3D as flagship operating-point evidence.
- [ ] Keep DAD as hard stress-test evidence with explicit limits.
- [ ] Keep actor-policy as support-only unless new aggregate evidence appears.

### P2 Best-Paper Lane (Optional, Guarded)

- [ ] Do not claim best-paper readiness in current cycle.
- [ ] If compute is explicitly approved: run only one predeclared mechanism block.
- [ ] Update claims only from full aggregate results, not checkpoint snapshots.

## Ownership + Deliverables

- Paper package integrity:
  - `paper/emnlp2026/submission_sanity_report.txt`
  - `paper/emnlp2026/claim_evidence_audit_report.md`
  - `paper/emnlp2026/pdf_first_read_audit_report.md`
- Strategy state:
  - `EMNLP_STAGE_STATUS.md`
  - `EMNLP_ORAL_BEST_PAPER_GAP_LEDGER_20260506.md`
- Rebuttal readiness:
  - `EMNLP_REVIEWER_QUICK_MAP.md`
  - `EMNLP_REVIEW_RESPONSE_PLAYBOOK.md`
  - `EMNLP_REVIEW_RESPONSE_TEMPLATES.md`

## This Turn: Started Actions

- [x] Baseline compile rerun completed (`insight_emnlp.pdf` refreshed).
- [x] Submission sanity rerun completed (`OK fatal_count=0`, generated `2026-05-06T06:53:30Z`).
- [x] Top-conference and oral-readiness audits rerun on latest workspace.
- [x] New 2026-05-06 gap ledger created.

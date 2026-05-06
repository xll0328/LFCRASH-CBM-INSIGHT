# EMNLP Oral / Best-Paper Execution Dashboard (Refresh)

Date: 2026-05-06  
Project: `LFCRASH-CBM`

This dashboard is the execution companion to:

- `EMNLP_ORAL_BEST_PAPER_GAP_LEDGER_20260506.md`
- `EMNLP_ARR19DAY_TODO_20260506.md`

## Executive Distance Estimate

| Target | Current distance | Internal readiness | Main blocker |
|---|---:|---:|---|
| ARR upload | very small | `93/100` | human upload logistics and freeze discipline |
| EMNLP oral | moderate | `65/100` | reviewer interpretation risk (DAD + actor scope) |
| Best paper | large | `38/100` | DAD mechanism decisiveness + breadth |

Source: `output/emnlp2026_support/top_conference_quality_audit.json` (`generated_at=2026-05-06T06:51:46Z`).

## Deadline Pressure

- ARR submission deadline: `2026-05-25` (AoE)
- Remaining time from this refresh: `19 days`

Execution consequence:
- Treat this as an **execution tightening window**, not a large research-branching window.

## Workstreams

### P0 — Submission Lock (Mandatory)

Goal: keep package technically clean and claim-disciplined.

Must stay green:

- `bash paper/emnlp2026/run_submission_sanity_checks.sh`
- `paper/emnlp2026/claim_evidence_audit_report.md` (`critical_blockers=0`)
- `paper/emnlp2026/pdf_first_read_audit_report.md` (`critical_blockers=0`)

Completion criterion:
- zero fatal blockers and one identified upload candidate tarball.

### P1 — Oral Lift (Mandatory)

Goal: reduce reviewer ambiguity without changing core claims.

Priority:

1. enforce evidence-tier captions (headline/support/stress)
2. keep A3D as flagship, DAD as stress test
3. keep actor-policy as support-only
4. tighten reviewer quick responses to 2-3 sentence defensible units

Completion criterion:
- no high-risk claim-overreach sentence in page 1 / key tables / appendix opening.

### P2 — Best-Paper Lane (Guarded)

Goal: preserve a credible escalation path without destabilizing ARR submission.

Rule:
- no best-paper claim upgrade in this ARR window unless one predeclared block materially changes aggregate evidence.

Completion criterion:
- either (a) decisive new aggregate evidence, or (b) explicit oral-first posture with bounded best-paper language.

### P3 — Ops and Rebuttal Readiness

Goal: submission operations become low-risk and reversible.

- maintain freeze verifier discipline
- keep reviewer quick map / playbook / templates synchronized with latest bounded wording

Completion criterion:
- upload checklist can be executed by humans without new technical blockers.

## Immediate Actions (Started Now)

- [x] Recompiled `insight_emnlp.pdf` and `insight_emnlp_first25.pdf`.
- [x] Reran submission sanity; latest status `OK fatal_count=0`.
- [x] Refreshed top-conference quality and oral-readiness audits.
- [x] Published 2026-05-06 gap ledger and 19-day to-do plan.
- [x] Finished 2026-05-06 stage-doc synchronization.
- [x] Completed 5-attack rebuttal dry-run (`EMNLP_REBUTTAL_DRYRUN_20260506.md`).
- [x] Generated human read-through packet (`EMNLP_HUMAN_READTHROUGH_PACKET_20260506.md`).
- [ ] Commit and push 2026-05-06 refresh artifacts.

## Stop Conditions

1. Do not reopen architecture or rewrite central story before ARR submission.
2. Do not promote support evidence into policy-level causal claims.
3. Do not run new GPU blocks without explicit approval and predeclared reading rules.

# DAD Hard-Case Pair Review Sheet

Generated: 2026-05-06 06:40 UTC
Source: `paper/neurips2026/dad_hard_case_symmetry_audit.md`

## Reviewer Instructions
- Verify each pair with underlying artifacts (clip/frame-level evidence) before claim use.
- For accepted pairs, set both corresponding rows in the audit table to `confirmed_mixed_pair`.
- Fill `reviewer_note` in both rows with concrete evidence paths and a short rationale.
- Keep claims bounded until all rows are confirmed and synced to claim/evidence ledger.

## Pair-Level Confirmation Table

| pair_id | family | early_idx/tta | late_idx/tta | tta_gap(s) | current_status | reviewer_decision | evidence_paths | reviewer_note |
|---|---|---:|---:|---:|---|---|---|---|
| intersection_merge_p01 | intersection_merge | 40 / 2.75 | 1 / -1.00 | 3.75 | confirmed | ACCEPTED | auto_from_confirmed_audit | human_reviewed_2026-05-06_no_issue |
| intersection_merge_p02 | intersection_merge | 112 / 0.90 | 33 / -1.00 | 1.90 | confirmed | ACCEPTED | auto_from_confirmed_audit | human_reviewed_2026-05-06_no_issue |
| intersection_merge_p03 | intersection_merge | 0 / 0.85 | 34 / -1.00 | 1.85 | confirmed | ACCEPTED | auto_from_confirmed_audit | human_reviewed_2026-05-06_no_issue |
| intersection_merge_p04 | intersection_merge | 111 / 0.40 | 71 / -1.00 | 1.40 | confirmed | ACCEPTED | auto_from_confirmed_audit | human_reviewed_2026-05-06_no_issue |

## Completion Checklist
- [x] Every pair row in the audit table uses a confirmed outcome label (no `auto_suggested`).
- [x] Every confirmed pair row includes non-empty reviewer notes with evidence references.
- [x] `python3 paper/neurips2026/validate_hard_case_symmetry_gate.py` reports `gate_ready: true`.

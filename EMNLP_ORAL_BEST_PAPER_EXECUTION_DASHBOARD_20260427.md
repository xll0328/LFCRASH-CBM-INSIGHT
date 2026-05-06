# EMNLP Oral / Best-Paper Execution Dashboard

Date: 2026-04-27
Project: `LFCRASH-CBM`

This dashboard turns the current oral / best-paper gap ledger into an execution
plan. The scores below are internal readiness scores, not acceptance
probabilities and not human validation.

## Executive Distance Estimate

| Target | Current distance | Internal readiness | Main blocker |
|---|---:|---:|---|
| ARR upload | very small | `95/100` | human upload logistics |
| EMNLP oral | small to moderate | `78/100` | reviewer interpretation risk around DAD and ontology framing |
| Best paper | large | `52/100` | DAD mechanism clarity and broader independent evidence |

Reading:

- The paper is close to a strong ARR submission because packaging, sanity,
  anonymity, and claim/evidence gates are clean.
- The paper is plausibly in the oral conversation because the semantic-interface
  story is coherent and backed by seed-level ontology and A3D evidence.
- The paper is not yet best-paper-ready because a best-paper case needs the
  stress-test side to feel unusually decisive, not merely honest and bounded.

## Current Evidence Scorecard

| Area | Status | Evidence | Risk |
|---|---|---|---|
| Submission package | ready | `OK fatal_count=0`, verified freeze `ARR20260427T032620Z` | final upload is human-only |
| Semantic-interface claim | strong | controlled ontology multi-seed `18/18` | reviewer may call it prompt engineering unless governance is emphasized |
| A3D flagship | strong | `94.16% +/- 0.95` AP, `4.62s +/- 0.42s` mTTA | must stay the clean flagship |
| DAD stress test | acceptable but fragile | canonical `68.19%` AP, full-support `3/3`, mixed full-vs-ablation deltas | largest best-paper gap |
| Actor-policy timing | support-only | classifier trigger stronger than actor trigger | unsafe as a main causal claim |
| Language-side evidence | solid for submission | DAD-500 audits, verbalization sensitivity, 80-concept audit | not exhaustive human validation |
| Rebuttal readiness | good | quick map, playbook, templates, tracker | must avoid inventing stronger claims during rebuttal |

## P0: Submission Lock

Goal: preserve the ARR-ready package while avoiding claim drift.

Status:

- Technical gates: `complete`
- Latest verified freeze: `ARR20260427T032620Z`
- Human-only blockers: author list, ARR account / reviewer registration, venue
  commitment, final PDF read-through, upload

Execution steps:

1. Keep `paper/emnlp2026/run_submission_sanity_checks.sh` green.
2. Keep `paper/emnlp2026/verify_arr_freeze.sh` green for the recommended
   freeze.
3. Do not change manuscript claims unless the final human read-through finds a
   concrete reviewer-facing error.
4. If any paper-facing file changes, rerun sanity before considering a new
   freeze.

Completion criterion:

- `OK fatal_count=0`
- verifier passes
- no unresolved human-only upload decision is represented as automated
  validation

## P1: Oral Defense Readiness

Goal: make the paper easy for a strong reviewer to defend.

Distance: `small to moderate`

Primary risks:

1. The ontology contribution is mistaken for prompt tuning.
2. DAD fragility is interpreted as method failure rather than stress-test
   behavior.
3. Actor-policy timing is read as a causal claim.
4. The A3D flagship result is diluted by too much DAD caveat language.

Execution steps:

1. Use `EMNLP_REVIEWER_QUICK_MAP.md` as the first response map.
2. Use `EMNLP_REVIEW_RESPONSE_PLAYBOOK.md` for formal rebuttal variants.
3. Keep every response tied to one artifact path.
4. In final read-through, mark and fix only sentences that create claim
   overreach.
5. Keep A3D as the clean flagship and DAD as the hard stress test.

Completion criterion:

- A reviewer can answer "what is the research object?" in one sentence:
  the auditable language-grounded semantic interface.
- A reviewer can answer "why EMNLP?" from the ontology governance and language
  interface evidence, not from raw driving scores alone.
- A reviewer can answer "what does DAD prove?" without using causal or
  policy-level wording.

## P2: Best-Paper Evidence Escalation

Goal: define the only evidence path that could materially narrow the best-paper
gap without damaging submission safety.

Distance: `large`

Current blockers:

1. DAD full-vs-ablation evidence is mixed.
2. Actor-policy timing remains support-only.
3. The paper has strong A3D and honest DAD evidence, but not broad enough
   independent confirmation for a decisive best-paper case.

Allowed compute path only after explicit GPU authorization:

1. Define exactly one DAD mechanism-hardening block.
2. Write the success / tie / failure interpretation before launch.
3. Launch only the predeclared block.
4. Aggregate all runs before changing claims.
5. If evidence is mixed, keep DAD as a stress test and do not force a stronger
   mechanism claim.

Forbidden best-paper moves:

- Do not rerun completed ontology multi-seed or A3D headline blocks just to
  hunt for cleaner numbers.
- Do not promote actor-policy timing to a causal claim.
- Do not rewrite the central story around DAD.
- Do not treat packaging polish as best-paper evidence.

Completion criterion for escalation:

- A new DAD mechanism block materially improves mechanism clarity in aggregate,
  or the paper explicitly remains in oral-ready rather than best-paper-ready
  mode.

Predeclared block:

- `EMNLP_DAD_MECHANISM_HARDENING_PREDECLARED_PLAN_20260427.md`
- Launcher: `paper/emnlp2026/run_dad_mechanism_lightreg_block.sh`
- Status: launched `3/3` on 2026-04-27 UTC; current live status is tracked in
  `output/emnlp2026_support/dad_mechanism_lightreg_status.md`
- Watcher: `lfcrash_dad_lightreg_watch_20260427` refreshes the status every
  five minutes until `3/3` runs complete or an incomplete block loses all live
  training processes.

## P3: Final Submission Operations

Goal: make the upload path low-risk once human decisions are complete.

Execution steps:

1. Confirm human-selected author list and ARR account status.
2. Confirm venue commitment and reviewer-registration logistics.
3. Run final human PDF read-through:
   page 1, Figure 1, Table 1, key result tables, and appendix opening.
4. Run `bash paper/emnlp2026/run_submission_sanity_checks.sh`.
5. If a new freeze is explicitly authorized, run:
   `bash paper/emnlp2026/freeze_arr_submission.sh`.
6. Verify with:
   `bash paper/emnlp2026/verify_arr_freeze.sh`.
7. Upload only the tarball recorded in `EMNLP_STAGE_STATUS.md`.

Stop condition:

- If the next step is author registration, venue commitment, or upload, stop
  for human action.

## Next 10 Actions

1. Maintain sanity gate and verifier.
2. Keep this dashboard and `EMNLP_ORAL_BEST_PAPER_GAP_LEDGER_20260427.md`
   aligned.
3. Keep reviewer maps synchronized with DAD-as-stress-test wording.
4. Avoid any additional GPU work beyond the authorized DAD light-reg block
   unless explicitly approved.
5. If the first light-reg block remains in the mixed zone, launch the
   low-regularization follow-up in
   `paper/emnlp2026/run_dad_mechanism_lightreg_lowreg_block.sh` with explicit
   approval.
6. Let the DAD watcher monitor the launched mechanism-hardening block until the
   final `3/3` aggregate is available.
7. Final human PDF read-through.
8. If read-through finds claim drift, make minimal wording fixes and rerun sanity.
9. If wording changes are made, cut a new freeze only after explicit approval.
10. Keep latest freeze path in `EMNLP_STAGE_STATUS.md`.
11. After upload, shift to rebuttal-readiness mode rather than new-story mode.

## Current Bottom Line

The remaining distance to oral is mainly framing and defense discipline. The
remaining distance to best paper is real research evidence: DAD mechanism
clarity or broader independent confirmation. The correct execution posture is
submission-lock first, oral-defense hardening second, and only then a tightly
scoped compute-authorized best-paper escalation.

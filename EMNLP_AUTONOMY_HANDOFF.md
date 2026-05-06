# EMNLP Autonomy Handoff

Date: 2026-04-29

## Current Autonomy State

- `lf24` is the correct LFCRASH-CBM Codex/Ralph loop launcher.
- The first `lf24` attempt failed because Ralph required `.omx/prd.json`.
- `.omx/prd.json` has been created and validated.
- A later `lf24` attempt passed the Ralph PRD gate and migrated the PRD into
  `.omx/plans/prd-lfcrash-cbm-arr-emnlp-submission-finalization.md`.
- The current blocker is Codex CLI authentication inside tmux, not the project
  plan or the LFCRASH launcher.
- As of 2026-04-28, the `lfcrash24-20260426T133214Z` tmux session is still
  blocked at the Codex sign-in prompt, so Ralph persistence has not progressed
  beyond startup.

## Active / Latest Tmux Session

- Session: `lfcrash24-20260426T133214Z`
- Attach command:
  `tmux attach -t lfcrash24-20260426T133214Z`
- Current state: waiting at the Codex CLI sign-in screen.

## What Was Advanced Without Waiting For Tmux Auth

- The submission sanity script now behaves as a real gate:
  it exits nonzero on missing core files, stale first-25 PDF, PDF page-count
  mismatch, nonempty PDF metadata fields, missing support artifacts, identity or
  link scan hits, or stale project-state references.
- The sanity report now includes:
  manuscript-source freshness, PDF page counts, PDF metadata fallback checks,
  expanded key support artifacts, claim/evidence text audit, stale-state scan,
  and a final fatal-count verdict.
- `paper/emnlp2026/run_claim_evidence_audit.py` now writes
  `paper/emnlp2026/claim_evidence_audit_report.md`; the latest audit reports
  `critical_blockers=0` and leaves only scoped causal/proof wording as
  advisory review items.
- The latest verified sanity run reports `OK fatal_count=0`.
- Freeze creation now runs strict sanity before packaging.
- Freeze creation now updates `EMNLP_STAGE_STATUS.md` to the new stamp before
  packaging, then verifies that the frozen stage status and manifest agree.
- Freeze creation now treats the stage-status update as transactional: if
  sanity, packaging, or freeze verification fails, the previous
  `EMNLP_STAGE_STATUS.md` is restored automatically.
- `paper/emnlp2026/verify_arr_freeze.sh` verifies tarballs, manifests,
  checksums, key artifacts, stage-status stamp coherence, and the frozen sanity
  verdict.
- Freeze packaging now copies only the two static crash visualizations actually
  referenced by the paper, excluding the old interactive HTML and gallery
  caption text from the ARR package.
- `EMNLP_REVIEWER_QUICK_MAP.md` is now the compact reviewer-navigation sheet:
  it maps EMNLP fit, ontology modeling, DAD fragility, actor-policy scope, and
  intervention scope to exact evidence artifacts.
- `EMNLP_REVIEW_RESPONSE_TEMPLATES.md` now contains guarded response drafts for
  likely reviewer concerns, with explicit "do not say" boundaries to prevent
  claim overreach.
- `EMNLP_REVIEW_RESPONSE_PLAYBOOK.md` now provides short and formal rebuttal
  variants for the likely reviewer attacks, tied to current artifact paths.
- `EMNLP_FINAL_READINESS_AUDIT.md` now records the current AI-checkable package
  readiness and explicitly separates remaining human-only upload logistics from
  technical blockers.
- `EMNLP_ORAL_BEST_PAPER_GAP_LEDGER_20260427.md` now records the current
  distance from ARR-ready submission to oral and best-paper competitiveness,
  replacing the root-level April 20 best-paper plan as the active planning
  ledger.
- `EMNLP_ORAL_BEST_PAPER_EXECUTION_DASHBOARD_20260427.md` turns the gap ledger
  into an execution dashboard: ARR upload `95/100` internal readiness, oral
  `78/100`, best paper `52/100`, with P0 submission lock, P1 oral defense,
  P2 compute-gated best-paper escalation, and P3 upload operations.
- `EMNLP_ORAL_ACCEPT_CASE_ONEPAGER_20260427.md` is the compact oral accept-case
  sheet: one core accept argument, one thirty-second pitch, evidence paths,
  safe numbers, boundaries, and forbidden lines.
- `EMNLP_DAD_MECHANISM_HARDENING_PREDECLARED_PLAN_20260427.md` now predeclares
  the DAD light-regularization mechanism block and its success / tie / failure
  criteria. The companion launcher
  `paper/emnlp2026/run_dad_mechanism_lightreg_block.sh` defaults to dry-run and
  refuses `--execute` unless `LFCRASH_ALLOW_GPU=1` is set.
- After explicit GPU authorization, the DAD light-regularization block was
  launched as three tmux-held runs on GPU `7`:
  `lfcrash_dad_lightreg_r1_20260427`,
  `lfcrash_dad_lightreg_r2_20260427`, and
  `lfcrash_dad_lightreg_r3_20260427`.
- `paper/emnlp2026/summarize_dad_mechanism_lightreg_status.py` now writes
  `output/emnlp2026_support/dad_mechanism_lightreg_status.md/json` so the live
  block can be monitored without manually inspecting tmux.
- `paper/emnlp2026/watch_dad_mechanism_lightreg_status.py` is running in tmux
  session `lfcrash_dad_lightreg_watch_20260427`; it refreshes the same status
  files every five minutes and exits when all three runs complete or the block
  loses all live training processes before completion.
- `paper/emnlp2026/run_pdf_first_read_audit.py` now writes
  `paper/emnlp2026/pdf_first_read_audit_report.md`; this gate checks the first
  reviewer-facing surfaces for the semantic-interface story before freeze.
- `paper/emnlp2026/run_reviewer_defense_audit.py` now writes
  `paper/emnlp2026/reviewer_defense_coverage_report.md`; this gate checks that
  the rebuttal preparation package covers EMNLP fit, ontology modeling, DAD
  fragility, actor-policy scope, intervention scope, human-audit scope, and
  the single strongest accept-case answer.
- `paper/emnlp2026/refresh_emnlp_status.py` now includes
  `paper/emnlp2026/visualize_experiment_portfolio.py`, so each full refresh
  regenerates `paper/figures/insight_fig9_experiment_portfolio.pdf/png`.
- `paper/emnlp2026/watch_arch_extension_status.py` now supports structured
  script output logging and transition-aware full refresh triggers.
  Specifically, when `(dad_completed, a3d_completed, running, failed)` changes,
  it can automatically run `refresh_emnlp_status.py`; it always runs a final
  refresh before exiting on completion or incomplete no-live-process failure.
- The architecture-extension watcher has been restarted with the new logic in
  tmux session `lf_arch_watch` (created 2026-04-28 07:33 UTC), still running at
  180-second cadence.
- On 2026-04-29, a full refresh and submission sanity run exposed three
  AI-checkable blockers: stale `insight_emnlp.pdf`, one false-positive
  hype/path hit from a CRASH artifact path containing `sota`, and two
  first-read caption coverage misses. These were fixed in the EMNLP sources
  without changing the central claim.
- The refreshed Figure 1 caption now explicitly defines the object as named
  risk concepts that can be inspected, compared, and audited as the scene
  evolves. The protocol-map caption now explicitly separates headline
  prediction, controlled support, and ontology science so it is not mistaken for
  one pooled leaderboard.
- Long support-status paths in the manuscript text were replaced by prose
  descriptions to avoid LaTeX overfull boxes and paper-facing path clutter.
- The EMNLP PDF has been recompiled and the first-25-page PDF regenerated
  (`insight_emnlp.pdf`: 35 pages; `insight_emnlp_first25.pdf`: 25 pages).
- Latest verified checks on 2026-04-29:
  `run_claim_evidence_audit.py` reports `critical_blockers=0`;
  `run_pdf_first_read_audit.py` reports `critical_blockers=0`;
  `run_reviewer_defense_audit.py` reports `critical_blockers=0`;
  `run_submission_sanity_checks.sh` reports `OK fatal_count=0`.
- Latest architecture-extension snapshot on 2026-04-29 03:10 UTC:
  DAD RWKV `0/3` completed with `3` running, A3D h384 `1/3` completed with
  `2` running, `0` failed. The active watcher remains `lf_arch_watch`.
- On 2026-04-29, an oral-focused expert sprint was added:
  `EMNLP_ORAL_PERFECTION_SPRINT_20260429.md`. It records the remaining oral
  gaps as first-read object clarity, evidence-tier discipline, DAD fragility,
  actor-policy scope, related-work authority, and human-validation boundaries.
- The main manuscript was polished to remove paper-facing acceptance-target
  meta language. In particular, the introduction no longer uses an
  "Oral-first, best-paper-in-progress" paragraph, and the conclusion no longer
  contains a review-readiness / acceptance-target summary. These were replaced
  with evidence-tier and scope language without changing the central claim.
- Experiments and appendix wording were tightened from "reviewer" /
  "submission" phrasing toward claim-tier, evidence-depth, and external-audit
  language. This is intended to make the paper read as a finished scientific
  manuscript rather than as an internal submission plan.
- After the oral-polish edits, the PDF was recompiled and
  `insight_emnlp_first25.pdf` regenerated. Latest verified checks on
  2026-04-29 03:36 UTC: `run_claim_evidence_audit.py` reports
  `critical_blockers=0`; `run_pdf_first_read_audit.py` reports
  `critical_blockers=0`; `run_reviewer_defense_audit.py` reports
  `critical_blockers=0`; `run_submission_sanity_checks.sh` reports
  `OK fatal_count=0`. LaTeX warning scan reports no `Overfull`,
  `LaTeX Warning`, undefined reference, or undefined citation hits.
- Latest architecture-extension snapshot on 2026-04-29 03:36 UTC:
  DAD RWKV `0/3` completed with `3` running, A3D h384 `1/3` completed with
  `2` running, `0` failed. The active watcher remains `lf_arch_watch`.
- On 2026-04-29, the defense package was synchronized with the current DAD
  support boards. `EMNLP_REVIEWER_QUICK_MAP.md`,
  `EMNLP_REVIEW_RESPONSE_PLAYBOOK.md`, and
  `EMNLP_REVIEW_RESPONSE_TEMPLATES.md` now use the current DAD evidence split:
  canonical `68.19%` AP / `1.75s` mTTA, clean three-seed diagnostic
  `62.31% +/- 1.90` AP / `2.07s +/- 0.09s` mTTA, recovery block
  `63.52% +/- 0.81` AP / `2.16s +/- 0.25s` mTTA, and matched full support
  block `63.19% +/- 1.21` AP / `2.17s +/- 0.05s` mTTA.
- PDF visual review was performed with PyMuPDF-rendered page images because
  `pdftoppm` was not available in the environment. Pages 1 and 3 looked clean
  for first-read story and claim-tier routing. Page 4 showed ACL review line
  numbers close to a continued sentence at the top of the page; the source text
  was shortened from "light human review all become..." to "light review are
  all recorded..." to reduce crowding. This is a review-style line-number
  artifact, not source-text contamination.
- Latest verified checks on 2026-04-29 04:29 UTC:
  `run_submission_sanity_checks.sh` reports `OK fatal_count=0`;
  stale defense-number scan reports no old `59.55%` / `2.26s` DAD diagnostic
  strings in the quick map, playbook, templates, or paper sources.
- Latest architecture-extension snapshot on 2026-04-29 04:29 UTC:
  DAD RWKV `0/3` completed with `3` running, A3D h384 `1/3` completed with
  `2` running, `0` failed. The active watcher remains `lf_arch_watch`.

## Current Submission State

- ARR-ready: yes.
- Oral-ready: yes, under the current project audit.
- Best-paper-ready: no; remaining gap is strategic evidence breadth and DAD
  mechanism clarity, not missing submission packaging or missing matched DAD
  full-support coverage.
- A3D remains the clean flagship result.
- DAD remains the harder stress test and should not be oversold.
- Actor-policy timing remains support evidence, not a main causal claim.

## Safe Next Commands

Run after any source/support edit:

```bash
bash paper/emnlp2026/run_submission_sanity_checks.sh
```

Run a focused claim/evidence text audit:

```bash
python3 paper/emnlp2026/run_claim_evidence_audit.py
```

Refresh support boards after any completed support run:

```bash
python paper/emnlp2026/refresh_emnlp_status.py
```

Cut a new freeze only after sanity is clean:

```bash
bash paper/emnlp2026/freeze_arr_submission.sh
```

Verify a freeze:

```bash
bash paper/emnlp2026/verify_arr_freeze.sh
```

## Stop Conditions

- ARR upload, author registration, venue commitment, or author-list decisions.
- Any GPU job, multi-seed rerun, checkpoint download, or large data regeneration.
- Any claim that would promote DAD mechanism evidence or actor-policy timing
  beyond the current support boards.

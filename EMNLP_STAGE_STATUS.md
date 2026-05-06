# EMNLP / ARR Stage Status

Date: 2026-05-06

## Current stage

The paper is in an **ARR-ready pre-submission stage**.

## 2026-05-06 refresh

- EMNLP compile and sanity were rerun after the latest appendix updates:
  `paper/emnlp2026/submission_sanity_report.txt` now reports
  `OK fatal_count=0` at `2026-05-06T06:53:30Z`.
- Top-conference and oral-readiness audits were refreshed:
  `output/emnlp2026_support/top_conference_quality_audit.json` and
  `output/emnlp2026_support/oral_readiness_audit.md`.
- Oral/best-paper distance was rebaselined in
  `EMNLP_ORAL_BEST_PAPER_GAP_LEDGER_20260506.md`.
- A deadline-aware execution checklist for the ARR window was created in
  `EMNLP_ARR19DAY_TODO_20260506.md` (T-19 to T-0 plan toward 2026-05-25 AoE).
- A new technical freeze candidate was created and verified:
  `ARR20260506T073630Z` with tarball
  `paper/emnlp2026/frozen/insight_emnlp_arr_freeze_ARR20260506T073630Z.tar.gz`.
- Latest sanity rerun reports `OK fatal_count=0` at `2026-05-06T08:06:17Z`.
- No-change rehearsal gate was rerun and passed for the current freeze candidate:
  `bash paper/emnlp2026/verify_arr_freeze.sh ARR20260506T073630Z`.
- Upload-day operations are now codified in:
  `EMNLP_UPLOAD_DAY_RUNBOOK_20260506.md`.
- Virtual pre-review sprint revision was applied on 2026-05-06 with:
  narrative compression, matched-protocol wording cleanup, explicit
  seed-aggregated ontology effect-size reporting, and a semantic-validity table.
- Latest post-revision sanity rerun reports `OK fatal_count=0` at
  `2026-05-06T09:04:24Z`.
- Phase-1 DAD stability extension has been launched:
  `dad_shared_perfect_v1` additional seeds `{7,11,2718,314,2026}` (running).
- Phase-2 DAD stability extension has been launched with queued workers:
  `dad_shared_historical_full` and `dad_shared_risk_core_v1` additional seeds
  `{7,11,2718,314,2026}` on GPU `{2,3}`.
- Virtual pre-review issue closure is tracked in:
  `EMNLP_VIRTUAL_REVIEW_ACTION_MATRIX_20260506.md`.
- A new size-matched ontology control track is now running to decouple source
  effects from concept-count effects:
  `paper/emnlp2026/run_ontology_size_matched_controls.sh`
  (DAD+A3D, 30/80 budgets, seeds `{42,123,3407}` on GPU `{5,7}`).
- Live monitor for this new block:
  `output/emnlp2026_support/ontology_size_matched_status.md`.
- Auto effect-size summary for this block:
  `output/emnlp2026_support/ontology_size_matched_effects.md`.
- Current extension progress snapshot:
  `output/emnlp2026_support/dad_ontology_seed_extension_status.md`
  now uses a completion-safe rule (finished process + result file):
  `historical_full: 3/8 completed + 2 running`,
  `risk_core_v1: 3/8 completed + 0 running`,
  `perfect_v1: 3/8 completed + 5 running`.
- Size-matched queue is live with `DAD historical-stratified (30): 0/3 completed + 2 running`
  under the same completion-safe accounting rule.
- Latest post-update sanity rerun reports `OK fatal_count=0` at
  `2026-05-06T10:11:32Z`.

## Upload Candidate Checklist (Human-Gated)

1. Sanity gate green:
   `bash paper/emnlp2026/run_submission_sanity_checks.sh` -> `OK fatal_count=0`.
2. Candidate freeze tarball path is explicit in this file.
3. Freeze package verifier passes:
   `bash paper/emnlp2026/verify_arr_freeze.sh <tarball>`.
4. Human read-through completed for page 1, Figure 1, Table 1, and appendix opening.
5. Human confirms author/reviewer-registration/venue-commit logistics before upload.

This is no longer an idea-stage draft or an experiment-exploration draft. The
main scientific story, evidence hierarchy, appendix framing, and submission
packaging are all in place. The work is now in the final submission lane:
freeze, sanity, manual read-through, and upload logistics.

## What is already stable

- The main framing has been re-centered around a **language-grounded risk
  concept interface** rather than a venue-shifted NeurIPS story.
- Main paper sections are aligned: abstract, introduction, method,
  experiments, and conclusion now tell the same story.
- The appendix has been aligned to the same semantic-interface framing.
- Controlled ontology evidence is complete and supports the operating-point
  claim under matched launchers.
- Low-cost semantic support analyses are complete and integrated into the paper
  narrative.
- The paper compiles cleanly on the remote machine and passes the current
  submission sanity script.
- The first-25-page PDF is now regenerated during compilation and checked for
  freshness by the submission sanity script.
- DAD hardening status now correctly records that the matched full support block
  is complete and that the remaining DAD issue is mechanism fragility, not
  missing coverage.
- The master execution plan, oral push plan, rerun plan, review response map,
  reviewer quick map, support-results note, and last-minute checklist have
  been refreshed to the current multi-seed / DAD-500 evidence state.
- Page-1 framing, Figure 1 caption, and the protocol-map caption now more
  directly position the paper as a semantic-interface contribution rather than
  a pooled leaderboard story.
- The appendix opening now includes a reader roadmap and keeps timing /
  intervention claims tied to scoped semantic-interface evidence.
- `EMNLP_FINAL_READINESS_AUDIT.md` records the current AI-checkable readiness
  state and separates remaining human-only upload logistics from technical
  blockers.
- `EMNLP_ORAL_BEST_PAPER_GAP_LEDGER_20260427.md` records the current
  ARR-to-oral and oral-to-best-paper distance after the matched DAD full
  support block completed, with DAD mechanism clarity and broader evidence
  still marked as the best-paper gaps.
- `EMNLP_ORAL_BEST_PAPER_EXECUTION_DASHBOARD_20260427.md` converts that gap
  assessment into a concrete execution dashboard with internal readiness
  scores, P0--P3 workstreams, stop conditions, and the next 10 actions.
- `EMNLP_DAD_MECHANISM_HARDENING_PREDECLARED_PLAN_20260427.md` predeclares the
  only currently justified DAD mechanism-hardening escalation block, including
  success / tie / failure criteria and a guarded dry-run-first launcher.
- `output/emnlp2026_support/dad_mechanism_lightreg_status.md` tracks the live
  predeclared DAD light-regularization block; it is execution status only until
  all three runs complete and are aggregated.
- The DAD light-regularization block has been launched as `3/3` tmux-held runs
  on GPU `7` after explicit GPU authorization; no paper claims should be
  upgraded until the completed aggregate is available.
- The bounded watcher `lfcrash_dad_lightreg_watch_20260427` refreshes DAD
  light-reg status every five minutes and exits on `3/3` completion or an
  incomplete no-live-process condition.
- `EMNLP_ORAL_ACCEPT_CASE_ONEPAGER_20260427.md` compresses the oral-level
  accept case into a single internal sheet with the core pitch, evidence spine,
  usable numbers, boundaries, and forbidden response lines.
- The submission sanity gate now includes a PDF first-read audit that checks
  whether page 1, Figure 1 caption, protocol-map caption, and appendix roadmap
  still present the semantic-interface story clearly.
- The submission sanity gate now includes a reviewer defense coverage audit
  that checks the reviewer quick map, response map, playbook, templates, and
  tracker cover the expected attack surfaces and cite existing evidence paths.
- The submission sanity gate now checks PDF page counts, first-25 freshness,
  manuscript-source freshness, PDF metadata via a Python fallback, expanded
  support artifacts, claim/evidence wording, and stale project-state
  references.
- The focused claim/evidence audit now reports zero critical blockers; the
  remaining strong-wording hits are scoped causal/proof/guarantee disclaimers
  rather than unbounded paper claims.
- Freeze creation now runs the strict submission sanity gate before packaging,
  so a failed sanity check blocks new ARR freeze tarballs.
- Freeze creation now updates `EMNLP_STAGE_STATUS.md` transactionally: a failed
  sanity, packaging, or verification step restores the previous recommended
  freeze paths.
- Freeze packages now have a standalone verifier:
  `paper/emnlp2026/verify_arr_freeze.sh`, which checks the tarball, manifest,
  package files, key artifacts, stage-status stamp, and sanity verdict.
- Freeze packages now include `EMNLP_REVIEWER_QUICK_MAP.md` as the compact
  rebuttal/navigation sheet for final human review and response preparation.
- Freeze packages now include `EMNLP_REVIEW_RESPONSE_TEMPLATES.md`, a guarded
  set of rebuttal-style response drafts tied to current evidence artifacts.
- Freeze packages now include `EMNLP_REVIEW_RESPONSE_PLAYBOOK.md`, which gives
  short-answer, formal-response, evidence-payload, and forbidden-response
  variants for likely reviewer attacks.
- Freeze packages now include `EMNLP_FINAL_READINESS_AUDIT.md` as an internal
  final-readiness ledger, without treating it as human validation.
- Freeze packages now include `paper/emnlp2026/pdf_first_read_audit_report.md`
  and its generator as a mechanical first-read guardrail.
- Freeze packaging now includes only the static crash visualizations referenced
  by the manuscript, avoiding stale gallery prose and external-link HTML in the
  ARR bundle.
- Final layout follow-up on 2026-05-01 added a strict ARR/EMNLP long-paper
  candidate at `paper/emnlp2026/insight_emnlp_arr8.tex` and
  `paper/emnlp2026/insight_emnlp_arr8.pdf`. This candidate keeps the main
  semantic-interface story, headline tables, controlled ontology table, and
  support/stress evidence in an ACL-style page-limit-aware flow, with
  Conclusion/Limitations/References before the appendix and appendix beginning
  on page 9. It is a candidate upload path, not yet a new freeze.
- Visual-layout rework on 2026-05-01 added a separate polished candidate at
  `paper/emnlp2026/insight_emnlp_polished.tex` and
  `paper/emnlp2026/insight_emnlp_polished.pdf`, with rebuilt compact Figure 1,
  semantic-interface pipeline, and AP--warning-time figure plus consolidated
  result tables. The render contact sheet is
  `paper/emnlp2026/layout_audit_polished/contact_sheet.png`, and the audit note
  is `paper/emnlp2026/POLISHED_LAYOUT_AUDIT.md`. This is a visual candidate
  only, not a new freeze.
- Follow-up visual inspection on 2026-05-01 rejected the polished candidate:
  its appendix starts on page 6, so it is too compressed for EMNLP/ARR
  long-paper review. The current strongest non-frozen read-through candidate is
  `paper/emnlp2026/insight_emnlp_oral8.tex` /
  `paper/emnlp2026/insight_emnlp_oral8.pdf`, with appendix starting on page 9,
  a rebuilt framework figure at
  `paper/figures/emnlp_fig2_framework_image2_overlay.pdf`, and a render audit
  at `paper/emnlp2026/layout_audit_oral8/contact_sheet.png`. GPT Image 2
  framework prompt plumbing is stored under
  `paper/emnlp2026/gpt_image2_framework/`, but the current AIHubMix generation
  calls failed during the long-running image request; the paper-facing figure
  therefore uses deterministic scientific overlay output until a successful
  Image 2 bitmap base is available.
- The next freeze cut from the current working tree will include
  `EMNLP_ORAL_ACCEPT_CASE_ONEPAGER_20260427.md`,
  `EMNLP_ORAL_BEST_PAPER_GAP_LEDGER_20260427.md` and
  `EMNLP_ORAL_BEST_PAPER_EXECUTION_DASHBOARD_20260427.md`,
  `EMNLP_DAD_MECHANISM_HARDENING_PREDECLARED_PLAN_20260427.md`, and the guarded
  DAD mechanism launcher as key planning artifacts; the latest verified tarball
  below remains the last fully frozen upload candidate.
- Ralph PRD startup state has been initialized via `.omx/prd.json`; the
  remaining `lf24` blocker is Codex CLI authentication inside tmux, documented
  in `EMNLP_AUTONOMY_HANDOFF.md`.
- Freeze packaging is now standardized around `ARR<UTC timestamp>` naming.

## Current strongest claims

- Ontology construction is a modeling choice, not hidden preprocessing.
- The semantic bottleneck is an auditable research object with provenance,
  family coverage, and matched ontology comparisons.
- Ontology choice changes the AP--mTTA operating point on DAD and A3D.
- The model reaches a strong interpretable operating point, especially on A3D.

## Current boundaries

- Actor-policy timing is support evidence, not the central pillar of the paper.
- Intervention evidence supports a structurally meaningful and partially
  intervenable concept interface, not a full causal timing proof.
- DAD is still the more fragile dataset and is treated that way in the paper.

## What remains before actual submission

- One final human read-through of page 1, key tables, main figures, and the
  appendix opening.
- Final authorship / ARR reviewer-registration / venue-commitment logistics.
- Optional micro-polish only if it clearly improves readability.

## What should probably *not* be reopened now

- Large new experiment branches that do not change the acceptance picture.
- Another full story rewrite.
- Leaderboard chasing at the cost of the semantic-interface narrative.

## Latest recommended freeze

- Remote freeze directory:
  `/data/sony/LFCRASH/LFCRASH-CBM/paper/emnlp2026/frozen/ARR20260506T073630Z`
- Remote tarball:
  `/data/sony/LFCRASH/LFCRASH-CBM/paper/emnlp2026/frozen/insight_emnlp_arr_freeze_ARR20260506T073630Z.tar.gz`

## Residual technical notes

- Submission sanity currently reports only underfull layout warnings; no
  overfull issues remain in the latest compiled log.
- `pdfinfo` is still unavailable on the remote machine, but the sanity script
  now falls back to Python PDF checks for page counts and metadata fields.

# LFCRASH-CBM Automation Notes

Use OMX/Ralph for bounded submission-lane work in this repository. The project is currently in ARR/EMNLP pre-submission finalization, not broad experiment exploration.

## Project North Star

The paper story is a language-grounded risk concept interface. The main claim is semantic-interface science: ontology choice changes the AP--mTTA operating point, with A3D as the clean flagship dataset and DAD framed honestly as more fragile.

## Default Loop Objective

When triggered with `loop`, `lfx`, `lfcrashx`, `@loop`, or `自动推进`, continue the submission-safe path:

- inspect `EMNLP_STAGE_STATUS.md`, `EMNLP_MASTER_EXECUTION_PLAN.md`, `EMNLP_SUPPORT_RESULTS.md`, and the latest paper freeze;
- refresh or run only low-risk sanity checks when needed;
- summarize blockers, final manual-readthrough items, upload logistics, and any stale freeze pointers;
- keep reviewer-facing claim wording aligned with the current evidence hierarchy.

## Safe Automation Targets

- Run or inspect submission sanity scripts after manuscript/support-document edits.
- Refresh status/checklist documents that summarize existing artifacts.
- Verify latest freeze paths and package freshness.
- Prepare final-readthrough checklists, reviewer-response maps, and bounded claim ledgers.
- Inspect existing output tables/figures and paper logs.

## Guardrails

- Do not reopen the central paper story.
- Do not launch new large experiments, Optuna sweeps, multi-seed reruns, or GPU jobs without explicit approval.
- Do not repeat completed ontology multi-seed, A3D headline, DAD full-support, or DAD-500 language-side audits unless a concrete artifact is corrupted.
- Do not overwrite frozen artifacts unless intentionally cutting a new freeze.
- Do not promote actor-policy timing from support evidence to a main causal claim.
- Do not chase CRASH-only leaderboard gains before submission.
- Do not hide DAD fragility; keep DAD wording bounded.

## Stop Conditions

Stop and report when the next step requires:

- author list, ARR reviewer registration, venue commitment, or upload logistics;
- a manual human read-through of PDF page 1, key tables, figures, or appendix opening;
- GPU compute or long-running training;
- cutting a new freeze;
- changing the paper's central claim.

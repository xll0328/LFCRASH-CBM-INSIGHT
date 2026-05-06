# NeurIPS 2026 Oral / Best Paper Execution Plan

Date: 2026-04-20  
Project: `LFCRASH-CBM` / `INSIGHT`  
Execution owner for this round: Codex

## Goal

The target is no longer just to keep the work in an EMNLP-safe lane.

The target is to turn the current repository into a **credible NeurIPS 2026 oral-level paper**
with a real, evidence-backed path toward **best-paper competitiveness**.

That requires three things simultaneously:

1. a tighter NeurIPS-facing thesis than the current EMNLP packaging,
2. a cleaner and more reviewer-proof evidence hierarchy,
3. continuous execution on the highest-value missing evidence block rather than compute drift.

This document is operational. It is meant to be executed continuously, not admired.

## 2026-05-01 Follow-up Status

Current action taken: added and ran a NeurIPS-specific lightweight sanity gate
at `paper/neurips2026/run_neurips_sanity_checks.py`. The gate checks that the
active NeurIPS paper remains tied to the current submission results ledger,
keeps the required actor-policy and intervention scope caveats, has a current
compiled PDF, and marks the legacy `paper/NEURIPS2026_full_draft.md` as a
warning source when it still contains stale high-risk language.

Latest local outcome:

- Active compiled paper: `paper/neurips2026/insight_main.pdf`
- Sanity report: `paper/neurips2026/neurips_sanity_report.md`
- Expected state: zero fatal blockers; legacy draft warnings are acceptable
  only if future writing continues to treat that file as stale reference text.

Next safe command:

```bash
python3 paper/neurips2026/run_neurips_sanity_checks.py
```

Stop condition: if the gate reports any fatal blocker, fix the active
NeurIPS paper or ledger alignment before cutting or sharing a new paper freeze.

## North-Star Thesis

The strongest repo-grounded NeurIPS thesis is:

> Interpretable accident anticipation should be built on an intervenable semantic state space,
> not post-hoc explanation over a black-box predictor.

This thesis is already consistent with:

- `paper/neurips2026/thesis_claims_evidence_matrix.md`
- `paper/neurips2026/claim_evidence_audit.json`
- `paper/neurips2026/reviewer_proof_experiment_manifest.md`
- the current A3D and controlled-ontology support artifacts

It is **not** consistent with the most optimistic legacy NeurIPS draft numbers in
`paper/NEURIPS2026_full_draft.md`. Those older draft values should be treated as stale unless they
are re-proven by current repo artifacts.

## Current Truth Set

As of `2026-04-20 08:50 UTC`, the repository supports the following stable facts.

### Stable paper-facing evidence

- EMNLP / ARR stage is effectively submission-ready:
  - `ARR-ready=True`
  - `Oral-ready=True`
  - `Best-paper-ready=False`
- Controlled ontology multi-seed block is complete:
  - `18/18` seeded cells completed
- A3D headline multi-seed is complete:
  - `94.16% +- 0.95` AP
  - `4.619s +- 0.417s` mTTA
- Canonical DAD submission line remains:
  - `68.19%` AP
  - `1.75s` mTTA
- DAD clean-seed diagnostic exists:
  - `62.31% +- 1.90` AP
  - `2.07s +- 0.09s` mTTA

Primary sources:

- `EMNLP_STAGE_STATUS.md`
- `output/emnlp2026_support/oral_readiness_audit.md`
- `output/emnlp2026_support/multiseed_ontology_status.md`
- `output/emnlp2026_support/a3d_headline_multiseed_status.md`

### Stable scope boundaries

- The paper's strongest predictive evidence is currently on `A3D`.
- `DAD` remains the harder and noisier stress test.
- Archived actor-branch evidence is still not strong enough for an aggressive
  policy-level timing-control claim.
- Intervention evidence is real but still belongs in the
  "partial structural intervenability" bucket, not "causal control solved".

Primary sources:

- `paper/neurips2026/timing_faithfulness_package.md`
- `paper/neurips2026/threats_to_validity.md`
- `paper/neurips2026/thesis_claims_evidence_matrix.md`

## Core Judgment

The project is **not yet NeurIPS best-paper ready** for one simple reason:

the repo already contains a strong thesis, usable A3D evidence, and ontology science,
but the full evidence package is still asymmetrical on the hardest dataset and still too weak on
policy-level timing proof.

Right now the best path is **not**:

- another ontology search branch,
- another old-draft rewrite,
- or pretending the legacy NeurIPS draft numbers are current.

Right now the best path is:

1. finish the matched `DAD full-support` block cleanly,
2. keep the paper's central thesis on semantic state construction and auditability,
3. strengthen timing/intervention evidence only where the repo can actually support it,
4. rewrite the NeurIPS paper around claims that survive hostile review.

## What "NeurIPS Oral" Actually Requires Here

For this repo, an oral-level NeurIPS package would require all of the following:

1. A memorable thesis that is broader than "language grounding helps" and sharper than
   "we are interpretable".
2. A reviewer-proof evidence map separating:
   - canonical headline runs,
   - controlled support blocks,
   - appendix-only diagnostics,
   - stale or support-only local search runs.
3. One clean flagship result family with multi-seed credibility:
   `A3D` already does most of this work.
4. One hard stress-test family where the main method does not look structurally fragile:
   `DAD` is the current bottleneck.
5. Honest scope boundaries around actor-policy evidence and intervention strength.
6. Figures and tables whose logic matches the claim hierarchy.

## What "Best Paper" Would Add Beyond Oral

Best-paper competitiveness would require more than being correct and polished.
For this project, it would require all of the following.

### 1. Claim-evidence tightness

Every memorable claim must be directly defended by a current artifact.
No speculative numbers. No "expected convergence". No outdated draft inflation.

### 2. A stronger hard-case story

`DAD` must stop reading like an unresolved weakness and start reading like
"hard but understood, bounded, and mostly controlled".

### 3. A convincing semantic-science contribution

The paper needs to feel like it changes how people think about accident anticipation:
semantic state construction, ontology governance, timing faithfulness, and intervention behavior
must feel like one coherent scientific object.

### 4. An unforgettable figure or table package

The repo already has the ingredients:

- ontology figures in `paper/neurips2026/`
- intervention casebank assets
- case-study visuals
- timing-faithfulness package

But the paper will only feel best-paper level if these are composed around one clean thesis.

## Non-Negotiable Truth Rules

These rules govern every future edit and experiment.

### Rule 1. Current artifacts beat old drafts

If a number in `paper/NEURIPS2026_full_draft.md` disagrees with:

- `results.json`
- current status boards
- live `train.log`
- paper-facing current `.tex`

then the draft loses.

Practical implication:

- `paper/neurips2026/sec_intro.tex` and `sec_experiments.tex` are already much closer to the
  current truth set
- the highest-risk stale artifact is `paper/NEURIPS2026_full_draft.md`, which still contains
  unsupported A3D and convergence claims

### Rule 2. Canonical lines stay canonical

The main DAD and A3D tables must not silently absorb:

- support-only search runs,
- local exploratory wins,
- incomplete runs,
- partially refreshed analysis outputs.

### Rule 3. Actor claims stay bounded

Until refreshed actor-policy evidence exists, the NeurIPS paper may claim:

- concept-conditioned timing architecture,
- prediction-branch timing faithfulness,
- partial intervention evidence,

but not:

- fully validated policy-level semantic control,
- decisive actor crossing dominance over thresholding,
- solved WHY+WHEN causal control.

### Rule 4. DAD fragility must be reduced, not hidden

If the full matched block underperforms, the paper must narrow the claim rather than bluff through it.

## Ranked Workstreams

### Workstream 1. Finish the matched DAD full-support block

This is still the single highest-value active compute block.

Objective:

- complete three matched `full` runs under the same support recipe already used for
  `no_cbm`, `no_align`, `no_sparse`, and `no_recon`

Current canonical tags:

- `insight_journal_dad_full_r1`
- `insight_journal_dad_full_r2`
- `insight_journal_dad_full_r3`

Success criterion:

- `3/3` completed `results.json`
- aggregate mean/std added to `dad_hardening_status`
- DAD mechanism story can be stated symmetrically against existing support ablations

Why this remains first:

- it directly reduces the biggest current reviewer vulnerability,
- it is already running,
- it is cheaper and more decisive than opening a new branch.

### Workstream 2. Rewrite the NeurIPS claim hierarchy around current evidence

Objective:

- align `paper/neurips2026/` and any new draft text with the current thesis/evidence matrix

Immediate required cleanup:

- remove stale "absolute SOTA" or inflated legacy A3D claims
- remove speculative DAD convergence claims
- downgrade full actor-control language to supported scope
- keep ontology + auditability + semantic-state framing as the paper spine

Primary sources to obey:

- `paper/neurips2026/thesis_claims_evidence_matrix.md`
- `paper/neurips2026/claim_evidence_audit.json`
- `paper/neurips2026/reviewer_proof_experiment_manifest.md`

### Workstream 3. Convert timing/intervention from appendix support into reviewer-proof evidence

Objective:

- keep current timing/intervention evidence, but phrase it with exact scope discipline

Current support:

- prediction-branch timing faithfulness exists and is usable
- partial intervention evidence exists and is usable

Current gaps:

- actor-policy branch evidence is still weak in archived outputs
- strong causal or policy-dominance language is not yet justified

Success criterion:

- main paper uses timing/intervention as convincing but honestly bounded evidence,
  not as overclaimed centerpiece

### Workstream 4. Figure and table package for oral / best-paper reading

Objective:

- make the NeurIPS package read fast and feel coherent

Priority visual assets:

1. one framework figure centered on semantic state and timing control
2. one ontology-governance / controlled-ontology figure
3. one timing-faithfulness figure
4. one intervention case figure with success + failure contrast
5. one DAD/A3D result table with strict protocol separation

The repo already contains most of these ingredients under `paper/neurips2026/`.

### Workstream 5. Submission-proof packaging

Objective:

- ensure the paper survives aggressive review and final packaging

Needed outputs:

- claim-evidence audit
- threats-to-validity section
- reviewer-proof experiment manifest
- final appendix asset index

These assets already partially exist; the work is alignment, not invention.

## Live Execution Status

This section is the execution heartbeat for the currently running DAD block.

### Status at `2026-04-20 08:50 UTC`

- `tmux` watchdog session: `lfcrash_dad_watch`
- current matched full block is alive on `GPU0/3/4`
- all three runs have now passed `epoch 5`
- `epoch 4 eval` is complete and already refreshed into the status board

Latest in-progress checkpoint:

- aggregate at `epoch 4`:
  - `49.85% +- 4.28` AP
  - `3.80s +- 0.39s` mTTA
  - `3.76s +- 0.13s` TTA@R80
  - `0.419 +- 0.015` P@R80
- per-run:
  - `r1`: `52.83%` AP, `3.60s` mTTA
  - `r2`: `52.92%` AP, `3.47s` mTTA
  - `r3`: `43.80%` AP, `4.34s` mTTA

Newest training progress after that eval:

- `r1`: `epoch 5` logged at `2026-04-20 08:48:51 UTC`
- `r2`: `epoch 5` logged at `2026-04-20 08:48:47 UTC`
- `r3`: `epoch 5` logged at `2026-04-20 08:48:42 UTC`

## Live Judgment

The active DAD block has now crossed the most important continuity threshold:

it did **not** die again after `epoch 2` or `epoch 4`.

That matters more than the raw numbers by themselves, because continuity was the main active failure mode.

Numerically, the block is also behaving sensibly:

- `epoch 2` aggregate: `34.70% +- 1.68` AP
- `epoch 4` aggregate: `49.85% +- 4.28` AP

So the curve is moving in the expected direction.

Current caution:

- `r3` remains a meaningfully weaker seed
- therefore the eventual mean/std is still not settled

## Next Checkpoint Logic

### Next meaningful checkpoint

The next decision-worthy checkpoint is `epoch 8 eval`.

Based on current observed pacing:

- `epoch 6 eval` landed around `08:58-09:00 UTC`
- one training epoch is still taking roughly `11-12` minutes
- the `epoch 8` training line should land roughly around `09:18-09:21 UTC`
- the `epoch 8 eval` write should land roughly around `09:20-09:28 UTC`

### What `epoch 8` will tell us

1. Whether the block remains healthy into the mid-run region where best checkpoints often start to matter
2. Whether the current aggregate continues to separate from the old interrupted family rather than just briefly matching it
3. Whether `r3` is closing the gap enough for the final mean/std story to read as robust rather than bimodal

### If `epoch 8` aggregate stays in the mid-to-high 50s or better

Interpretation:

- continuity is now no longer the main DAD blocker
- DAD full-support becomes credible support evidence for a stronger NeurIPS story

### If `epoch 8` stalls or one seed collapses

Interpretation:

- keep the continuity fix in place
- diagnose that seed specifically
- do not over-read `r1/r2`

## Continuous Execution Protocol

This is the default order of execution from here.

1. Keep `lfcrash_dad_watch` alive.
2. Let the current three-run block proceed without interference between meaningful checkpoints.
3. Refresh `output/emnlp2026_support/` after each new synchronized eval checkpoint.
4. Update this document with:
   - the new checkpoint
   - the new judgment
   - the new ETA
5. Do not edit NeurIPS paper claims based on incomplete or stale numbers.
6. After `3/3` runs complete:
   - refresh DAD hardening summaries
   - compare matched `full` vs `no_cbm` / `no_align`
   - rewrite the NeurIPS claim hierarchy using the finished block

## Commands

Refresh support summaries:

```bash
python paper/emnlp2026/refresh_emnlp_status.py
```

Read the DAD hardening board:

```bash
sed -n '1,240p' output/emnlp2026_support/dad_hardening_status.md
```

Check live watchdog:

```bash
tmux capture-pane -pt lfcrash_dad_watch | tail -n 80
```

Check current canonical runs:

```bash
for tag in r1 r2 r3; do
  tail -n 60 output/dad_full_support_block/insight_journal_dad_full_${tag}/train.log
done
```

## Execution Log

### 2026-04-20 08:50 UTC

- reframed the target explicitly to `NeurIPS 2026 Oral / Best Paper`
- audited the current EMNLP stage docs, NeurIPS claim-evidence files, and live DAD block together
- confirmed that the repo already has a strong NeurIPS thesis and artifact scaffold, but not yet a
  best-paper-complete evidence package
- confirmed that the strongest live bottleneck remains the matched `DAD full-support` completion
- confirmed that the current DAD block has reached `epoch 5` on all three seeds
- updated the next decision window to `epoch 6 eval`, expected roughly around `09:03-09:10 UTC`

### 2026-04-20 08:52 UTC

- audited the NeurIPS paper assets themselves
- confirmed that the actively maintained `paper/neurips2026/` TeX files are already relatively
  disciplined about not claiming global AP leadership
- confirmed that the major stale-overclaim source is the legacy standalone draft
  `paper/NEURIPS2026_full_draft.md`, which still contains unsupported numbers such as
  `97.36% AP`, `9.59s mTTA`, and speculative DAD convergence language
- execution consequence:
  future NeurIPS writing work should revise `paper/neurips2026/` first and either archive,
  quarantine, or explicitly deprecate the legacy full draft rather than using it as a source

### 2026-04-20 09:01 UTC

- the synchronized `epoch 6` evaluation is now complete for the active `DAD full-support` block
- per-run metrics:
  - `r1`: `57.23%` AP, `2.32s` mTTA, `2.96s` TTA@R80, `0.443` P@R80
  - `r2`: `59.07%` AP, `2.58s` mTTA, `3.42s` TTA@R80, `0.411` P@R80
  - `r3`: `49.05%` AP, `3.29s` mTTA, `3.97s` TTA@R80, `0.424` P@R80
- aggregate:
  `55.12% +- 4.36` AP,
  `2.73s +- 0.41s` mTTA,
  `3.45s +- 0.41s` TTA@R80,
  `0.426 +- 0.013` P@R80
- scientific interpretation:
  the block has now moved from "continuity repair" to "useful paper evidence"
  because it has surpassed the old interrupted family's rough `epoch 8` mean AP range while
  still preserving synchronized three-run execution
- paper implication:
  if the block continues to improve through `epoch 8+`, the DAD side will become much easier to
  position as a hard stress test with bounded fragility rather than as an unresolved weakness
- next execution window:
  wait for `epoch 8 eval`, expected roughly around `09:20-09:28 UTC`

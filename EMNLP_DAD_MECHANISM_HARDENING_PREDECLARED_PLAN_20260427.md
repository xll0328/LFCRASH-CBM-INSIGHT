# DAD Mechanism-Hardening Predeclared Plan

Date: 2026-04-27
Project: `LFCRASH-CBM`

This is a predeclared plan for the only currently justified DAD-side
best-paper escalation block. It is not an instruction to launch GPU jobs.
Launch requires explicit human approval and the launcher guard described below.

## Execution State

- GPU authorization received in chat on 2026-04-27.
- Launched runs: `3/3`
- GPU placement: all three runs currently use GPU `7`.
- Tmux sessions:
  - `lfcrash_dad_lightreg_r1_20260427`
  - `lfcrash_dad_lightreg_r2_20260427`
  - `lfcrash_dad_lightreg_r3_20260427`
- Live status artifact:
  `output/emnlp2026_support/dad_mechanism_lightreg_status.md`
- Paper-claim status: execution only, not paper evidence until all three runs
  complete and are aggregated.

## Why This Block Exists

The current ARR package is already submission-ready and plausibly oral-ready.
The remaining best-paper gap is DAD mechanism clarity, not missing submission
packaging.

Current DAD facts:

- Canonical DAD headline: `68.19%` AP, `1.75s` mTTA.
- Matched DAD full support block: `3/3` completed runs.
- Matched full aggregate: `63.19% +/- 1.21` AP, `2.17s +/- 0.05s` mTTA.
- `no_cbm` aggregate: `64.55% +/- 1.00` AP, `2.36s +/- 0.22s` mTTA.
- `no_align` aggregate: `63.36% +/- 0.75` AP, `2.39s +/- 0.18s` mTTA.
- Actor trigger evidence remains support-only and should not become a main
  causal timing claim.

Reading:

- DAD is not missing coverage anymore.
- The full semantic interface is competitive in timing, but its AP is not
  cleanly above the strongest ablations.
- The next useful question is whether the DAD mechanism becomes clearer when
  semantic regularization is lightened rather than removed.

## Predeclared Block

Block name:

- `dad_mechanism_lightreg_block`

Output directory:

- `output/dad_mechanism_lightreg_block`

Run tags:

- `insight_journal_dad_lightreg_r1`
- `insight_journal_dad_lightreg_r2`
- `insight_journal_dad_lightreg_r3`

Launcher:

- `paper/emnlp2026/run_dad_mechanism_lightreg_block.sh`

Default behavior:

- dry-run only

Execution guard:

- `--execute` is rejected unless `LFCRASH_ALLOW_GPU=1` is set in the
  environment.

## Exact Recipe

The block keeps the matched 40-epoch DAD support recipe and changes only the
semantic regularization strength:

- dataset: `dad`
- epochs: `40`
- batch size: `16`
- learning rate: `2e-4`
- weight decay: `3e-5`
- h_dim: `256`
- z_dim: `128`
- num_concepts: `837`
- num_workers: `0`
- eval_every: `2`
- patience: `10`
- lambda_align: `3e-5`
- lambda_sparse: `1e-4`
- lambda_recon: `5e-4`

Rationale:

- `no_align` is roughly tied with full in AP but slower in timing, so removing
  semantic alignment entirely is not the target.
- `no_cbm` is stronger in AP but slower in timing, so the goal is not to drop
  the semantic interface.
- The block tests whether lighter semantic regularization can preserve the
  auditable interface while reducing the DAD AP cost.

## Success / Tie / Failure Rules

All rules must be evaluated on the 3-run aggregate. Do not change paper claims
from a single run.

### Strong Success

Requirements:

- `3/3` runs complete.
- AP mean is at least `64.0%`.
- mTTA mean is at most `2.30s`.
- AP standard deviation is at most `2.0` percentage points.
- P@R80 is not worse than the matched full aggregate by more than `0.02`.

Allowed interpretation:

- DAD mechanism clarity improves.
- The paper may say the DAD semantic interface has a calibratable
  regularization tradeoff.
- Still do not claim DAD is the clean flagship or that the actor branch gives
  causal timing control.

### Useful Tie

Requirements:

- `3/3` runs complete.
- AP mean is in `[62.5%, 64.0%)`.
- mTTA mean is at most `2.35s`.
- No run shows a failed-training signature.

Allowed interpretation:

- The semantic interface has bounded cost on DAD.
- The paper should remain oral-ready rather than best-paper-ready.
- DAD stays framed as a stress test.

### Failure

Any of the following:

- AP mean below `62.5%`.
- AP standard deviation above `2.5` percentage points.
- mTTA mean above `2.50s`.
- Fewer than `3/3` runs complete.
- Training instability, missing `results.json`, or corrupted logs.

Allowed interpretation:

- Do not upgrade DAD mechanism claims.
- Keep DAD as a fragile stress test.
- Best-paper readiness remains blocked by mechanism clarity.

## Paper-Claim Rules

The result may only affect the paper after:

1. all three runs finish;
2. a summary board is written;
3. the result is compared against matched full, `no_cbm`, and `no_align`;
4. `paper/emnlp2026/run_submission_sanity_checks.sh` passes.

Forbidden claims:

- DAD is solved.
- DAD is the flagship dataset.
- Actor-policy timing is causal.
- The semantic bottleneck universally improves AP.
- Best-paper readiness is achieved from this block alone.

## Dry-Run Command

Use this to inspect the exact commands without launching:

```bash
bash paper/emnlp2026/run_dad_mechanism_lightreg_block.sh --dry-run --gpus 0,1,2
```

## Execution Command After Explicit Approval Only

Only after explicit GPU approval:

```bash
LFCRASH_ALLOW_GPU=1 bash paper/emnlp2026/run_dad_mechanism_lightreg_block.sh --execute --gpus 0,1,2
```

## Escalation Candidate (Second Light-Regularization Probe)

For higher-confidence mechanism clarity, the next predeclared probe is:

```bash
LFCRASH_ALLOW_GPU=1 bash paper/emnlp2026/run_dad_mechanism_lightreg_lowreg_block.sh --execute --gpus 2,3,4
```

Difference vs. the predeclared block:

- output directory: `output/dad_mechanism_lightreg_block_lowreg`
- tags: `insight_journal_dad_lightreg_lowreg_r{1,2,3}`
- hyperparameters: `lambda_align=1e-6`, `lambda_sparse=0`, `lambda_recon=1e-4`
- status artifact:
  - `output/emnlp2026_support/dad_mechanism_lightreg_lowreg_status.json`
  - `output/emnlp2026_support/dad_mechanism_lightreg_lowreg_status.md`

Success/failure interpretation follows the same 3-run aggregate gate as the
first block, but with the low-regularization block as the candidate evidence.

## Stop Rule

If GPU approval, GPU placement, data availability, or author-facing claim
interpretation is unclear, stop and report. Do not launch the block by default.

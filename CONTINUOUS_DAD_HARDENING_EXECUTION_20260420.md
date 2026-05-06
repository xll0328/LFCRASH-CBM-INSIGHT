# Continuous DAD Hardening Execution Plan

Date: 2026-04-20
Project: `LFCRASH-CBM`
Owner for this execution round: Codex

## Goal

Close the highest-value remaining evidence gap for the current EMNLP oral-to-best-paper push:
finish a clean matched 3-run `DAD full-support` block, keep status artifacts fresh, and avoid
another silent stall.

This plan is intentionally operational rather than aspirational. It is written to be executed
continuously, not just read.

## Current Repo-Grounded State

### Stable paper-level state

- ARR-ready: `True`
- Oral-ready: `True`
- Best-paper-ready: `False`
- Controlled ontology seeded block: `18/18` complete
- A3D headline multi-seed block: complete
- Canonical DAD line remains usable, but DAD is still the fragile side of the story

Primary sources:

- `output/emnlp2026_support/oral_readiness_audit.{md,json}`
- `output/emnlp2026_support/multiseed_ontology_status.{md,json}`
- `output/emnlp2026_support/dad_hardening_status.{md,json}`
- `EMNLP_BEST_PAPER_EXECUTION_PLAN_20260420.md`

### DAD hardening state at the moment of writing

The matched `full` block was launched and trained healthily through `epoch 8`, but all three runs
stopped abnormally before producing `results.json`.

Interrupted live metrics from the last completed checkpoint:

| Run | Last eval epoch | AP | mTTA | TTA@R80 | P@R80 | Last log write |
|---|---:|---:|---:|---:|---:|---|
| `insight_journal_dad_full_r1` | 8 | 50.03% | 3.16s | 3.52s | 0.439 | 2026-04-20 04:42 UTC |
| `insight_journal_dad_full_r2` | 8 | 58.98% | 2.09s | 3.05s | 0.436 | 2026-04-20 04:41 UTC |
| `insight_journal_dad_full_r3` | 8 | 49.10% | 3.31s | 3.31s | 0.455 | 2026-04-20 04:40 UTC |

### Why this is confirmed to be an abnormal interruption

- `train.py` always writes final completion lines and `results.json` on a normal exit.
- None of the three directories contains `results.json`.
- None of the three `train.log` files contains `Training complete`.
- The three `.launch.log` files are empty, which is consistent with a process killed externally
  rather than a normal Python exception path dumping a traceback to stderr.

## Observed vs. Assumed

### Observed

- The matched `full` recipe is numerically healthy through `epoch 8`.
- The current bottleneck is continuity, not immediate divergence.
- The machine has real GPU headroom right now.
- GPU0 is the safest placement at this instant.
- GPUs `3`, `4`, `6`, and `7` each still have about `16 GB` free and low reported GPU utilization.
- `/data` is very full (`99%` used) but still has about `457 GB` available, so disk exhaustion is
  not the leading explanation for the interruption.

### Assumed, but still uncertain

- The previous three runs were likely killed externally or lost due to process lifecycle issues,
  not because the recipe itself is invalid.
- A relaunch can still produce useful evidence if continuity is protected.

## Ranked Cause Hypotheses

1. External process termination or lifecycle loss after launch.
   Evidence:
   abnormal stop pattern across all three runs, no Python traceback, no normal finalization.
2. Shared-machine interference.
   Evidence:
   the box is heavily loaded and multiple long-running jobs occupy most GPUs.
3. Training-script logic bug.
   Current evidence against this:
   the same script trained stably through multiple evaluations and the normal completion path
   should have written a final result artifact.
4. Hard OOM or disk-full event.
   Current evidence against this:
   no result of memory exhaustion is visible now, and the box still has large absolute disk headroom.

## Execution Principles

1. Preserve evidence before restarting.
   Interrupted run directories must be archived out of the active matched-block path before relaunch.
2. Keep canonical run names.
   The active matched block should still use
   `insight_journal_dad_full_r1`,
   `insight_journal_dad_full_r2`,
   `insight_journal_dad_full_r3`
   so that the existing summarizers continue to work without paper-facing rewrites.
3. Prefer automated recovery to manual babysitting.
   A guard loop should refresh status, detect stale runs, archive them, and relaunch missing runs.
4. Do not update paper claims from partial live checkpoints.
   Status boards can mention live progress; paper claims must wait for completed `results.json`.

## Active GPU Placement Decision

### Current practical choice

- Primary safe GPU: `0`
- Secondary pool for parallel relaunch: `3,4`

Rationale:

- `GPU0` is almost empty now.
- `GPU3` and `GPU4` each have about `8.5 GB` already occupied but still retain enough headroom for
  another LFCRASH DAD run.
- `GPU5` is too crowded.
- `GPU1` and `GPU2` are viable, but they are already tied to one multi-GPU training job and are not
  the cleanest first choice for this relaunch.

Fallback pool if `3/4` become problematic:

- `6,7`

## Clean Restart Protocol

### Step 1. Archive interrupted active runs

Move any stale incomplete directory matching `insight_journal_dad_full_r*` out of
`output/dad_full_support_block/` into a sibling archive directory.

Reason:

- reusing the same directory in place would append new logs to old logs and contaminate the status
  summarizer.

### Step 2. Relaunch the canonical matched block

Use the existing launcher:

```bash
bash paper/emnlp2026/run_dad_full_support_block.sh --execute --gpus 0,3,4 --stale-after-min 35
```

### Step 3. Refresh status immediately after launch

```bash
python paper/emnlp2026/refresh_emnlp_status.py
```

### Step 4. Do not interfere before the first meaningful checkpoint

For this recipe on the current shared machine, the first decision-worthy checkpoint is `epoch 2`.
Anything earlier is just proof of life.

## Expected Timing Windows

These are grounded in the real interrupted run logs from `2026-04-20`, not optimistic historical
best cases.

| Milestone | ETA after relaunch | What it means |
|---|---:|---|
| log creation | 1-3 min | launch is real |
| `epoch 1` completion | 12-18 min | throughput looks normal |
| `epoch 2` eval | 25-35 min | first health checkpoint |
| `epoch 4` eval | 55-70 min | first meaningful trend |
| `epoch 6` eval | 80-100 min | mid-run trajectory |
| `epoch 8` eval | 100-130 min | comparable to interrupted run point |
| full 40-epoch completion or early stop | 8-11 h | usable aggregate evidence |

## Continuous Monitoring Rules

### Healthy run

A run is treated as healthy if all of the following are true:

- `train.log` exists
- either log age is less than `35` minutes, or the matching tagged training PID is still alive
- latest lines show epoch progress or evaluation writes
- there is no NaN stop

### Stale run

A run is treated as stale if:

- `results.json` is absent, and
- `train.log` exists, and
- log age is `>= 35` minutes, and
- no live `train.py --tag <canonical_tag>` PID exists for that run

This rule is intentionally stricter than plain log-age checking so that the watchdog does not
archive a still-running process just because `train.log` has been quiet for a long epoch window.

### Recovery action for stale run

1. Archive the stale directory out of the active block.
2. Relaunch the missing canonical tag.
3. Refresh status files.
4. Start a new wait window based on relaunch time.

## Decision Rules By Checkpoint

### After `epoch 2`

- If all three runs reach `epoch 2` eval:
  do nothing except keep monitoring.
- If one run is missing while others are healthy:
  archive only that stale run and relaunch it.
- If all three die again before `epoch 2`:
  treat this as an environment continuity problem, not a recipe problem.

### After `epoch 4`

- If AP aggregate is still rising and all three runs are alive:
  keep the block untouched.
- If one seed collapses while two are healthy:
  keep it running unless there is a clear NaN or crash signature.

### After `epoch 8`

- Compare trajectory against the interrupted `epoch 8` snapshot.
- If the new block reaches or exceeds the old `epoch 8` mean AP of about `52.70%`,
  continuity is now the only remaining concern.

### After completion

- Refresh `dad_hardening_status`
- Refresh all EMNLP status files
- Compare matched `full` aggregate against existing `no_cbm` and `no_align`
- Only then update any paper-facing interpretation

## Paper Claim Discipline

What can be said after live partial checkpoints:

- the matched `full` DAD hardening block is running
- the recipe is numerically stable
- the evidence gap is actively being closed

What must wait for completed runs:

- whether `full` beats `no_cbm`
- whether the semantic bottleneck helps DAD prediction under the matched recipe
- any new main-text wording about DAD mechanism

## Commands

### Status refresh

```bash
python paper/emnlp2026/refresh_emnlp_status.py
```

### DAD hardening board only

```bash
python paper/emnlp2026/summarize_dad_hardening_status.py
```

### Manual launcher

```bash
bash paper/emnlp2026/run_dad_full_support_block.sh --execute --gpus 0,3,4 --stale-after-min 35
```

### Continuous watchdog

```bash
tmux new-session -d -s lfcrash_dad_watch \
  'cd /data/sony/LFCRASH/LFCRASH-CBM && bash paper/emnlp2026/continuous_dad_full_support_loop.sh'
```

## Immediate Next Actions

1. Write a continuous watchdog script that archives stale partial runs and relaunches the matched
   block automatically.
2. Start the watchdog on GPUs `0,3,4`.
3. Wait for the first `epoch 2` evaluation window.
4. Refresh status files again.
5. Reassess whether the block is progressing normally or being interrupted again.

## Execution Log For This Round

### 2026-04-20 06:35 UTC

- confirmed the three matched `full` runs stopped abnormally after `epoch 8`
- confirmed no `results.json` exists for any of the three active canonical run dirs
- confirmed the current safest restart topology is `GPU0` plus parallel slots on `GPU3/4`
- committed to an automated recovery loop rather than another one-shot manual relaunch

### 2026-04-20 06:39 UTC

- archived the interrupted `epoch 8` runs into `output/dad_full_support_block_archive/`
- started the new continuous watchdog loop
- first relaunch exposed a hidden execution bug:
  `train.py` imported `torch` before changing `CUDA_VISIBLE_DEVICES`, so
  `--gpu 3` and `--gpu 4` did not reliably bind to physical GPUs `3` and `4`
- practical symptom:
  new CUDA memory only appeared on `GPU0`, while `GPU3/4` showed no LFCRASH allocation
- conclusion:
  this was an execution-correctness bug, not a modeling issue

### 2026-04-20 06:41 UTC

- patched `train.py` so that it now uses:
  - explicit `torch.device(f'cuda:{args.gpu}')`
  - `torch.cuda.set_device(device)`
  - launch-time logging of the selected CUDA device
- stopped the bad relaunch, archived those partial directories as
  `*_bad_gpu_binding_*`
- restarted the watchdog with the fixed device-binding logic

### 2026-04-20 06:42 UTC

- confirmed the corrected relaunch created fresh canonical run directories
- confirmed all three runs progressed past dataset loading to `Trainable params`
- confirmed actual CUDA allocations now appear on:
  - `r1 -> GPU0`
  - `r2 -> GPU3`
  - `r3 -> GPU4`
- updated ETA anchor:
  the first meaningful checkpoint remains the `epoch 2` evaluation window, now
  expected roughly around `07:07-07:17 UTC` if the current shared-machine throughput holds

### 2026-04-20 06:47 UTC

- confirmed from `launch.log` progress bars that all three runs are actively training inside
  `epoch 1`, rather than idling after initialization
- live approximate positions:
  - `r1`: `29/80`
  - `r2`: `24/80`
  - `r3`: `29/80`
- corrected near-term ETA:
  - `epoch 1` completion is likely around `06:53-06:57 UTC`
  - `epoch 2` evaluation is more realistically around `07:12-07:20 UTC`
- no intervention is justified before that window because the block is behaving like a normal,
  slow shared-machine run rather than a failed launch

### 2026-04-20 07:02 UTC

- `epoch 1` is now confirmed for all three runs
- per-run `epoch 1` summaries:
  - `r1`: `loss=1.6150`, `ce=0.4567`
  - `r2`: `loss=1.6316`, `ce=0.4754`
  - `r3`: `loss=1.6738`, `ce=0.4956`
- interpretation:
  the block is numerically stable, seed spread is modest at this stage, and there is still no
  evidence of the earlier silent-interruption pattern recurring
- next decision point remains unchanged:
  wait for the first `epoch 2` evaluation rather than over-reading raw training loss

### 2026-04-20 07:14 UTC

- the first `epoch 2` evaluation checkpoint is now complete for all three runs
- per-run metrics:
  - `r1`: `34.39%` AP, `5.00s` mTTA, `5.00s` TTA@R80, `0.349` P@R80
  - `r2`: `34.47%` AP, `5.00s` mTTA, `5.00s` TTA@R80, `0.368` P@R80
  - `r3`: `29.28%` AP, `5.00s` mTTA, `5.00s` TTA@R80, `0.349` P@R80
- aggregate:
  `32.71% +- 2.43` AP,
  `5.00s +- 0.00s` mTTA,
  `5.00s +- 0.00s` TTA@R80,
  `0.355 +- 0.009` P@R80
- interpretation:
  this is a normal cold-start checkpoint, not a claim point, but it confirms that the repaired
  three-run block is healthy and synchronized enough to keep running untouched
- next checkpoint:
  the first trend-level read should come from `epoch 4` evaluation, now expected roughly around
  `07:35-07:45 UTC` if the current throughput holds

### 2026-04-20 07:49 UTC

- the repaired block did not reach `epoch 3` logging after the `epoch 2` evaluation window
- the original watchdog therefore archived all three canonical run dirs as stale after about
  `35-36` minutes without new `train.log` writes and relaunched the block on the same `0/3/4`
  topology
- important interpretation correction:
  this does not prove the watchdog was wrong; the absence of any `epoch 3` summary for roughly
  half an hour after `epoch 2` still points to a real continuity problem
- but it did expose an operational weakness:
  the watchdog relied only on `train.log` age and could have duplicated work if a matching live
  process still existed
- new active run start times:
  - `r1`: `2026-04-20 07:49:58 UTC`
  - `r2`: `2026-04-20 07:49:59 UTC`
  - `r3`: `2026-04-20 07:49:59 UTC`

### 2026-04-20 07:56 UTC

- hardened both launcher and watchdog logic:
  - added live-PID detection keyed by canonical `--tag`
  - changed launch stderr/stdout capture from a single overwritten `*.launch.log` file to
    timestamped `*.launch.<UTC>.log` files plus a `*.launch.latest.log` symlink
  - started writing `*.pid` files on launch
- restarted the persistent watchdog cleanly under `tmux` session `lfcrash_dad_watch`
- confirmed the new watchdog no longer tries to relaunch while the current tagged PIDs are alive:
  - `r1 -> pid 654776`
  - `r2 -> pid 654777`
  - `r3 -> pid 654778`
- updated near-term ETA from the current relaunch anchor:
  - `epoch 1` should appear roughly around `08:03-08:07 UTC`
  - `epoch 2` evaluation should appear roughly around `08:17-08:24 UTC`
- current decision:
  do not intervene before that window unless the processes disappear or the box loses the GPUs

### 2026-04-20 08:03 UTC

- `epoch 1` is now confirmed for all three runs from the `07:49 UTC` relaunch
- per-run summaries:
  - `r1`: `loss=1.6039`, `ce=0.4409` at `08:01:54 UTC`
  - `r2`: `loss=1.6352`, `ce=0.4665` at `08:02:38 UTC`
  - `r3`: `loss=1.6210`, `ce=0.4616` at `08:02:16 UTC`
- interpretation:
  throughput is consistent with the current shared-machine expectation and there is no sign of an
  immediate relaunch failure
- operational note:
  for several minutes before `epoch 1`, `train.log` stayed unchanged while the three tagged PIDs
  remained active and held CUDA memory on `GPU0/3/4`; this directly validates the live-PID
  protection added at `07:56 UTC`
- next checkpoint:
  the first decision-worthy read remains the synchronized `epoch 2` evaluation, now most likely
  around `08:14-08:22 UTC`

### 2026-04-20 08:23 UTC

- the synchronized `epoch 2` evaluation is now complete for all three runs from the `07:49 UTC`
  relaunch
- per-run metrics:
  - `r1`: `36.48%` AP, `5.00s` mTTA, `5.00s` TTA@R80, `0.356` P@R80
  - `r2`: `35.17%` AP, `4.9965s` mTTA, `4.9722s` TTA@R80, `0.359` P@R80
  - `r3`: `32.44%` AP, `5.00s` mTTA, `5.00s` TTA@R80, `0.347` P@R80
- aggregate:
  `34.70% +- 1.68` AP,
  `5.00s +- 0.00s` mTTA,
  `4.99s +- 0.01s` TTA@R80,
  `0.354 +- 0.005` P@R80
- interpretation:
  this is still a cold-start checkpoint, but it matters operationally because the block has now
  crossed `epoch 2` evaluation cleanly after the watchdog hardening changes instead of vanishing
  around the same stage
- relative to the previous repaired attempt at `07:14 UTC`, the new `epoch 2` AP mean improved from
  `32.71%` to `34.70%`
- next checkpoint:
  the first trend-level comparison should now come from `epoch 4` evaluation, likely around
  `08:45-08:52 UTC` if the current machine contention remains similar

### 2026-04-20 08:46 UTC

- the synchronized `epoch 4` evaluation is now complete for all three runs
- per-run metrics:
  - `r1`: `52.83%` AP, `3.60s` mTTA, `3.81s` TTA@R80, `0.401` P@R80
  - `r2`: `52.92%` AP, `3.47s` mTTA, `3.89s` TTA@R80, `0.438` P@R80
  - `r3`: `43.80%` AP, `4.34s` mTTA, `3.59s` TTA@R80, `0.419` P@R80
- aggregate:
  `49.85% +- 4.28` AP,
  `3.80s +- 0.39s` mTTA,
  `3.76s +- 0.13s` TTA@R80,
  `0.419 +- 0.015` P@R80
- interpretation:
  this is the first real trend checkpoint and it is encouraging
  because the block is now moving in the expected direction instead of dying near `epoch 2`
- relative to the current-round `epoch 2` aggregate, AP improved from `34.70%` to `49.85%`
- relative to the old interrupted run family, the new block is already close to the previous
  interrupted `epoch 8` mean AP of about `52.70%`, but continuity is still the main open question
- the seed spread is still asymmetric:
  `r3` remains clearly weaker than `r1/r2`, so the eventual mean will still depend on whether the
  weaker seed catches up in later epochs
- next checkpoint:
  the next high-value read is `epoch 6` evaluation, likely around `09:00-09:08 UTC` if the
  current pace holds

### 2026-04-20 09:01 UTC

- the synchronized `epoch 6` evaluation is now complete for all three runs
- per-run metrics:
  - `r1`: `57.23%` AP, `2.32s` mTTA, `2.96s` TTA@R80, `0.443` P@R80
  - `r2`: `59.07%` AP, `2.58s` mTTA, `3.42s` TTA@R80, `0.411` P@R80
  - `r3`: `49.05%` AP, `3.29s` mTTA, `3.97s` TTA@R80, `0.424` P@R80
- aggregate:
  `55.12% +- 4.36` AP,
  `2.73s +- 0.41s` mTTA,
  `3.45s +- 0.41s` TTA@R80,
  `0.426 +- 0.013` P@R80
- interpretation:
  this is the strongest sign so far that the repaired block is healthy
  because the three runs have now progressed well beyond the previously fragile window and the
  aggregate has moved decisively upward from `epoch 4`
- relative to the current-round `epoch 4` aggregate, AP improved from `49.85%` to `55.12%`
- relative to the old interrupted family, the current block has now surpassed the previous
  interrupted `epoch 8` mean AP of about `52.70%`, although the comparison is still checkpoint-
  to-checkpoint rather than final-to-final
- `r3` is still the weakest seed, but it also improved materially from `43.80%` to `49.05%`,
  so the weak-seed story is now "lagging but recovering", not "collapsed"
- next checkpoint:
  `epoch 8` evaluation is now the next meaningful read, likely around `09:20-09:28 UTC`

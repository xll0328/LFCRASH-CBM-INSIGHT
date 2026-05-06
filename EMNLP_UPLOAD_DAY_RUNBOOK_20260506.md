# EMNLP Upload Day Runbook

Date: 2026-05-06  
Scope: ARR upload-day execution for `LFCRASH-CBM`  
Reference freeze candidate: `ARR20260506T073630Z`

## Goal

Run a deterministic upload-day sequence with clear stop rules:

- preserve technical integrity of the package,
- keep claims bounded to current evidence,
- avoid last-minute drift from the frozen candidate.

## Hard Stop Rules

- Stop if `run_submission_sanity_checks.sh` is not green.
- Stop if `verify_arr_freeze.sh ARR20260506T073630Z` fails.
- Stop if any human logistics owner is missing.
- Stop if last-minute edits introduce new claim-overreach wording.

## Upload-Day Sequence

### 1) Repository and branch check

Run in `/data/sony/LFCRASH/LFCRASH-CBM`:

```bash
git status --short --branch
git log --oneline -n 1
```

Expected:

- working tree clean,
- `main` aligned with `origin/main`,
- latest commit recorded in signoff notes.

### 2) Final no-change technical gate

```bash
bash paper/emnlp2026/run_submission_sanity_checks.sh
bash paper/emnlp2026/verify_arr_freeze.sh ARR20260506T073630Z
```

Expected:

- `OK fatal_count=0`,
- freeze verifier pass.

### 3) Freeze artifact identity check

```bash
ls -lh paper/emnlp2026/frozen/insight_emnlp_arr_freeze_ARR20260506T073630Z.tar.gz
sha256sum paper/emnlp2026/frozen/insight_emnlp_arr_freeze_ARR20260506T073630Z.tar.gz
```

Record:

- size,
- hash,
- timestamp in signoff sheet.

### 4) Human content check (required)

Human reviewers read:

- page 1 narrative framing,
- Figure 1 caption wording,
- Table 1/headline table wording and tier boundaries,
- appendix opening boundary language.

Rule:

- if wording is edited, rerun Step 2 and cut a new freeze before upload.

### 5) Human logistics check (required)

Confirm and record:

- final author list owner,
- ARR account and reviewer-registration owner,
- upload operator and fallback operator,
- venue commit/lock decision owner.

### 6) ARR metadata check (required)

Human operator verifies metadata fields are consistent with latest manuscript:

- title,
- abstract,
- keywords/topics,
- conflict/ethics/disclosure fields (if present in portal),
- supplementary/appendix upload pairing.

Rule:

- metadata mismatch means stop and fix before upload.

### 7) Upload execution

Upload only:

- `paper/emnlp2026/frozen/insight_emnlp_arr_freeze_ARR20260506T073630Z.tar.gz`

After upload, capture:

- upload timestamp (UTC),
- uploader identity,
- venue/portal confirmation screenshot or confirmation ID.

### 8) Post-upload ledger update

Immediately update:

- `EMNLP_UPLOAD_SIGNOFF_20260506.md`
- `EMNLP_STAGE_STATUS.md`

Record:

- final decision,
- upload timestamp,
- confirmation reference,
- whether rebuttal mode freeze is now active.

## Decision Outcomes

Only one outcome can be selected:

- `APPROVED_AS_IS`
- `APPROVED_AFTER_MINOR_EDIT_AND_REFREEZE`
- `HOLD_PENDING_HUMAN_DECISION`

## Owner Fields

- Decision owner:
- Upload operator:
- Fallback operator:
- Final decision:
- Decision timestamp (UTC):
- Upload confirmation reference:

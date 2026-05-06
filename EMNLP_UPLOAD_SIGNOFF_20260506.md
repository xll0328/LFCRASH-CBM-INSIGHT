# EMNLP Upload Signoff Checklist

Date: 2026-05-06
Freeze candidate: `ARR20260506T073630Z`
Tarball: `paper/emnlp2026/frozen/insight_emnlp_arr_freeze_ARR20260506T073630Z.tar.gz`
Runbook: `EMNLP_UPLOAD_DAY_RUNBOOK_20260506.md`

## A. Technical Gate (already passed)

- [x] `bash paper/emnlp2026/run_submission_sanity_checks.sh` -> `OK fatal_count=0`
- [x] `bash paper/emnlp2026/verify_arr_freeze.sh ARR20260506T073630Z` -> pass
- [x] PDF metadata anonymity checks pass (`Author/Title/Subject/Keywords` empty)

## B. Human Read-Through (required)

- [ ] Page 1 message is accurate and bounded.
- [ ] Figure 1 caption has no overclaim wording.
- [ ] Table 1 / headline tables match intended claim tier.
- [ ] Appendix opening still states bounded intervention/timing scope.

## C. Ownership & Logistics (required)

- [ ] Final author list confirmed by humans.
- [ ] ARR account and reviewer-registration ownership confirmed.
- [ ] Venue commitment and upload operator confirmed.

## D. Upload Decision

- [ ] Approved for upload as-is.
- [ ] Approved with minor wording edits (if yes, rerun sanity and re-freeze).
- [ ] Hold upload pending additional human decision.

## Signoff

- Decision owner:
- Decision timestamp (UTC):
- Notes:

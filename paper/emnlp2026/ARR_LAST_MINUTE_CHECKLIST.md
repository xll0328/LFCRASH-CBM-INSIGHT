# ARR Last-Minute Checklist

Use this in the final 10 minutes before upload.

## Paper-facing checks

- Open `insight_emnlp.pdf` and read page 1, Figure 1, Table 1, and the first appendix page as a human reader.
- Confirm the title, abstract, and contributions all center the same claim: a language-grounded risk concept interface.
- Confirm the main paper does not overclaim actor-policy causality on DAD.
- Confirm the ontology comparison table still reads as an operating-point result, not a leaderboard trick.
- Confirm `EMNLP_FINAL_READINESS_AUDIT.md` lists only human-only blockers and no technical blockers.
- Confirm `pdf_first_read_audit_report.md` reports zero critical blockers.

## Anonymity and packaging

- Run `bash run_submission_sanity_checks.sh` inside `paper/emnlp2026`.
- Confirm `claim_evidence_audit_report.md` reports zero critical blockers.
- Confirm the report shows no obvious author names, home directories, or private URLs.
- Upload the frozen tarball from `paper/emnlp2026/frozen/` rather than reassembling files manually.
- If a new freeze is needed, regenerate it with `bash freeze_arr_submission.sh`.

## Artifact sanity

- Confirm `insight_emnlp.pdf`, `insight_emnlp_first25.pdf`, `insight_emnlp.tex`, and `insight.bib` are all present in the freeze manifest.
- Confirm the freeze manifest includes the review response map, review tracker, DAD-500 support files, multi-seed ontology status, A3D headline status, and DAD hardening status.
- Confirm the freeze manifest includes the reviewer quick map, response templates, and final readiness audit.
- Confirm the freeze manifest includes the PDF first-read audit report.
- Confirm appendix references point to the semantic-interface framing, not the older WHY/WHEN wording.
- Confirm the written evidence still matches the support claims.
- Controlled ontology block should read as complete.
- Intervention evidence should read as partial and scoped.
- DAD multi-seed numbers should stay diagnostic, not headline replacements.

## Visual sanity

- Check that the safety-utility plot, ontology evolution figure, and concept case-study figure are crisp and readable at normal zoom.
- Check that table text remains readable after the last layout pass.
- Scan for obvious overfull boxes or captions spilling into margins.

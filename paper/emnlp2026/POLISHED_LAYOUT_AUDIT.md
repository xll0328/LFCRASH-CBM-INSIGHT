# Polished EMNLP Layout Audit

Generated: 2026-05-01

## Current Verdict

- Rejected/superseded on 2026-05-01 after visual inspection: this candidate is
  too compressed for an EMNLP/ARR long-paper read-through because the appendix
  starts on page 6.
- Do not use `insight_emnlp_polished.pdf` as an upload candidate. Use
  `insight_emnlp_oral8.pdf` for the next human read-through unless the upload
  path is changed explicitly.

## Candidate

- Source: `paper/emnlp2026/insight_emnlp_polished.tex`
- PDF: `paper/emnlp2026/insight_emnlp_polished.pdf`
- Compile script: `paper/emnlp2026/compile_emnlp_polished.sh`
- Page render contact sheet: `paper/emnlp2026/layout_audit_polished/contact_sheet.png`

## Visual Changes

- Replaced the oversized diagnostic Figure 1 with a compact paper-scale case
  interface: real frames, risk/alert trajectory, and concept activations in one
  controlled visual.
- Replaced the previous dense framework figure with a reduced semantic-interface
  pipeline focused on the paper's central claim.
- Replaced the large safety-utility figure with a tighter two-panel AP vs.
  warning-time view.
- Reduced main-text float count and removed the prior pages with large
  white-space gaps around Figure 3/Table 2/Table 3.
- Consolidated results into two main tables: headline predictive evidence and
  controlled ontology/support evidence.

## Verification

- `bash paper/emnlp2026/compile_emnlp_polished.sh` completed successfully.
- `insight_emnlp_polished.pdf` has 16 total pages with appendix starting on
  page 6.
- Main body, limitations, and references finish by page 5.
- Latest polished LaTeX log has no overfull boxes, undefined references,
  undefined citations, duplicate labels, or float-too-large warnings matching
  the strict scan.

## Human-Review Notes

- The polished candidate is much more visually controlled than
  `insight_emnlp_arr8.pdf`, but it is intentionally more compressed.
- It should be treated as a rejected visual experiment, not a new freeze.

# Oral8 EMNLP Layout Audit

Generated: 2026-05-01

## Candidate

- Source: `paper/emnlp2026/insight_emnlp_oral8.tex`
- PDF: `paper/emnlp2026/insight_emnlp_oral8.pdf`
- Compile script: `paper/emnlp2026/compile_emnlp_oral8.sh`
- Figure builder: `paper/emnlp2026/make_emnlp_oral_figures.py`
- Render contact sheet: `paper/emnlp2026/layout_audit_oral8/contact_sheet.png`

## Verdict

- Current role: strongest layout/content candidate for human read-through.
- Appendix starts on page 9.
- Main paper no longer collapses into a five-page compressed draft.
- The rejected `insight_emnlp_polished.pdf` should not be treated as an upload
  candidate because its appendix starts on page 6.

## Visual And Layout Checks

- Figure 1 uses a compact real-case semantic-interface view: frame strip,
  risk/alert trajectory, and auditable concept activations.
- Figure 2 is rebuilt as a deterministic conference-style framework figure with
  readable labels and controlled geometry. GPT Image 2 prompt and output
  plumbing are stored in `paper/emnlp2026/gpt_image2_framework/`; the current
  AIHubMix calls failed at the long-running generation request, so the promoted
  figure uses the deterministic overlay path without an Image 2 bitmap base.
- Figure 3 keeps ontology construction visible as an EMNLP-facing language
  artifact rather than hidden preprocessing.
- Figure 4 uses a compact AP--warning-time view split by DAD stress and A3D
  cleaner settings.
- Tables remain in the main paper and keep headline, ontology, support, and
  stress evidence separated.

## Compile Checks

- `bash paper/emnlp2026/compile_emnlp_oral8.sh` completed successfully.
- Strict warning scan found no overfull boxes, undefined references, undefined
  citations, duplicate labels, float-too-large warnings, or LaTeX warning lines
  matching the audit regex.
- Render inspection used the first 12 pages at
  `paper/emnlp2026/layout_audit_oral8/contact_sheet.png`.

## Remaining Human Checks

- Read page 1, Figure 2, Tables 2--4, the Discussion section, and page 9 at
  normal PDF zoom.
- Decide explicitly whether this oral8 candidate replaces the prior strict
  ARR8 upload path.
- Do not cut a new freeze until that upload-path decision is explicit.

## 2026-05-02 Progress Note

- Strengthened the Figure 1 first-read bridge in
  `paper/emnlp2026/insight_emnlp_oral8.tex`: the motivated case now points
  explicitly to the headline, ontology, and support/stress evidence blocks
  rather than standing as an implicit qualitative claim.
- Added a short protocol-discipline paragraph after the evidence map so the
  risk trace, concept activations, and alert marker cannot be read as pooled
  leaderboard evidence or deployment-level policy validation.

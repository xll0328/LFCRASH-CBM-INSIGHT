# ARR8 Layout Audit

Date: 2026-05-01

## Scope

This audit covers the strict ARR/EMNLP long-paper candidate:

- `paper/emnlp2026/insight_emnlp_arr8.tex`
- `paper/emnlp2026/insight_emnlp_arr8.pdf`

It does not replace the existing full evidence package
`paper/emnlp2026/insight_emnlp.pdf`; it provides a page-limit-friendly main
paper candidate built from the same claim hierarchy.

## Current Verdict

- Official-style main-paper candidate: `ready for human read-through`
- Critical LaTeX issues: `0`
- Overfull boxes: `0`
- Undefined references/citations: `0`
- Main flow: title/abstract + main body + limitations/references before appendix
- Appendix starts: page 9

## Figure/Table Design Audit

- Figure 1 is a motivated real-case example, not generated art. It is appropriate
  as the first visual anchor because it exposes the semantic interface on an
  actual DAD case before the quantitative tables.
- Figure 2 is the solution overview. It provides a clean method bridge from
  language artifact to model state and audit output.
- Figure 3 is the ontology-construction pipeline. It makes the EMNLP-facing
  language contribution explicit and prevents the ontology from reading as
  hidden preprocessing.
- Figure 4 is the safety--utility evidence view. It anchors the quantitative
  story without turning the paper into a leaderboard claim.
- Table 1 is a compact evidence-tier map. It protects claim discipline.
- Table 2 combines DAD and A3D headline evidence in one readable table.
- Table 3 is the controlled ontology comparison, the strongest evidence for
  ontology-as-modeling.
- Table 4 consolidates timing, ablation, and intervention diagnostics into one
  support/stress table to avoid sparse pages and dense table clutter.

## Remaining Human Checks

- Read page 1, Figure 1, Tables 2--4, and the first appendix page at normal PDF
  zoom.
- Decide whether to promote the strict candidate to the main upload path or keep
  it as a page-limit-safe alternative beside the existing full package.
- Do not cut a new ARR freeze until the strict candidate vs. full package choice
  is explicit.

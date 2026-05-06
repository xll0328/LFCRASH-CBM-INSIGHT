# EMNLP Claim/Evidence Audit

- Generated at: `2026-05-06T07:27:49Z`
- Scope: live EMNLP source files plus the shared appendix source.
- Nature: mechanical pre-submission scan, not human validation.

## Verdict
- Source availability: OK
- Critical blockers: `0`
- Advisory strong-claim hits: `9`

## Missing Sources

- OK no missing source files.

## BLOCKER: Direct links or private identity tokens

Anonymous submissions should not expose private paths, URLs, or affiliations.

- OK no matches.

## BLOCKER: Unresolved placeholders

Unresolved notes are submission blockers.

- OK no matches.

## BLOCKER: Banned hype / AI-tone terms

These terms are high-risk unless directly quoted; use concrete evidence instead.

- OK no matches.

## ADVISORY: Strong causal/proof/guarantee wording

Allowed when scoped or negated, but each instance should remain tied to evidence.

- `paper/emnlp2026/sec_intro_emnlp.tex:165`: \item We keep the central claim intentionally constrained: \method{} advances semantic-interface methodology and auditing, with the strongest current contribution in interpretab...
- `paper/emnlp2026/sec_method_emnlp.tex:319`: This does not prove that every trained model will respond monotonically to
- `paper/emnlp2026/sec_experiments_emnlp.tex:700`: Top-risk edit & 25 edited cases & mean shift = 0.00 frames & naive top-risk edits alone do not guarantee movement \\
- `paper/emnlp2026/sec_experiments_emnlp.tex:715`: timing proof. That asymmetry is intentional in our presentation: the paper is
- `paper/neurips2026/sec_appendix.tex:21`: policy-level causal benchmark.
- `paper/neurips2026/sec_appendix.tex:484`: We do not claim a formal global theorem guaranteeing monotone policy shifts for
- `paper/neurips2026/sec_appendix.tex:486`: for such a guarantee to be informative without unrealistic assumptions. The
- `paper/neurips2026/sec_appendix.tex:502`: completed policy-level causal benchmark.
- `paper/neurips2026/sec_appendix.tex:531`: \emph{partial structural intervenability} rather than a completed causal proof.

## BLOCKER: Unicode dash punctuation

Use LaTeX ASCII dashes in source files for style and portability.

- OK no matches.

## Submission Reading

- If critical blockers are zero, the scanned sources do not show obvious anonymity leaks, placeholders, banned hype terms, or Unicode dash punctuation.
- Advisory causal/proof/guarantee hits should remain scoped to structural or negative claims, not upgraded into full policy-level causality.
- This audit does not replace the final human read-through required before upload.

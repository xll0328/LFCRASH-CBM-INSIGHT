# EMNLP Review Response Templates

Date: 2026-05-06

Purpose: prepare concise, evidence-grounded response drafts for likely ARR /
EMNLP reviewer concerns. These are internal templates, not final rebuttal text.
They should be adapted to the actual reviewer wording and never used to claim
human validation, new experiments, or policy-level causality beyond the current
evidence boards.
For response-ready short and formal variants, use
`EMNLP_REVIEW_RESPONSE_PLAYBOOK.md`.

## Response Principles

- Lead with the paper's central object: a governed language-grounded semantic
  interface for accident anticipation.
- Answer with evidence paths, not broad claims.
- Keep A3D as the clean flagship result.
- Treat DAD as a harder stress test with explicit diagnostics.
- Keep actor-policy timing and intervention evidence scoped.
- Never merge headline, support, search, and diagnostic runs into one
  leaderboard claim.
- Keep the response reading order explicit: headline evidence, then controlled
  support, then stress evidence.

## Template 1: "Why is this an EMNLP paper?"

Core response:

Thank you for raising the venue-fit question. The main contribution of the
paper is not only an accident anticipation architecture, but a
language-grounded semantic interface: the model exposes named risk concepts,
constructs and reviews an ontology, tests canonical naming stability, and
studies how ontology choice changes downstream behavior. We have revised the
front of the paper to make this object explicit and to separate it from a
standard vision leaderboard framing.

Evidence to cite:

- Introduction: `paper/emnlp2026/sec_intro_emnlp.tex`
- Method: `paper/emnlp2026/sec_method_emnlp.tex`
- Concept pipeline figure: `paper/figures/insight_fig_concept_pipeline.pdf`
- Family coverage figure: `paper/figures/insight_fig_concept_family_coverage.pdf`
- Verbalization audit: `output/emnlp2026_support/concept_verbalization_sensitivity_dad500.json`

Possible paper change:

- Add one sentence in the introduction or limitations clarifying that the
  language layer is evaluated as a governed interface, not as a free-form
  prompt artifact.

Do not say:

- "This is primarily an NLP benchmark."
- "The language model solves accident anticipation."

## Template 2: "Is ontology construction just prompt engineering?"

Core response:

We agree that an ungoverned prompt-generated vocabulary would be weak evidence.
This is why the paper treats ontology construction as a governed modeling
component: candidate concepts are mined, canonicalized, merged, family-balanced,
reviewed, and then evaluated under matched training launchers. The controlled
ontology block asks whether changing only the ontology source changes the
AP--mTTA operating point; the completed multi-seed extension covers `18/18`
dataset--ontology cells.

Evidence to cite:

- Controlled ontology status: `output/emnlp2026_support/controlled_ontology_status.md`
- Multi-seed ontology status: `output/emnlp2026_support/multiseed_ontology_status.md`
- Ontology manifest: `output/concept_sets/neurips2026_controlled_ontology_manifest.json`
- Experiments: `paper/emnlp2026/sec_experiments_emnlp.tex`

Possible paper change:

- Add a short phrase in the ontology table caption: "only the ontology source is
  varied under the matched launcher."

Do not say:

- "The ontology is optimal."
- "Prompt choice does not matter."

## Template 3: "The method is not the highest-scoring system on DAD."

Core response:

The paper does not claim best overall DAD performance. DAD is presented as the
harder stress test. The canonical DAD result shows that the semantic interface
can remain competitive, while the support diagnostics make the dataset's
fragility explicit. The strongest positive evidence for the paper's central
semantic-interface claim comes from the controlled ontology block and the A3D
headline stability, not from claiming a DAD leaderboard win.

Evidence to cite:

- DAD hardening status: `output/emnlp2026_support/dad_hardening_status.md`
- Review map: `EMNLP_REVIEW_RESPONSE_MAP.md`
- Experiments: `paper/emnlp2026/sec_experiments_emnlp.tex`

Key facts:

- Canonical DAD: `68.19%` AP / `1.75s` mTTA.
- Clean three-seed diagnostic: `62.31% +/- 1.90` AP and
  `2.07s +/- 0.09s` mTTA.
- Recovery block: `63.52% +/- 0.81` AP and
  `2.16s +/- 0.25s` mTTA over `6/6` completed runs.
- Matched full support block (paper stress summary): `63.18% +/- 1.32` AP and
  `2.27s +/- 0.20s` mTTA over `3/3` completed runs.

Possible paper change:

- Clarify that DAD diagnostics are included to bound fragility, not to replace
  the canonical headline line.

Do not say:

- "DAD is solved."
- "INSIGHT is best on all accident anticipation metrics."

## Template 4: "Are the timing and actor-policy claims overstated?"

Core response:

We have intentionally scoped the timing claims. The classifier trajectory
supports measurable anticipation timing, while the actor-policy branch is
reported as downstream support evidence rather than the main claim. The
extended trigger-source diagnostic shows why this distinction matters: the
classifier trigger is substantially stronger than the actor trigger over the
same support block. We therefore avoid presenting the actor branch as a mature
replacement for classifier-derived anticipation scores.

Evidence to cite:

- Intervention status: `EMNLP_INTERVENTION_STATUS.md`
- Trigger-source support: `output/emnlp2026_support/dad_trigger_compare_extended_summary.md`
- Appendix: `paper/neurips2026/sec_appendix.tex`
- Claim audit: `paper/emnlp2026/claim_evidence_audit_report.md`

Key facts:

- Trigger-source diagnostic: classifier `61.11% +/- 4.58` AP vs actor
  `37.41% +/- 7.35` AP over `n=6` checkpoints.
- Actor-policy timing remains support evidence.

Possible paper change:

- Strengthen one limitations sentence saying that mature policy-level timing is
  future work.

Do not say:

- "The actor proves causal timing."
- "The policy head is the main contribution."

## Template 5: "Does the intervention evidence prove causality?"

Core response:

No. The paper does not claim a completed policy-level causal benchmark. The
intervention analysis is used as evidence that the concept layer is
structurally meaningful and partially intervenable: concept edits act on the
state consumed by downstream prediction and can move alerts in some archived
settings. We explicitly report the heterogeneity, including the fact that the
median shift remains zero in the strongest archived hybrid setting.

Evidence to cite:

- Appendix intervention section: `paper/neurips2026/sec_appendix.tex`
- Intervention status: `EMNLP_INTERVENTION_STATUS.md`
- Review quick map: `EMNLP_REVIEWER_QUICK_MAP.md`

Key facts:

- Strongest archived hybrid boost: mean alert shift `-2.04` frames over
  `64` cases.
- Median shift remains `0.0`.
- Null and risk baselines both report `0.0` mean shift over `32` cases.

Possible paper change:

- If needed, replace "causal" with "structural" or "intervention" in any
  remaining ambiguous sentence.
- Preserve caption-level evidence-tier tags to prevent pooled-reading
  misinterpretation.

Do not say:

- "This proves causal control."
- "Every concept edit produces the expected timing behavior."

## Template 6: "Is the ontology human-verified?"

Core response:

The ontology is lightly reviewed, not a finished human-verified benchmark. The
paper is careful about this distinction. Human review covers all 80 released
concepts across nine families, and the language-side audits test pseudo-label
sparsity and paraphrase robustness. We use these assets to support the claim
that the ontology is governed and inspectable, not to claim exhaustive
frame-level human supervision.

Evidence to cite:

- Human audit: `output/emnlp2026_support/human_ontology_audit_summary.md`
- Support results: `EMNLP_SUPPORT_RESULTS.md`
- Pseudo-label audit: `output/emnlp2026_support/topm_pseudolabel_sensitivity_dad500.json`
- Verbalization audit: `output/emnlp2026_support/concept_verbalization_sensitivity_dad500.json`

Key facts:

- Human review covers `80` concepts across `9` families.
- Pseudo-label audit uses `500` DAD frames.
- Verbalization audit reports mean text cosine `0.9389` and mean
  frame-score correlation `0.8872`.

Possible paper change:

- Keep "light human review" phrasing and avoid "human-verified concept
  benchmark."

Do not say:

- "The ontology is fully human validated."
- "The pseudo-labels are exact semantic ground truth."

## Template 7: "What is the main reason to accept?"

Core response:

The paper introduces and evaluates a governed semantic interface for a
safety-critical anticipation task. The contribution is not only that the model
has concepts, but that ontology construction, naming, family structure,
language-side robustness, and matched ontology comparisons are made explicit
and auditable. This changes how accident anticipation systems can be evaluated:
not only by AP and mTTA, but also by whether their semantic evidence layer is a
stable research object.

Evidence to cite:

- Page 1 framing: `paper/emnlp2026/insight_emnlp.tex`
- Motivation figure: `paper/emnlp2026/sec_intro_emnlp.tex`
- Protocol map: `paper/emnlp2026/sec_experiments_emnlp.tex`
- Reviewer quick map: `EMNLP_REVIEWER_QUICK_MAP.md`

Do not say:

- "The main reason is that we outperform every baseline."

## Final Rebuttal Guardrails

- Tie every response to a current artifact path.
- Preserve the DAD/A3D asymmetry.
- Use exact numbers only from current support boards.
- If a reviewer asks for a human decision, do not fabricate it.
- If a reviewer asks for a new GPU experiment, only promise it after the run
  exists and has been summarized.

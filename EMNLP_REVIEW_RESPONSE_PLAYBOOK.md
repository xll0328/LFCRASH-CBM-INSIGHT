# EMNLP Review Response Playbook

Date: 2026-05-06

Purpose: provide response-ready variants for the most likely ARR / EMNLP
reviewer concerns. This playbook is internal preparation. It must be adapted to
actual reviews and must not be presented as human validation, new experimental
evidence, or a promise to run uncompleted GPU work.

## How To Use

- Start with the 30-second answer when the review raises a broad concern.
- Use the formal response when writing the actual rebuttal paragraph.
- Attach only the evidence payload that matches the reviewer's question.
- Add a paper-change commitment only if the change is already made or is a
  small wording/caption clarification that does not require new experiments.
- Do not broaden claims beyond the current support boards.
- Keep the evidence-tier reading rule explicit: headline first, then controlled
  support, then stress evidence.

## Attack 1: "This is not really an EMNLP paper."

30-second answer:

The central object is language-grounded: INSIGHT constructs, canonicalizes,
audits, and evaluates a named risk-concept interface. The paper is not only
about predicting accidents; it studies whether the semantic layer itself is a
governed, trainable, and comparable research object.

Formal response:

We agree that venue fit depends on whether the language layer is central rather
than decorative. In INSIGHT, the language-grounded concept interface is the
object under study: concepts are mined from driving frames, canonicalized into
short risk phrases, organized into semantic families, lightly reviewed, tested
for paraphrase stability, and evaluated under matched ontology comparisons.
The model then uses this ontology as the bottleneck vocabulary, so language-side
design choices affect downstream AP--mTTA behavior rather than only figure
captions.

Evidence payload:

- Intro and abstract: `paper/emnlp2026/insight_emnlp.tex`,
  `paper/emnlp2026/sec_intro_emnlp.tex`
- Method: `paper/emnlp2026/sec_method_emnlp.tex`
- Language-side audit: `EMNLP_SUPPORT_RESULTS.md`
- Verbalization audit:
  `output/emnlp2026_support/concept_verbalization_sensitivity_dad500.json`

Safe paper-change commitment:

- Clarify in the introduction that the ontology is evaluated as a governed
  semantic interface, not as a prompt artifact.

Forbidden response:

- Do not claim the paper is mainly an NLP benchmark.
- Do not claim the language model alone solves anticipation.

## Attack 2: "The ontology is just prompt engineering."

30-second answer:

The ontology is treated as a controlled modeling variable. Under matched
training launchers, changing the ontology source changes the AP--mTTA operating
point, and the multi-seed controlled block is complete across `18/18` cells.

Formal response:

An ungoverned prompted word list would indeed be weak evidence. Our pipeline is
designed to avoid that failure mode. Candidate concepts are mined, filtered,
merged, family-balanced, canonicalized, lightly reviewed, and exported as a
trainable ontology. We then compare ontology sources under matched launchers so
that the intervention is the semantic interface itself. The completed
multi-seed extension covers all six dataset--ontology cells over three seeds,
which supports the claim that ontology construction is a modeling choice rather
than hidden preprocessing.

Evidence payload:

- `output/emnlp2026_support/controlled_ontology_status.md`
- `output/emnlp2026_support/multiseed_ontology_status.md`
- `output/concept_sets/neurips2026_controlled_ontology_manifest.json`
- Table and discussion in `paper/emnlp2026/sec_experiments_emnlp.tex`

Safe paper-change commitment:

- Add or preserve the statement that the ontology source is the only controlled
  change in the matched-launcher block.

Forbidden response:

- Do not claim the ontology is globally optimal.
- Do not claim prompt choice is irrelevant.

## Attack 3: "DAD is weak or unstable."

30-second answer:

DAD is intentionally presented as the harder stress test. The paper does not
claim a DAD leaderboard win; it reports the canonical DAD line together with
multi-seed diagnostics so reviewers can see the fragility rather than infer
that it was hidden.

Formal response:

We agree that DAD is the less stable dataset. Our presentation separates the
canonical DAD line from synchronized-epoch and same-family diagnostics to avoid
pooling search, support, and headline runs. The DAD evidence shows that the
semantic interface remains competitive and measurable under stress, but the
cleanest mechanism story comes from A3D and the controlled ontology block. We
therefore frame DAD as a stress test rather than as the strongest mechanism
proof.

Evidence payload:

- Canonical DAD: `68.19%` AP / `1.75s` mTTA
- Clean three-seed diagnostic: `62.31% +/- 1.90` AP,
  `2.07s +/- 0.09s` mTTA
- Recovery block: `63.52% +/- 0.81` AP,
  `2.16s +/- 0.25s` mTTA over `6/6` completed runs
- Matched full support block (paper stress summary): `63.18% +/- 1.32` AP,
  `2.27s +/- 0.20s` mTTA over `3/3` completed runs
- `output/emnlp2026_support/dad_hardening_status.md`
- `output/emnlp2026_support/dad_curriculum_recovery_status.md`
- DAD discussion in `paper/emnlp2026/sec_experiments_emnlp.tex`

Safe paper-change commitment:

- Clarify that DAD diagnostics bound sensitivity and do not replace the
  canonical headline line.

Forbidden response:

- Do not claim DAD is solved.
- Do not imply DAD evidence is as clean as A3D.

## Attack 4: "The paper overclaims timing or actor-policy behavior."

30-second answer:

The current paper does not make actor-policy timing the main claim. It reports
actor behavior as scoped support evidence and uses the classifier trajectory
for headline AP/mTTA evaluation.

Formal response:

We intentionally separate classifier-triggered anticipation from actor-policy
diagnostics. The classifier trajectory supports the main AP/mTTA numbers, while
the actor-policy branch is included as downstream timing support. The extended
trigger-source diagnostic shows why this distinction is necessary: classifier
triggers remain substantially stronger than actor triggers across the same
support family. We therefore keep actor-policy behavior out of the central
claim and present mature policy-level timing as future work.

Evidence payload:

- Classifier trigger diagnostic: `61.11% +/- 4.58` AP over `n=6`
- Actor trigger diagnostic: `37.41% +/- 7.35` AP over `n=6`
- `EMNLP_INTERVENTION_STATUS.md`
- `output/emnlp2026_support/dad_trigger_compare_extended_summary.md`

Safe paper-change commitment:

- Strengthen the limitations sentence that mature policy-level timing remains
  future work.

Forbidden response:

- Do not claim the actor proves timing causality.
- Do not describe the actor branch as the main contribution.

## Attack 5: "The intervention analysis is causal overclaiming."

30-second answer:

The paper explicitly does not claim a complete policy-level causal benchmark.
The intervention evidence supports partial structural intervenability: concept
edits operate on the state consumed downstream and sometimes move warnings, but
the effects are heterogeneous.

Formal response:

We use intervention analysis to test whether the exposed concept state is more
than post-hoc explanation. Because concept edits are applied to the same state
consumed by downstream prediction, the test is structurally meaningful. At the
same time, the quantitative evidence is deliberately scoped: the strongest
archived hybrid boost has a negative mean alert shift but a zero median shift,
and null-style controls do not move the warning systematically. We therefore
describe the evidence as partial structural intervenability, not completed
policy-level causal control.

Evidence payload:

- Strongest archived hybrid boost: mean alert shift `-2.04` frames over
  `64` cases, median `0.0`
- Null and risk baselines: mean shift `0.0` over `32` cases each
- `paper/neurips2026/sec_appendix.tex`
- `EMNLP_INTERVENTION_STATUS.md`

Safe paper-change commitment:

- Replace ambiguous "causal" phrasing with "structural" or "intervention" if
  a reviewer flags a specific sentence.
- Keep figure/table captions explicitly tagged as headline / controlled support
  / stress evidence where applicable.

Forbidden response:

- Do not say the intervention proof is complete.
- Do not claim every concept edit causes earlier warnings.

## Attack 6: "The human ontology audit is too light."

30-second answer:

We agree it is light review, not a finished concept benchmark. The claim is
that the ontology is governed and inspectable: all 80 released concepts are
reviewed at light-touch level, and the language-side audits test sparsity and
paraphrase stability.

Formal response:

The paper does not present the ontology as exhaustive frame-level human ground
truth. Instead, it presents the ontology as a governed semantic interface:
concepts have canonical names, family assignments, merge provenance, and
light-touch review. We further evaluate pseudo-label sparsity over 500 DAD
frames and test paraphrase robustness over canonical concepts. These checks are
intended to support auditability and naming stability, not to claim complete
human verification.

Evidence payload:

- Human audit: `80` concepts across `9` families
- Pseudo-label audit: `500` DAD frames
- Verbalization audit: mean text cosine `0.9389`, frame-score correlation
  `0.8872`, mean absolute score difference `0.0134`
- `output/emnlp2026_support/human_ontology_audit_summary.md`
- `EMNLP_SUPPORT_RESULTS.md`

Safe paper-change commitment:

- Preserve "light human review" language and add a limitation sentence if
  needed.

Forbidden response:

- Do not call the ontology fully human validated.
- Do not call pseudo-labels exact semantic ground truth.

## Attack 7: "What is the single strongest reason to accept?"

30-second answer:

The paper makes the semantic evidence layer itself evaluable. It shows how to
construct a governed risk-concept interface, audit its language side, and test
whether ontology choice changes downstream AP--mTTA behavior.

Formal response:

The main contribution is a shift in what accident anticipation systems expose
and how they can be evaluated. Rather than treating the model as an opaque risk
score, INSIGHT exposes a governed language-grounded semantic interface and
studies it as a modeling variable. The paper combines ontology construction,
canonical naming, family coverage, paraphrase robustness, matched ontology
comparisons, and interpretable operating-point results. This package lets
reviewers evaluate not only prediction and timing, but also whether the
semantic evidence layer is stable, inspectable, and tied to downstream
behavior.

Evidence payload:

- Page 1: `paper/emnlp2026/insight_emnlp.tex`
- Figure 1: `paper/emnlp2026/sec_intro_emnlp.tex`
- Protocol map: `paper/emnlp2026/sec_experiments_emnlp.tex`
- Reviewer quick map: `EMNLP_REVIEWER_QUICK_MAP.md`

Forbidden response:

- Do not say the primary reason is outperforming every baseline.

## Rapid Response Checklist

Before sending any response:

- Does it preserve the A3D/DAD asymmetry?
- Does it keep actor-policy timing scoped?
- Does it avoid broad causal wording?
- Does it cite a current artifact path?
- Does it avoid promising unrun experiments?
- Does it separate human-only decisions from AI-assisted checks?

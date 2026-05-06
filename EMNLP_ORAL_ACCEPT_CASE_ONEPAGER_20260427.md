# EMNLP Oral Accept Case One-Pager

Date: 2026-04-27
Project: `LFCRASH-CBM`

This is an internal one-page accept-case sheet for oral-level positioning. It
is not a substitute for actual reviewer comments, human validation, or final
upload approval.

## Core Accept Case

INSIGHT deserves serious oral consideration because it makes the semantic
evidence layer itself auditable. The paper is not only an accident anticipation
model; it exposes a governed language-grounded risk-concept interface and
tests whether ontology construction changes the downstream AP--mTTA operating
point.

The central research object is:

- a named risk-concept interface;
- a governed ontology construction pipeline;
- family coverage and canonical naming;
- language-side stability checks;
- matched ontology comparisons that connect the semantic interface to
  prediction and timing behavior.

## Thirty-Second Oral Pitch

Current accident anticipation systems usually expose a risk score. INSIGHT
instead exposes a language-grounded semantic interface: named risk concepts
with provenance, family structure, and audit hooks. We show that ontology
construction is a modeling choice, not hidden preprocessing: under matched
launchers, ontology choice changes the AP--mTTA operating point across DAD and
A3D. The clean flagship evidence is A3D; DAD is treated honestly as a harder
stress test. The contribution is a way to evaluate not only whether a model
warns early, but what semantic evidence layer it exposes and how that layer
behaves.

## Evidence Spine

1. Semantic interface framing:
   `paper/emnlp2026/insight_emnlp.tex`,
   `paper/emnlp2026/sec_intro_emnlp.tex`,
   `paper/emnlp2026/sec_method_emnlp.tex`
2. Controlled ontology evidence:
   `output/emnlp2026_support/controlled_ontology_status.md`,
   `output/emnlp2026_support/multiseed_ontology_status.md`
3. Clean flagship result:
   `output/emnlp2026_support/a3d_headline_multiseed_status.md`
4. Language-side support:
   `EMNLP_SUPPORT_RESULTS.md`,
   `output/emnlp2026_support/concept_verbalization_sensitivity_dad500.json`,
   `output/emnlp2026_support/human_ontology_audit_summary.md`
5. DAD stress-test boundary:
   `output/emnlp2026_support/dad_hardening_status.md`
6. Reviewer defense package:
   `EMNLP_REVIEWER_QUICK_MAP.md`,
   `EMNLP_REVIEW_RESPONSE_PLAYBOOK.md`,
   `paper/emnlp2026/reviewer_defense_coverage_report.md`

## Numbers To Use

- Controlled ontology multi-seed coverage: `18/18` seeded cells.
- A3D headline multi-seed: `94.16% +/- 0.95` AP,
  `4.62s +/- 0.42s` mTTA over `3/3` seeds.
- Canonical DAD: `68.19%` AP, `1.75s` mTTA.
- DAD matched full support: `63.19% +/- 1.21` AP,
  `2.17s +/- 0.05s` mTTA over `3/3` runs.
- Verbalization audit: text cosine `0.9389`, frame-score correlation `0.8872`,
  mean absolute score difference `0.0134`.
- Human ontology audit: `80` reviewed concepts across `9` families.

## Boundaries

- A3D is the clean flagship.
- DAD is the harder stress test.
- Actor-policy timing is support evidence, not a main causal claim.
- Intervention evidence supports partial structural intervenability, not a
  finished policy-level causal benchmark.
- The ontology is lightly reviewed and governed, not an exhaustive
  human-verified concept benchmark.
- Best-paper readiness is not achieved by packaging quality alone.

## What To Say If Asked For The Single Strongest Reason To Accept

The strongest reason to accept is that the paper changes the evaluation object:
it makes the semantic evidence layer inspectable, comparable, and tied to
downstream prediction/timing behavior. This is the part that makes the work
more than a standard accident anticipation system.

## Do Not Say

- Do not say INSIGHT is best on every accident anticipation benchmark.
- Do not say DAD is solved.
- Do not say the actor branch proves causal timing.
- Do not say the ontology is fully human validated.
- Do not promise unrun GPU experiments during rebuttal.

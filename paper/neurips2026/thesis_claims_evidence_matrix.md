# INSIGHT Thesis / Claims / Evidence Matrix

## One-sentence thesis

> Interpretable accident anticipation should be built on an intervenable semantic state space, not post-hoc explanation over a black-box predictor.

---

## Reviewer-facing two-sentence summary

INSIGHT formulates accident anticipation as concept-grounded sequential decision making: a semantic risk state explains **why** the scene is dangerous, and a concept-conditioned timing policy determines **when** to alert. The paper's contribution is not just an interpretable predictor, but an auditable semantic interface whose ontology quality, intervention behavior, and timing dynamics can be analyzed directly.

---

## Core claim 1 — Semantic state construction matters

### Claim
Accident anticipation should use an explicit semantic risk state rather than a purely latent hidden state.

### Evidence already available
- `paper/neurips2026/sec_method.tex` defines the concept bottleneck and concept-augmented state.
- `paper/neurips2026/sec_experiments.tex` reports competitive DAD/A3D operating points in an interpretable regime.
- Ontology pipeline, family coverage, and semantic compression figures already exist in `paper/neurips2026/`.

### What this supports now
- Strong framing and method novelty
- A credible WHY-layer story
- A clean argument that ontology construction belongs inside the method story

### What it does **not** yet prove
- That the semantic state is universally superior to strong black-box alternatives on raw AP
- That all concept channels are equally meaningful or calibrated

---

## Core claim 2 — Timing is a policy problem, not thresholded classification

### Claim
Alert timing should be handled as a sequential control problem over a semantic state, rather than as post-hoc thresholding of a score curve.

### Evidence already available
- `paper/neurips2026/sec_method.tex` defines the CAAC state and reward.
- `paper/neurips2026/sec_experiments.tex` positions mTTA as a first-class metric.
- `output/insight_analysis/dad_timing_v4_enhanced/timing_faithfulness_summary.json` shows event-aligned prediction timing summaries.

### What this supports now
- The architectural and conceptual argument
- A preliminary timing-faithfulness section based on prediction timing
- An explicit hard-case symmetry audit path (`paper/neurips2026/dad_hard_case_symmetry_audit.md`) that keeps stronger symmetry claims gated on manual paired evidence

### What it does **not** yet prove
- A strong actor-policy crossing story in archived outputs, because current stored actor signals are effectively flat
- A definitive claim that policy timing is measurably earlier than threshold timing under a fully validated actor head
- DAD hard-case symmetry across scenario families under a matched, fixed protocol

### Paper language implication
Use this as a **framing claim** plus partial empirical support, not as a definitive fully verified policy-level theorem.

---

## Core claim 3 — Intervenable semantics are more valuable than passive explanation

### Claim
Semantically meaningful concept edits should induce semantically aligned changes in alert timing.

### Evidence already available
- `insight_concept_intervention.py` implements a reproducible intervention analysis path.
- `paper/neurips2026/sec_appendix.tex` includes stored-trajectory and backbone-level intervention framing.
- `output/insight_analysis/dad_intervention_v4_hybrid_boost1/concept_intervention_summary.json` shows mean alert shift of `-2.04` frames over 64 samples under one hybrid boost setting.
- `paper/neurips2026/v4_case_bank_shortlist.json` provides case candidates for qualitative intervention figures.

### What this supports now
- The semantic interface is not purely decorative
- There is already non-trivial evidence that some concept edits can alter prediction timing
- Appendix-level intervention material is worth keeping

### What it does **not** yet prove
- Stable family-level intervention controllability across many cases
- A reliable actor-level intervention effect
- A field-defining causal statement unless the intervention protocol is strengthened

### Paper language implication
Present this as **partial structural intervenability evidence** with honest limits.

---

## Claim hierarchy for the final paper

### Tier A — Can be stated confidently now
1. Accident anticipation benefits from being framed as semantic state construction plus timing control.
2. INSIGHT provides an intrinsically interpretable semantic interface through concepts, ontology metadata, and policy conditioning.
3. Ontology construction is a substantive methodological component, not a hidden preprocessing choice.

### Tier B — Can be stated with careful wording
1. The concept-conditioned design yields meaningful timing-oriented evidence beyond plain descriptive explanations.
2. Semantic edits can alter warning behavior in at least some realistic cases.
3. Compact discovered ontologies can be made trainable, auditable, and paper-ready.

### Tier C — Do **not** state as settled fact yet
1. Fully verified policy-level semantic control is solved.
2. INSIGHT is the global AP leader across all accident anticipation methods.
3. The current repository already proves strong causal timing intervention at scale.
4. DAD hard-case symmetry is already solved by the present evidence package.

---

## Hero figure sentence

> Edit a semantically meaningful risk concept, and the warning policy moves in a semantically aligned direction.

This should be the figure-level takeaway for the main memorable visualization.

---

## Immediate writing implications

1. Use the thesis sentence in the abstract, introduction, and conclusion.
2. Keep the paper centered on three evidence blocks:
   - ontology quality
   - timing faithfulness
   - intervention behavior
3. Treat leaderboard results as support, not as the sole narrative spine.
4. Separate canonical submission results from support runs and search runs.

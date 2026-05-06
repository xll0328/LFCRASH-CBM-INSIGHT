# GPT Image 2 Teaser Quality Gate

Use this gate after every generated candidate. A candidate must pass all
CRITICAL items and most MAJOR items before it can replace a paper figure.

## CRITICAL

- Labels are readable at single-column or full-width NeurIPS scale.
- The figure says concepts are in the decision pathway, not post-hoc.
- The figure keeps DAD / policy-level claims bounded; no "solved", "causal
  proof", "global SOTA", or fake benchmark values.
- The traffic scene shows anticipation before impact, not crash aftermath.
- No hallucinated architecture modules beyond the paper's vocabulary.

## MAJOR

- Clear left-to-right story: video -> concepts -> concept-augmented state ->
  timing -> audit trail.
- WHY and WHEN are visually distinguishable.
- Color roles are consistent with the paper: blue for WHY/risk, green for WHEN,
  amber for agents, red only for impact.
- Figure remains readable after conversion into the LaTeX PDF.
- There is enough white space; no crowded prompt-art clutter.

## MINOR

- Visual tone matches the existing paper figures.
- The dashcam scene is plausible and not stock-like.
- The mini timing chart looks schematic but not childish.
- The generated image does not contain extra decorative text or watermarks.

## Iteration Notes Template

Candidate path:

Pass / fail:

Strong points:

Failure points:

Next prompt delta:

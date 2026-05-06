Create a polished NeurIPS-style graphical abstract for a research paper titled:

INSIGHT: Interpretable Accident Anticipation with Concept-Guided Risk Reasoning and Timing

Generate a wide, high-resolution academic paper figure, not a square poster.
The target layout is a clean 1536x1024 landscape teaser that can span two
columns in a NeurIPS paper. It should look like a mature ML systems/vision paper
figure: precise, readable, restrained, and immediately interpretable.

Main scientific message:
Accident anticipation should expose an intervenable semantic risk state. The
model should explain WHY risk is increasing through named concepts, and decide
WHEN to warn through concept-conditioned timing. The figure must communicate
that semantic concepts are part of the decision pathway, not a post-hoc
explanation attached after a black-box prediction.

Composition requirements:
1. Overall structure: three large panels from left to right with a thin,
   understated flow arrow between panels. Use a single compact footer for audit,
   not a second full pipeline.
2. Left panel, "Dashcam sequence": show three realistic dashcam video frames
   stacked vertically or slightly staggered along a time axis. The frames should
   imply increasing pre-crash risk on a highway or urban road: close following,
   brake-light onset, a nearby vehicle entering the lane, limited visibility, or
   a vehicle conflict. Use small gold bounding boxes around relevant vehicles.
   No visible crash, no damage, no smoke, no gore. This is anticipation before
   impact.
3. Middle panel, "WHY: semantic risk concepts": show a compact semantic risk
   state with 5 readable concept chips and subtle activation bars. Required
   concept chip labels:
   - brake-light onset
   - close headway
   - lane intrusion
   - visibility limit
   - rear-end risk
   Include a simple neural/semantic icon above or beside the chips, but avoid
   decorative 3D objects.
4. Right panel, "WHEN: concept-conditioned alert timing": show one clean
   schematic chart. The blue risk curve rises over time. A green alert policy
   switches before a red dashed impact boundary. Add a green dashed "alert" line
   before the red "impact" line, and label the interval "Early warning".
   Keep the chart schematic and do not include numeric benchmark values.
5. Footer: one short audit strip, only 10-15% of figure height, reading
   "Audit trail" with a concise chain:
   Video state -> Semantic concepts -> Concept-augmented state -> Alert timing.
   Do not repeat the word "Audit trail" twice.

Text labels that must appear exactly and be readable:
- INSIGHT
- WHY: semantic risk concepts
- WHEN: concept-conditioned alert timing
- Early warning
- Audit trail

Style:
- White or very light gray background, thin gray panel dividers, clean sans-serif
  typography, publication-quality alignment.
- Blue represents semantic concepts and risk score; green represents alert
  timing; amber/gold represents object boxes; red only marks impact.
- Prefer crisp vector-like overlays on realistic dashcam frames.
- Use generous spacing and fewer arrows. Avoid crowded infographic clutter.
- The figure should remain readable when reduced to paper width.

Scientific guardrails:
- Do not claim global SOTA, solved safety, causal proof, human validation, or
  policy readiness.
- Do not include fake AP, mTTA, accuracy, or leaderboard numbers.
- Do not include equations unless they are simple unlabeled schematic marks.
- Do not use painterly, Fauvist, anime, comic, cinematic, glossy marketing, or
  web landing page aesthetics.

Negative prompt:
blurry labels, misspelled labels, tiny text, overfilled diagram, duplicated audit
trail, fake metrics, fake equations, crash aftermath, fire, smoke, blood, gore,
robot faces, autonomous dashboard UI, stock collage, neon gradients, 3D cubes,
decorative swirls, Fauvism, painterly brushstrokes, saturated art poster.

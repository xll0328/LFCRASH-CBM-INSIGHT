Create a polished NeurIPS-style graphical abstract for a research paper titled:

INSIGHT: Interpretable Accident Anticipation with Concept-Guided Risk Reasoning and Timing

The image should look like a top-tier AI conference paper teaser, not a marketing
poster and not a decorative painting. It must be clean, technical, readable,
and suitable for a two-column machine learning paper. Use a wide 3:2 canvas.

Core thesis to communicate:
Interpretable accident anticipation should be built on an intervenable semantic
risk state, not post-hoc explanation over a black-box predictor.

Scientific story:
1. A dashcam video approaches a traffic accident.
2. A WHY layer converts visual evidence into named semantic risk concepts.
3. A WHEN layer uses the concept-augmented state to decide alert timing.
4. The output is an early warning plus an audit trail that a safety engineer can inspect.

Composition:
- Left third: a realistic dashcam road scene sequence, shown as 3 small frames
  along a timeline. The frames should imply an emerging hazard such as close
  following, brake-light onset, lane intrusion, or pedestrian/vehicle conflict.
  Use plausible bounding boxes in gold, but do not make the scene graphic or
  violent. No crash aftermath. The point is anticipation before impact.
- Middle third: a transparent semantic layer over the scene. Show 4-6 named
  concept chips arranged as a compact risk state, with subtle activation bars:
  "brake-light onset", "close headway", "lane intrusion", "visibility limit",
  "rear-end risk". These concept names should be readable.
- Right third: a timing decision panel. Show a clean mini line chart with a blue
  "risk score" curve rising, a green "alert policy" curve, a dashed vertical
  "alert" line before a dashed red "impact" line, and a small arrow labeled
  "early warning". The chart should be schematic, not numerically exact.
- Bottom strip: a concise audit trail flowing left to right:
  "Video state" -> "Semantic concepts" -> "Concept-augmented state" ->
  "Alert timing" -> "Audit trail".

Text labels that must appear exactly:
- INSIGHT
- WHY: semantic risk concepts
- WHEN: concept-conditioned alert timing
- Early warning
- Audit trail

Important correctness constraints:
- Do not claim global SOTA.
- Do not show "causal proof", "solved", "fully reliable", or any stronger
  policy-level claim.
- Do not imply the model has completed human validation.
- Do not use fake numeric AP/mTTA values.
- Do not write long paragraphs inside the figure.
- Do not use a Fauvist, painterly, comic, anime, sci-fi, or glossy marketing style.
- Do not make the image look like a web landing page.

Visual style:
- Academic infographic with realistic dashcam imagery and crisp vector-like
  overlays.
- White or very light gray background, restrained palette, high contrast labels.
- Color roles: blue for WHY/risk score, green for WHEN/alert timing, amber/gold
  for agent boxes, red only for impact boundary.
- Typography should be clean sans-serif, readable at paper scale, with no tiny text.
- Use generous spacing and avoid clutter.
- The final image should be immediately understandable to a NeurIPS reviewer in
  five seconds: semantic concepts are inside the decision pathway, not added
  after prediction.

Negative prompt:
blurry labels, illegible text, extra claims, decorative swirls, gradient blobs,
3D cubes, robot faces, autonomous car dashboard UI, stock photo collage, crash
violence, blood, fire, gore, overfilled diagram, too many arrows, tiny unreadable
captions, fake equations, fake benchmark numbers, painterly style, Fauvism.

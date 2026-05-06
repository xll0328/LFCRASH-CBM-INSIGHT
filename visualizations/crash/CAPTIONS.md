# CG-CRASH Visualization Gallery — CCD (Crash) Dataset

## Dataset Overview
- **Dataset**: Car Crash Dataset (CCD) — dashcam videos of real traffic accidents
- **Task**: Binary accident prediction from multi-object visual features
- **Model**: CG-CRASH (Concept-Guided CRASH) with CBM interpretability module
- **Key result**: AP = 99.22%, mTTA = 4.66s (vs CRASH baseline: AP=99.6%, mTTA=4.91s)

---

## Figure 1: `best_case_study.png` / `best_case_study_sample2.png` / `best_case_study_sample3.png`

**Caption**: Comprehensive case study of CG-CRASH's accident prediction on a single CCD video sequence. **(Top row)** Eight evenly-sampled video frames spanning the full clip. Red borders indicate post-crash frames; orange borders indicate the alert zone (P>0.5). Probability scores are shown above each frame. **(Middle row)** Accident prediction confidence curve P(t) over time. Green fill = safe phase; orange = alert phase (P>0.5 before crash); red = post-crash. Dashed vertical lines mark the alert trigger and crash onset. **(Bottom left)** Concept activation heatmap showing the temporal evolution of the top-8 safety-critical concepts (ranked by Safe→Crash activation delta). **(Bottom right)** Side-by-side comparison of concept activation at safe baseline vs. crash moment, with Δ annotations indicating which concepts spike at the crash.

**Analysis**: The model issues early warning ~2–4 seconds before crash onset, consistent with LFCRASH-CBM's temporal advantage. Concepts such as "Congested traffic with vehicles merging" and "Nighttime driving with glare" show strong activation increase at crash time, providing human-interpretable explanations for the prediction.

---

## Figure 2: `multi_case.png`

**Caption**: Multi-case visualization of the top-5 best-predicted accident samples from the CCD test set. Each row corresponds to one sample: **(left)** 8-frame strip with P-score labels and color-coded borders (red=crash, orange=alert); **(center)** prediction confidence curve with crash/alert markers; **(right)** concept activation heatmap for the globally most discriminative concepts across all shown samples.

**Analysis**: All five samples achieve near-perfect peak probability (P>0.99). The concept heatmap reveals consistent activation patterns across different accident scenarios — nighttime glare, lane merging, and wet roads are recurrent safety-critical concepts. The diversity of video content demonstrates the model's generalisation ability.

---

## Figure 3: `concept_importance.png`

**Caption**: Concept-level discriminability analysis. **(Left)** Mean concept activation for accident (red) vs. normal (blue) samples across the top-15 most discriminative concepts. Double-headed arrows indicate the activation gap for the top-5 concepts. **(Right)** Cohen's d discriminability score for each concept, color-coded from high (dark red) to low (light green).

**Analysis**: The top concepts — including nighttime visibility issues, traffic congestion, and wet road conditions — are semantically meaningful safety hazards with high face validity. The large gap between accident and normal activations (Cohen's d > 2.5 for top concepts) confirms that the CBM has learned genuinely discriminative, human-interpretable representations rather than spurious correlations.

---

## Figure 4: `paper_strip.png` / `paper_strip_sample2.png`

**Caption**: CVPR-style four-panel strip. **(Top row)** Four key frames at Normal, Pre-Alert, Alert, and Crash stages, with colored borders and stage labels. **(Middle row)** Per-stage concept activation bars comparing current activation (colored) against safe baseline (light blue). Only the leftmost panel shows concept labels; subsequent panels share the same order. The Crash panel annotates concepts with Δ values (red=increase, blue=decrease). **(Bottom row)** Full prediction confidence timeline with stage markers.

**Analysis**: This figure is designed for the main paper figure slot, compactly communicating both the temporal prediction behavior and the concept-level explanation in one coherent layout. The concept ordering is fixed across stages, enabling direct comparison of how individual concepts evolve from normal driving to the crash moment.

---

## Figure 5: `timeline_concepts.png`

**Caption**: Three-stage concept evolution analysis for the best-predicted sample. **(Top)** Prediction confidence curve with stage annotations. **(Rows 2–4)** Per-stage mean concept activation bars (colored) overlaid on safe baseline (light blue), for Normal, Alert, and Crash phases respectively. Δ annotations with colored badges show the direction and magnitude of change from safe baseline.

**Analysis**: The progressive increase in accident-relevant concept activations from Normal → Alert → Crash provides a clear narrative of how the model's internal representations evolve as the accident unfolds. This is a key figure for demonstrating the interpretability advantage of CG-CRASH over black-box methods.

---

## Figure 6: `animation_keyframes.gif`

**Caption**: Animated visualization showing real-time accident prediction. Each frame shows: **(left)** the current video frame with status badge (NORMAL/ALERT/CRASH); **(center)** the growing prediction confidence curve with a live cursor dot; **(right)** the live concept bar chart showing which safety concepts are currently active.

**Analysis**: The animation demonstrates the model's real-time interpretability — as the scene evolves, both the prediction confidence and the active concept set update dynamically, providing a transparent and human-readable explanation of each prediction step.

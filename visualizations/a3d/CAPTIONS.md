# CG-CRASH Visualization Gallery — A3D Dataset

## Dataset Overview
- **Dataset**: Anticipating Accidents in Dashcam Videos (A3D) — diverse dashcam recordings
- **Task**: Accident anticipation; high positive rate (91% positive in test set)
- **Model**: CG-CRASH with CBM interpretability module
- **Key result**: AP = 94.75%, mTTA = 4.81s, TTA@R80 = 4.44s
- **Note**: No raw video available for A3D — all visualizations are feature-based (no frame strips).

---

## Figure 1: `best_case_study.png` / `best_case_study_sample2.png` / `best_case_study_sample3.png`

**Caption**: Comprehensive case study for A3D. Since raw video frames are unavailable, the top panel shows a placeholder. **(Middle row)** Prediction confidence curve with Safe/Alert/Crash phase coloring and stage markers. **(Bottom left)** Concept activation heatmap — top-8 concepts by Safe→Crash delta. **(Bottom right)** Safe vs. crash concept comparison with Δ annotations.

**Analysis**: Despite the absence of raw video, the concept-level analysis remains highly informative. A3D's top concepts include "Poor nighttime visibility", "Pedestrians crossing at night", "Wet roads and reduced visibility" — consistent with A3D's focus on nighttime and adverse-weather accident scenarios. The high positive rate (91%) makes A3D's curves characteristically high-baseline.

---

## Figure 2: `multi_case.png`

**Caption**: Top-5 best-predicted A3D samples. Each row: feature-based placeholder (no video) + prediction confidence curve + concept heatmap.

**Analysis**: A3D's prediction confidence curves show characteristically high values throughout, reflecting the dataset's high positive rate. The concept heatmap consistently highlights nighttime and weather-related hazard concepts, with "Poor nighttime visibility" and "Wet roads" appearing as the most universally active concepts across samples.

---

## Figure 3: `concept_importance.png`

**Caption**: Top-15 discriminative concept analysis for A3D. Mean activation comparison between accident and normal samples, with Cohen's d discriminability scores.

**Analysis**: A3D's Concept Completeness Score of 100% (linear probe AP) confirms that the learned concept activations are fully sufficient for predicting accidents. Cohen's d ≈ 0.97 for top concepts is intermediate between CCD (2.80) and DAD (0.24), reflecting A3D's medium difficulty. The dominance of nighttime/weather concepts aligns with A3D's scenario distribution.

---

## Figure 4: `paper_strip.png` / `paper_strip_sample2.png`

**Caption**: Four-stage concept evolution strip for A3D. Since no video frames are available, the top row shows informational placeholders. Per-stage concept bars demonstrate activation shifts from Normal to Crash. Only the first column shows concept labels.

**Analysis**: The concept bars demonstrate that A3D accidents are associated with a clear progression of environmental hazard concepts. The crash panel's Δ values show strong positive changes for nighttime visibility and pedestrian-related concepts, consistent with A3D's accident scenarios.

---

## Figure 5: `timeline_concepts.png`

**Caption**: Three-stage concept evolution (Normal → Alert → Crash) for the best A3D sample. Per-stage activation bars with Δ badges relative to safe baseline.

**Analysis**: The A3D timeline shows a distinctive pattern where most concepts are already moderately active in the Normal phase (due to the high positive base rate), with a concentrated spike in nighttime and pedestrian concepts at the Crash phase. This is structurally different from CCD and DAD, reflecting A3D's dataset characteristics.

---

## Figure 6: `animation_keyframes.gif`

**Caption**: Animated visualization for the best A3D sample. Since video frames are unavailable, the left panel shows a status indicator. The center panel shows the growing confidence curve and the right panel shows live concept activation bars.

**Analysis**: Despite the lack of raw video, the animated confidence curve and live concept bars provide a clear temporal narrative of the accident prediction. The concept bars reveal which safety-critical concepts the model attends to at each timestep.

---

## Cross-Dataset Summary

| Metric | CCD | DAD | A3D |
|--------|-----|-----|-----|
| AP (%) | 99.22 | 63.20 | 94.75 |
| mTTA (s) | 4.66 | **1.72** | **4.81** |
| TTA@R80 (s) | 4.26 | **2.64** | **4.44** |
| Top concept theme | Nighttime/Glare | Motorcycles/Pedestrians | Nighttime/Weather |
| Concept Completeness | 100% | 87.5% | 100% |
| Cohen's d (top-20) | 2.80 | 0.24 | 0.97 |

CG-CRASH achieves competitive accuracy while providing full concept-level interpretability — a unique advantage over all existing accident prediction methods.

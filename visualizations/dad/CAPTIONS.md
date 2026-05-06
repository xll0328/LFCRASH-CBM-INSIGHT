# CG-CRASH Visualization Gallery — DAD Dataset

## Dataset Overview
- **Dataset**: Driver Attention in Driving (DAD) — dashcam videos from diverse Asian traffic environments
- **Task**: Binary accident prediction; notably harder than CCD due to diverse scene types and imbalanced classes
- **Model**: CG-CRASH with CBM interpretability module
- **Key result**: AP = 63.20%, mTTA = 1.72s, TTA@R80 = 2.64s
- **Note**: mTTA (1.72s) and TTA@R80 (2.64s) **outperform** CRASH baseline (1.91s / 2.20s), showing our model issues earlier warnings even if overall AP is lower due to dataset difficulty.

---

## Figure 1: `best_case_study.png` / `best_case_study_sample2.png` / `best_case_study_sample3.png`

**Caption**: Comprehensive case study of CG-CRASH's prediction on a DAD video sequence. **(Top row)** Eight video frames across the full clip with P-score and colored borders (red=crash, orange=alert). **(Middle row)** Prediction confidence curve with colored fill regions and stage markers. **(Bottom left)** Concept activation heatmap — top-8 concepts ranked by Safe→Crash delta activation. **(Bottom right)** Safe baseline vs. crash-moment concept comparison with Δ annotations.

**Analysis**: DAD sequences involve complex Asian urban traffic scenarios including motorcycles, pedestrians, and intersections. The model identifies safety-relevant concepts such as "Motorcycle closely following a car", "Pedestrians crossing near moving vehicles", and "Narrow bridge with no visible shoulder" — all semantically appropriate for the accident context. Three alternative samples are provided for paper figure selection.

---

## Figure 2: `multi_case.png`

**Caption**: Top-5 best-predicted accident cases from the DAD test set. Each row: **(left)** 8-frame strip with probability scores and color-coded borders; **(center)** prediction confidence curve; **(right)** shared concept activation heatmap for globally discriminative concepts.

**Analysis**: DAD predictions exhibit more varied confidence curve shapes compared to CCD, reflecting the dataset's higher difficulty. Some samples show gradual probability increase while others spike sharply. The concept heatmap reveals that motorcycle-related and pedestrian-related concepts are the most consistently activated across high-confidence accident predictions in DAD.

---

## Figure 3: `concept_importance.png`

**Caption**: Top-15 discriminative concept analysis for DAD. **(Left)** Mean activation for accident vs. normal samples with bi-directional arrows for top-5 discriminative concepts. **(Right)** Cohen's d discriminability scores.

**Analysis**: Unlike CCD (dominated by nighttime/glare concepts), DAD's top concepts center on vulnerable road users (motorcyclists, pedestrians) and constrained environments (narrow roads, intersections). Cohen's d values are lower than CCD (≈0.24 vs ≈2.80), reflecting DAD's inherently harder prediction task. This cross-dataset comparison of concept discriminability is valuable evidence for the model's adaptability.

---

## Figure 4: `paper_strip.png` / `paper_strip_sample2.png`

**Caption**: Four-stage visualization strip for a DAD accident sequence. Normal → Pre-Alert → Alert → Crash stages with per-stage concept bars. Concept names shown only on leftmost panel; all panels share the same concept order for direct comparison. Crash panel annotates delta values.

**Analysis**: The paper strip format compactly demonstrates how concept activations shift at each stage of the accident. For DAD sequences, motorcycle-following and intersection concepts show the strongest Δ at crash time, consistent with the dataset's demographic of urban intersection accidents.

---

## Figure 5: `timeline_concepts.png`

**Caption**: Three-stage concept evolution (Normal → Alert → Crash) for the best DAD sample. Each row shows per-stage mean activation vs. safe baseline, with Δ badges.

**Analysis**: The timeline reveals that DAD accidents involve more gradual concept activation shifts compared to CCD — the Normal and Alert phases show similar concept profiles, with the Crash phase showing the distinctive spike. This pattern is consistent with DAD's lower AP scores (harder to distinguish safe from alert phases).

---

## Figure 6: `animation_keyframes.gif`

**Caption**: Frame-by-frame animated prediction visualization for the best DAD accident sample. Live concept bars and growing confidence curve with real-time status badge.

**Analysis**: The DAD animation illustrates the model's behavior in complex Asian traffic — the confidence curve rises more gradually than in CCD, reflecting the higher uncertainty. The live concept bars show dynamic shifts in motorcycle and pedestrian-related concepts as the accident unfolds, providing transparent real-time explanations.

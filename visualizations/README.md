# CG-CRASH Visualization Gallery

This directory contains all publication-quality visualizations for the **CG-CRASH** paper.

## Structure

```
visualizations/
├── crash/          # CCD (Car Crash Dataset)
│   ├── CAPTIONS.md
│   ├── best_case_study.png          # Hero figure - sample 1
│   ├── best_case_study_sample2.png  # Hero figure - sample 2
│   ├── best_case_study_sample3.png  # Hero figure - sample 3
│   ├── multi_case.png               # Top-5 cases, wide 8-frame strip
│   ├── concept_importance.png       # Top-15 discriminative concepts
│   ├── paper_strip.png              # CVPR-style 4-stage strip - sample 1
│   ├── paper_strip_sample2.png      # CVPR-style 4-stage strip - sample 2
│   ├── timeline_concepts.png        # Concept evolution Normal->Alert->Crash
│   └── animation_keyframes.gif      # Animated real-time prediction
├── dad/            # DAD (Driver Attention in Driving)
│   └── ... (same structure)
└── a3d/            # A3D (Anticipating Accidents in Dashcam)
    └── ... (same structure)
```

## Key Results

| Dataset | AP (%) | mTTA (s) | TTA@R80 (s) | vs. CRASH baseline |
|---------|--------|----------|-------------|--------------------|
| CCD     | 99.22  | 4.66     | 4.26        | -0.4% AP (interpretable) |
| DAD     | 63.20  | 1.72     | 2.64        | +earlier warning |
| A3D     | 94.75  | 4.81     | 4.44        | -1.25% AP (interpretable) |

## Recommended Figures for Paper

| Paper Section | Recommended Figure |
|---------------|-------------------|
| Main figure (teaser) | crash/paper_strip.png or crash/best_case_study.png |
| Ablation visualization | output/paper_tables/ablation_heatmap.png |
| SOTA comparison | output/paper_tables/sota_scatter.png |
| Concept interpretability | crash/concept_importance.png + crash/timeline_concepts.png |
| Multi-case evidence | crash/multi_case.png |
| Supplementary animation | crash/animation_keyframes.gif |

## Figure Descriptions

See CAPTIONS.md in each dataset folder for detailed captions and analysis text.

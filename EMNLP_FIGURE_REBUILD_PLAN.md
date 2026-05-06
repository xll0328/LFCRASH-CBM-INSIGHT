# EMNLP Figure Rebuild Plan

## Goal

Use the existing figure assets to support the new EMNLP narrative:
the paper is about a language-grounded risk concept interface, not about a
maximal actor-policy claim.

## Current Best Assets

### Semantic interface / ontology assets

- `paper/figures/insight_fig_concept_pipeline.pdf`
- `paper/figures/insight_fig_concept_family_coverage.pdf`
- `paper/figures/insight_fig_concept_case_study.pdf`
- `paper/figures/insight_fig_concept_evolution.pdf`

Generation scripts:

- `paper/neurips2026/make_concept_pipeline_fig.py`
- `paper/neurips2026/make_concept_family_coverage_fig.py`
- `paper/neurips2026/make_concept_case_study_fig.py`
- `paper/neurips2026/make_concept_evolution_fig.py`

### Predictive / operating-point assets

- `paper/figures/insight_fig5_safety_utility.pdf`
- `output/paper_tables/sota_scatter.pdf`
- `output/paper_tables/sota_comparison.pdf`
- `output/paper_tables/ablation_heatmap.pdf`

### Case-study assets

- `paper/figures/insight_fig1_hero_crash.pdf`
- `paper/figures/insight_fig1_hero_dad_real.pdf`
- `paper/figures/insight_fig2_multi_dad_real.pdf`

## Locked Main-Paper Lineup

### Figure 1: semantic interface in action

Locked asset:

- `visualizations/crash/paper_strip.png`
- `visualizations/crash/timeline_concepts.png`

Role:

- teach the reader what the interface looks like on a real case before any
  architecture or leaderboard discussion

### Figure 2: model-side semantic interface pipeline

Locked asset:

- `paper/figures/insight_framework.pdf`

Role:

- show how ontology, CBM, CRS/CGTA, and timing connect inside one model

### Figure 3: ontology construction pipeline

Locked asset:

- `paper/figures/insight_fig_concept_pipeline.pdf`

Role:

- make ontology construction visibly part of the method

### Figure 4: predictive operating point

Locked asset:

- `paper/figures/insight_fig5_safety_utility.pdf`

Role:

- position the model as a strong interpretable operating point, not absolute
  SOTA

### Figure 5: ontology family evidence

Locked asset:

- `paper/figures/insight_fig_concept_family_coverage.pdf`

Role:

- show that the paper-facing ontology is balanced, task-facing, and auditable

### Figure 6: ontology evolution and canonicalization

Locked asset:

- `paper/figures/insight_fig_concept_evolution.pdf`
- `paper/figures/insight_fig_concept_case_study.pdf`

Role:

- show that the final ontology is governed through merge provenance and
  canonical naming, not just selected by taste

## Figures To Move Out Of The Main Narrative

- actor-heavy intervention composites
- DAD score-heavy hero figures that mainly replay the old WHY+WHEN rhetoric
- appendix-only casebank figures
- t-SNE figures unless a reviewer specifically asks for them

Reason:

- these figures are support assets, not the cleanest route to the new EMNLP thesis

## Immediate Implication For The Draft

1. Keep the intro composite as the only "live prediction" hero figure.
2. Keep ontology pipeline, safety-utility, family coverage, and ontology evolution in the main text.
3. Use concept canonicalization examples as the qualitative ontology evidence.
4. Move actor-heavy and intervention-heavy visuals to appendix/support material.

# EMNLP Support Results

## Scope

These are the low-cost support analyses that strengthen the EMNLP framing
without launching new large training jobs.

Generated with:

- `paper/emnlp2026/run_emnlp_support_analyses.sh`

Outputs:

- `output/emnlp2026_support/dad500_frame_manifest.json`
- `output/emnlp2026_support/topm_pseudolabel_sensitivity_dad500.json`
- `output/emnlp2026_support/concept_verbalization_sensitivity_dad500.json`

Older 120-frame audit files are retained for provenance, but the paper-facing
numbers below use the current 500-frame DAD audit.

## Top-m pseudo-label sensitivity

Audit setup:

- 500 DAD frames
- paper-facing `perfect_concept_set_v1` ontology
- compare `top-m` for `m in {1, 3, 5, 10}`

Key numbers:

- `top-1`: average family diversity `1.00`, relative score mass vs top-20 `0.0548`
- `top-3`: average family diversity `2.336`, relative score mass vs top-20 `0.1602`
- `top-5`: average family diversity `3.35`, relative score mass vs top-20 `0.2630`
- `top-10`: average family diversity `5.136`, relative score mass vs top-20 `0.5146`

Interpretation:

- `top-1` is too concentrated to serve as a stable semantic bootstrap.
- `top-10` is much more diffuse and pulls in substantially more low-confidence
  mass.
- `top-3` is the clean middle ground for the current recipe.

## Concept verbalization sensitivity

Audit setup:

- 12 canonical paper-facing concepts
- 2 paraphrase variants per canonical concept
- 500 DAD frames

Aggregate results:

- mean text cosine: `0.9389`
- mean frame-score correlation: `0.8872`
- mean top-10 frame overlap: `0.5250`
- mean absolute score difference: `0.0134`

Observed weaker paraphrases:

- `unsafe following distance` vs `insufficient following gap`
- `limited lateral clearance` vs `narrow side clearance`
- `lane change maneuver` vs `vehicle changing lanes`
- `visibility reduction` vs `poor visibility conditions`
- `motorcycle proximity` vs `nearby motorcycle`

Interpretation:

- canonical names are reasonably stable anchors for the semantic interface
  under light paraphrase variation
- shorter canonical risk phrases remain preferable for paper-facing ontology
  naming and intervention design

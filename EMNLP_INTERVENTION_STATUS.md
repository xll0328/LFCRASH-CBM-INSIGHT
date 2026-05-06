# EMNLP Intervention Status

This note freezes the current interpretation boundary for timing and
intervention evidence so the paper and future rebuttal stay consistent.

## Timing-faithfulness

Source:

- `output/insight_analysis/dad_timing_v4_enhanced/timing_faithfulness_summary.json`

Usable evidence now:

- prediction-branch lead time over valid DAD positives:
  `pred_to_toa.mean = 39.41` frames over `39` valid cases
- at `20 fps`, this is roughly `1.97s` mean prediction lead time

Boundary:

- actor crossing is not available in the archived summary
- actor probabilities remain effectively flat:
  `actor_peak.mean = 0.4981`, `actor_mean.mean = 0.4873`

Interpretation:

- prediction timing is measurable and paper-usable
- actor-policy timing is not yet strong enough for a hard claim

## Intervention

Primary source:

- `output/insight_analysis/dad_intervention_v4_hybrid_boost1/concept_intervention_summary.json`

Support baselines:

- `output/insight_analysis/dad_intervention_v4_hybrid/concept_intervention_summary.json`
- `output/insight_analysis/dad_intervention_v4_risk/concept_intervention_summary.json`

Current strongest support:

- hybrid boost setting
  - `num_samples = 64`
  - `mean_alert_shift = -2.04` frames
  - `median_alert_shift = 0.0`
  - `mean_peak_prob_delta = 0.00224`

Control-style comparisons:

- hybrid zero-baseline
  - `num_samples = 32`
  - `mean_alert_shift = 0.0`
- risk-weight baseline
  - `num_samples = 32`
  - `mean_alert_shift = 0.0`

Boundary:

- actor alert shifts are `null` in these archived outputs
- intervention effects are heterogeneous and often zero at the case level

Interpretation:

- the semantic interface is not merely decorative
- some positive concept edits can move prediction timing earlier
- the evidence supports partial structural intervenability, not large-scale
  policy-level causal control

## Writing rule

Allowed wording:

- partial structural intervenability
- support evidence for a meaningful semantic control surface
- prediction-branch timing evidence

Avoid:

- strong policy-level intervention proof
- robust actor-level timing control
- large-scale causal timing guarantee

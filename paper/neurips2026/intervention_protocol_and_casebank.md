# Semantic Intervention Protocol and Casebank

## Objective
Test whether semantically meaningful concept edits induce semantically aligned changes in warning behavior.

## Available implementation assets
- `insight_concept_intervention.py`
- `paper/neurips2026/intervention_v4_backbone.py`
- `paper/neurips2026/intervention_scan.py`
- `paper/neurips2026/scan_v4_intervention_cases.json`
- `paper/neurips2026/v4_case_bank_shortlist.json`
- archived outputs in `output/insight_analysis/*intervention*`

## Intervention families

### Family A — Visibility / weather
Examples:
- reduced visibility due to rain
- wet-road glare
- blocked forward view

Expected effect:
- amplification may advance warnings in low-visibility near-crash cases
- suppression may delay warnings when visibility degradation is a primary cue

### Family B — Pedestrian crossing / vulnerable road users
Examples:
- pedestrian crossing zone
- jaywalking risk
- cyclist crossing conflict

Expected effect:
- amplification should advance warnings in crosswalk / pedestrian conflict cases
- suppression should reduce or delay alerts in those same cases

### Family C — Merge / trajectory conflict
Examples:
- merge conflict
- cut-in behavior
- lane intrusion
- crossing trajectory conflict

Expected effect:
- amplification should move warnings earlier in interaction-heavy cases
- strongest candidate for the hero intervention figure

### Family D — Proximity / following distance
Examples:
- closely spaced vehicles
- insufficient following distance
- closing-speed conflict

Expected effect:
- amplification should advance rear-end or congestion-related alerts

## Intervention modes
1. **Suppression**: set target concepts toward zero over a pre-crash window
2. **Amplification**: raise target concepts gradually in a pre-crash window
3. **One-family-at-a-time edits**: edit only one semantic family per case
4. **Ontology granularity comparison**: apply analogous edits under 30-concept and 80-concept ontologies when available

## Metrics
For each intervention, record:
- original alert frame
- edited alert frame
- alert shift (edited - original)
- original peak probability
- edited peak probability
- peak probability delta
- actor alert shift when valid actor outputs are available
- qualitative note on semantic alignment

## Current evidence boundary
Archived DAD intervention outputs show:
- hybrid suppression-like setting: mostly zero shift
- hybrid amplification setting (`intervention_value=1.0`): mean alert shift `-2.04` frames across 64 samples, but with heterogeneous case behavior
- actor branch is not yet reliable in archived outputs and should not be overclaimed

## Casebank selection policy
Use `paper/neurips2026/v4_case_bank_shortlist.json` as the primary shortlist.

### Named paper-facing slots
- `strong_primary`
- `strong_secondary`
- `borderline_positive`
- `borderline_negative`
- `near_threshold_alt`
- `night_visibility_alt`

## Recommended case use
- Main paper hero figure: `strong_primary`
- Main paper supporting case: `strong_secondary`
- Appendix contrast cases: `borderline_positive`, `borderline_negative`, `night_visibility_alt`

## Success criteria for paper-facing intervention evidence
1. At least one family-level amplification case with clear earlier warning
2. At least one suppression case showing delayed or weakened warning in the expected scenario family
3. At least one failure case included honestly
4. Clear separation between prediction-branch evidence and full actor-policy evidence

## Language policy for the paper
Allowed now:
- semantic edits can alter warning behavior in some realistic cases
- INSIGHT exposes an intervention-ready semantic interface

Not yet allowed without stronger reruns:
- semantic intervention reliably controls the full actor policy at scale
- strong causal timing control is fully established across datasets

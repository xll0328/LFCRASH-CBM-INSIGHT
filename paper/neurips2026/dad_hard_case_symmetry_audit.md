# DAD Hard-Case Symmetry Audit (Scaffold)

Generated: 2026-05-06 05:51 UTC
Input: `paper/neurips2026/v4_case_bank_scan.json`

## Scope
- This file is a scaffold from archived casebank artifacts only.
- It does not claim hard-case symmetry is solved.
- Primary-family assignment is a keyword heuristic for audit organization, not a causal label.
- Auto-suggested pairs require manual reviewer confirmation.

## Raw Availability Snapshot
- `strong_top`: 30
- `failed_top`: 0
- `late_top`: 0
- auto-suggested mixed pairs: 4
- Interpretation: when `failed_top`/`late_top` are sparse or empty,
  symmetry claims must remain bounded.

## Primary Family × Timing Bucket Counts (from `strong_top`)

| Primary family | early_strong | early_marginal | late_or_missed | auto_pair_count |
|---|---:|---:|---:|---:|
| heavy_vehicle_conflict | 0 | 0 | 1 | 0 |
| intersection_merge | 1 | 3 | 10 | 4 |
| pedestrian_conflict | 0 | 0 | 7 | 0 |
| road_surface_weather | 0 | 0 | 2 | 0 |
| visibility_night | 0 | 0 | 6 | 0 |

## Auto-Suggested Mixed Pairs (Manual Confirmation Required)

| pair_id | family | early_idx | early_tta(s) | late_idx | late_tta(s) | tta_gap(s) |
|---|---|---:|---:|---:|---:|---:|
| intersection_merge_p01 | intersection_merge | 40 | 2.75 | 1 | -1.00 | 3.75 |
| intersection_merge_p02 | intersection_merge | 112 | 0.90 | 33 | -1.00 | 1.90 |
| intersection_merge_p03 | intersection_merge | 0 | 0.85 | 34 | -1.00 | 1.85 |
| intersection_merge_p04 | intersection_merge | 111 | 0.40 | 71 | -1.00 | 1.40 |

## Case-Level Manual Audit Table

| idx | pred_tta(s) | alert_tta(s) | pred_max | bucket | primary_family | all_families | top cue | symmetry_pair_id | paired_outcome | reviewer_note |
|---:|---:|---:|---:|---|---|---|---|---|---|---|
| 40 | 2.75 | 4.45 | 0.611 | early_strong | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk | intersection_merge_p01 | auto_suggested_mixed_pair |  |
| 112 | 0.90 | 4.45 | 0.590 | early_marginal | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk | intersection_merge_p02 | auto_suggested_mixed_pair |  |
| 0 | 0.85 | 4.45 | 0.613 | early_marginal | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. | intersection_merge_p03 | auto_suggested_mixed_pair |  |
| 111 | 0.40 | 4.45 | 0.629 | early_marginal | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk | intersection_merge_p04 | auto_suggested_mixed_pair |  |
| 160 | -0.15 | 4.45 | 0.635 | late_or_missed | road_surface_weather | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  | unpaired_needs_manual_match |  |
| 42 | -0.20 | 4.45 | 0.521 | late_or_missed | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  | unpaired_needs_manual_match |  |
| 104 | -0.20 | 4.45 | 0.517 | late_or_missed | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  | unpaired_needs_manual_match |  |
| 1 | -1.00 | 4.45 | 0.409 | late_or_missed | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk | intersection_merge_p01 | auto_suggested_mixed_pair |  |
| 10 | -1.00 | 4.45 | 0.388 | late_or_missed | heavy_vehicle_conflict | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, road_surface_weather, intersection_merge | large truck merging into traffic. |  | unpaired_needs_manual_match |  |
| 12 | -1.00 | 4.45 | 0.294 | late_or_missed | pedestrian_conflict | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  | unpaired_needs_manual_match |  |
| 33 | -1.00 | 4.45 | 0.279 | late_or_missed | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk | intersection_merge_p02 | auto_suggested_mixed_pair |  |
| 34 | -1.00 | 4.45 | 0.329 | late_or_missed | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. | intersection_merge_p03 | auto_suggested_mixed_pair |  |
| 64 | -1.00 | 4.45 | 0.350 | late_or_missed | visibility_night | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  | unpaired_needs_manual_match |  |
| 71 | -1.00 | 4.45 | 0.452 | late_or_missed | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk | intersection_merge_p04 | auto_suggested_mixed_pair |  |
| 73 | -1.00 | 4.45 | 0.355 | late_or_missed | visibility_night | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  | unpaired_needs_manual_match |  |
| 81 | -1.00 | 4.45 | 0.276 | late_or_missed | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  | unpaired_needs_manual_match |  |
| 82 | -1.00 | 4.45 | 0.309 | late_or_missed | visibility_night | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  | unpaired_needs_manual_match |  |
| 90 | -1.00 | 4.45 | 0.376 | late_or_missed | pedestrian_conflict | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, road_surface_weather, intersection_merge | large truck merging into traffic. |  | unpaired_needs_manual_match |  |
| 91 | -1.00 | 4.45 | 0.227 | late_or_missed | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  | unpaired_needs_manual_match |  |
| 102 | -1.00 | 4.45 | 0.186 | late_or_missed | pedestrian_conflict | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  | unpaired_needs_manual_match |  |
| 122 | -1.00 | 4.45 | 0.364 | late_or_missed | visibility_night | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  | unpaired_needs_manual_match |  |
| 142 | -1.00 | 4.45 | 0.302 | late_or_missed | road_surface_weather | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  | unpaired_needs_manual_match |  |
| 143 | -1.00 | 4.45 | 0.392 | late_or_missed | pedestrian_conflict | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  | unpaired_needs_manual_match |  |
| 150 | -1.00 | 4.45 | 0.199 | late_or_missed | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  | unpaired_needs_manual_match |  |
| 151 | -1.00 | 4.45 | 0.273 | late_or_missed | pedestrian_conflict | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  | unpaired_needs_manual_match |  |
| 152 | -1.00 | 4.45 | 0.243 | late_or_missed | pedestrian_conflict | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  | unpaired_needs_manual_match |  |
| 161 | -1.00 | 4.45 | 0.318 | late_or_missed | intersection_merge | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  | unpaired_needs_manual_match |  |
| 170 | -1.00 | 4.45 | 0.250 | late_or_missed | pedestrian_conflict | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  | unpaired_needs_manual_match |  |
| 171 | -1.00 | 4.45 | 0.394 | late_or_missed | visibility_night | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  | unpaired_needs_manual_match |  |
| 174 | -1.00 | 4.45 | 0.247 | late_or_missed | visibility_night | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  | unpaired_needs_manual_match |  |

## Completion Rule
- Do not promote hard-case symmetry claims until:
  1. family-level pair IDs are reviewed/edited by a human,
  2. paired outcomes include both success/failure evidence,
  3. the resulting summary is synchronized into the claim/evidence ledger.

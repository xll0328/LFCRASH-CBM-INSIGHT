# DAD Hard-Case Symmetry Audit (Scaffold)

Generated: 2026-05-06 05:44 UTC
Input: `paper/neurips2026/v4_case_bank_scan.json`

## Scope
- This file is a scaffold from archived casebank artifacts only.
- It does not claim hard-case symmetry is solved.
- Use it to complete paired family-level success/failure auditing.

## Raw Availability Snapshot
- `strong_top`: 30
- `failed_top`: 0
- `late_top`: 0
- Interpretation: when `failed_top`/`late_top` are sparse or empty,
  symmetry claims must remain bounded.

## Family × Timing Bucket Counts (from `strong_top`)

| Family | early_strong | early_marginal | late_or_missed |
|---|---:|---:|---:|
| heavy_vehicle_conflict | 1 | 3 | 26 |
| intersection_merge | 1 | 3 | 26 |
| motorcyclist_conflict | 1 | 3 | 26 |
| pedestrian_conflict | 1 | 3 | 26 |
| road_surface_weather | 1 | 2 | 17 |
| visibility_night | 0 | 1 | 12 |

## Case-Level Manual Audit Table

| idx | pred_tta(s) | pred_max | bucket | inferred_families | top cue | symmetry_pair_id | paired_outcome | reviewer_note |
|---:|---:|---:|---|---|---|---|---|---|
| 40 | 2.75 | 0.611 | early_strong | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 112 | 0.90 | 0.590 | early_marginal | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 0 | 0.85 | 0.613 | early_marginal | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  |  |  |
| 111 | 0.40 | 0.629 | early_marginal | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 160 | -0.15 | 0.635 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 42 | -0.20 | 0.521 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 104 | -0.20 | 0.517 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 1 | -1.00 | 0.409 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 10 | -1.00 | 0.388 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, road_surface_weather, intersection_merge | large truck merging into traffic. |  |  |  |
| 12 | -1.00 | 0.294 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 33 | -1.00 | 0.279 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 34 | -1.00 | 0.329 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  |  |  |
| 64 | -1.00 | 0.350 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  |  |  |
| 71 | -1.00 | 0.452 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 73 | -1.00 | 0.355 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  |  |  |
| 81 | -1.00 | 0.276 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 82 | -1.00 | 0.309 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  |  |  |
| 90 | -1.00 | 0.376 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, road_surface_weather, intersection_merge | large truck merging into traffic. |  |  |  |
| 91 | -1.00 | 0.227 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 102 | -1.00 | 0.186 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  |  |  |
| 122 | -1.00 | 0.364 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  |  |  |
| 142 | -1.00 | 0.302 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 143 | -1.00 | 0.392 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 150 | -1.00 | 0.199 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 151 | -1.00 | 0.273 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 152 | -1.00 | 0.243 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 161 | -1.00 | 0.318 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, road_surface_weather, intersection_merge | parked vehicle partially obstructing the sidewalk |  |  |  |
| 170 | -1.00 | 0.250 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  |  |  |
| 171 | -1.00 | 0.394 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  |  |  |
| 174 | -1.00 | 0.247 | late_or_missed | motorcyclist_conflict, pedestrian_conflict, heavy_vehicle_conflict, visibility_night, intersection_merge | large truck merging into traffic. |  |  |  |

## Completion Rule
- Do not promote hard-case symmetry claims until:
  1. family-level pair IDs are filled,
  2. paired outcomes include both success/failure evidence,
  3. the resulting summary is synchronized into the claim/evidence ledger.

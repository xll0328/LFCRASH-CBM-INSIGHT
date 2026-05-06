#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

FAMILY_RULES = [
    ("vulnerable_road_users", ["pedestrian", "cyclist", "bicycle", "motorcycle", "motorcyclist", "scooter", "two-wheeler", "rider", "crosswalk"]),
    ("surface_weather", ["wet", "slippery", "rain", "fog", "glare", "hydroplan", "traction", "surface", "nighttime", "low-light", "precipitation", "pavement"]),
    ("occlusion_visibility", ["visibility", "occlusion", "occluded", "blind", "sightline", "sight distance", "glare", "illumination", "lighting"]),
    ("right_of_way_conflict", ["right-of-way", "intersection", "cross-traffic", "oncoming", "yield", "signal", "stop-line", "conflict zone"]),
    ("relative_motion", ["proximity", "deceleration", "braking", "following distance", "rear-end", "closing speed"]),
    ("agent_behavior", ["lane change", "encroachment", "merge", "turn", "cut-in", "brake light", "crossing intent", "entry conflict", "filtering"]),
    ("road_layout_constraint", ["lane boundary", "lane marking", "clearance", "road curvature", "shoulder", "narrow", "junction", "roadside", "carriageway", "lane width"]),
    ("traffic_density_obstacle", ["traffic density", "congestion", "parked vehicle", "large vehicle", "obstacle", "mixed traffic", "heavy vehicle"]),
    ("imminent_crash_cue", ["collision risk", "crash", "emergency", "sudden deceleration", "conflict point"]),
]


def assign_family(concept: str) -> str:
    c = concept.lower()
    for fam, kws in FAMILY_RULES:
        if any(k in c for k in kws):
            return fam
    return "agent_behavior"


def stable_key(seed: int, concept: str) -> str:
    return hashlib.sha256(f"{seed}:{concept}".encode("utf-8")).hexdigest()


def allocate_quotas(size: int, buckets: dict[str, list[str]]) -> dict[str, int]:
    non_empty = {k: v for k, v in buckets.items() if v}
    total = sum(len(v) for v in non_empty.values())
    if total == 0:
        return {k: 0 for k in buckets}

    # Start with proportional floor allocation.
    quotas = {k: int(size * len(v) / total) for k, v in non_empty.items()}

    # Ensure minimal family coverage when budget permits.
    fam_count = len(non_empty)
    if size >= fam_count:
        for k in non_empty:
            quotas[k] = max(1, quotas[k])

    # Cap by available bucket sizes.
    for k, v in non_empty.items():
        quotas[k] = min(quotas[k], len(v))

    current = sum(quotas.values())

    # Fill remaining budget by largest residual capacity.
    while current < size:
        candidates = sorted(
            [(len(non_empty[k]) - quotas[k], k) for k in non_empty if quotas[k] < len(non_empty[k])],
            reverse=True,
        )
        if not candidates:
            break
        _, k = candidates[0]
        quotas[k] += 1
        current += 1

    # Trim if over-allocated.
    while current > size:
        candidates = sorted([(quotas[k], k) for k in non_empty if quotas[k] > 0], reverse=True)
        if not candidates:
            break
        _, k = candidates[0]
        quotas[k] -= 1
        current -= 1

    out = {k: 0 for k in buckets}
    out.update(quotas)
    return out


def build_subset(concepts: list[str], size: int, seed: int) -> tuple[list[str], dict]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for c in concepts:
        buckets[assign_family(c)].append(c)

    # Deterministic ordering within each family bucket.
    for fam in list(buckets.keys()):
        buckets[fam] = sorted(buckets[fam], key=lambda x: stable_key(seed, x))

    quotas = allocate_quotas(size=size, buckets=buckets)

    selected: list[str] = []
    for fam in sorted(buckets.keys()):
        selected.extend(buckets[fam][: quotas.get(fam, 0)])

    # Top up globally if any shortfall remains.
    if len(selected) < size:
        selected_set = set(selected)
        leftovers = [c for c in concepts if c not in selected_set]
        leftovers = sorted(leftovers, key=lambda x: stable_key(seed + 7919, x))
        selected.extend(leftovers[: size - len(selected)])

    selected = selected[:size]

    family_counts = defaultdict(int)
    for c in selected:
        family_counts[assign_family(c)] += 1

    meta = {
        "size": size,
        "seed": seed,
        "total_source_concepts": len(concepts),
        "selection": "family-stratified deterministic sampling",
        "family_counts": dict(sorted(family_counts.items())),
        "quotas": {k: int(v) for k, v in sorted(quotas.items()) if v > 0},
    }
    return selected, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Build deterministic size-matched subsets from historical ontology")
    ap.add_argument("--input", default="/data/sony/LFCRASH/000_all_concept_set.txt")
    ap.add_argument("--sizes", default="30,80")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output_dir", default="/data/sony/LFCRASH/LFCRASH-CBM/output/concept_sets")
    args = ap.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    if not sizes:
        raise ValueError("No sizes parsed from --sizes")

    src = Path(args.input)
    lines = [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # De-duplicate while preserving first appearance.
    dedup: list[str] = []
    seen = set()
    for c in lines:
        if c not in seen:
            seen.add(c)
            dedup.append(c)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_file": str(src),
        "source_unique_concepts": len(dedup),
        "seed": args.seed,
        "outputs": [],
    }

    for size in sizes:
        selected, meta = build_subset(dedup, size=size, seed=args.seed)
        txt_path = out_dir / f"historical_full_stratified_{size}.txt"
        meta_path = out_dir / f"historical_full_stratified_{size}.meta.json"

        txt_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        summary["outputs"].append(
            {
                "size": size,
                "txt": str(txt_path),
                "meta": str(meta_path),
                "selected": len(selected),
            }
        )

    summary_path = out_dir / "historical_full_size_matched_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

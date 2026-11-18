
from pathlib import Path
from collections import defaultdict
from bisect import bisect_left
import csv
import json

from src.geospatial.candidate_generation import (
    load_track,
    load_nodes,
    load_arcs,
    generate_candidates,
    build_spatial_index,
    generate_candidates_for_observation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "student_folder" / "train"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SEARCH_RADIUS_M = 50.0
MAX_CANDIDATES = 8

FEATURE_COLUMNS = [
    "distance_m",
    "projection_fraction",
    "segment_length_m",
    "candidate_rank",
    "distance_to_best_m",
    "distance_margin_m",
]


def load_route(route_path):
    """Read chronological, zero-based reference arc IDs."""
    values = []

    with route_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            for token in line.split():
                value = float(token)
                if not value.is_integer():
                    raise ValueError(
                        f"Non-integer route arc ID at line {line_number}"
                    )
                values.append(int(value))

    if not values:
        raise ValueError(f"Empty route file: {route_path}")

    return values


def align_observations_to_route(
    observations,
    nodes,
    arcs,
    route_ids,
    search_radius_m=SEARCH_RADIUS_M,
):
    """
    Find the nearest reference-route arc for each GPS observation,
    then enforce non-decreasing route position.

    This is a geometric, monotonic alignment heuristic. It is not
    a full HMM/Viterbi alignment and must be validated.
    """
    arc_by_id = {arc["arc_id"]: arc for arc in arcs}

    invalid_ids = sorted(set(route_ids) - set(arc_by_id))
    if invalid_ids:
        raise ValueError(
            f"Route references arc IDs absent from .arcs: {invalid_ids[:10]}"
        )

    route_arcs = list({
        arc_id: arc_by_id[arc_id] for arc_id in route_ids
    }.values())

    reference_latitude = sum(
        node["latitude"] for node in nodes
    ) / len(nodes)

    grid, segments = build_spatial_index(
        nodes,
        route_arcs,
        reference_latitude,
        search_radius_m,
    )

    positions_by_arc = defaultdict(list)
    for position, arc_id in enumerate(route_ids):
        positions_by_arc[arc_id].append(position)

    aligned_arc_ids = []
    aligned_positions = []
    last_position = 0

    for observation in observations:
        nearest = generate_candidates_for_observation(
            observation,
            grid,
            segments,
            reference_latitude,
            search_radius_m,
            1,
        )

        if not nearest:
            aligned_arc_ids.append(None)
            aligned_positions.append(None)
            continue

        nearest_arc_id = nearest[0]["arc_id"]
        possible_positions = positions_by_arc[nearest_arc_id]

        index = bisect_left(possible_positions, last_position)

        if index >= len(possible_positions):
            aligned_arc_ids.append(None)
            aligned_positions.append(None)
            continue

        route_position = possible_positions[index]
        last_position = route_position

        aligned_arc_ids.append(nearest_arc_id)
        aligned_positions.append(route_position)

    return aligned_arc_ids, aligned_positions


def process_trip(trip_dir):
    track_files = list(trip_dir.glob("*.track"))
    node_files = list(trip_dir.glob("*.nodes"))
    arc_files = list(trip_dir.glob("*.arcs"))
    route_files = list(trip_dir.glob("*.route"))

    if not all(len(files) == 1 for files in [
        track_files, node_files, arc_files, route_files
    ]):
        raise ValueError(
            f"{trip_dir}: expected exactly one .track, .nodes, "
            ".arcs and .route file"
        )

    observations = load_track(str(track_files[0]))
    nodes = load_nodes(str(node_files[0]))
    arcs = load_arcs(str(arc_files[0]))
    route_ids = load_route(route_files[0])

    valid_arc_ids = {arc["arc_id"] for arc in arcs}
    if not set(route_ids).issubset(valid_arc_ids):
        raise ValueError(f"{trip_dir.name}: route contains invalid arc IDs")

    aligned_ids, aligned_positions = align_observations_to_route(
        observations, nodes, arcs, route_ids
    )

    candidates_by_observation = generate_candidates(
        observations,
        nodes,
        arcs,
        search_radius_m=SEARCH_RADIUS_M,
        max_candidates=MAX_CANDIDATES,
    )

    rows = []
    stats = {
        "trip_id": trip_dir.name,
        "gps_observations": len(observations),
        "route_entries": len(route_ids),
        "route_aligned_observations": 0,
        "candidate_misses": 0,
        "training_observations": 0,
        "training_candidates": 0,
    }

    for observation_id, (observation, candidate_group) in enumerate(
        zip(observations, candidates_by_observation)
    ):
        reference_arc_id = aligned_ids[observation_id]

        if reference_arc_id is None:
            continue

        stats["route_aligned_observations"] += 1

        candidates = candidate_group["candidates"]
        candidate_arc_ids = {c["arc_id"] for c in candidates}

        if reference_arc_id not in candidate_arc_ids:
            stats["candidate_misses"] += 1
            continue

        best_distance = min(
            c["features"]["distance_m"] for c in candidates
        )

        sorted_candidates = sorted(
            candidates,
            key=lambda c: c["features"]["distance_m"]
        )

        for rank, candidate in enumerate(sorted_candidates):
            features = candidate["features"]
            distance = features["distance_m"]

            rows.append({
                "trip_id": trip_dir.name,
                "observation_id": observation_id,
                "timestamp": observation["timestamp"],
                "longitude": observation["longitude"],
                "latitude": observation["latitude"],
                "arc_id": candidate["arc_id"],
                "reference_arc_id": reference_arc_id,
                "route_position": aligned_positions[observation_id],
                "label": int(candidate["arc_id"] == reference_arc_id),
                "distance_m": distance,
                "projection_fraction": features["projection_fraction"],
                "segment_length_m": features["segment_length_m"],
                "candidate_rank": rank,
                "distance_to_best_m": best_distance,
                "distance_margin_m": distance - best_distance,
            })

        stats["training_observations"] += 1
        stats["training_candidates"] += len(sorted_candidates)

    return rows, stats


def main():
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_stats = []

    trip_dirs = sorted(p for p in TRAIN_DIR.iterdir() if p.is_dir())

    if not trip_dirs:
        raise RuntimeError(f"No trip directories found in {TRAIN_DIR}")

    for trip_dir in trip_dirs:
        print(f"Processing {trip_dir.name} ...")
        rows, stats = process_trip(trip_dir)
        all_rows.extend(rows)
        all_stats.append(stats)

        print(
            f"  aligned={stats['route_aligned_observations']}, "
            f"misses={stats['candidate_misses']}, "
            f"usable_obs={stats['training_observations']}, "
            f"candidate_rows={stats['training_candidates']}"
        )

    dataset_path = OUTPUT_DIR / "bee_training_candidates.csv"
    diagnostics_path = OUTPUT_DIR / "bee_label_diagnostics.json"

    if not all_rows:
        raise RuntimeError(
            "No labelled candidate rows generated. "
            "Do not train models until the alignment issue is investigated."
        )

    with dataset_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    total_aligned = sum(s["route_aligned_observations"] for s in all_stats)
    total_misses = sum(s["candidate_misses"] for s in all_stats)
    total_usable = sum(s["training_observations"] for s in all_stats)

    print("\nDATASET BUILD COMPLETE")
    print(f"Trips processed: {len(all_stats)}")
    print(f"Candidate rows: {len(all_rows)}")
    print(f"Route-aligned observations: {total_aligned}")
    print(f"Candidate misses: {total_misses}")
    print(f"Usable training observations: {total_usable}")
    print(f"CSV: {dataset_path}")
    print(f"Diagnostics: {diagnostics_path}")
    print(
        "\nIMPORTANT: Labels use monotonic nearest-route geometric alignment. "
        "Review diagnostics before interpreting model metrics."
    )


if __name__ == "__main__":
    main()

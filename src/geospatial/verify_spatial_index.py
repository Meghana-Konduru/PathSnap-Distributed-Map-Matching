
from pathlib import Path

from src.geospatial.candidate_generation import (
    load_track,
    load_nodes,
    load_arcs,
    project_coordinates,
    point_to_segment,
    build_spatial_index,
    generate_candidates_for_observation,
)


# ============================================================
# 1. LOAD TRIP 00000054
# ============================================================

TRIP_ID = "00000054"

TRIP_DIR = Path("data/raw/student_folder/train") / TRIP_ID

track = load_track(str(TRIP_DIR / f"{TRIP_ID}.track"))
nodes = load_nodes(str(TRIP_DIR / f"{TRIP_ID}.nodes"))
arcs = load_arcs(str(TRIP_DIR / f"{TRIP_ID}.arcs"))

SEARCH_RADIUS_M = 50.0
MAX_CANDIDATES = 8

print("=" * 65)
print("SPATIAL INDEX VERIFICATION")
print("=" * 65)

print("GPS observations:", len(track))
print("Road nodes:", len(nodes))
print("Road arcs:", len(arcs))


# ============================================================
# 2. BUILD THE SAME SPATIAL INDEX AS THE MAIN IMPLEMENTATION
# ============================================================

reference_latitude = sum(
    node["latitude"] for node in nodes
) / len(nodes)

grid, segments = build_spatial_index(
    nodes,
    arcs,
    reference_latitude,
    SEARCH_RADIUS_M,
)

print("\nSpatial index built successfully.")
print("Indexed road segments:", len(segments))
print("Occupied grid cells:", len(grid))


# ============================================================
# 3. SELECT REPRESENTATIVE GPS OBSERVATIONS
# ============================================================

# These include observations from the large missing sequences
# and observations from other parts of the trip.

sample_indices = [
    0,
    2710,
    3720,
    8113,
    10000,
    14148,
    16888,
    19000,
]

sample_indices = [
    index for index in sample_indices
    if index < len(track)
]


# ============================================================
# 4. COMPARE INDEX SEARCH WITH BRUTE-FORCE SEARCH
# ============================================================

for observation_index in sample_indices:

    observation = track[observation_index]

    px, py = project_coordinates(
        observation["longitude"],
        observation["latitude"],
        reference_latitude,
    )

    # --------------------------------------------------------
    # A. Current spatial-index search
    # --------------------------------------------------------

    indexed_candidates = generate_candidates_for_observation(
        observation,
        grid,
        segments,
        reference_latitude,
        SEARCH_RADIUS_M,
        MAX_CANDIDATES,
    )

    indexed_arc_ids = [
        candidate["arc_id"]
        for candidate in indexed_candidates
    ]

    # --------------------------------------------------------
    # B. Independent brute-force search
    # --------------------------------------------------------

    brute_force_candidates = []

    for arc_id, (ax, ay, bx, by) in segments.items():

        distance, fraction, projected_x, projected_y = (
            point_to_segment(px, py, ax, ay, bx, by)
        )

        if distance <= SEARCH_RADIUS_M:

            brute_force_candidates.append({
                "arc_id": arc_id,
                "distance_m": distance,
            })

    brute_force_candidates.sort(
        key=lambda candidate: candidate["distance_m"]
    )

    brute_force_arc_ids = [
        candidate["arc_id"]
        for candidate in brute_force_candidates[:MAX_CANDIDATES]
    ]

    # --------------------------------------------------------
    # C. Compare results
    # --------------------------------------------------------

    missing_from_index = (
        set(brute_force_arc_ids) - set(indexed_arc_ids)
    )

    extra_in_index = (
        set(indexed_arc_ids) - set(brute_force_arc_ids)
    )

    print("\n" + "-" * 65)
    print("GPS observation index:", observation_index)
    print("Timestamp:", observation["timestamp"])

    print(
        "Brute-force candidates within 50m:",
        len(brute_force_candidates),
    )

    print(
        "Spatial-index candidates returned:",
        len(indexed_candidates),
    )

    if brute_force_candidates:
        print(
            "Nearest brute-force distance:",
            round(brute_force_candidates[0]["distance_m"], 3),
            "metres",
        )
    else:
        print("Nearest brute-force distance: greater than 50 metres")

    print("Brute-force top arc IDs:", brute_force_arc_ids)
    print("Spatial-index arc IDs:", indexed_arc_ids)

    print("Missing from index:", sorted(missing_from_index))
    print("Extra in index:", sorted(extra_in_index))

    if missing_from_index or extra_in_index:
        print("RESULT: MISMATCH")
    else:
        print("RESULT: MATCH")


print("\n" + "=" * 65)
print("SPATIAL INDEX VERIFICATION COMPLETED")
print("=" * 65)

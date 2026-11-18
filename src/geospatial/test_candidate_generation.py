
"""
PathSnap - BIS
Test script for GPS loading and candidate generation.
"""

from pathlib import Path

from src.geospatial.candidate_generation import (
    load_track,
    load_nodes,
    load_arcs,
    generate_candidates
)


# ---------------------------------------------------------
# DATASET PATH
# ---------------------------------------------------------

TRIP_DIR = Path(
    "data/raw/student_folder/train/00000001"
)

track_path = TRIP_DIR / "00000001.track"
nodes_path = TRIP_DIR / "00000001.nodes"
arcs_path = TRIP_DIR / "00000001.arcs"


# ---------------------------------------------------------
# TEST 1: LOAD DATA
# ---------------------------------------------------------

print("=" * 60)
print("PATHSNAP - BIS CANDIDATE GENERATION TEST")
print("=" * 60)

observations = load_track(str(track_path))
nodes = load_nodes(str(nodes_path))
arcs = load_arcs(str(arcs_path))

print("\nDATA LOADING")
print("-" * 40)

print("GPS observations:", len(observations))
print("Road nodes:", len(nodes))
print("Road arcs:", len(arcs))


# ---------------------------------------------------------
# TEST 2: INSPECT FIRST RECORDS
# ---------------------------------------------------------

print("\nFIRST GPS OBSERVATION")
print("-" * 40)

print(observations[0])

print("\nFIRST ROAD NODE")
print("-" * 40)

print(nodes[0])

print("\nFIRST ROAD ARC")
print("-" * 40)

print(arcs[0])


# ---------------------------------------------------------
# TEST 3: VERIFY CHRONOLOGICAL ORDER
# ---------------------------------------------------------

timestamps = [
    observation["timestamp"]
    for observation in observations
]

assert timestamps == sorted(timestamps), (
    "GPS timestamps are not chronological"
)

print("\nTimestamp ordering: PASSED")


# ---------------------------------------------------------
# TEST 4: VERIFY ARC IDS
# ---------------------------------------------------------

for expected_arc_id, arc in enumerate(arcs):

    assert arc["arc_id"] == expected_arc_id, (
        "Original arc IDs were not preserved"
    )

print("Original zero-based arc IDs: PASSED")


# ---------------------------------------------------------
# TEST 5: GENERATE CANDIDATES
# ---------------------------------------------------------

print("\nGENERATING ROAD CANDIDATES")
print("-" * 40)

results = generate_candidates(
    observations=observations,
    nodes=nodes,
    arcs=arcs,
    search_radius_m=50.0,
    max_candidates=8
)

print("Generated observations:", len(results))


# ---------------------------------------------------------
# TEST 6: VERIFY OUTPUT STRUCTURE
# ---------------------------------------------------------

assert len(results) == len(observations), (
    "Output observation count does not match input"
)

for observation in results:

    assert "timestamp" in observation
    assert "longitude" in observation
    assert "latitude" in observation
    assert "candidates" in observation

    for candidate in observation["candidates"]:

        assert "arc_id" in candidate
        assert "features" in candidate

        assert 0 <= candidate["arc_id"] < len(arcs)

        assert "distance_m" in candidate["features"]
        assert "projection_fraction" in candidate["features"]
        assert "segment_length_m" in candidate["features"]


# ---------------------------------------------------------
# TEST 7: PRINT SAMPLE RESULT
# ---------------------------------------------------------

print("\nFIRST OBSERVATION WITH CANDIDATES")
print("-" * 40)

print(results[0])

print(
    "\nCandidates for first observation:",
    len(results[0]["candidates"])
)


# ---------------------------------------------------------
# TEST 8: VERIFY OUTPUT CHRONOLOGY
# ---------------------------------------------------------

result_timestamps = [
    item["timestamp"] for item in results
]

assert result_timestamps == sorted(result_timestamps)

print("\nOutput chronological order: PASSED")


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ALL BASIC TESTS PASSED")
print("=" * 60)

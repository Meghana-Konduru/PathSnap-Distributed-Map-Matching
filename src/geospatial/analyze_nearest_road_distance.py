
from pathlib import Path

from src.geospatial.candidate_generation import (
    load_track,
    load_nodes,
    load_arcs,
    project_coordinates,
    point_to_segment,
)

TRIP_ID = "00000054"
TRIP_DIR = Path("data/raw/student_folder/train") / TRIP_ID

track = load_track(str(TRIP_DIR / f"{TRIP_ID}.track"))
nodes = load_nodes(str(TRIP_DIR / f"{TRIP_ID}.nodes"))
arcs = load_arcs(str(TRIP_DIR / f"{TRIP_ID}.arcs"))

# Representative observations from missing-candidate sequences
SAMPLE_INDICES = [2710, 3720, 8113, 10000, 14148, 16888, 19000]

reference_latitude = sum(
    node["latitude"] for node in nodes
) / len(nodes)

# Project all road nodes once
projected_nodes = {
    node["node_id"]: project_coordinates(
        node["longitude"],
        node["latitude"],
        reference_latitude,
    )
    for node in nodes
}

# Build road segments
segments = []

for arc in arcs:
    ax, ay = projected_nodes[arc["from_node_id"]]
    bx, by = projected_nodes[arc["to_node_id"]]

    segments.append((arc["arc_id"], ax, ay, bx, by))

print("=" * 70)
print("NEAREST ROAD DISTANCE ANALYSIS - TRIP 00000054")
print("=" * 70)

for index in SAMPLE_INDICES:

    observation = track[index]

    px, py = project_coordinates(
        observation["longitude"],
        observation["latitude"],
        reference_latitude,
    )

    nearest_distance = float("inf")
    nearest_arc_id = None
    counts = {50: 0, 100: 0, 200: 0, 500: 0}

    for arc_id, ax, ay, bx, by in segments:

        distance, _, _, _ = point_to_segment(
            px, py, ax, ay, bx, by
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_arc_id = arc_id

        for radius in counts:
            if distance <= radius:
                counts[radius] += 1

    print("\n" + "-" * 60)
    print("Observation index:", index)
    print("Timestamp:", observation["timestamp"])
    print("GPS:", observation["longitude"], observation["latitude"])
    print("Nearest road arc ID:", nearest_arc_id)
    print("Nearest road distance:", round(nearest_distance, 2), "m")

    for radius, count in counts.items():
        print(f"Road segments within {radius}m:", count)

print("\nANALYSIS COMPLETED")

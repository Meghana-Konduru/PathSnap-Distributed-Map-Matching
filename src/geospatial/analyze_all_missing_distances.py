
from pathlib import Path
import csv
from collections import Counter

from src.geospatial.candidate_generation import (
    load_track,
    load_nodes,
    load_arcs,
    generate_candidates,
    project_coordinates,
    point_to_segment,
)

TRIP_ID = "00000054"
RADIUS = 100.0

TRIP_DIR = Path("data/raw/student_folder/train") / TRIP_ID
GAP_FILE = Path("reports/bis/candidate_gap_analysis.csv")
OUTPUT_FILE = Path("reports/bis/all_missing_distances_00000054.csv")

track = load_track(str(TRIP_DIR / f"{TRIP_ID}.track"))
nodes = load_nodes(str(TRIP_DIR / f"{TRIP_ID}.nodes"))
arcs = load_arcs(str(TRIP_DIR / f"{TRIP_ID}.arcs"))

# Load missing observation indices from the validated gap report.
with GAP_FILE.open("r", encoding="utf-8", newline="") as file:
    missing_indices = [
        int(row["observation_index"])
        for row in csv.DictReader(file)
        if row["trip_id"] == TRIP_ID
    ]

print("Missing observations:", len(missing_indices))

reference_latitude = sum(
    node["latitude"] for node in nodes
) / len(nodes)

projected_nodes = {
    node["node_id"]: project_coordinates(
        node["longitude"],
        node["latitude"],
        reference_latitude,
    )
    for node in nodes
}

# Build road segments once.
segments = []

for arc in arcs:
    ax, ay = projected_nodes[arc["from_node_id"]]
    bx, by = projected_nodes[arc["to_node_id"]]

    segments.append((arc["arc_id"], ax, ay, bx, by))

rows = []

for count, index in enumerate(missing_indices, start=1):
    observation = track[index]

    px, py = project_coordinates(
        observation["longitude"],
        observation["latitude"],
        reference_latitude,
    )

    nearest_distance = float("inf")
    nearest_arc_id = None
    within_counts = {100: 0, 150: 0, 200: 0, 300: 0, 500: 0}

    for arc_id, ax, ay, bx, by in segments:
        distance, _, _, _ = point_to_segment(
            px, py, ax, ay, bx, by
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_arc_id = arc_id

        for radius in within_counts:
            if distance <= radius:
                within_counts[radius] += 1

    rows.append({
        "trip_id": TRIP_ID,
        "observation_index": index,
        "timestamp": observation["timestamp"],
        "longitude": observation["longitude"],
        "latitude": observation["latitude"],
        "nearest_arc_id": nearest_arc_id,
        "nearest_road_distance_m": round(nearest_distance, 2),
        "segments_within_100m": within_counts[100],
        "segments_within_150m": within_counts[150],
        "segments_within_200m": within_counts[200],
        "segments_within_300m": within_counts[300],
        "segments_within_500m": within_counts[500],
    })

    if count % 100 == 0:
        print(f"Processed {count}/{len(missing_indices)}")


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

distances = [row["nearest_road_distance_m"] for row in rows]

print("\nDISTANCE SUMMARY")
print("Total analysed:", len(rows))
print("Minimum distance:", min(distances))
print("Maximum distance:", max(distances))
print("Average distance:", round(sum(distances) / len(distances), 2))

for threshold in [100, 150, 200, 300, 500]:
    count = sum(distance <= threshold for distance in distances)
    print(f"Within {threshold}m: {count}/{len(rows)}")

print("\nReport saved:", OUTPUT_FILE)

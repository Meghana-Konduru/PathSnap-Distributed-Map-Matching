
from pathlib import Path
import csv

from src.geospatial.candidate_generation import (
    load_track,
    load_nodes,
    load_arcs,
    generate_candidates
)


TRAIN_DIR = Path("data/raw/student_folder/train")

OUTPUT_DIR = Path("reports/bis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "candidate_gap_analysis.csv"

rows = []

for trip_dir in sorted(TRAIN_DIR.iterdir()):

    if not trip_dir.is_dir():
        continue

    trip_id = trip_dir.name

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"

    observations = load_track(str(track_file))
    nodes = load_nodes(str(nodes_file))
    arcs = load_arcs(str(arcs_file))

    results = generate_candidates(
        observations,
        nodes,
        arcs,
        search_radius_m=100.0,
        max_candidates=8
    )

    empty_indices = [
        i for i, result in enumerate(results)
        if not result["candidates"]
    ]

    if not empty_indices:
        continue

    for index in empty_indices:

        observation = results[index]

        previous_time = (
            results[index - 1]["timestamp"]
            if index > 0 else None
        )

        next_time = (
            results[index + 1]["timestamp"]
            if index + 1 < len(results) else None
        )

        rows.append({
            "trip_id": trip_id,
            "observation_index": index,
            "timestamp": observation["timestamp"],
            "longitude": observation["longitude"],
            "latitude": observation["latitude"],
            "previous_timestamp": previous_time,
            "next_timestamp": next_time,
        })


fieldnames = [
    "trip_id",
    "observation_index",
    "timestamp",
    "longitude",
    "latitude",
    "previous_timestamp",
    "next_timestamp",
]

with REPORT_PATH.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


print("Candidate gap analysis completed.")
print("Zero-candidate observations:", len(rows))
print("Report saved:", REPORT_PATH)

print("\nObservations by trip:")

counts = {}

for row in rows:
    trip_id = row["trip_id"]
    counts[trip_id] = counts.get(trip_id, 0) + 1

for trip_id, count in sorted(
    counts.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]:

    print(trip_id, ":", count)

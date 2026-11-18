
from pathlib import Path
from collections import defaultdict, deque
from math import radians, sin, cos, sqrt, atan2
import csv

from src.geospatial.candidate_generation import load_track, load_nodes, load_arcs

TRIP_ID = "00000095"
BASE_DIR = Path("data/raw/student_folder/train") / TRIP_ID
OUTPUT_FILE = Path("reports/bis") / "component_proximity_00000095.csv"
RADIUS_METERS = 100.0


def haversine(lon1, lat1, lon2, lat2):
    earth_radius = 6371000.0

    lon1, lat1, lon2, lat2 = map(
        radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * atan2(sqrt(a), sqrt(1 - a))


def main():
    track = load_track(str(BASE_DIR / f"{TRIP_ID}.track"))
    nodes = load_nodes(str(BASE_DIR / f"{TRIP_ID}.nodes"))
    arcs = load_arcs(str(BASE_DIR / f"{TRIP_ID}.arcs"))

    node_lookup = {node["node_id"]: node for node in nodes}

    adjacency = defaultdict(set)

    for arc in arcs:
        source = arc["from_node_id"]
        destination = arc["to_node_id"]

        adjacency[source].add(destination)
        adjacency[destination].add(source)

    visited = set()
    components = []

    for start in node_lookup:
        if start in visited:
            continue

        queue = deque([start])
        visited.add(start)
        component = []

        while queue:
            current = queue.popleft()
            component.append(current)

            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        components.append(component)

    components.sort(key=len, reverse=True)

    results = []

    print(f"GPS observations: {len(track)}")
    print(f"Connected components: {len(components)}")
    print(f"Distance threshold: {RADIUS_METERS} metres")

    for rank, component in enumerate(components, start=1):
        component_nodes = [
            node_lookup[node_id] for node_id in component
        ]

        distances = []

        for observation in track:
            nearest_distance = min(
                haversine(
                    observation["longitude"],
                    observation["latitude"],
                    node["longitude"],
                    node["latitude"],
                )
                for node in component_nodes
            )
            distances.append(nearest_distance)

        within_radius = sum(
            distance <= RADIUS_METERS
            for distance in distances
        )

        row = {
            "component_rank": rank,
            "component_nodes": len(component),
            "gps_observations_within_100m": within_radius,
            "gps_coverage_percentage": round(
                within_radius / len(track) * 100, 2
            ),
            "minimum_distance_m": round(min(distances), 2),
            "median_nearest_distance_m": round(
                sorted(distances)[len(distances) // 2], 2
            ),
        }

        results.append(row)

        print(
            f"Component {rank}: "
            f"{len(component)} nodes | "
            f"{within_radius}/{len(track)} GPS observations within "
            f"{RADIUS_METERS}m | "
            f"minimum distance: {row['minimum_distance_m']}m"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nReport saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

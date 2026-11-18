
from pathlib import Path
from collections import defaultdict, deque
import csv

from src.geospatial.candidate_generation import load_nodes, load_arcs

TRAIN_DIR = Path("data/raw/student_folder/train")
OUTPUT_DIR = Path("reports/bis")
OUTPUT_FILE = OUTPUT_DIR / "fragmented_network_details.csv"

TRIP_IDS = ["00000095", "00000043", "00000001"]


def inspect_trip(trip_id):

    trip_dir = TRAIN_DIR / trip_id

    nodes = load_nodes(str(trip_dir / f"{trip_id}.nodes"))
    arcs = load_arcs(str(trip_dir / f"{trip_id}.arcs"))

    node_lookup = {node["node_id"]: node for node in nodes}
    adjacency = defaultdict(set)

    invalid_arcs = 0

    for arc in arcs:
        source = arc["from_node_id"]
        destination = arc["to_node_id"]

        if source not in node_lookup or destination not in node_lookup:
            invalid_arcs += 1
            continue

        adjacency[source].add(destination)
        adjacency[destination].add(source)

    for node_id in node_lookup:
        adjacency[node_id]

    visited = set()
    components = []

    for start in node_lookup:

        if start in visited:
            continue

        queue = deque([start])
        visited.add(start)
        component_nodes = []

        while queue:
            current = queue.popleft()
            component_nodes.append(current)

            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        components.append(component_nodes)

    components.sort(key=len, reverse=True)

    print("\n" + "=" * 65)
    print("TRIP:", trip_id)
    print("=" * 65)
    print("Total nodes:", len(nodes))
    print("Total arcs:", len(arcs))
    print("Connected components:", len(components))
    print("Invalid arcs:", invalid_arcs)

    rows = []

    for rank, component in enumerate(components[:10], start=1):

        component_nodes = [
            node_lookup[node_id] for node_id in component
        ]

        longitudes = [node["longitude"] for node in component_nodes]
        latitudes = [node["latitude"] for node in component_nodes]

        row = {
            "trip_id": trip_id,
            "component_rank": rank,
            "component_nodes": len(component),
            "percentage_of_network": round(
                len(component) / len(nodes) * 100, 2
            ),
            "min_longitude": min(longitudes),
            "max_longitude": max(longitudes),
            "min_latitude": min(latitudes),
            "max_latitude": max(latitudes),
        }

        rows.append(row)

        print(
            f"Component {rank}: "
            f"{len(component)} nodes "
            f"({row['percentage_of_network']}%)"
        )

    return rows


def main():

    all_rows = []

    for trip_id in TRIP_IDS:
        all_rows.extend(inspect_trip(trip_id))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "trip_id",
        "component_rank",
        "component_nodes",
        "percentage_of_network",
        "min_longitude",
        "max_longitude",
        "min_latitude",
        "max_latitude",
    ]

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nInvestigation completed.")
    print("Report saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

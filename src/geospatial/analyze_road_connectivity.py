
from pathlib import Path
from collections import defaultdict, deque
import csv

from src.geospatial.candidate_generation import load_nodes, load_arcs


TRAIN_DIR = Path("data/raw/student_folder/train")
OUTPUT_DIR = Path("reports/bis")
OUTPUT_FILE = OUTPUT_DIR / "road_network_connectivity.csv"


def analyze_connectivity(nodes, arcs):

    node_ids = {node["node_id"] for node in nodes}

    adjacency = defaultdict(set)
    degree = Counter() if False else defaultdict(int)

    invalid_arcs = 0
    self_loops = 0

    for arc in arcs:

        source = arc["from_node_id"]
        destination = arc["to_node_id"]

        if source not in node_ids or destination not in node_ids:
            invalid_arcs += 1
            continue

        if source == destination:
            self_loops += 1
            continue

        # Treat arcs as undirected for physical connectivity analysis.
        adjacency[source].add(destination)
        adjacency[destination].add(source)

    # Include every node, even nodes with no connections.
    for node_id in node_ids:
        adjacency[node_id]

    degrees = {
        node_id: len(neighbours)
        for node_id, neighbours in adjacency.items()
    }

    isolated_nodes = sum(
        value == 0 for value in degrees.values()
    )

    degree_one_nodes = sum(
        value == 1 for value in degrees.values()
    )

    visited = set()
    component_sizes = []

    for start_node in node_ids:

        if start_node in visited:
            continue

        queue = deque([start_node])
        visited.add(start_node)
        component_size = 0

        while queue:

            current = queue.popleft()
            component_size += 1

            for neighbour in adjacency[current]:

                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        component_sizes.append(component_size)

    component_sizes.sort(reverse=True)

    largest_component = (
        component_sizes[0] if component_sizes else 0
    )

    largest_component_percentage = (
        largest_component / len(nodes) * 100
        if nodes else 0
    )

    return {
        "nodes": len(nodes),
        "arcs": len(arcs),
        "connected_components": len(component_sizes),
        "largest_component_nodes": largest_component,
        "largest_component_percentage": round(
            largest_component_percentage, 2
        ),
        "isolated_nodes": isolated_nodes,
        "degree_one_nodes": degree_one_nodes,
        "invalid_arcs": invalid_arcs,
        "self_loops": self_loops,
    }


def main():

    trip_directories = sorted(
        path for path in TRAIN_DIR.iterdir()
        if path.is_dir()
    )

    if not trip_directories:
        raise RuntimeError("No training trip directories found.")

    rows = []

    print("=" * 70)
    print("ROAD NETWORK CONNECTIVITY ANALYSIS")
    print("=" * 70)

    for trip_dir in trip_directories:

        trip_id = trip_dir.name

        nodes = load_nodes(
            str(trip_dir / f"{trip_id}.nodes")
        )

        arcs = load_arcs(
            str(trip_dir / f"{trip_id}.arcs")
        )

        result = analyze_connectivity(nodes, arcs)
        result["trip_id"] = trip_id

        rows.append(result)

        print(
            f"{trip_id}: "
            f"components={result['connected_components']}, "
            f"largest_component={result['largest_component_percentage']}%, "
            f"isolated_nodes={result['isolated_nodes']}, "
            f"invalid_arcs={result['invalid_arcs']}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "trip_id",
        "nodes",
        "arcs",
        "connected_components",
        "largest_component_nodes",
        "largest_component_percentage",
        "isolated_nodes",
        "degree_one_nodes",
        "invalid_arcs",
        "self_loops",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print("CONNECTIVITY ANALYSIS COMPLETED")
    print("Trips analysed:", len(rows))
    print("Report saved to:", OUTPUT_FILE)
    print("=" * 70)


if __name__ == "__main__":
    main()

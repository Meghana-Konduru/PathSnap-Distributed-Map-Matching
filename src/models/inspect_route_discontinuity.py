
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

from src.geospatial.candidate_generation import load_nodes, load_arcs


def haversine_m(lon1, lat1, lon2, lat2):
    radius = 6_371_000

    lon1, lat1, lon2, lat2 = map(
        radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return 2 * radius * atan2(sqrt(a), sqrt(1 - a))


def inspect_pair(trip_id, first_arc_id, second_arc_id):
    trip_dir = Path("data/raw/student_folder/train") / trip_id

    nodes = load_nodes(str(trip_dir / f"{trip_id}.nodes"))
    arcs = load_arcs(str(trip_dir / f"{trip_id}.arcs"))

    node_lookup = {node["node_id"]: node for node in nodes}
    arc_lookup = {arc["arc_id"]: arc for arc in arcs}

    first = arc_lookup[first_arc_id]
    second = arc_lookup[second_arc_id]

    first_from = node_lookup[first["from_node_id"]]
    first_to = node_lookup[first["to_node_id"]]

    second_from = node_lookup[second["from_node_id"]]
    second_to = node_lookup[second["to_node_id"]]

    gap_m = haversine_m(
        first_to["longitude"],
        first_to["latitude"],
        second_from["longitude"],
        second_from["latitude"],
    )

    print(f"\nTrip: {trip_id}")
    print(f"Route arc pair: {first_arc_id} -> {second_arc_id}")

    print("\nFirst arc:")
    print(first)

    print("\nSecond arc:")
    print(second)

    print("\nEndpoint coordinates:")
    print("First arc start:", first_from)
    print("First arc end:  ", first_to)
    print("Second arc start:", second_from)
    print("Second arc end:  ", second_to)

    print(f"\nDistance: first arc end -> second arc start = {gap_m:.3f} m")

    print("\nConnectivity checks:")
    print(
        "Forward node connectivity:",
        first["to_node_id"] == second["from_node_id"],
    )
    print(
        "Reverse node connectivity:",
        first["to_node_id"] == second["to_node_id"],
    )
    print(
        "Same start node:",
        first["from_node_id"] == second["from_node_id"],
    )


if __name__ == "__main__":
    inspect_pair("00000002", 1848, 10863)

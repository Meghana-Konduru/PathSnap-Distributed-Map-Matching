
from pathlib import Path
from collections import Counter

from src.geospatial.candidate_generation import (
    load_nodes,
    load_arcs,
    load_track,
)


def inspect_trip(trip_id):
    trip_dir = (
        Path("data/raw/student_folder/train") / trip_id
    )

    track = load_track(str(trip_dir / f"{trip_id}.track"))
    nodes = load_nodes(str(trip_dir / f"{trip_id}.nodes"))
    arcs = load_arcs(str(trip_dir / f"{trip_id}.arcs"))

    route_path = trip_dir / f"{trip_id}.route"

    route = [
        int(line.strip())
        for line in route_path.read_text().splitlines()
        if line.strip()
    ]

    arc_lookup = {arc["arc_id"]: arc for arc in arcs}

    invalid_ids = [
        arc_id for arc_id in route
        if arc_id not in arc_lookup
    ]

    valid_route = [
        arc_id for arc_id in route
        if arc_id in arc_lookup
    ]

    disconnected_pairs = []

    for first, second in zip(valid_route, valid_route[1:]):
        first_arc = arc_lookup[first]
        second_arc = arc_lookup[second]

        if first_arc["to_node_id"] != second_arc["from_node_id"]:
            disconnected_pairs.append((first, second))

    print(f"\n========== TRIP {trip_id} ==========")
    print("GPS observations:", len(track))
    print("Road nodes:", len(nodes))
    print("Road arcs:", len(arcs))
    print("Route entries:", len(route))
    print("Invalid route arc IDs:", len(invalid_ids))
    print("Disconnected consecutive route pairs:", len(disconnected_pairs))
    print("Unique route arc IDs:", len(set(route)))
    print("Repeated route arc IDs:", len(route) - len(set(route)))

    if invalid_ids:
        print("Invalid ID examples:", invalid_ids[:10])

    if disconnected_pairs:
        print("Disconnected pair examples:", disconnected_pairs[:10])


if __name__ == "__main__":
    inspect_trip("00000001")
    inspect_trip("00000002")
    inspect_trip("00000054")

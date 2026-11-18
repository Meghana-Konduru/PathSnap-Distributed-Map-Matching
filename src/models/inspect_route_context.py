
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

from src.geospatial.candidate_generation import (
    load_nodes,
    load_arcs,
    load_track,
)


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


def inspect_trip(trip_id, first_arc_id, second_arc_id):
    trip_dir = Path("data/raw/student_folder/train") / trip_id

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
    node_lookup = {node["node_id"]: node for node in nodes}

    pair_index = next(
        i for i in range(len(route) - 1)
        if route[i] == first_arc_id
        and route[i + 1] == second_arc_id
    )

    print(f"\nTrip: {trip_id}")
    print(f"Discontinuity at route position: {pair_index}")

    print("\n--- Route context ---")

    start = max(0, pair_index - 5)
    end = min(len(route), pair_index + 7)

    for i in range(start, end):
        arc_id = route[i]
        arc = arc_lookup[arc_id]

        if i > 0:
            previous = arc_lookup[route[i - 1]]
            connected = (
                previous["to_node_id"] == arc["from_node_id"]
            )
            status = "CONNECTED" if connected else "DISCONNECTED"
        else:
            status = "FIRST"

        print(
            f"Route position {i}: arc_id={arc_id}, "
            f"from={arc['from_node_id']}, "
            f"to={arc['to_node_id']}, "
            f"status={status}"
        )

    first_arc = arc_lookup[first_arc_id]
    second_arc = arc_lookup[second_arc_id]

    first_end = node_lookup[first_arc["to_node_id"]]
    second_start = node_lookup[second_arc["from_node_id"]]

    print("\n--- GPS observations nearest to the two endpoints ---")

    endpoints = [
        ("First arc end", first_end),
        ("Second arc start", second_start),
    ]

    for label, endpoint in endpoints:
        distances = []

        for observation_index, observation in enumerate(track):
            distance = haversine_m(
                endpoint["longitude"],
                endpoint["latitude"],
                observation["longitude"],
                observation["latitude"],
            )

            distances.append((distance, observation_index, observation))

        distances.sort(key=lambda item: item[0])

        print(f"\n{label}:")
        for distance, index, observation in distances[:5]:
            print(
                f"  GPS index={index}, "
                f"timestamp={observation['timestamp']}, "
                f"distance={distance:.2f} m, "
                f"lon={observation['longitude']}, "
                f"lat={observation['latitude']}"
            )


if __name__ == "__main__":
    inspect_trip("00000002", 1848, 10863)

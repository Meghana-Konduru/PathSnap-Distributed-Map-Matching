
"""
PathSnap - BIS
GPS loading, road-network loading and candidate generation.

Dataset formats:
    .track : longitude latitude timestamp
    .nodes : longitude latitude
    .arcs  : from_node_id to_node_id

Important:
    arc_id is the original zero-based row index in .arcs.
"""

from pathlib import Path
from math import cos, radians, sqrt
from collections import defaultdict
from typing import Dict, List, Tuple, Any


EARTH_RADIUS_M = 6_371_000.0


# =========================================================
# 1. GENERIC NUMERIC FILE LOADER
# =========================================================

def read_numeric_rows(
    file_path: str,
    expected_columns: int
) -> List[List[float]]:

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != expected_columns:
                raise ValueError(
                    f"{path}, line {line_number}: "
                    f"expected {expected_columns} columns, "
                    f"found {len(values)}"
                )

            try:
                row = [float(value) for value in values]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in {path}, "
                    f"line {line_number}"
                ) from exc

            rows.append(row)

    return rows


# =========================================================
# 2. LOAD GPS TRACK
# =========================================================

def load_track(track_path: str) -> List[Dict[str, float]]:
    """
    Loads GPS observations.

    Format:
        longitude latitude timestamp

    Returns observations in chronological order.
    """

    rows = read_numeric_rows(track_path, 3)

    observations = []

    for longitude, latitude, timestamp in rows:

        if not -180 <= longitude <= 180:
            raise ValueError(f"Invalid longitude: {longitude}")

        if not -90 <= latitude <= 90:
            raise ValueError(f"Invalid latitude: {latitude}")

        observations.append({
            "longitude": longitude,
            "latitude": latitude,
            "timestamp": timestamp
        })

    observations.sort(key=lambda item: item["timestamp"])

    return observations


# =========================================================
# 3. LOAD ROAD NODES
# =========================================================

def load_nodes(nodes_path: str) -> List[Dict[str, float]]:
    """
    Loads road nodes.

    Format:
        longitude latitude

    Node IDs are zero-based row indices.
    """

    rows = read_numeric_rows(nodes_path, 2)

    nodes = []

    for node_id, (longitude, latitude) in enumerate(rows):

        if not -180 <= longitude <= 180:
            raise ValueError(f"Invalid longitude at node {node_id}")

        if not -90 <= latitude <= 90:
            raise ValueError(f"Invalid latitude at node {node_id}")

        nodes.append({
            "node_id": node_id,
            "longitude": longitude,
            "latitude": latitude
        })

    return nodes


# =========================================================
# 4. LOAD ROAD ARCS
# =========================================================

def load_arcs(arcs_path: str) -> List[Dict[str, int]]:
    """
    Loads road connections.

    Format:
        from_node_id to_node_id

    IMPORTANT:
        arc_id is the original zero-based row index
        in the .arcs file.
    """

    rows = read_numeric_rows(arcs_path, 2)

    arcs = []

    for arc_id, (from_node, to_node) in enumerate(rows):

        if not from_node.is_integer():
            raise ValueError(f"Invalid source node in arc {arc_id}")

        if not to_node.is_integer():
            raise ValueError(f"Invalid destination node in arc {arc_id}")

        arcs.append({
            "arc_id": arc_id,
            "from_node_id": int(from_node),
            "to_node_id": int(to_node)
        })

    return arcs


# =========================================================
# 5. GEOGRAPHICAL PROJECTION
# =========================================================

def project_coordinates(
    longitude: float,
    latitude: float,
    reference_latitude: float
) -> Tuple[float, float]:
    """
    Convert longitude/latitude into approximate local metres.

    Uses a local equirectangular projection.
    """

    x = (
        EARTH_RADIUS_M
        * radians(longitude)
        * cos(radians(reference_latitude))
    )

    y = EARTH_RADIUS_M * radians(latitude)

    return x, y


def inverse_project(
    x: float,
    y: float,
    reference_latitude: float
) -> Tuple[float, float]:

    latitude = y / EARTH_RADIUS_M * 180.0 / 3.141592653589793

    longitude = (
        x
        / (EARTH_RADIUS_M * cos(radians(reference_latitude)))
        * 180.0
        / 3.141592653589793
    )

    return longitude, latitude


# =========================================================
# 6. POINT-TO-ROAD-SEGMENT DISTANCE
# =========================================================

def point_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float
) -> Tuple[float, float, float, float]:
    """
    Calculate the nearest point on a road segment.

    Returns:
        distance_m,
        projection_fraction,
        projected_x,
        projected_y
    """

    dx = bx - ax
    dy = by - ay

    length_squared = dx * dx + dy * dy

    if length_squared == 0:
        fraction = 0.0
    else:
        fraction = (
            (px - ax) * dx + (py - ay) * dy
        ) / length_squared

        fraction = max(0.0, min(1.0, fraction))

    projected_x = ax + fraction * dx
    projected_y = ay + fraction * dy

    distance = sqrt(
        (px - projected_x) ** 2
        + (py - projected_y) ** 2
    )

    return distance, fraction, projected_x, projected_y


# =========================================================
# 7. BUILD SPATIAL INDEX
# =========================================================

def build_spatial_index(
    nodes: List[Dict[str, float]],
    arcs: List[Dict[str, int]],
    reference_latitude: float,
    cell_size_m: float
):
    """
    Creates a uniform grid index for road segments.

    The index reduces the number of segments checked
    for each GPS observation.
    """

    projected_nodes = {}

    for node in nodes:

        projected_nodes[node["node_id"]] = project_coordinates(
            node["longitude"],
            node["latitude"],
            reference_latitude
        )

    grid = defaultdict(list)
    segments = {}

    for arc in arcs:

        arc_id = arc["arc_id"]

        from_id = arc["from_node_id"]
        to_id = arc["to_node_id"]

        if from_id not in projected_nodes:
            raise ValueError(
                f"Arc {arc_id} references missing node {from_id}"
            )

        if to_id not in projected_nodes:
            raise ValueError(
                f"Arc {arc_id} references missing node {to_id}"
            )

        ax, ay = projected_nodes[from_id]
        bx, by = projected_nodes[to_id]

        segments[arc_id] = (ax, ay, bx, by)

        min_x = int(min(ax, bx) // cell_size_m)
        max_x = int(max(ax, bx) // cell_size_m)

        min_y = int(min(ay, by) // cell_size_m)
        max_y = int(max(ay, by) // cell_size_m)

        for cell_x in range(min_x, max_x + 1):

            for cell_y in range(min_y, max_y + 1):

                grid[(cell_x, cell_y)].append(arc_id)

    return grid, segments


# =========================================================
# 8. GENERATE CANDIDATES FOR ONE GPS OBSERVATION
# =========================================================

def generate_candidates_for_observation(
    observation: Dict[str, float],
    grid,
    segments,
    reference_latitude: float,
    search_radius_m: float,
    max_candidates: int
) -> List[Dict[str, Any]]:

    px, py = project_coordinates(
        observation["longitude"],
        observation["latitude"],
        reference_latitude
    )

    min_x = int((px - search_radius_m) // search_radius_m)
    max_x = int((px + search_radius_m) // search_radius_m)

    min_y = int((py - search_radius_m) // search_radius_m)
    max_y = int((py + search_radius_m) // search_radius_m)

    possible_arc_ids = set()

    for cell_x in range(min_x, max_x + 1):

        for cell_y in range(min_y, max_y + 1):

            possible_arc_ids.update(
                grid.get((cell_x, cell_y), [])
            )

    candidates = []

    for arc_id in possible_arc_ids:

        ax, ay, bx, by = segments[arc_id]

        (
            distance,
            fraction,
            projected_x,
            projected_y
        ) = point_to_segment(
            px, py, ax, ay, bx, by
        )

        if distance > search_radius_m:
            continue

        projected_lon, projected_lat = inverse_project(
            projected_x,
            projected_y,
            reference_latitude
        )

        segment_length = sqrt(
            (bx - ax) ** 2 + (by - ay) ** 2
        )

        candidates.append({
            "arc_id": arc_id,
            "features": {
                "distance_m": distance,
                "projection_fraction": fraction,
                "projected_longitude": projected_lon,
                "projected_latitude": projected_lat,
                "segment_length_m": segment_length
            }
        })

    candidates.sort(
        key=lambda candidate: candidate["features"]["distance_m"]
    )

    return candidates[:max_candidates]


# =========================================================
# 9. GENERATE CANDIDATES FOR COMPLETE GPS TRACK
# =========================================================

def generate_candidates(
    observations: List[Dict[str, float]],
    nodes: List[Dict[str, float]],
    arcs: List[Dict[str, int]],
    search_radius_m: float = 50.0,
    max_candidates: int = 8
) -> List[Dict[str, Any]]:
    """
    Generates candidate road segments for every observation.

    Output:
        [
            {
                "timestamp": ...,
                "longitude": ...,
                "latitude": ...,
                "candidates": [
                    {
                        "arc_id": ...,
                        "features": {...}
                    }
                ]
            }
        ]
    """

    if not observations:
        return []

    if not nodes:
        raise ValueError("Road network has no nodes")

    if not arcs:
        raise ValueError("Road network has no arcs")

    if search_radius_m <= 0:
        raise ValueError("Search radius must be positive")

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")

    reference_latitude = sum(
        node["latitude"] for node in nodes
    ) / len(nodes)

    grid, segments = build_spatial_index(
        nodes,
        arcs,
        reference_latitude,
        search_radius_m
    )

    chronological_observations = sorted(
        observations,
        key=lambda item: item["timestamp"]
    )

    results = []

    for observation in chronological_observations:

        candidates = generate_candidates_for_observation(
            observation,
            grid,
            segments,
            reference_latitude,
            search_radius_m,
            max_candidates
        )

        results.append({
            "timestamp": observation["timestamp"],
            "longitude": observation["longitude"],
            "latitude": observation["latitude"],
            "candidates": candidates
        })

    return results


# =========================================================
# 10. LOAD COMPLETE TRIP
# =========================================================

def load_trip_data(
    trip_directory: str,
    search_radius_m: float = 50.0,
    max_candidates: int = 8
) -> List[Dict[str, Any]]:
    """
    Loads one trip and generates candidates.

    Expected files:
        trip_id.track
        trip_id.nodes
        trip_id.arcs

    .route is intentionally not loaded here.
    """

    trip_dir = Path(trip_directory)

    if not trip_dir.is_dir():
        raise NotADirectoryError(
            f"Trip directory not found: {trip_dir}"
        )

    track_files = list(trip_dir.glob("*.track"))
    node_files = list(trip_dir.glob("*.nodes"))
    arc_files = list(trip_dir.glob("*.arcs"))

    if len(track_files) != 1:
        raise ValueError(
            f"Expected one .track file in {trip_dir}"
        )

    if len(node_files) != 1:
        raise ValueError(
            f"Expected one .nodes file in {trip_dir}"
        )

    if len(arc_files) != 1:
        raise ValueError(
            f"Expected one .arcs file in {trip_dir}"
        )

    observations = load_track(str(track_files[0]))
    nodes = load_nodes(str(node_files[0]))
    arcs = load_arcs(str(arc_files[0]))

    return generate_candidates(
        observations,
        nodes,
        arcs,
        search_radius_m,
        max_candidates
    )

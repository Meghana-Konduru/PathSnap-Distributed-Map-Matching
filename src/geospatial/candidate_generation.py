
"""
PathSnap - BIS
GPS preprocessing, road-network loading, spatial indexing,
candidate generation and movement feature engineering.

Dataset formats:
    .track : longitude latitude timestamp
    .nodes : longitude latitude
    .arcs  : from_node_id to_node_id

Important:
    - arc_id is the original zero-based row index in .arcs.
    - GPS observations are sorted chronologically.
    - .route is intentionally not loaded by this module.
    - Search radius default: 125 metres (provisional).
    - Maximum candidates default: 8.

Candidate features:
    1. distance_m
    2. projection_fraction
    3. projected_longitude
    4. projected_latitude
    5. segment_length_m
    6. gps_heading_deg
    7. gps_speed_mps
    8. time_delta_s
    9. segment_heading_deg
    10. heading_difference_deg
"""

from pathlib import Path
from math import (
    cos,
    radians,
    sqrt,
    sin,
    atan2,
    degrees,
)
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


EARTH_RADIUS_M = 6_371_000.0

DEFAULT_SEARCH_RADIUS_M = 125.0
DEFAULT_MAX_CANDIDATES = 8


# =========================================================
# 1. GENERIC NUMERIC FILE LOADER
# =========================================================

def read_numeric_rows(
    file_path: str,
    expected_columns: int
) -> List[List[float]]:
    """
    Read a whitespace-separated numeric file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a row has an incorrect column count
                    or contains a non-numeric value.
    """

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

def load_track(
    track_path: str
) -> List[Dict[str, float]]:
    """
    Load GPS observations.

    Format:
        longitude latitude timestamp

    Returns:
        Chronologically sorted GPS observations.
    """

    rows = read_numeric_rows(track_path, 3)

    observations = []

    for longitude, latitude, timestamp in rows:

        if not -180 <= longitude <= 180:
            raise ValueError(
                f"Invalid longitude: {longitude}"
            )

        if not -90 <= latitude <= 90:
            raise ValueError(
                f"Invalid latitude: {latitude}"
            )

        observations.append({
            "longitude": longitude,
            "latitude": latitude,
            "timestamp": timestamp,
        })

    observations.sort(
        key=lambda item: item["timestamp"]
    )

    return observations


# =========================================================
# 3. LOAD ROAD NODES
# =========================================================

def load_nodes(
    nodes_path: str
) -> List[Dict[str, float]]:
    """
    Load road-network nodes.

    Format:
        longitude latitude

    Node IDs are zero-based row indices.
    """

    rows = read_numeric_rows(nodes_path, 2)

    nodes = []

    for node_id, (longitude, latitude) in enumerate(rows):

        if not -180 <= longitude <= 180:
            raise ValueError(
                f"Invalid longitude at node {node_id}"
            )

        if not -90 <= latitude <= 90:
            raise ValueError(
                f"Invalid latitude at node {node_id}"
            )

        nodes.append({
            "node_id": node_id,
            "longitude": longitude,
            "latitude": latitude,
        })

    return nodes


# =========================================================
# 4. LOAD ROAD ARCS
# =========================================================

def load_arcs(
    arcs_path: str
) -> List[Dict[str, int]]:
    """
    Load road-network arcs.

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
            raise ValueError(
                f"Invalid source node in arc {arc_id}"
            )

        if not to_node.is_integer():
            raise ValueError(
                f"Invalid destination node in arc {arc_id}"
            )

        arcs.append({
            "arc_id": arc_id,
            "from_node_id": int(from_node),
            "to_node_id": int(to_node),
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

    Suitable for compact geographical regions.
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
    """
    Convert projected local coordinates back to longitude/latitude.
    """

    latitude = (
        y / EARTH_RADIUS_M
        * 180.0 / 3.141592653589793
    )

    longitude = (
        x
        / (
            EARTH_RADIUS_M
            * cos(radians(reference_latitude))
        )
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

        fraction = max(
            0.0,
            min(1.0, fraction)
        )

    projected_x = ax + fraction * dx
    projected_y = ay + fraction * dy

    distance = sqrt(
        (px - projected_x) ** 2
        + (py - projected_y) ** 2
    )

    return (
        distance,
        fraction,
        projected_x,
        projected_y,
    )


# =========================================================
# 7. BEARING AND ANGULAR DIFFERENCE
# =========================================================

def calculate_bearing(
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> Optional[float]:
    """
    Calculate bearing clockwise from geographic north.

    Returns:
        Bearing between 0 and 360 degrees.

    Returns None if both points are identical.
    """

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return None

    bearing = degrees(
        atan2(dx, dy)
    )

    return (bearing + 360.0) % 360.0


def angular_difference(
    angle1: Optional[float],
    angle2: Optional[float]
) -> Optional[float]:
    """
    Calculate the smallest absolute angular difference.

    Example:
        Difference between 350 and 10 degrees = 20 degrees.
    """

    if angle1 is None or angle2 is None:
        return None

    difference = abs(angle1 - angle2)

    return min(
        difference,
        360.0 - difference
    )


# =========================================================
# 8. GPS MOVEMENT FEATURE ENGINEERING
# =========================================================

def calculate_observation_features(
    observations: List[Dict[str, float]],
    reference_latitude: float
) -> List[Dict[str, Optional[float]]]:
    """
    Calculate GPS movement features.

    Features:
        gps_heading_deg
        gps_speed_mps
        time_delta_s

    The first observation has no previous observation,
    so its movement features are None.

    Duplicate timestamps and non-positive time intervals
    do not produce a speed value.
    """

    features = []

    previous_projected = None
    previous_timestamp = None

    for observation in observations:

        current_projected = project_coordinates(
            observation["longitude"],
            observation["latitude"],
            reference_latitude,
        )

        current_timestamp = observation["timestamp"]

        if previous_projected is None:

            features.append({
                "gps_heading_deg": None,
                "gps_speed_mps": None,
                "time_delta_s": None,
            })

        else:

            x1, y1 = previous_projected
            x2, y2 = current_projected

            time_delta = (
                current_timestamp - previous_timestamp
            )

            distance = sqrt(
                (x2 - x1) ** 2
                + (y2 - y1) ** 2
            )

            heading = calculate_bearing(
                x1, y1, x2, y2
            )

            if time_delta > 0:
                speed = distance / time_delta
                stored_time_delta = time_delta

            else:
                speed = None
                stored_time_delta = None

            features.append({
                "gps_heading_deg": heading,
                "gps_speed_mps": speed,
                "time_delta_s": stored_time_delta,
            })

        previous_projected = current_projected
        previous_timestamp = current_timestamp

    return features


# =========================================================
# 9. BUILD SPATIAL INDEX
# =========================================================

def build_spatial_index(
    nodes: List[Dict[str, float]],
    arcs: List[Dict[str, int]],
    reference_latitude: float,
    cell_size_m: float
):
    """
    Create a uniform grid spatial index for road segments.

    Each road segment is inserted into every grid cell
    intersected by its bounding rectangle.

    This reduces the number of segments checked for
    each GPS observation.
    """

    if cell_size_m <= 0:
        raise ValueError(
            "Spatial index cell size must be positive"
        )

    projected_nodes = {}

    for node in nodes:

        projected_nodes[node["node_id"]] = project_coordinates(
            node["longitude"],
            node["latitude"],
            reference_latitude,
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

        segments[arc_id] = (
            ax, ay, bx, by
        )

        min_x = int(
            min(ax, bx) // cell_size_m
        )

        max_x = int(
            max(ax, bx) // cell_size_m
        )

        min_y = int(
            min(ay, by) // cell_size_m
        )

        max_y = int(
            max(ay, by) // cell_size_m
        )

        for cell_x in range(min_x, max_x + 1):

            for cell_y in range(min_y, max_y + 1):

                grid[(cell_x, cell_y)].append(arc_id)

    return grid, segments


# =========================================================
# 10. GENERATE CANDIDATES FOR ONE GPS OBSERVATION
# =========================================================

def generate_candidates_for_observation(
    observation: Dict[str, float],
    grid,
    segments,
    reference_latitude: float,
    search_radius_m: float,
    max_candidates: int
) -> List[Dict[str, Any]]:
    """
    Generate nearby road-segment candidates for one GPS point.

    Candidate output:
        {
            "arc_id": ...,
            "features": {
                ...
            }
        }
    """

    px, py = project_coordinates(
        observation["longitude"],
        observation["latitude"],
        reference_latitude,
    )

    min_x = int(
        (px - search_radius_m) // search_radius_m
    )

    max_x = int(
        (px + search_radius_m) // search_radius_m
    )

    min_y = int(
        (py - search_radius_m) // search_radius_m
    )

    max_y = int(
        (py + search_radius_m) // search_radius_m
    )

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
            projected_y,
        ) = point_to_segment(
            px,
            py,
            ax,
            ay,
            bx,
            by,
        )

        if distance > search_radius_m:
            continue

        projected_lon, projected_lat = inverse_project(
            projected_x,
            projected_y,
            reference_latitude,
        )

        segment_length = sqrt(
            (bx - ax) ** 2
            + (by - ay) ** 2
        )

        segment_heading = calculate_bearing(
            ax, ay, bx, by
        )

        candidates.append({
            "arc_id": arc_id,
            "features": {
                "distance_m": distance,
                "projection_fraction": fraction,
                "projected_longitude": projected_lon,
                "projected_latitude": projected_lat,
                "segment_length_m": segment_length,
                "segment_heading_deg": segment_heading,
            },
        })

    candidates.sort(
        key=lambda candidate: (
            candidate["features"]["distance_m"]
        )
    )

    return candidates[:max_candidates]


# =========================================================
# 11. GENERATE CANDIDATES FOR COMPLETE GPS TRACK
# =========================================================

def generate_candidates(
    observations: List[Dict[str, float]],
    nodes: List[Dict[str, float]],
    arcs: List[Dict[str, int]],
    search_radius_m: float = DEFAULT_SEARCH_RADIUS_M,
    max_candidates: int = DEFAULT_MAX_CANDIDATES
) -> List[Dict[str, Any]]:
    """
    Generate road-segment candidates for every GPS observation.

    Output:
        [
            {
                "timestamp": ...,
                "longitude": ...,
                "latitude": ...,
                "movement_features": {...},
                "candidates": [
                    {
                        "arc_id": ...,
                        "features": {
                            ...
                        }
                    }
                ]
            }
        ]

    The candidate feature dictionary contains:
        5 spatial features
        3 GPS movement features
        2 road-direction features

    The observation-level movement_features field preserves
    movement information even when an observation has no
    road candidates.
    """

    if not observations:
        return []

    if not nodes:
        raise ValueError(
            "Road network has no nodes"
        )

    if not arcs:
        raise ValueError(
            "Road network has no arcs"
        )

    if search_radius_m <= 0:
        raise ValueError(
            "Search radius must be positive"
        )

    if max_candidates < 1:
        raise ValueError(
            "max_candidates must be at least 1"
        )

    # Use the average node latitude as the local
    # projection reference.
    reference_latitude = sum(
        node["latitude"] for node in nodes
    ) / len(nodes)

    # Build spatial index once for the complete trip.
    grid, segments = build_spatial_index(
        nodes,
        arcs,
        reference_latitude,
        search_radius_m,
    )

    # Ensure chronological ordering.
    chronological_observations = sorted(
        observations,
        key=lambda item: item["timestamp"],
    )

    # Calculate movement features for every observation.
    observation_features = calculate_observation_features(
        chronological_observations,
        reference_latitude,
    )

    results = []

    for index, observation in enumerate(
        chronological_observations
    ):

        candidates = generate_candidates_for_observation(
            observation,
            grid,
            segments,
            reference_latitude,
            search_radius_m,
            max_candidates,
        )

        movement_features = observation_features[index]

        for candidate in candidates:

            candidate_features = candidate["features"]

            # Add GPS movement features.
            candidate_features.update(
                movement_features
            )

            # Compare GPS movement direction with
            # the direction of the road arc.
            candidate_features[
                "heading_difference_deg"
            ] = angular_difference(
                movement_features["gps_heading_deg"],
                candidate_features["segment_heading_deg"],
            )

        results.append({
            "timestamp": observation["timestamp"],
            "longitude": observation["longitude"],
            "latitude": observation["latitude"],
            "movement_features": movement_features,
            "candidates": candidates,
        })

    return results


# =========================================================
# 12. LOAD COMPLETE TRIP
# =========================================================

def load_trip_data(
    trip_directory: str,
    search_radius_m: float = DEFAULT_SEARCH_RADIUS_M,
    max_candidates: int = DEFAULT_MAX_CANDIDATES
) -> List[Dict[str, Any]]:
    """
    Load one trip and generate road-segment candidates.

    Expected files:
        trip_id.track
        trip_id.nodes
        trip_id.arcs

    The .route file is intentionally not loaded here.

    Default search radius:
        125 metres (provisional experimental setting).
    """

    trip_dir = Path(trip_directory)

    if not trip_dir.is_dir():
        raise NotADirectoryError(
            f"Trip directory not found: {trip_dir}"
        )

    track_files = list(
        trip_dir.glob("*.track")
    )

    node_files = list(
        trip_dir.glob("*.nodes")
    )

    arc_files = list(
        trip_dir.glob("*.arcs")
    )

    if len(track_files) != 1:
        raise ValueError(
            f"Expected one .track file in {trip_dir}, "
            f"found {len(track_files)}"
        )

    if len(node_files) != 1:
        raise ValueError(
            f"Expected one .nodes file in {trip_dir}, "
            f"found {len(node_files)}"
        )

    if len(arc_files) != 1:
        raise ValueError(
            f"Expected one .arcs file in {trip_dir}, "
            f"found {len(arc_files)}"
        )

    observations = load_track(
        str(track_files[0])
    )

    nodes = load_nodes(
        str(node_files[0])
    )

    arcs = load_arcs(
        str(arc_files[0])
    )

    return generate_candidates(
        observations=observations,
        nodes=nodes,
        arcs=arcs,
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
    )


from pathlib import Path
from collections import Counter

from src.geospatial.candidate_generation import (
    load_track,
    load_nodes,
    load_arcs,
    generate_candidates,
)

TRIP_ID = "00000054"
TRIP_DIR = Path("data/raw/student_folder/train") / TRIP_ID

track = load_track(str(TRIP_DIR / f"{TRIP_ID}.track"))
nodes = load_nodes(str(TRIP_DIR / f"{TRIP_ID}.nodes"))
arcs = load_arcs(str(TRIP_DIR / f"{TRIP_ID}.arcs"))

results = generate_candidates(
    track,
    nodes,
    arcs,
    search_radius_m=50.0,
    max_candidates=8,
)

# --------------------------------------------------
# 1. GPS AND ROAD NETWORK BOUNDING BOXES
# --------------------------------------------------

def bounds(records, lon_key, lat_key):
    longitudes = [r[lon_key] for r in records]
    latitudes = [r[lat_key] for r in records]

    return {
        "min_lon": min(longitudes),
        "max_lon": max(longitudes),
        "min_lat": min(latitudes),
        "max_lat": max(latitudes),
    }


gps_bounds = bounds(track, "longitude", "latitude")
node_bounds = bounds(nodes, "longitude", "latitude")

print("=" * 60)
print("TRIP 00000054 DIAGNOSTIC")
print("=" * 60)

print("\nGPS BOUNDING BOX")
print(gps_bounds)

print("\nROAD NETWORK BOUNDING BOX")
print(node_bounds)

print("\nGPS OBSERVATIONS:", len(track))
print("ROAD NODES:", len(nodes))
print("ROAD ARCS:", len(arcs))

# --------------------------------------------------
# 2. TIMESTAMP ANALYSIS
# --------------------------------------------------

timestamps = [item["timestamp"] for item in track]

time_differences = [
    timestamps[i] - timestamps[i - 1]
    for i in range(1, len(timestamps))
]

print("\nTIMESTAMP ANALYSIS")
print("First timestamp:", timestamps[0])
print("Last timestamp:", timestamps[-1])
print("Minimum interval:", min(time_differences))
print("Maximum interval:", max(time_differences))
print("Non-positive intervals:", sum(
    gap <= 0 for gap in time_differences
))
print("Intervals greater than 60 seconds:", sum(
    gap > 60 for gap in time_differences
))

# --------------------------------------------------
# 3. CANDIDATE COVERAGE
# --------------------------------------------------

candidate_counts = [
    len(item["candidates"]) for item in results
]

print("\nCANDIDATE ANALYSIS")
print("Observations with candidates:", sum(
    count > 0 for count in candidate_counts
))
print("Observations without candidates:", sum(
    count == 0 for count in candidate_counts
))
print("Maximum candidates:", max(candidate_counts))
print("Average candidates:", sum(candidate_counts) / len(candidate_counts))

# --------------------------------------------------
# 4. CONTIGUOUS ZERO-CANDIDATE SEQUENCES
# --------------------------------------------------

missing_indices = [
    i for i, count in enumerate(candidate_counts)
    if count == 0
]

runs = []

if missing_indices:
    start = previous = missing_indices[0]

    for index in missing_indices[1:]:
        if index == previous + 1:
            previous = index
        else:
            runs.append((start, previous))
            start = previous = index

    runs.append((start, previous))

print("\nZERO-CANDIDATE SEQUENCES")
print("Number of separate sequences:", len(runs))

for start, end in sorted(
    runs,
    key=lambda item: item[1] - item[0],
    reverse=True
)[:15]:

    print({
        "start_index": start,
        "end_index": end,
        "observations": end - start + 1,
        "start_timestamp": timestamps[start],
        "end_timestamp": timestamps[end],
        "start_coordinate": (
            track[start]["longitude"],
            track[start]["latitude"],
        ),
        "end_coordinate": (
            track[end]["longitude"],
            track[end]["latitude"],
        ),
    })

# --------------------------------------------------
# 5. SAMPLE MISSING OBSERVATIONS
# --------------------------------------------------

print("\nFIRST 10 MISSING OBSERVATIONS")

for index in missing_indices[:10]:
    print({
        "index": index,
        "timestamp": timestamps[index],
        "longitude": track[index]["longitude"],
        "latitude": track[index]["latitude"],
    })

# --------------------------------------------------
# 6. CHECK GPS POINTS OUTSIDE NETWORK BOUNDS
# --------------------------------------------------

outside_bounds = []

for index, observation in enumerate(track):
    outside = (
        observation["longitude"] < node_bounds["min_lon"]
        or observation["longitude"] > node_bounds["max_lon"]
        or observation["latitude"] < node_bounds["min_lat"]
        or observation["latitude"] > node_bounds["max_lat"]
    )

    if outside:
        outside_bounds.append(index)

print("\nGPS POINTS OUTSIDE ROAD NETWORK BOUNDING BOX")
print(len(outside_bounds))

print("\nDIAGNOSTIC COMPLETE")

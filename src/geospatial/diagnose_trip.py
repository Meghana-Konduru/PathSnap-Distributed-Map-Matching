
from pathlib import Path
from collections import Counter

from src.geospatial.candidate_generation import (
    load_track,
    load_nodes,
    load_arcs,
    generate_candidates,
)


# ============================================================
# 1. SELECT THE TRIP
# ============================================================

TRIP_ID = "00000054"

TRIP_DIR = Path("data/raw/student_folder/train") / TRIP_ID

print("=" * 65)
print(f"DIAGNOSTIC ANALYSIS FOR TRIP {TRIP_ID}")
print("=" * 65)


# ============================================================
# 2. LOAD GPS, ROAD NODES AND ROAD ARCS
# ============================================================

track = load_track(
    str(TRIP_DIR / f"{TRIP_ID}.track")
)

nodes = load_nodes(
    str(TRIP_DIR / f"{TRIP_ID}.nodes")
)

arcs = load_arcs(
    str(TRIP_DIR / f"{TRIP_ID}.arcs")
)

print("\nDATASET INFORMATION")
print("-" * 40)

print("GPS observations:", len(track))
print("Road nodes:", len(nodes))
print("Road arcs:", len(arcs))


# ============================================================
# 3. GENERATE CANDIDATES
# ============================================================

results = generate_candidates(
    observations=track,
    nodes=nodes,
    arcs=arcs,
    search_radius_m=50.0,
    max_candidates=8,
)

print("\nCANDIDATE GENERATION")
print("-" * 40)

total_observations = len(results)

candidate_counts = [
    len(item["candidates"])
    for item in results
]

total_candidates = sum(candidate_counts)

observations_with_candidates = sum(
    count > 0 for count in candidate_counts
)

observations_without_candidates = sum(
    count == 0 for count in candidate_counts
)

coverage = (
    observations_with_candidates / total_observations * 100
    if total_observations else 0
)

print("Total observations:", total_observations)
print("Total candidates:", total_candidates)
print("Observations with candidates:", observations_with_candidates)
print("Observations without candidates:", observations_without_candidates)
print(f"Candidate coverage: {coverage:.2f}%")


# ============================================================
# 4. COMPARE GPS AND ROAD-NETWORK GEOGRAPHICAL BOUNDARIES
# ============================================================

def get_bounds(records):
    longitudes = [item["longitude"] for item in records]
    latitudes = [item["latitude"] for item in records]

    return {
        "min_longitude": min(longitudes),
        "max_longitude": max(longitudes),
        "min_latitude": min(latitudes),
        "max_latitude": max(latitudes),
    }


gps_bounds = get_bounds(track)
node_bounds = get_bounds(nodes)

print("\nGPS GEOGRAPHICAL BOUNDARIES")
print("-" * 40)
print(gps_bounds)

print("\nROAD-NETWORK GEOGRAPHICAL BOUNDARIES")
print("-" * 40)
print(node_bounds)


# ============================================================
# 5. CHECK WHETHER GPS POINTS FALL OUTSIDE NODE BOUNDARIES
# ============================================================

outside_boundary = []

for index, observation in enumerate(track):

    lon = observation["longitude"]
    lat = observation["latitude"]

    outside = (
        lon < node_bounds["min_longitude"]
        or lon > node_bounds["max_longitude"]
        or lat < node_bounds["min_latitude"]
        or lat > node_bounds["max_latitude"]
    )

    if outside:
        outside_boundary.append(index)


print("\nGPS POINTS OUTSIDE ROAD-NODE BOUNDARIES")
print("-" * 40)

print("Outside boundary:", len(outside_boundary))
print("Inside boundary:", len(track) - len(outside_boundary))


# ============================================================
# 6. ANALYSE TIMESTAMPS
# ============================================================

timestamps = [
    observation["timestamp"]
    for observation in track
]

time_differences = [
    timestamps[i] - timestamps[i - 1]
    for i in range(1, len(timestamps))
]

print("\nTIMESTAMP ANALYSIS")
print("-" * 40)

print("First timestamp:", timestamps[0])
print("Last timestamp:", timestamps[-1])

if time_differences:

    print("Minimum time interval:", min(time_differences))
    print("Maximum time interval:", max(time_differences))

    print(
        "Average time interval:",
        sum(time_differences) / len(time_differences),
    )

    print(
        "Non-positive time intervals:",
        sum(diff <= 0 for diff in time_differences),
    )


# ============================================================
# 7. IDENTIFY OBSERVATIONS WITH NO CANDIDATES
# ============================================================

missing_indices = [
    index
    for index, item in enumerate(results)
    if len(item["candidates"]) == 0
]

print("\nMISSING-CANDIDATE ANALYSIS")
print("-" * 40)

print("Number of missing observations:", len(missing_indices))

print("\nFIRST 10 OBSERVATIONS WITHOUT CANDIDATES")

for index in missing_indices[:10]:

    observation = results[index]

    print(
        f"Index: {index}, "
        f"Timestamp: {observation['timestamp']}, "
        f"Longitude: {observation['longitude']}, "
        f"Latitude: {observation['latitude']}"
    )


# ============================================================
# 8. FIND CONTIGUOUS MISSING-CANDIDATE SEQUENCES
# ============================================================

missing_runs = []

if missing_indices:

    run_start = missing_indices[0]
    previous = missing_indices[0]

    for index in missing_indices[1:]:

        if index == previous + 1:
            previous = index

        else:
            missing_runs.append((run_start, previous))
            run_start = index
            previous = index

    missing_runs.append((run_start, previous))


print("\nCONTIGUOUS MISSING-CANDIDATE SEQUENCES")
print("-" * 40)

print("Number of missing sequences:", len(missing_runs))

for start, end in sorted(
    missing_runs,
    key=lambda run: run[1] - run[0],
    reverse=True,
)[:10]:

    print(
        f"Start index: {start}, "
        f"End index: {end}, "
        f"Length: {end - start + 1}"
    )


# ============================================================
# 9. CANDIDATE COUNT DISTRIBUTION
# ============================================================

distribution = Counter(candidate_counts)

print("\nCANDIDATE COUNT DISTRIBUTION")
print("-" * 40)

for number_of_candidates in sorted(distribution):

    print(
        f"{number_of_candidates} candidates: "
        f"{distribution[number_of_candidates]} observations"
    )


# ============================================================
# 10. SUMMARY
# ============================================================

print("\nFINAL DIAGNOSTIC SUMMARY")
print("=" * 65)

print("Trip ID:", TRIP_ID)
print("GPS observations:", total_observations)
print("Road nodes:", len(nodes))
print("Road arcs:", len(arcs))
print("Total candidates:", total_candidates)
print("Candidate coverage:", f"{coverage:.2f}%")
print("Missing candidates:", observations_without_candidates)
print("GPS points outside node bounding box:", len(outside_boundary))
print("Missing-candidate sequences:", len(missing_runs))

print("=" * 65)
print("DIAGNOSTIC COMPLETED")
print("=" * 65)

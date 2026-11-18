
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

from src.geospatial.candidate_generation import load_track


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


def inspect_track_gap(trip_id, start_index, end_index):
    trip_dir = Path("data/raw/student_folder/train") / trip_id

    track = load_track(str(trip_dir / f"{trip_id}.track"))

    start_index = max(0, start_index)
    end_index = min(len(track) - 1, end_index)

    print(f"\nTrip: {trip_id}")
    print(f"Inspecting GPS indices {start_index} to {end_index}")

    start = track[start_index]
    end = track[end_index]

    print("\n--- Interval summary ---")
    print("Start:", start)
    print("End:", end)
    print("Elapsed time:", round(
        end["timestamp"] - start["timestamp"], 3
    ), "seconds")
    print("Number of GPS observations:", end_index - start_index + 1)

    movements = []

    for i in range(start_index + 1, end_index + 1):
        previous = track[i - 1]
        current = track[i]

        distance = haversine_m(
            previous["longitude"],
            previous["latitude"],
            current["longitude"],
            current["latitude"],
        )

        elapsed = current["timestamp"] - previous["timestamp"]
        speed = distance / elapsed if elapsed > 0 else float("inf")

        movements.append((distance, speed, i, elapsed))

    print("\n--- Largest consecutive GPS movements ---")

    for distance, speed, index, elapsed in sorted(
        movements, reverse=True
    )[:15]:
        print(
            f"GPS index={index}, "
            f"distance={distance:.2f} m, "
            f"time_delta={elapsed:.3f} s, "
            f"speed={speed:.2f} m/s "
            f"({speed * 3.6:.2f} km/h)"
        )

    print("\n--- Sampled GPS trajectory ---")

    step = 25

    for i in range(start_index, end_index + 1, step):
        observation = track[i]

        print(
            f"index={i}, "
            f"time={observation['timestamp']:.3f}, "
            f"lon={observation['longitude']:.6f}, "
            f"lat={observation['latitude']:.6f}"
        )

    if end_index not in range(start_index, end_index + 1, step):
        observation = track[end_index]

        print(
            f"index={end_index}, "
            f"time={observation['timestamp']:.3f}, "
            f"lon={observation['longitude']:.6f}, "
            f"lat={observation['latitude']:.6f}"
        )


if __name__ == "__main__":
    inspect_track_gap("00000002", 650, 1090)

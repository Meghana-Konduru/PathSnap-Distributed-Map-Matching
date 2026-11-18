
from pathlib import Path
from src.geospatial.candidate_generation import load_trip_data

TRAIN_DIR = Path("data/raw/student_folder/train")

trip_dirs = sorted(
    path for path in TRAIN_DIR.iterdir()
    if path.is_dir()
)

print("Training trips found:", len(trip_dirs))

total_observations = 0
total_candidates = 0
observations_without_candidates = 0

for index, trip_dir in enumerate(trip_dirs, start=1):

    try:
        results = load_trip_data(
            str(trip_dir),
            search_radius_m=50.0,
            max_candidates=8
        )

        observation_count = len(results)

        candidate_count = sum(
            len(item["candidates"]) for item in results
        )

        empty_count = sum(
            len(item["candidates"]) == 0 for item in results
        )

        total_observations += observation_count
        total_candidates += candidate_count
        observations_without_candidates += empty_count

        print(
            f"[{index:02d}/{len(trip_dirs)}] "
            f"{trip_dir.name}: "
            f"observations={observation_count}, "
            f"candidates={candidate_count}, "
            f"no_candidates={empty_count}"
        )

    except Exception as error:
        print(f"FAILED: {trip_dir.name}: {error}")
        raise

print("\n========== TRAINING DATA SUMMARY ==========")
print("Trips processed:", len(trip_dirs))
print("Total observations:", total_observations)
print("Total candidates:", total_candidates)
print(
    "Observations without candidates:",
    observations_without_candidates
)
print("All training trips processed successfully.")

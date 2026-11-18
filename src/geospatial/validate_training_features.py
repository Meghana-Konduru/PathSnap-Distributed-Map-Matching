
from pathlib import Path
from src.geospatial.candidate_generation import load_trip_data

root = Path("data/raw/student_folder/train")

total = 0
covered = 0
candidates = 0
feature_counts = set()

print("TRAINING FEATURE VALIDATION")

for trip_path in sorted(root.iterdir()):
    if not trip_path.is_dir():
        continue

    observations = load_trip_data(
        str(trip_path),
        search_radius_m=125
    )

    trip_covered = sum(
        bool(obs["candidates"]) for obs in observations
    )
    trip_candidates = sum(
        len(obs["candidates"]) for obs in observations
    )

    for obs in observations:
        for candidate in obs["candidates"]:
            feature_counts.add(len(candidate["features"]))

    total += len(observations)
    covered += trip_covered
    candidates += trip_candidates

    print(
        f"{trip_path.name}: "
        f"observations={len(observations)}, "
        f"covered={trip_covered}, "
        f"candidates={trip_candidates}"
    )

print("\nFINAL SUMMARY")
print("Trips:", sum(1 for p in root.iterdir() if p.is_dir()))
print("Observations:", total)
print("Covered:", covered)
print("Uncovered:", total - covered)
print("Coverage:", round(covered / total * 100, 4) if total else 0, "%")
print("Candidates:", candidates)
print("Feature counts:", sorted(feature_counts))
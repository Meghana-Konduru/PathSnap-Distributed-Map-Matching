
from pathlib import Path
from src.geospatial.candidate_generation import load_trip_data

root = Path("data/raw/student_folder/student_test")

total = 0
covered = 0
candidates = 0

print("STUDENT-TEST FEATURE VALIDATION")

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

    feature_counts = {
        len(candidate["features"])
        for obs in observations
        for candidate in obs["candidates"]
    }

    total += len(observations)
    covered += trip_covered
    candidates += trip_candidates

    coverage = (
        trip_covered / len(observations) * 100
        if observations else 0
    )

    print(
        f"{trip_path.name}: "
        f"observations={len(observations)}, "
        f"coverage={coverage:.2f}%, "
        f"candidates={trip_candidates}, "
        f"feature_counts={sorted(feature_counts)}"
    )

print("\nTOTAL")
print("Observations:", total)
print("Covered:", covered)
print("Uncovered:", total - covered)
print("Coverage:", round(covered / total * 100, 2) if total else 0, "%")
print("Candidates:", candidates)
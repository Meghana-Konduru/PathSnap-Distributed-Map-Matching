
from pathlib import Path
import csv

from src.geospatial.candidate_generation import (
    load_track,
    load_nodes,
    load_arcs,
    generate_candidates,
)

TEST_DIR = Path("data/raw/student_folder/student_test")
OUTPUT_FILE = Path("reports/bis/radius_comparison_student_test.csv")

RADII = [100.0, 110.0, 125.0, 150.0]
MAX_CANDIDATES = 8

trip_directories = sorted(
    path for path in TEST_DIR.iterdir() if path.is_dir()
)

summary = []

print("=" * 65)
print("STUDENT-TEST DATASET: RADIUS COMPARISON")
print("Student-test trips:", len(trip_directories))
print("=" * 65)

for radius in RADII:
    total_observations = 0
    total_covered = 0
    total_candidates = 0

    print(f"\nTesting radius: {radius:.0f}m")

    for trip_dir in trip_directories:
        trip_id = trip_dir.name

        track = load_track(str(trip_dir / f"{trip_id}.track"))
        nodes = load_nodes(str(trip_dir / f"{trip_id}.nodes"))
        arcs = load_arcs(str(trip_dir / f"{trip_id}.arcs"))

        generated = generate_candidates(
            observations=track,
            nodes=nodes,
            arcs=arcs,
            search_radius_m=radius,
            max_candidates=MAX_CANDIDATES,
        )

        covered = sum(bool(item["candidates"]) for item in generated)
        candidates = sum(len(item["candidates"]) for item in generated)

        total_observations += len(generated)
        total_covered += covered
        total_candidates += candidates

        trip_coverage = covered / len(generated) * 100 if generated else 0

        print(
            f"Trip {trip_id}: {trip_coverage:.2f}% coverage"
        )

    coverage = (
        total_covered / total_observations * 100
        if total_observations else 0
    )

    row = {
        "radius_m": radius,
        "observations": total_observations,
        "covered_observations": total_covered,
        "missing_observations": total_observations - total_covered,
        "total_candidates": total_candidates,
        "coverage_percentage": round(coverage, 4),
    }

    summary.append(row)

    print("\nOverall coverage:", f"{coverage:.2f}%")
    print("Total candidates:", total_candidates)
    print("Missing observations:", total_observations - total_covered)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print("\nFINAL STUDENT-TEST COMPARISON")

for row in summary:
    print(row)

print("\nReport saved:", OUTPUT_FILE)

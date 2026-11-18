
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)

from src.features.feature_engineering import process_trip


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STUDENT_TEST_DIR = (
    PROJECT_ROOT / "data" / "raw" / "student_folder" / "student_test"
)

MODEL_PATH = (
    PROJECT_ROOT / "models" / "artifacts" / "pathsnap_random_forest.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT / "models" / "artifacts" / "pathsnap_random_forest_metadata.json"
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "bee_student_test"


def flatten_rows(rows):
    """Flatten nested feature dictionaries if present."""
    flattened = []

    for row in rows:
        item = dict(row)

        features = item.pop("features", None)

        if isinstance(features, dict):
            item.update(features)

        flattened.append(item)

    return flattened


def calculate_metrics(data):
    """Calculate candidate-level and observation-level metrics."""
    y_true = data["label"].astype(int).to_numpy()
    y_pred = data["predicted_label"].astype(int).to_numpy()
    probabilities = data["positive_probability"].to_numpy()

    candidate_metrics = {
        "candidate_accuracy": float(accuracy_score(y_true, y_pred)),
        "candidate_precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "candidate_recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "candidate_f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "candidate_log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
    }

    # Select the highest-probability candidate for each GPS observation.
    best_indices = data.groupby(
        ["trip_id", "observation_id"]
    )["positive_probability"].idxmax()

    best_candidates = data.loc[best_indices]

    top1_accuracy = float(
        (
            best_candidates["arc_id"].astype(int)
            == best_candidates["reference_arc_id"].astype(int)
        ).mean()
    )

    candidate_metrics["observation_top1_accuracy"] = top1_accuracy
    candidate_metrics["evaluated_observations"] = int(len(best_candidates))
    candidate_metrics["evaluated_candidates"] = int(len(data))

    return candidate_metrics


def main():
    print("PathSnap BEE - Student-Test Evaluation")
    print("=" * 50)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]

    print(f"Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    trip_directories = sorted(
        path for path in STUDENT_TEST_DIR.iterdir()
        if path.is_dir()
    )

    if len(trip_directories) != 10:
        raise ValueError(
            f"Expected 10 student-test trips, found {len(trip_directories)}"
        )

    all_rows = []
    trip_diagnostics = []

    for trip_dir in trip_directories:
        print(f"\nProcessing trip {trip_dir.name}...")

        rows, stats = process_trip(trip_dir)

        if not rows:
            print("WARNING: No usable candidate rows. Skipping.")
            trip_diagnostics.append(stats)
            continue

        trip_rows = flatten_rows(rows)
        dataframe = pd.DataFrame(trip_rows)

        required_columns = [
            "trip_id",
            "observation_id",
            "arc_id",
            "reference_arc_id",
            "label",
        ]

        missing = [
            column for column in required_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Trip {trip_dir.name} is missing required columns: {missing}"
            )

        missing_features = [
            column for column in feature_columns
            if column not in dataframe.columns
        ]

        if missing_features:
            raise ValueError(
                f"Trip {trip_dir.name} is missing model features: "
                f"{missing_features}"
            )

        # Prediction receives ONLY model features.
        X = dataframe[feature_columns]

        probabilities = model.predict_proba(X)

        classes = model[-1].classes_
        positive_index = list(classes).index(1)

        positive_probabilities = probabilities[:, positive_index]
        predictions = model.predict(X)

        dataframe["positive_probability"] = positive_probabilities
        dataframe["predicted_label"] = predictions

        all_rows.append(dataframe)
        trip_diagnostics.append(stats)

        print(f"Candidate rows: {len(dataframe)}")
        print(
            "Usable observations:",
            dataframe["observation_id"].nunique()
        )

    if not all_rows:
        raise RuntimeError("No student-test candidate rows were generated.")

    results = pd.concat(all_rows, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Overall metrics.
    overall_metrics = calculate_metrics(results)

    total_aligned = sum(
        int(item.get("route_aligned_observations", 0))
        for item in trip_diagnostics
    )

    total_misses = sum(
        int(item.get("candidate_misses", 0))
        for item in trip_diagnostics
    )

    overall_metrics["route_aligned_observations"] = total_aligned
    overall_metrics["candidate_misses"] = total_misses
    overall_metrics["candidate_miss_rate"] = (
        total_misses / total_aligned if total_aligned else None
    )

    # Per-trip metrics.
    per_trip_results = []

    for trip_id, trip_data in results.groupby("trip_id"):
        metrics = calculate_metrics(trip_data)
        metrics["trip_id"] = str(trip_id)
        per_trip_results.append(metrics)

    pd.DataFrame(per_trip_results).to_csv(
        OUTPUT_DIR / "student_test_metrics.csv",
        index=False,
    )

    # Save predictions separately from the reference labels.
    prediction_columns = [
        "trip_id",
        "observation_id",
        "arc_id",
        "positive_probability",
        "predicted_label",
    ]

    results[prediction_columns].to_csv(
        OUTPUT_DIR / "student_test_predictions.csv",
        index=False,
    )

    summary = {
        "evaluation": "student_test_development_evaluation",
        "model": "Random Forest",
        "trip_count_processed": int(results["trip_id"].nunique()),
        "features": feature_columns,
        "overall_metrics": overall_metrics,
        "trip_diagnostics": trip_diagnostics,
        "label_limitation": (
            "Reference labels use monotonic nearest-route geometric "
            "alignment. Metrics are development-stage results against "
            "approximate labels, not independent ground-truth validation."
        ),
    }

    with (OUTPUT_DIR / "student_test_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 50)
    print("STUDENT-TEST EVALUATION COMPLETE")
    print("=" * 50)

    for metric, value in overall_metrics.items():
        print(f"{metric}: {value}")

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


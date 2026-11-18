
import json
from pathlib import Path

import joblib

from src.models.models import make_model_registry
from src.models.train import (
    load_feature_data,
    prepare_training_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bee_training_candidates.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "models" / "artifacts"

FEATURE_COLUMNS = [
    "distance_m",
    "projection_fraction",
    "segment_length_m",
    "candidate_rank",
    "distance_to_best_m",
    "distance_margin_m",
]

TARGET_COLUMN = "label"
GROUP_COLUMN = "trip_id"
OBSERVATION_COLUMN = "observation_id"

RANDOM_FOREST_PARAMETERS = {
    "classifier__n_estimators": 200,
    "classifier__max_depth": 12,
    "classifier__min_samples_split": 7,
    "classifier__min_samples_leaf": 1,
    "classifier__max_features": "sqrt",
    "classifier__n_jobs": 2,
}


def main():
    print("PathSnap BEE - Final Model Training")
    print("=" * 45)

    dataframe = load_feature_data(DATA_PATH)

    X, y, groups, _ = prepare_training_data(
        dataframe=dataframe,
        target_column=TARGET_COLUMN,
        group_column=GROUP_COLUMN,
        observation_column=OBSERVATION_COLUMN,
        feature_columns=FEATURE_COLUMNS,
    )

    print(f"Candidate rows: {len(X)}")
    print(f"Training trips: {groups.nunique()}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Class distribution:\n{y.value_counts()}")

    registry = make_model_registry()

    model = registry["Random Forest"]

    # Apply the best completed Optuna trial parameters.
    model.set_params(**RANDOM_FOREST_PARAMETERS)

    print("\nSelected Random Forest parameters:")
    print(json.dumps(RANDOM_FOREST_PARAMETERS, indent=4))

    print("\nFitting on all available training trips...")
    model.fit(X, y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    artifact_path = OUTPUT_DIR / "pathsnap_random_forest.joblib"
    metadata_path = OUTPUT_DIR / "pathsnap_random_forest_metadata.json"

    # Export the complete fitted sklearn pipeline.
    joblib.dump(model, artifact_path)

    metadata = {
        "project": "PathSnap",
        "model_name": "Random Forest",
        "model_status": "provisional_final_model",
        "training_dataset": "data/processed/bee_training_candidates.csv",
        "training_rows": int(len(X)),
        "training_trip_count": int(groups.nunique()),
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "hyperparameters": RANDOM_FOREST_PARAMETERS,
        "random_state": 42,
        "label_method": (
            "Monotonic nearest-route geometric alignment; "
            "derived approximate labels."
        ),
        "validation_note": (
            "Hyperparameters selected from the best of three completed "
            "Random Forest Optuna trials on a 20-trip subset. "
            "Not yet confirmed by identical-split validation."
        ),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nMODEL EXPORT COMPLETE")
    print(f"Model: {artifact_path}")
    print(f"Metadata: {metadata_path}")
    print("\nThe exported artifact includes the fitted imputer and classifier.")


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def save_experiment_manifest(
    groups,
    feature_columns,
    n_splits,
    output_dir,
    experiment_name
):
    groups = pd.Series(groups).reset_index(drop=True)

    if groups.isna().any():
        raise ValueError("Trip IDs contain missing values.")

    if groups.nunique() < n_splits:
        raise ValueError("Not enough unique trips for requested folds.")

    splitter = GroupKFold(n_splits=n_splits)

    # Assign each trip to exactly one validation fold.
    dummy_X = np.zeros((len(groups), 1))
    fold_assignments = []

    for fold, (_, val_idx) in enumerate(
        splitter.split(dummy_X, groups=groups),
        start=1
    ):
        validation_trips = groups.iloc[val_idx].unique()

        for trip_id in validation_trips:
            fold_assignments.append({
                "trip_id": str(trip_id),
                "validation_fold": fold
            })

    manifest = pd.DataFrame(fold_assignments)

    if manifest["trip_id"].duplicated().any():
        raise RuntimeError("A trip appears in multiple validation folds.")

    if len(manifest) != groups.nunique():
        raise RuntimeError("Some trips are missing from the fold manifest.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / f"{experiment_name}_fold_manifest.csv"

    manifest.to_csv(manifest_path, index=False)

    metadata = {
        "experiment_name": experiment_name,
        "n_splits": n_splits,
        "unique_trip_count": int(groups.nunique()),
        "candidate_row_count": int(len(groups)),
        "feature_columns": list(feature_columns),
        "fold_manifest_file": manifest_path.name
    }

    metadata_path = output_dir / f"{experiment_name}_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Fold manifest saved: {manifest_path}")
    print(f"Metadata saved: {metadata_path}")

    return manifest, metadata

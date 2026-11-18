
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.models import make_model_registry
from src.models.cross_validation import run_group_cv
from src.models.experiment_tracking import save_experiment_manifest


def load_feature_data(file_path):
    """Load BIS features from CSV or Parquet."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        dataframe = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        dataframe = pd.read_parquet(path)
    else:
        raise ValueError("Only CSV and Parquet files are supported.")

    if dataframe.empty:
        raise ValueError("The feature dataset is empty.")

    return dataframe


def prepare_training_data(
    dataframe,
    target_column,
    group_column,
    observation_column,
    feature_columns
):
    """Validate the schema and separate features from metadata."""

    if not feature_columns:
        raise ValueError("Provide at least one feature column.")

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Duplicate feature column names were provided.")

    metadata_columns = {
        target_column,
        group_column,
        observation_column
    }

    if len(metadata_columns) != 3:
        raise ValueError(
            "Target, group and observation columns must be different."
        )

    if metadata_columns.intersection(feature_columns):
        raise ValueError(
            "Target and grouping metadata cannot be model features."
        )

    required = metadata_columns.union(feature_columns)
    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Required columns missing from dataset: {sorted(missing)}"
        )

    X = dataframe[feature_columns].copy()
    y = dataframe[target_column].copy()
    groups = dataframe[group_column].copy()
    observation_ids = dataframe[observation_column].copy()

    if y.isna().any() or groups.isna().any():
        raise ValueError("Target labels or trip IDs contain missing values.")

    if observation_ids.isna().any():
        raise ValueError("Observation IDs contain missing values.")

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            "Expected binary labels encoded as 0 and 1. "
            f"Found: {sorted(y.unique().tolist())}"
        )

    non_numeric = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Non-numeric features need preprocessing: {non_numeric}"
        )

    for column in X.select_dtypes(include=["bool"]).columns:
        X[column] = X[column].astype(int)

    # Let the model pipelines' SimpleImputer handle missing feature values.
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().all().any():
        all_missing = X.columns[X.isna().all()].tolist()
        raise ValueError(
            f"These features contain only missing values: {all_missing}"
        )

    return X, y, groups, observation_ids


def run_experiments(
    dataframe,
    target_column,
    group_column,
    observation_column,
    feature_columns,
    n_splits,
    output_dir
):
    X, y, groups, observation_ids = prepare_training_data(
        dataframe=dataframe,
        target_column=target_column,
        group_column=group_column,
        observation_column=observation_column,
        feature_columns=feature_columns
    )

    if groups.nunique() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique trips; "
            f"found {groups.nunique()}."
        )

    print("\nPathSnap BEE Baseline Experiments")
    print("=" * 45)
    print(f"Candidate rows: {len(X)}")
    print(f"Features: {feature_columns}")
    print(f"Unique trips: {groups.nunique()}")
    print(f"Class distribution:\n{y.value_counts().sort_index()}")

    registry = make_model_registry()
    all_results = []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_experiment_manifest(
    groups=groups,
    feature_columns=feature_columns,
    n_splits=n_splits,
    output_dir=output_dir,
    experiment_name="baseline"
    )

    for model_name, estimator in registry.items():
        print(f"\nRunning model: {model_name}")
        print("-" * 45)

        fold_results = run_group_cv(
            estimator=estimator,
            X=X,
            y=y,
            groups=groups,
            observation_ids=observation_ids,
            n_splits=n_splits
        )

        fold_results.insert(0, "model", model_name)
        all_results.append(fold_results)

    results = pd.concat(all_results, ignore_index=True)

    fold_path = output_dir / "baseline_fold_results.csv"
    results.to_csv(fold_path, index=False)

    metric_columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "log_loss",
        "top1_accuracy",
        "training_time",
        "inference_time"
    ]

    summary = (
        results.groupby("model")[metric_columns]
        .agg(["mean", "std"])
        .round(6)
    )

    summary_path = output_dir / "baseline_model_comparison.csv"
    summary.to_csv(summary_path)

    print("\nBaseline model comparison (mean and standard deviation):")
    print(summary)

    print(f"\nFold results saved: {fold_path}")
    print(f"Summary saved: {summary_path}")

    return results, summary


def main():
    parser = argparse.ArgumentParser(
        description="PathSnap BEE baseline model experiments"
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument(
        "--features",
        required=True,
        help="Comma-separated list of verified BIS feature columns"
    )
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument(
        "--output",
        default="experiments/bee"
    )

    args = parser.parse_args()

    feature_columns = [
        item.strip()
        for item in args.features.split(",")
        if item.strip()
    ]

    dataframe = load_feature_data(args.data)

    run_experiments(
        dataframe=dataframe,
        target_column=args.target,
        group_column=args.group,
        observation_column=args.observation,
        feature_columns=feature_columns,
        n_splits=args.splits,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()


import time
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.models.evaluate import (
    evaluate_classifier,
    observation_top1_accuracy
)


def run_group_cv(
    estimator,
    X,
    y,
    groups,
    observation_ids,
    n_splits=5
):
    """
    Group-aware cross-validation for PathSnap.

    All candidates belonging to the same trip stay
    within the same fold.

    Returns one metrics row per fold.
    """

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    observation_ids = pd.Series(
        observation_ids
    ).reset_index(drop=True)

    if not (
        len(X) == len(y) == len(groups) == len(observation_ids)
    ):
        raise ValueError("Input lengths do not match.")

    if X.isna().any().any():
        raise ValueError("Missing values found in X.")

    if y.isna().any() or groups.isna().any():
        raise ValueError("Missing labels or trip IDs.")

    if y.nunique() != 2:
        raise ValueError("Expected two target classes.")

    if groups.nunique() < n_splits:
        raise ValueError("Not enough unique trips for GroupKFold.")

    splitter = GroupKFold(n_splits=n_splits)

    results = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1
    ):

        # Explicitly verify no trip leakage.
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])

        if train_groups.intersection(val_groups):
            raise RuntimeError("Trip leakage detected!")

        model = clone(estimator)

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        if y_train.nunique() != 2:
            raise ValueError(
                f"Fold {fold}: training data has only one class."
            )

        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        start = time.perf_counter()

        predictions = model.predict(X_val)

        probabilities_all = model.predict_proba(X_val)

        inference_time = time.perf_counter() - start

        # Locate positive-class probability correctly.
        classes = list(model.classes_)
        positive_index = classes.index(1)

        probabilities = probabilities_all[:, positive_index]

        metrics = evaluate_classifier(
            y_val,
            predictions,
            probabilities
        )

        # Build observation-level validation data.
        validation_df = pd.DataFrame({
            "trip_id": groups.iloc[val_idx].values,
            "observation_id": observation_ids.iloc[val_idx].values,
            "label": y_val.values,
            "probability": probabilities
        })

        validation_df["observation_key"] = list(zip(
            validation_df["trip_id"],
            validation_df["observation_id"]
        ))

        top1 = observation_top1_accuracy(
        validation_df,
        observation_column="observation_key"
        )

        results.append({
            "fold": fold,
            **metrics,
            "top1_accuracy": top1,
            "training_time": training_time,
            "inference_time": inference_time,
            "train_trips": len(train_groups),
            "validation_trips": len(val_groups)
        })

        print(
            f"Fold {fold}/{n_splits} completed | "
            f"Macro F1: {metrics['f1_macro']:.4f} | "
            f"Top-1: {top1:.4f}"
        )

    return pd.DataFrame(results)

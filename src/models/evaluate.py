
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)


def evaluate_classifier(y_true, y_pred, y_proba=None):
    """
    Calculate binary candidate-classification metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths differ.")

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Expected binary labels 0 and 1.")

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        )
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)

        if len(y_proba) != len(y_true):
            raise ValueError("Probability length mismatch.")

        if not np.isfinite(y_proba).all():
            raise ValueError("Probabilities contain invalid values.")

        if np.any((y_proba < 0) | (y_proba > 1)):
            raise ValueError("Probabilities must be between 0 and 1.")

        results["log_loss"] = log_loss(
            y_true,
            np.column_stack([1 - y_proba, y_proba]),
            labels=[0, 1]
        )
    else:
        results["log_loss"] = np.nan

    return results


def observation_top1_accuracy(
    dataframe,
    probability_column="probability",
    label_column="label",
    observation_column="observation_id"
):
    """
    Select the highest-probability candidate for each
    GPS observation and calculate Top-1 accuracy.

    Assumes exactly one positive candidate per observation.
    """

    required = [
        probability_column,
        label_column,
        observation_column
    ]

    missing = set(required) - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if dataframe[required].isna().any().any():
        raise ValueError("Missing values in evaluation data.")

    if not dataframe[label_column].isin([0, 1]).all():
        raise ValueError("Labels must be binary.")

    probabilities = dataframe[probability_column].to_numpy(
        dtype=float
    )

    if not np.isfinite(probabilities).all():
        raise ValueError("Invalid probabilities.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    positive_counts = dataframe.groupby(
        observation_column
    )[label_column].sum()

    if not (positive_counts == 1).all():
        raise ValueError(
            "Expected exactly one positive candidate "
            "per observation. Verify BIS label generation."
        )

    selected_indices = dataframe.groupby(
        observation_column,
        sort=False
    )[probability_column].idxmax()

    selected = dataframe.loc[selected_indices]

    return float(selected[label_column].mean())

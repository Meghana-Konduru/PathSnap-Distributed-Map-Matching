
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.models.models import make_model_registry
from src.models.train import load_feature_data, prepare_training_data
from src.models.evaluate import evaluate_classifier
from src.models.experiment_tracking import save_experiment_manifest


RANDOM_STATE = 42


def suggest_parameters(trial, model_name):
    """Define Bayesian search spaces for the five registered models."""

    if model_name == "Logistic Regression":
        return {
            "classifier__C": trial.suggest_float(
                "classifier__C", 1e-3, 100.0, log=True
            )
        }

    if model_name == "KNN":
        return {
            "classifier__n_neighbors": trial.suggest_int(
                "classifier__n_neighbors", 3, 15
            ),
            "classifier__weights": trial.suggest_categorical(
                "classifier__weights", ["uniform", "distance"]
            ),
            "classifier__p": trial.suggest_categorical(
                "classifier__p", [1, 2]
            )
        }

    if model_name == "Decision Tree":
        return {
            "classifier__max_depth": trial.suggest_categorical(
                "classifier__max_depth", [None, 5, 8, 12, 16, 24]
            ),
            "classifier__min_samples_split": trial.suggest_int(
                "classifier__min_samples_split", 2, 20
            ),
            "classifier__min_samples_leaf": trial.suggest_int(
                "classifier__min_samples_leaf", 1, 10
            ),
            "classifier__class_weight": trial.suggest_categorical(
                "classifier__class_weight", [None, "balanced"]
            )
        }

    if model_name == "Random Forest":
        return {
            "classifier__n_estimators": trial.suggest_int(
                "classifier__n_estimators", 100, 500, step=50
            ),
            "classifier__max_depth": trial.suggest_categorical(
                "classifier__max_depth", [None, 8, 12, 16, 24]
            ),
            "classifier__min_samples_split": trial.suggest_int(
                "classifier__min_samples_split", 2, 10
            ),
            "classifier__min_samples_leaf": trial.suggest_int(
                "classifier__min_samples_leaf", 1, 5
            ),
            "classifier__max_features": trial.suggest_categorical(
                "classifier__max_features", ["sqrt", "log2", None]
            )
        }

    if model_name == "HistGradientBoosting":
        return {
            "classifier__max_iter": trial.suggest_int(
                "classifier__max_iter", 100, 400, step=50
            ),
            "classifier__learning_rate": trial.suggest_float(
                "classifier__learning_rate", 0.02, 0.2, log=True
            ),
            "classifier__max_leaf_nodes": trial.suggest_int(
                "classifier__max_leaf_nodes", 7, 31
            ),
            "classifier__l2_regularization": trial.suggest_float(
                "classifier__l2_regularization", 1e-4, 10.0, log=True
            )
        }

    raise ValueError(f"Unknown model: {model_name}")


def objective_factory(
    base_estimator,
    model_name,
    X,
    y,
    groups,
    n_splits
):
    """Create an Optuna objective using trip-grouped CV."""

    splitter = GroupKFold(n_splits=n_splits)

    def objective(trial):
        parameters = suggest_parameters(trial, model_name)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(X, y, groups), start=1
        ):
            train_groups = set(groups.iloc[train_idx])
            val_groups = set(groups.iloc[val_idx])

            if train_groups.intersection(val_groups):
                raise RuntimeError("Trip leakage detected.")

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            if y_train.nunique() != 2:
                raise ValueError(
                    f"Fold {fold} training data has only one class."
                )

            estimator = clone(base_estimator)
            estimator.set_params(**parameters)
            estimator.fit(X_train, y_train)

            predictions = estimator.predict(X_val)

            metrics = evaluate_classifier(
                y_true=y_val,
                y_pred=predictions
            )

            fold_scores.append(metrics["f1_macro"])

        mean_score = float(np.mean(fold_scores))
        trial.set_user_attr("fold_macro_f1", fold_scores)

        return mean_score

    return objective


def run_optimization(
    dataframe,
    model_name,
    target_column,
    group_column,
    observation_column,
    feature_columns,
    n_splits,
    n_trials,
    output_dir
):
    X, y, groups, _ = prepare_training_data(
        dataframe=dataframe,
        target_column=target_column,
        group_column=group_column,
        observation_column=observation_column,
        feature_columns=feature_columns
    )

    if groups.nunique() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique trips."
        )

    registry = make_model_registry()

    if model_name not in registry:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(registry.keys())}"
        )

    output_dir = Path(output_dir)

    safe_model_name = (
    model_name.lower()
    .replace(" ", "_")
    .replace("-", "_")
    )

    save_experiment_manifest(
    groups=groups,
    feature_columns=feature_columns,
    n_splits=n_splits,
    output_dir=output_dir,
    experiment_name=f"{safe_model_name}_optuna"
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )

    objective = objective_factory(
        base_estimator=registry[model_name],
        model_name=model_name,
        X=X,
        y=y,
        groups=groups,
        n_splits=n_splits
    )

    print(f"\nOptimising: {model_name}")
    print(f"Trials: {n_trials}")
    print(f"CV folds: {n_splits}")
    print(f"Training trips: {groups.nunique()}")

    study.optimize(objective, n_trials=n_trials)

    trials_path = output_dir / "optuna_trials.csv"
    best_path = output_dir / "best_hyperparameters.json"

    study.trials_dataframe().to_csv(
        trials_path,
        index=False
    )

    best_result = {
        "model": model_name,
        "best_macro_f1_cv": study.best_value,
        "best_parameters": study.best_params,
        "n_trials": n_trials,
        "n_splits": n_splits,
        "random_state": RANDOM_STATE,
        "training_trip_count": int(groups.nunique()),
        "feature_columns": feature_columns
    }

    with open(best_path, "w", encoding="utf-8") as file:
        json.dump(best_result, file, indent=4)

    print("\nOptimisation completed.")
    print(f"Best mean CV Macro F1: {study.best_value:.6f}")
    print("Best parameters:")
    print(json.dumps(study.best_params, indent=4))
    print(f"\nTrials saved: {trials_path}")
    print(f"Best parameters saved: {best_path}")

    return study



def main():
    parser = argparse.ArgumentParser(
        description="PathSnap Bayesian hyperparameter optimisation"
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--trials", type=int, default=8)

    parser.add_argument(
        "--max-trips",
        type=int,
        default=20,
        help="Number of complete training trips used for optimisation"
    )

    parser.add_argument(
        "--output",
        default="experiments/bee/optuna"
    )

    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1.")

    if args.splits < 2:
        parser.error("--splits must be at least 2.")

    if args.max_trips < args.splits:
        parser.error("--max-trips must be >= --splits.")

    feature_columns = [
        item.strip()
        for item in args.features.split(",")
        if item.strip()
    ]

    dataframe = load_feature_data(args.data)

    # Select complete trips reproducibly.
    available_trips = sorted(dataframe[args.group].unique())

    if args.max_trips > len(available_trips):
        parser.error(
            f"Requested {args.max_trips} trips, "
            f"but only {len(available_trips)} are available."
        )

    rng = np.random.RandomState(RANDOM_STATE)

    selected_trips = rng.choice(
        available_trips,
        size=args.max_trips,
        replace=False
    )

    dataframe = dataframe[
        dataframe[args.group].isin(selected_trips)
    ].copy()

    print("\nOPTIMISATION DATA")
    print("-----------------")
    print(f"Selected trips: {len(selected_trips)}")
    print(f"Candidate rows: {len(dataframe)}")
    print(f"Model: {args.model}")

    run_optimization(
        dataframe=dataframe,
        model_name=args.model,
        target_column=args.target,
        group_column=args.group,
        observation_column=args.observation,
        feature_columns=feature_columns,
        n_splits=args.splits,
        n_trials=args.trials,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()

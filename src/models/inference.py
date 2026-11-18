
import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "artifacts" / "pathsnap_random_forest.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT / "models" / "artifacts" / "pathsnap_random_forest_metadata.json"
)


class PathSnapPredictor:

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

        with METADATA_PATH.open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.feature_columns = self.metadata["feature_columns"]
        self.model = joblib.load(MODEL_PATH)

        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Model does not support predict_proba.")

        print("PathSnap Random Forest loaded successfully.")

    def predict_observation(self, observation):

        candidates = observation.get("candidates", [])

        if not candidates:
            return {
                "timestamp": observation.get("timestamp"),
                "longitude": observation.get("longitude"),
                "latitude": observation.get("latitude"),
                "candidates": [],
            }

        feature_rows = []

        for candidate in candidates:
            features = candidate.get("features", candidate)

            missing = [
                column for column in self.feature_columns
                if column not in features
            ]

            if missing:
                raise ValueError(
                    f"Missing candidate features: {missing}"
                )

            feature_rows.append({
                column: features[column]
                for column in self.feature_columns
            })

        X = pd.DataFrame(
            feature_rows,
            columns=self.feature_columns,
        )

        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)

        classes = list(self.model.classes_)
        positive_index = classes.index(1)

        results = []

        for index, candidate in enumerate(candidates):
            result = dict(candidate)
            result["predicted_label"] = int(predictions[index])
            result["positive_probability"] = float(
                probabilities[index, positive_index]
            )
            results.append(result)

        results.sort(
            key=lambda item: item["positive_probability"],
            reverse=True,
        )

        return {
            "timestamp": observation.get("timestamp"),
            "longitude": observation.get("longitude"),
            "latitude": observation.get("latitude"),
            "candidates": results,
        }


if __name__ == "__main__":
    predictor = PathSnapPredictor()
    print("Inference interface loading test passed.")
    print("Feature columns:", predictor.feature_columns)

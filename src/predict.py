"""
Prediction module for MLOps Anomaly Detection.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
from datetime import datetime

import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging, load_config, get_project_root


class AnomalyPredictor:
    """Predictor class for anomaly detection inference."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        config: Optional[dict] = None
    ):
        self.logger = setup_logging("predictor")
        self.config = config or load_config()

        project_root = get_project_root()

        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = project_root / self.config["model"]["output_path"] / self.config["model"]["filename"]

        if scaler_path:
            self.scaler_path = Path(scaler_path)
        else:
            self.scaler_path = project_root / self.config["data"]["processed_path"] / "scaler.joblib"

        self.model = None
        self.scaler = None
        self.n_features = self.config["features"]["n_features"]

        self._load_model()
        self._load_scaler()

    def _load_model(self) -> None:
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.model = joblib.load(self.model_path)
        self.logger.info(f"Model loaded from {self.model_path}")

    def _load_scaler(self) -> None:
        """Load the fitted scaler."""
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
            self.logger.info(f"Scaler loaded from {self.scaler_path}")
        else:
            self.logger.warning(f"Scaler not found at {self.scaler_path}")

    def predict(self, features: Union[List[float], np.ndarray]) -> Tuple[int, float]:
        """Make a single prediction."""
        features = np.array(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {features.shape[1]}")

        if self.scaler is not None:
            features = self.scaler.transform(features)

        prediction_raw = self.model.predict(features)[0]
        score = float(self.model.decision_function(features)[0])
        prediction = 1 if prediction_raw == -1 else 0

        return prediction, score

    def predict_batch(self, features_batch: Union[List[List[float]], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Make batch predictions."""
        X = np.array(features_batch, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        if self.scaler is not None:
            X = self.scaler.transform(X)

        predictions_raw = self.model.predict(X)
        predictions = (predictions_raw == -1).astype(int)
        scores = self.model.decision_function(X)

        return predictions, scores

    def predict_with_confidence(self, features: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """Make prediction with confidence information."""
        prediction, score = self.predict(features)
        confidence = min(abs(score) * 50, 99.9)

        return {
            "prediction": prediction,
            "is_anomaly": prediction == 1,
            "label": "Anomaly" if prediction == 1 else "Normal",
            "anomaly_score": score,
            "confidence": round(confidence, 2),
            "timestamp": datetime.now().isoformat()
        }


def run_demo():
    """Run prediction demo."""
    logger = setup_logging("demo")
    logger.info("Running prediction demo...")

    try:
        predictor = AnomalyPredictor()

        # Normal sample
        normal_sample = [0.1, 0.2, 0.3, 0.1, -0.1, 0.2, 0.1, 0.3, -0.2, 0.1]
        result_normal = predictor.predict_with_confidence(normal_sample)
        logger.info(f"Normal sample: {result_normal['label']} (score: {result_normal['anomaly_score']:.4f})")

        # Anomaly sample
        anomaly_sample = [10.0, -8.0, 15.0, -12.0, 20.0, -15.0, 10.0, -10.0, 8.0, -7.0]
        result_anomaly = predictor.predict_with_confidence(anomaly_sample)
        logger.info(f"Anomaly sample: {result_anomaly['label']} (score: {result_anomaly['anomaly_score']:.4f})")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please train the model first: python src/train.py")


if __name__ == "__main__":
    run_demo()

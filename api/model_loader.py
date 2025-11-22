"""
Model loader module for FastAPI application.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, get_project_root, setup_logging


class ModelLoader:
    """Model loader and manager for inference."""

    _instance = None
    _model = None
    _scaler = None
    _model_info = None
    _loaded_at = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.logger = setup_logging("model_loader")
        self.config = load_config()
        self.project_root = get_project_root()

        self.model_path = self._get_model_path()
        self.scaler_path = self._get_scaler_path()
        self._initialized = True

    def _get_model_path(self) -> Path:
        """Get model path from config or environment."""
        env_path = os.environ.get("MODEL_PATH")
        if env_path and Path(env_path).exists():
            return Path(env_path)
        return self.project_root / self.config["model"]["output_path"] / self.config["model"]["filename"]

    def _get_scaler_path(self) -> Path:
        """Get scaler path from config or environment."""
        env_path = os.environ.get("SCALER_PATH")
        if env_path and Path(env_path).exists():
            return Path(env_path)
        return self.project_root / self.config["data"]["processed_path"] / "scaler.joblib"

    def load_model(self, force_reload: bool = False) -> bool:
        """Load the trained model."""
        if self._model is not None and not force_reload:
            return True

        try:
            if not self.model_path.exists():
                self.logger.error(f"Model file not found: {self.model_path}")
                return False

            self.logger.info(f"Loading model from {self.model_path}")
            self._model = joblib.load(self.model_path)
            self._loaded_at = datetime.now()

            self._model_info = {
                "model_name": "IsolationForest",
                "model_path": str(self.model_path),
                "loaded_at": self._loaded_at.isoformat(),
                "n_features": getattr(self._model, 'n_features_in_', 10)
            }

            self.logger.info("Model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def load_scaler(self, force_reload: bool = False) -> bool:
        """Load the fitted scaler."""
        if self._scaler is not None and not force_reload:
            return True

        try:
            if not self.scaler_path.exists():
                self.logger.warning(f"Scaler not found: {self.scaler_path}")
                return True

            self.logger.info(f"Loading scaler from {self.scaler_path}")
            self._scaler = joblib.load(self.scaler_path)
            self.logger.info("Scaler loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load scaler: {e}")
            return False

    def initialize(self) -> bool:
        """Initialize model loader."""
        model_loaded = self.load_model()
        self.load_scaler()
        return model_loaded

    @property
    def model(self):
        return self._model

    @property
    def scaler(self):
        return self._scaler

    @property
    def model_info(self):
        return self._model_info

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """Make a prediction."""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self._scaler is not None:
            features = self._scaler.transform(features)

        prediction_raw = self._model.predict(features)[0]
        score = float(self._model.decision_function(features)[0])
        prediction = 1 if prediction_raw == -1 else 0

        return prediction, score

    def predict_batch(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make batch predictions."""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self._scaler is not None:
            features = self._scaler.transform(features)

        predictions_raw = self._model.predict(features)
        scores = self._model.decision_function(features)
        predictions = (predictions_raw == -1).astype(int)

        return predictions, scores


model_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    return model_loader

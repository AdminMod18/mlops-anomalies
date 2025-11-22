"""
Utility functions for MLOps Anomaly Detection project.
"""

import os
import sys
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "requirements.txt").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        logging.warning(f"Config file not found at {config_path}, using defaults")
        return get_default_config()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        "data": {
            "raw_path": "data/raw",
            "processed_path": "data/processed",
            "raw_filename": "anomaly_data.csv",
            "n_samples": 10000,
            "anomaly_ratio": 0.1,
            "test_size": 0.2,
            "random_state": 42
        },
        "features": {
            "n_features": 10,
            "feature_names": [f"feature_{i}" for i in range(1, 11)]
        },
        "model": {
            "name": "IsolationForest",
            "output_path": "models",
            "filename": "isolation_forest_model.joblib",
            "params": {
                "n_estimators": 100,
                "contamination": 0.1,
                "max_samples": "auto",
                "random_state": 42,
                "n_jobs": -1
            }
        },
        "mlflow": {
            "tracking_uri": "sqlite:///mlruns/mlflow.db",
            "experiment_name": "anomaly_detection",
            "registered_model_name": "isolation_forest_anomaly_detector"
        }
    }


def setup_logging(
    name: str = "mlops_anomaly",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def generate_synthetic_data(
    output_path: str,
    n_samples: int = 10000,
    n_features: int = 10,
    anomaly_ratio: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate synthetic dataset for anomaly detection."""
    logger = setup_logging("data_generator")
    logger.info(f"Generating synthetic data: {n_samples} samples, {anomaly_ratio*100}% anomalies")

    np.random.seed(random_state)

    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # Generate normal samples
    normal_data = np.zeros((n_normal, n_features))
    for i in range(n_features):
        mean = np.random.uniform(-2, 2)
        std = np.random.uniform(0.5, 2.0)
        normal_data[:, i] = np.random.normal(mean, std, n_normal)

    # Generate anomalies
    anomaly_data = np.zeros((n_anomalies, n_features))
    for i in range(n_anomalies):
        anomaly_data[i] = np.random.normal(0, 1, n_features)
        n_extreme = np.random.randint(2, n_features // 2 + 1)
        extreme_features = np.random.choice(n_features, n_extreme, replace=False)
        for feat in extreme_features:
            direction = np.random.choice([-1, 1])
            magnitude = np.random.uniform(4, 8)
            anomaly_data[i, feat] = direction * magnitude

    # Combine and shuffle
    data = np.vstack([normal_data, anomaly_data])
    labels = np.array([0] * n_normal + [1] * n_anomalies)

    shuffle_idx = np.random.permutation(n_samples)
    data = data[shuffle_idx]
    labels = labels[shuffle_idx]

    # Create DataFrame
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_names)
    df['is_anomaly'] = labels

    df['timestamp'] = pd.date_range(
        start=datetime.now() - pd.Timedelta(days=30),
        periods=n_samples,
        freq='3min'
    )
    df['sample_id'] = [f"sample_{i:06d}" for i in range(n_samples)]

    cols = ['sample_id', 'timestamp'] + feature_names + ['is_anomaly']
    df = df[cols]

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Data saved to {output_path}")
    logger.info(f"Normal samples: {n_normal}, Anomalies: {n_anomalies}")

    return df


def validate_features(features: List[float], expected_length: int = 10) -> bool:
    """Validate input features for prediction."""
    if not isinstance(features, (list, np.ndarray)):
        return False
    if len(features) != expected_length:
        return False
    try:
        features = np.array(features, dtype=np.float64)
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return False
    except (ValueError, TypeError):
        return False
    return True


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate classification metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
    }

    if y_scores is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, -y_scores))
        except ValueError:
            metrics["roc_auc"] = 0.0

    cm = confusion_matrix(y_true, y_pred)
    metrics["true_negatives"] = int(cm[0, 0]) if cm.shape[0] > 0 else 0
    metrics["false_positives"] = int(cm[0, 1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
    metrics["false_negatives"] = int(cm[1, 0]) if cm.shape[0] > 1 else 0
    metrics["true_positives"] = int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

    return metrics


def save_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """Save metrics to JSON file."""
    import json
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    project_root = get_project_root()
    output_path = project_root / "data" / "raw" / "anomaly_data.csv"
    df = generate_synthetic_data(str(output_path), n_samples=10000, anomaly_ratio=0.1)
    print(f"\nGenerated {len(df)} samples")
    print(f"Saved to: {output_path}")

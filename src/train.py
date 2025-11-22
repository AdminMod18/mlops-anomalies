"""
Model training module for MLOps Anomaly Detection.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_curve, auc

import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging, load_config, get_project_root, calculate_metrics, save_metrics


class AnomalyModelTrainer:
    """Trainer class for Isolation Forest anomaly detection model."""

    def __init__(
        self,
        processed_data_path: Optional[str] = None,
        model_output_path: Optional[str] = None,
        config: Optional[dict] = None
    ):
        self.logger = setup_logging("trainer")
        self.config = config or load_config()

        project_root = get_project_root()

        if processed_data_path:
            self.processed_data_path = Path(processed_data_path)
        else:
            self.processed_data_path = project_root / self.config["data"]["processed_path"]

        if model_output_path:
            self.model_output_path = Path(model_output_path)
        else:
            self.model_output_path = project_root / self.config["model"]["output_path"]

        self.model = None
        self.model_params = self.config["model"]["params"]
        self.mlflow_config = self.config["mlflow"]
        self.model_output_path.mkdir(parents=True, exist_ok=True)
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        try:
            tracking_uri = self.mlflow_config.get("tracking_uri", "sqlite:///mlruns/mlflow.db")
            if "sqlite" in tracking_uri:
                Path("mlruns").mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.mlflow_config.get("experiment_name", "anomaly_detection"))
            self.logger.info(f"MLflow tracking URI: {tracking_uri}")
        except Exception as e:
            self.logger.warning(f"MLflow setup warning: {e}")

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load processed training data."""
        self.logger.info(f"Loading data from {self.processed_data_path}")

        X_train = np.load(self.processed_data_path / "X_train.npy")
        X_test = np.load(self.processed_data_path / "X_test.npy")
        y_train = np.load(self.processed_data_path / "y_train.npy")
        y_test = np.load(self.processed_data_path / "y_test.npy")

        self.logger.info(f"Loaded - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def create_model(self) -> IsolationForest:
        """Create Isolation Forest model."""
        self.logger.info("Creating Isolation Forest model")
        return IsolationForest(
            n_estimators=self.model_params.get("n_estimators", 100),
            contamination=self.model_params.get("contamination", 0.1),
            max_samples=self.model_params.get("max_samples", "auto"),
            random_state=self.model_params.get("random_state", 42),
            n_jobs=self.model_params.get("n_jobs", -1),
            bootstrap=self.model_params.get("bootstrap", False),
            max_features=self.model_params.get("max_features", 1.0)
        )

    def train_model(self, X_train: np.ndarray) -> IsolationForest:
        """Train the Isolation Forest model."""
        self.logger.info("Training model...")
        self.model = self.create_model()
        start_time = datetime.now()
        self.model.fit(X_train)
        training_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Model trained in {training_time:.2f} seconds")
        return self.model

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the trained model."""
        self.logger.info("Evaluating model...")

        y_pred_raw = self.model.predict(X_test)
        y_pred = (y_pred_raw == -1).astype(int)
        y_scores = self.model.decision_function(X_test)

        metrics = calculate_metrics(y_test, y_pred, y_scores)
        metrics["model_name"] = "IsolationForest"
        metrics["n_estimators"] = self.model_params.get("n_estimators", 100)
        metrics["contamination"] = self.model_params.get("contamination", 0.1)
        metrics["test_samples"] = len(y_test)

        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall: {metrics['recall']:.4f}")
        self.logger.info(f"F1-Score: {metrics['f1_score']:.4f}")

        return metrics

    def generate_plots(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, str]:
        """Generate evaluation plots."""
        self.logger.info("Generating plots...")

        plots_dir = get_project_root() / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {}

        y_pred_raw = self.model.predict(X_test)
        y_pred = (y_pred_raw == -1).astype(int)
        y_scores = -self.model.decision_function(X_test)

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Normal', 'Anomaly'])
        plt.yticks([0, 1], ['Normal', 'Anomaly'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = plots_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        plot_paths["confusion_matrix"] = str(cm_path)

        # ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = plots_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=150)
        plt.close()
        plot_paths["roc_curve"] = str(roc_path)

        self.logger.info(f"Plots saved to {plots_dir}")
        return plot_paths

    def save_model(self) -> str:
        """Save trained model to disk."""
        model_filename = self.config["model"]["filename"]
        model_path = self.model_output_path / model_filename
        joblib.dump(self.model, model_path)
        self.logger.info(f"Model saved to {model_path}")
        return str(model_path)

    def log_to_mlflow(self, metrics: Dict[str, Any], plot_paths: Dict[str, str]) -> Optional[str]:
        """Log training run to MLflow."""
        self.logger.info("Logging to MLflow...")
        try:
            with mlflow.start_run() as run:
                mlflow.log_params(self.model_params)
                mlflow.log_metric("accuracy", metrics["accuracy"])
                mlflow.log_metric("precision", metrics["precision"])
                mlflow.log_metric("recall", metrics["recall"])
                mlflow.log_metric("f1_score", metrics["f1_score"])
                if "roc_auc" in metrics:
                    mlflow.log_metric("roc_auc", metrics["roc_auc"])

                mlflow.sklearn.log_model(self.model, "model")

                for plot_name, plot_path in plot_paths.items():
                    if os.path.exists(plot_path):
                        mlflow.log_artifact(plot_path, "plots")

                self.logger.info(f"MLflow run ID: {run.info.run_id}")
                return run.info.run_id
        except Exception as e:
            self.logger.warning(f"MLflow logging failed: {e}")
            return None

    def train(self) -> Tuple[IsolationForest, Dict[str, Any]]:
        """Run full training pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING PIPELINE")

        X_train, X_test, y_train, y_test = self.load_data()
        self.train_model(X_train)
        metrics = self.evaluate_model(X_test, y_test)
        plot_paths = self.generate_plots(X_test, y_test)
        model_path = self.save_model()
        metrics["model_path"] = model_path

        save_metrics(metrics, str(get_project_root() / "metrics.json"))
        run_id = self.log_to_mlflow(metrics, plot_paths)
        if run_id:
            metrics["mlflow_run_id"] = run_id

        self.logger.info("TRAINING PIPELINE COMPLETED!")
        return self.model, metrics


if __name__ == "__main__":
    trainer = AnomalyModelTrainer()
    try:
        model, metrics = trainer.train()
        print(f"\nModel: {metrics['model_name']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Model saved to: {metrics['model_path']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run preprocessing first: python src/preprocess.py")
        sys.exit(1)

"""
Data preprocessing module for MLOps Anomaly Detection.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging, load_config, get_project_root


class DataPreprocessor:
    """Data preprocessing pipeline for anomaly detection."""

    def __init__(
        self,
        raw_data_path: Optional[str] = None,
        processed_data_path: Optional[str] = None,
        config: Optional[dict] = None
    ):
        self.logger = setup_logging("preprocessor")
        self.config = config or load_config()

        project_root = get_project_root()

        if raw_data_path:
            self.raw_data_path = Path(raw_data_path)
        else:
            self.raw_data_path = project_root / self.config["data"]["raw_path"] / self.config["data"]["raw_filename"]

        if processed_data_path:
            self.processed_data_path = Path(processed_data_path)
        else:
            self.processed_data_path = project_root / self.config["data"]["processed_path"]

        self.scaler = None
        self.imputer = None
        self.feature_names = self.config["features"]["feature_names"]
        self.test_size = self.config["data"]["test_size"]
        self.random_state = self.config["data"]["random_state"]

    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        self.logger.info(f"Loading data from {self.raw_data_path}")

        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.raw_data_path}")

        df = pd.read_csv(self.raw_data_path)
        self.logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data structure and quality."""
        self.logger.info("Validating data...")

        required_cols = self.feature_names + ['is_anomaly']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        if len(df) == 0:
            raise ValueError("DataFrame is empty")

        self.logger.info("Data validation passed")
        return True

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data."""
        self.logger.info("Cleaning data...")

        df_clean = df.copy()
        columns_to_keep = self.feature_names + ['is_anomaly']
        df_clean = df_clean[columns_to_keep]

        if df_clean[self.feature_names].isnull().sum().sum() > 0:
            self.imputer = SimpleImputer(strategy='median')
            df_clean[self.feature_names] = self.imputer.fit_transform(df_clean[self.feature_names])

        n_before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()

        self.logger.info(f"Cleaned data: {len(df_clean)} samples")
        return df_clean

    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using RobustScaler."""
        self.logger.info("Scaling features")
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def split_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        self.logger.info(f"Splitting data (test_size={self.test_size})")

        X = df[self.feature_names].values
        y = df['is_anomaly'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        self.logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def save_processed_data(self, X_train, X_test, y_train, y_test) -> None:
        """Save processed data to disk."""
        self.logger.info(f"Saving processed data to {self.processed_data_path}")

        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        np.save(self.processed_data_path / "X_train.npy", X_train)
        np.save(self.processed_data_path / "X_test.npy", X_test)
        np.save(self.processed_data_path / "y_train.npy", y_train)
        np.save(self.processed_data_path / "y_test.npy", y_test)

        if self.scaler:
            joblib.dump(self.scaler, self.processed_data_path / "scaler.joblib")

        self.logger.info("Processed data saved successfully")

    def run_preprocessing(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run full preprocessing pipeline."""
        self.logger.info("=" * 50)
        self.logger.info("Starting preprocessing pipeline")

        df = self.load_data()
        self.validate_data(df)
        df_clean = self.clean_data(df)
        X_train, X_test, y_train, y_test = self.split_data(df_clean)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        self.save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test)

        self.logger.info("Preprocessing pipeline completed!")
        return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    try:
        X_train, X_test, y_train, y_test = preprocessor.run_preprocessing()
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run data generation first: python src/utils.py")
        sys.exit(1)

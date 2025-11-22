"""Unit tests for model training and prediction."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import generate_synthetic_data
from src.preprocess import DataPreprocessor
from src.train import AnomalyModelTrainer


class TestDataGeneration:
    def test_generate_synthetic_data(self, tmp_path):
        output_path = tmp_path / "test_data.csv"
        df = generate_synthetic_data(str(output_path), n_samples=1000, anomaly_ratio=0.1)

        assert os.path.exists(output_path)
        assert len(df) == 1000
        assert 'is_anomaly' in df.columns

    def test_data_columns(self, tmp_path):
        output_path = tmp_path / "test_data.csv"
        df = generate_synthetic_data(str(output_path), n_samples=100)

        expected_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4',
                          'feature_5', 'feature_6', 'feature_7', 'feature_8',
                          'feature_9', 'feature_10', 'is_anomaly']

        for col in expected_columns:
            assert col in df.columns


class TestPreprocessing:
    @pytest.fixture
    def sample_data(self, tmp_path):
        output_path = tmp_path / "sample_data.csv"
        generate_synthetic_data(str(output_path), n_samples=500)
        return str(output_path)

    def test_preprocessing_pipeline(self, sample_data, tmp_path):
        preprocessor = DataPreprocessor(
            raw_data_path=sample_data,
            processed_data_path=str(tmp_path)
        )
        X_train, X_test, y_train, y_test = preprocessor.run_preprocessing()

        assert X_train is not None
        assert X_test is not None
        assert len(X_train) > len(X_test)


class TestModelTraining:
    @pytest.fixture
    def training_data(self, tmp_path):
        data_path = tmp_path / "data.csv"
        generate_synthetic_data(str(data_path), n_samples=500)

        preprocessor = DataPreprocessor(
            raw_data_path=str(data_path),
            processed_data_path=str(tmp_path)
        )
        preprocessor.run_preprocessing()
        return str(tmp_path)

    def test_model_training(self, training_data, tmp_path):
        trainer = AnomalyModelTrainer(
            processed_data_path=training_data,
            model_output_path=str(tmp_path)
        )

        model, metrics = trainer.train()

        assert model is not None
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

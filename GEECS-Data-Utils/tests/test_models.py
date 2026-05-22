"""Tests for RegressionTrainer and ModelArtifact."""

import numpy as np
import pytest

from geecs_data_utils.modeling.ml.models import ModelArtifact, RegressionTrainer


class TestRegressionTrainer:
    """Tests for RegressionTrainer."""

    def test_fit_linear(self, sample_df, feature_columns):
        """Linear regression fits and returns an artifact."""
        trainer = RegressionTrainer(model="linear")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )
        assert isinstance(artifact, ModelArtifact)
        assert artifact.metrics.r2 is not None
        assert artifact.metrics.r2 > 0.9  # strong linear relationship

    def test_fit_ridge(self, sample_df, feature_columns):
        """Ridge regression fits and returns an artifact."""
        trainer = RegressionTrainer(model="ridge", model_params={"alpha": 1.0})
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )
        assert artifact.metrics.r2 > 0.8
        assert artifact.metadata.model_type == "ridge"

    def test_fit_elasticnet(self, sample_df, feature_columns):
        """ElasticNet fits and returns an artifact."""
        trainer = RegressionTrainer(model="elasticnet", model_params={"alpha": 0.1})
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )
        assert artifact.metrics.r2 > 0.5

    def test_cross_validation(self, sample_df, feature_columns):
        """Cross-validation metrics are computed when cv is set."""
        trainer = RegressionTrainer(model="ridge")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
            cv=3,
        )
        assert artifact.metrics.cv_r2_mean is not None
        assert artifact.metrics.cv_r2_std is not None

    def test_metrics_populated(self, sample_df, feature_columns):
        """All basic metrics are populated."""
        trainer = RegressionTrainer(model="linear")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )
        assert artifact.metrics.mae is not None
        assert artifact.metrics.rmse is not None
        assert artifact.metrics.mae >= 0
        assert artifact.metrics.rmse >= 0

    def test_schema_populated(self, sample_df, feature_columns):
        """Feature schema is correctly populated."""
        trainer = RegressionTrainer(model="linear")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )
        assert artifact.feature_schema.feature_names == feature_columns
        assert artifact.feature_schema.target_name == "charge"

    def test_unknown_model_raises(self):
        """Unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            RegressionTrainer(model="xgboost")

    def test_predict(self, sample_df, feature_columns):
        """Artifact.predict works on new data."""
        trainer = RegressionTrainer(model="ridge")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )
        preds = artifact.predict(sample_df[feature_columns])
        assert len(preds) == len(sample_df)
        assert isinstance(preds, np.ndarray)

    def test_predict_schema_mismatch(self, sample_df, feature_columns):
        """Prediction with wrong columns raises ValueError."""
        trainer = RegressionTrainer(model="ridge")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )
        with pytest.raises(ValueError, match="Missing features"):
            artifact.predict(sample_df[["feature_a"]])

    def test_numpy_input(self, sample_df, feature_columns):
        """Training with numpy arrays works."""
        trainer = RegressionTrainer(model="linear")
        X = sample_df[feature_columns].values
        y = sample_df["charge"].values
        artifact = trainer.fit(X, y, target_name="charge")
        assert artifact.metrics.r2 > 0.9
        assert artifact.feature_schema.feature_names[0] == "feature_0"

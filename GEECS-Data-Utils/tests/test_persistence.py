"""Tests for model persistence (save/load roundtrip)."""

import json

import numpy as np
import pytest

from geecs_data_utils.modeling.ml.models import RegressionTrainer
from geecs_data_utils.modeling.ml.persistence import (
    load_model_artifact,
    save_model_artifact,
)


class TestPersistence:
    """Tests for save_model_artifact and load_model_artifact."""

    def test_roundtrip(self, sample_df, feature_columns, tmp_path):
        """Save then load produces equivalent predictions."""
        trainer = RegressionTrainer(model="ridge")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
            cv=3,
        )

        # Save
        artifact_dir = tmp_path / "test_model"
        save_model_artifact(artifact, artifact_dir)

        # Verify directory contents
        assert (artifact_dir / "model.joblib").exists()
        assert (artifact_dir / "metadata.json").exists()
        assert (artifact_dir / "feature_schema.json").exists()
        assert (artifact_dir / "metrics.json").exists()

        # Load
        loaded = load_model_artifact(artifact_dir)

        # Predictions match
        original_preds = artifact.predict(sample_df[feature_columns])
        loaded_preds = loaded.predict(sample_df[feature_columns])
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_metadata_preserved(self, sample_df, feature_columns, tmp_path):
        """Metadata is preserved through save/load."""
        trainer = RegressionTrainer(model="ridge", model_params={"alpha": 2.0})
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )

        artifact_dir = tmp_path / "test_model"
        save_model_artifact(artifact, artifact_dir)
        loaded = load_model_artifact(artifact_dir)

        assert loaded.metadata.model_type == "ridge"
        assert loaded.metadata.target_name == "charge"
        assert loaded.metadata.feature_count == 3
        assert loaded.metadata.training_rows == len(sample_df)

    def test_schema_preserved(self, sample_df, feature_columns, tmp_path):
        """Feature schema is preserved through save/load."""
        trainer = RegressionTrainer(model="linear")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )

        artifact_dir = tmp_path / "test_model"
        save_model_artifact(artifact, artifact_dir)
        loaded = load_model_artifact(artifact_dir)

        assert loaded.feature_schema.feature_names == feature_columns
        assert loaded.feature_schema.target_name == "charge"

    def test_metrics_preserved(self, sample_df, feature_columns, tmp_path):
        """Metrics are preserved through save/load."""
        trainer = RegressionTrainer(model="ridge")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
            cv=3,
        )

        artifact_dir = tmp_path / "test_model"
        save_model_artifact(artifact, artifact_dir)
        loaded = load_model_artifact(artifact_dir)

        assert loaded.metrics.r2 == pytest.approx(artifact.metrics.r2)
        assert loaded.metrics.mae == pytest.approx(artifact.metrics.mae)
        assert loaded.metrics.cv_r2_mean is not None

    def test_json_files_readable(self, sample_df, feature_columns, tmp_path):
        """JSON files are human-readable and valid."""
        trainer = RegressionTrainer(model="ridge")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )

        artifact_dir = tmp_path / "test_model"
        save_model_artifact(artifact, artifact_dir)

        # All JSON files parse correctly
        for name in ("metadata.json", "feature_schema.json", "metrics.json"):
            data = json.loads((artifact_dir / name).read_text())
            assert isinstance(data, dict)

    def test_load_missing_dir_raises(self, tmp_path):
        """Loading from a nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Artifact directory not found"):
            load_model_artifact(tmp_path / "nonexistent")

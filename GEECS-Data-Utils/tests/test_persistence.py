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

    def test_runtime_versions_captured(self, sample_df, feature_columns, tmp_path):
        """New artifacts record sklearn / joblib / numpy / python versions."""
        import joblib
        import numpy
        import sklearn

        from geecs_data_utils.modeling.ml.schemas import (
            ARTIFACT_VERSION,
            _python_runtime_version,
        )

        trainer = RegressionTrainer(model="ridge")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )

        artifact_dir = tmp_path / "test_model"
        save_model_artifact(artifact, artifact_dir)
        loaded = load_model_artifact(artifact_dir)

        assert loaded.metadata.artifact_version == ARTIFACT_VERSION
        assert loaded.metadata.sklearn_version == sklearn.__version__
        assert loaded.metadata.joblib_version == joblib.__version__
        assert loaded.metadata.numpy_version == numpy.__version__
        assert loaded.metadata.python_version == _python_runtime_version()

    def test_load_old_artifact_without_versions(
        self, sample_df, feature_columns, tmp_path, caplog
    ):
        """Artifacts saved before version capture load with 'unknown' sentinels."""
        # Manually write an artifact whose metadata.json omits the version
        # fields, simulating an artifact saved by a pre-0.7.0 build.
        trainer = RegressionTrainer(model="ridge")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )

        artifact_dir = tmp_path / "old_model"
        save_model_artifact(artifact, artifact_dir)

        # Strip the version fields from the metadata file.
        meta_path = artifact_dir / "metadata.json"
        data = json.loads(meta_path.read_text())
        for key in (
            "artifact_version",
            "sklearn_version",
            "joblib_version",
            "numpy_version",
            "python_version",
        ):
            data.pop(key, None)
        meta_path.write_text(json.dumps(data))

        loaded = load_model_artifact(artifact_dir)

        # Missing version fields are reported as "unknown" — not the
        # current runtime versions, so callers can tell what happened.
        assert loaded.metadata.sklearn_version == "unknown"
        assert loaded.metadata.joblib_version == "unknown"
        assert loaded.metadata.numpy_version == "unknown"
        assert loaded.metadata.python_version == "unknown"
        assert loaded.metadata.artifact_version == "unknown"

    def test_load_warns_on_version_drift(
        self, sample_df, feature_columns, tmp_path, caplog
    ):
        """Version mismatch on load emits a warning naming the drifted packages."""
        trainer = RegressionTrainer(model="ridge")
        artifact = trainer.fit(
            sample_df[feature_columns],
            sample_df["charge"],
            target_name="charge",
        )

        artifact_dir = tmp_path / "drifted_model"
        save_model_artifact(artifact, artifact_dir)

        # Rewrite metadata with a fake older sklearn version.
        meta_path = artifact_dir / "metadata.json"
        data = json.loads(meta_path.read_text())
        data["sklearn_version"] = "0.0.0-fake"
        meta_path.write_text(json.dumps(data))

        with caplog.at_level("WARNING"):
            load_model_artifact(artifact_dir)

        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("sklearn 0.0.0-fake" in m for m in warnings), warnings

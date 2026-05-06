"""End-to-end integration test for the ML workflow."""

import numpy as np
import pytest

from geecs_data_utils.data import OutlierConfig
from geecs_data_utils.modeling.ml import (
    BeamPredictionDatasetBuilder,
    CorrelationReport,
    ModelArtifact,
    RegressionTrainer,
    load_model_artifact,
    save_model_artifact,
)


class TestEndToEnd:
    """Full workflow: build dataset -> rank features -> train -> save -> load -> predict."""

    def test_full_workflow(self, sample_df, tmp_path):
        """Complete ML pipeline from raw data to loaded predictions."""
        # Step 1: Build dataset
        ds = BeamPredictionDatasetBuilder.from_dataframe(
            sample_df,
            feature_columns=["feature_a", "feature_b", "feature_c"],
            target_column="charge",
            outlier_config=OutlierConfig(method="nan", sigma=5.0),
        )
        assert ds.rows_final > 0

        # Step 2: Feature ranking
        report = CorrelationReport.from_dataframe(
            ds.frame,
            target="charge",
            method="spearman",
            exclude_terms=["timestamp", "shot"],
        )
        top = report.top_features(n=3)
        assert len(top) > 0

        # Step 3: Train
        trainer = RegressionTrainer(model="ridge", model_params={"alpha": 1.0})
        artifact = trainer.fit(
            ds.frame[top],
            ds.frame["charge"],
            target_name="charge",
            cv=3,
        )
        assert artifact.metrics.r2 > 0.5

        # Step 4: Save
        model_path = tmp_path / "charge_ridge_v1"
        save_model_artifact(artifact, model_path)
        assert (model_path / "model.joblib").exists()
        assert (model_path / "metadata.json").exists()

        # Step 5: Load
        loaded = load_model_artifact(model_path)
        assert isinstance(loaded, ModelArtifact)
        assert loaded.feature_schema.feature_names == top

        # Step 6: Predict on "new" data
        new_data = ds.frame[top].head(10)
        preds = loaded.predict(new_data)
        assert len(preds) == 10
        assert isinstance(preds, np.ndarray)

        # Step 7: Schema validation catches mismatched columns
        with pytest.raises(ValueError, match="Missing features"):
            loaded.predict(ds.frame[["feature_a"]])

    def test_multiple_model_types(self, sample_df, feature_columns, tmp_path):
        """Train, save, load works for all supported model types."""
        for model_name in ("linear", "ridge", "elasticnet"):
            trainer = RegressionTrainer(
                model=model_name,
                model_params={"alpha": 0.5} if model_name != "linear" else {},
            )
            artifact = trainer.fit(
                sample_df[feature_columns],
                sample_df["charge"],
                target_name="charge",
            )

            path = tmp_path / f"{model_name}_model"
            save_model_artifact(artifact, path)
            loaded = load_model_artifact(path)

            preds = loaded.predict(sample_df[feature_columns])
            original = artifact.predict(sample_df[feature_columns])
            np.testing.assert_array_almost_equal(preds, original)

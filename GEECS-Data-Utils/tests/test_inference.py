"""Tests for ``predict_from_scan``."""

from types import SimpleNamespace

import numpy as np
import pytest

from geecs_data_utils.modeling.ml.inference import predict_from_scan
from geecs_data_utils.modeling.ml.models import RegressionTrainer


def _train_artifact(sample_df, feature_columns):
    return RegressionTrainer(model="ridge").fit(
        sample_df[feature_columns],
        sample_df["charge"],
        target_name="charge",
    )


def test_predict_from_scan_with_matching_columns(sample_df, feature_columns):
    """A scan whose columns match the trained schema produces predictions."""
    artifact = _train_artifact(sample_df, feature_columns)
    scan = SimpleNamespace(data_frame=sample_df[feature_columns])

    preds = predict_from_scan(artifact, scan)

    direct = artifact.predict(sample_df[feature_columns])
    np.testing.assert_array_almost_equal(preds, direct)


def test_predict_from_scan_selects_only_schema_columns(sample_df, feature_columns):
    """Extra columns on the scan are ignored; predictions use schema order."""
    artifact = _train_artifact(sample_df, feature_columns)
    extra_df = sample_df.assign(unrelated=np.arange(len(sample_df)))
    scan = SimpleNamespace(data_frame=extra_df)

    preds = predict_from_scan(artifact, scan)
    direct = artifact.predict(sample_df[feature_columns])
    np.testing.assert_array_almost_equal(preds, direct)


def test_predict_from_scan_raises_on_missing_feature(sample_df, feature_columns):
    """Missing a trained feature surfaces a clear ValueError naming it."""
    artifact = _train_artifact(sample_df, feature_columns)
    dropped = sample_df.drop(columns=[feature_columns[0]])
    scan = SimpleNamespace(data_frame=dropped)

    with pytest.raises(ValueError, match="missing required feature columns"):
        predict_from_scan(artifact, scan)


def test_predict_from_scan_raises_on_missing_data_frame(sample_df, feature_columns):
    """A scan with no loaded data_frame raises with a clear message."""
    artifact = _train_artifact(sample_df, feature_columns)
    scan = SimpleNamespace(data_frame=None)

    with pytest.raises(ValueError, match="no loaded data_frame"):
        predict_from_scan(artifact, scan)

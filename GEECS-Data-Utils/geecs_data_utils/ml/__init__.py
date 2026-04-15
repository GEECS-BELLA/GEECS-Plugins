"""Machine-learning utilities for GEECS scan data.

This subpackage provides a lightweight framework for:

- Assembling ML-ready datasets from :class:`~geecs_data_utils.ScanData`
- Correlation-based feature ranking and outlier handling
- Training sklearn-pipeline regression models
- Persisting and loading model artifacts with metadata
- Running inference with schema validation

Requires the ``ml`` extras: ``pip install geecs-data-utils[ml]``.
"""

from geecs_data_utils.ml.dataset import (
    BeamPredictionDatasetBuilder,
    DatasetResult,
    OutlierConfig,
)
from geecs_data_utils.ml.feature_selection import (
    CorrelationReport,
    sigma_clip_frame,
    sigma_nan_frame,
)
from geecs_data_utils.ml.inference import predict_from_scan
from geecs_data_utils.ml.models import ModelArtifact, RegressionTrainer
from geecs_data_utils.ml.persistence import load_model_artifact, save_model_artifact
from geecs_data_utils.ml.schemas import FeatureSchema, ModelMetadata, TrainingMetrics

__all__ = [
    "BeamPredictionDatasetBuilder",
    "CorrelationReport",
    "DatasetResult",
    "FeatureSchema",
    "ModelArtifact",
    "ModelMetadata",
    "OutlierConfig",
    "RegressionTrainer",
    "TrainingMetrics",
    "load_model_artifact",
    "predict_from_scan",
    "save_model_artifact",
    "sigma_clip_frame",
    "sigma_nan_frame",
]

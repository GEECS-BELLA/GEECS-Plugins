"""Machine-learning backend: sklearn regression, artifact I/O, and inference.

Builds on :mod:`geecs_data_utils.data` for DataFrame assembly and column
matching; does not duplicate that layer.

Notes
-----
Typical workflow:

1. :class:`~geecs_data_utils.modeling.ml.dataset.MLDatasetBuilder`
   — build ``X``/``y`` tabular data.
2. :class:`~geecs_data_utils.modeling.ml.models.RegressionTrainer` →
   :class:`~geecs_data_utils.modeling.ml.models.ModelArtifact` — fit.
3. :func:`~geecs_data_utils.modeling.ml.persistence.save_model_artifact` —
   persist.
4. :func:`~geecs_data_utils.modeling.ml.persistence.load_model_artifact` and
   :func:`~geecs_data_utils.modeling.ml.inference.predict_from_scan` — deploy.

For correlation ranking (non-ML), import from
:mod:`geecs_data_utils.analysis` directly — there's no re-export here, so the
analysis path stays free of sklearn weight.

See Also
--------
geecs_data_utils.data : Shared dataset assembly and column resolution.
geecs_data_utils.analysis : Correlation and other analysis helpers.
"""

from geecs_data_utils.modeling.ml.dataset import (
    DatasetResult,
    MLDatasetBuilder,
)
from geecs_data_utils.modeling.ml.inference import predict_from_scan
from geecs_data_utils.modeling.ml.models import ModelArtifact, RegressionTrainer
from geecs_data_utils.modeling.ml.persistence import (
    load_model_artifact,
    save_model_artifact,
)
from geecs_data_utils.modeling.ml.schemas import (
    FeatureSchema,
    ModelMetadata,
    TrainingMetrics,
)

__all__ = [
    "DatasetResult",
    "FeatureSchema",
    "MLDatasetBuilder",
    "ModelArtifact",
    "ModelMetadata",
    "RegressionTrainer",
    "TrainingMetrics",
    "load_model_artifact",
    "predict_from_scan",
    "save_model_artifact",
]

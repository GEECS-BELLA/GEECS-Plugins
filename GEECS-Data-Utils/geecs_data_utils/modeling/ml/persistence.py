"""Persist :class:`~geecs_data_utils.modeling.ml.models.ModelArtifact` to disk.

Notes
-----
Each artifact is a directory (caller-defined path) containing:

* ``model.joblib`` — pickled sklearn ``Pipeline`` via ``joblib``
* ``metadata.json`` — :class:`~geecs_data_utils.modeling.ml.schemas.ModelMetadata`
* ``feature_schema.json`` — :class:`~geecs_data_utils.modeling.ml.schemas.FeatureSchema`
* ``metrics.json`` — :class:`~geecs_data_utils.modeling.ml.schemas.TrainingMetrics`

See Also
--------
geecs_data_utils.modeling.ml.models.ModelArtifact
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import joblib

from geecs_data_utils.modeling.ml.models import ModelArtifact
from geecs_data_utils.modeling.ml.schemas import (
    FeatureSchema,
    ModelMetadata,
    TrainingMetrics,
)

_MODEL_FILE = "model.joblib"
_METADATA_FILE = "metadata.json"
_SCHEMA_FILE = "feature_schema.json"
_METRICS_FILE = "metrics.json"


def save_model_artifact(artifact: ModelArtifact, path: Union[str, Path]) -> Path:
    """Write pipeline and JSON sidecars under ``path``.

    Parameters
    ----------
    artifact : ModelArtifact
        Object produced by :meth:`~geecs_data_utils.modeling.ml.models.RegressionTrainer.fit`.
    path : str or pathlib.Path
        Directory to create or reuse.

    Returns
    -------
    pathlib.Path
        Absolute-style resolved directory ``path`` used for writes.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact.pipeline, out / _MODEL_FILE)

    _write_json(out / _METADATA_FILE, artifact.metadata.to_dict())
    _write_json(out / _SCHEMA_FILE, artifact.feature_schema.to_dict())
    _write_json(out / _METRICS_FILE, artifact.metrics.to_dict())

    return out


def load_model_artifact(path: Union[str, Path]) -> ModelArtifact:
    """Reload artifact written by :func:`save_model_artifact`.

    Parameters
    ----------
    path : str or pathlib.Path
        Artifact directory containing the four standard filenames.

    Returns
    -------
    ModelArtifact

    Raises
    ------
    FileNotFoundError
        If ``path`` is not an existing directory.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {root}")

    pipeline = joblib.load(root / _MODEL_FILE)
    metadata = ModelMetadata.from_dict(_read_json(root / _METADATA_FILE))
    schema = FeatureSchema.from_dict(_read_json(root / _SCHEMA_FILE))
    metrics = TrainingMetrics.from_dict(_read_json(root / _METRICS_FILE))

    return ModelArtifact(
        pipeline=pipeline,
        feature_schema=schema,
        metadata=metadata,
        metrics=metrics,
    )


def _write_json(path: Path, data: dict) -> None:
    """Serialize ``data`` as indented UTF-8 JSON at ``path``."""
    path.write_text(json.dumps(data, indent=2, default=str))


def _read_json(path: Path) -> dict:
    """Parse JSON object from ``path``."""
    return json.loads(path.read_text())

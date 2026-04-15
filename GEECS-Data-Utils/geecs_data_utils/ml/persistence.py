"""Model artifact persistence — save and load trained models to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import joblib

from geecs_data_utils.ml.models import ModelArtifact
from geecs_data_utils.ml.schemas import FeatureSchema, ModelMetadata, TrainingMetrics

_MODEL_FILE = "model.joblib"
_METADATA_FILE = "metadata.json"
_SCHEMA_FILE = "feature_schema.json"
_METRICS_FILE = "metrics.json"


def save_model_artifact(artifact: ModelArtifact, path: Union[str, Path]) -> Path:
    """Persist a :class:`ModelArtifact` to a directory.

    The directory layout is::

        <path>/
            model.joblib          # fitted sklearn pipeline
            metadata.json         # training metadata
            feature_schema.json   # feature contract
            metrics.json          # evaluation metrics

    Parameters
    ----------
    artifact : ModelArtifact
        The trained artifact to save.
    path : str or Path
        Directory to write the artifact into.  Created if it does not exist.

    Returns
    -------
    Path
        The resolved artifact directory.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact.pipeline, out / _MODEL_FILE)

    _write_json(out / _METADATA_FILE, artifact.metadata.to_dict())
    _write_json(out / _SCHEMA_FILE, artifact.feature_schema.to_dict())
    _write_json(out / _METRICS_FILE, artifact.metrics.to_dict())

    return out


def load_model_artifact(path: Union[str, Path]) -> ModelArtifact:
    """Load a :class:`ModelArtifact` from a directory written by :func:`save_model_artifact`.

    Parameters
    ----------
    path : str or Path
        Path to the artifact directory.

    Returns
    -------
    ModelArtifact

    Raises
    ------
    FileNotFoundError
        If the directory or any required file is missing.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    """Write a dict as pretty-printed JSON."""
    path.write_text(json.dumps(data, indent=2, default=str))


def _read_json(path: Path) -> dict:
    """Read a JSON file and return a dict."""
    return json.loads(path.read_text())

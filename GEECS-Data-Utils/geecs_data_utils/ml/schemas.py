"""Data structures for ML model metadata, feature schemas, and training metrics."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class FeatureSchema:
    """Describes the exact feature contract expected by a trained model.

    Parameters
    ----------
    feature_names : list of str
        Ordered feature column names the model expects at inference time.
    target_name : str
        Name of the target column the model was trained to predict.
    dtypes : dict, optional
        Mapping of column name to dtype string (e.g. ``{"col": "float64"}``).
    """

    feature_names: List[str]
    target_name: str
    dtypes: Optional[Dict[str, str]] = None

    def validate_columns(self, columns: List[str]) -> None:
        """Raise ``ValueError`` if *columns* don't match the schema.

        Parameters
        ----------
        columns : list of str
            Column names present in the incoming data.

        Raises
        ------
        ValueError
            If there are missing or extra columns.
        """
        expected = set(self.feature_names)
        actual = set(columns)
        missing = expected - actual
        extra = actual - expected
        parts: list[str] = []
        if missing:
            parts.append(f"Missing features: {sorted(missing)}")
        if extra:
            parts.append(f"Unexpected features: {sorted(extra)}")
        if parts:
            raise ValueError("; ".join(parts))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureSchema:
        """Deserialize from a dictionary."""
        return cls(**data)


@dataclass
class TrainingMetrics:
    """Stores evaluation metrics produced during training.

    Parameters
    ----------
    r2 : float or None
        Coefficient of determination.
    mae : float or None
        Mean absolute error.
    rmse : float or None
        Root mean squared error.
    cv_r2_mean : float or None
        Mean R² across cross-validation folds.
    cv_r2_std : float or None
        Standard deviation of R² across cross-validation folds.
    extra : dict
        Any additional metrics.
    """

    r2: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    cv_r2_mean: Optional[float] = None
    cv_r2_std: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingMetrics:
        """Deserialize from a dictionary."""
        return cls(**data)


@dataclass
class ModelMetadata:
    """Metadata stored alongside a persisted model artifact.

    Parameters
    ----------
    model_type : str
        Estimator name (e.g. ``"ridge"``).
    created_at : str
        ISO-format timestamp of when the artifact was saved.
    feature_count : int
        Number of input features.
    target_name : str
        Name of the target column.
    training_rows : int or None
        Number of rows used for training.
    package_version : str
        Version of ``geecs-data-utils`` that created the artifact.
    scan_info : dict
        Optional metadata about the source scans.
    training_params : dict
        Estimator hyperparameters used during training.
    metrics : dict
        Training metrics (serialized :class:`TrainingMetrics`).
    """

    model_type: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    feature_count: int = 0
    target_name: str = ""
    training_rows: Optional[int] = None
    package_version: str = ""
    scan_info: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelMetadata:
        """Deserialize from a dictionary."""
        return cls(**data)

"""Dataclasses serialized beside ``model.joblib`` for artifacts and contracts.

Notes
-----
These types are plain containers; persistence logic lives in
:mod:`~geecs_data_utils.modeling.ml.persistence`.

See Also
--------
geecs_data_utils.modeling.ml.persistence.save_model_artifact
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class FeatureSchema:
    """Column contract for :meth:`~geecs_data_utils.modeling.ml.models.ModelArtifact.predict`.

    Attributes
    ----------
    feature_names : list of str
        Ordered predictor columns required at inference.
    target_name : str
        Human-readable target label (metadata).
    dtypes : dict or None
        Optional column-to-dtype-string map for future strictness (unused today).
    """

    feature_names: List[str]
    target_name: str
    dtypes: Optional[Dict[str, str]] = None

    def validate_columns(self, columns: List[str]) -> None:
        """Check that ``columns`` equals ``feature_names`` as a set.

        Parameters
        ----------
        columns : list of str
            Typically ``df.columns.tolist()`` from the inference batch.

        Raises
        ------
        ValueError
            On missing features, extra columns, or both.
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
        """Serialize all fields using :func:`dataclasses.asdict`.

        Returns
        -------
        dict
            JSON-friendly mapping.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureSchema:
        """Instantiate from decoded JSON or compatible mapping.

        Parameters
        ----------
        data : dict
            Keys must match dataclass fields.

        Returns
        -------
        FeatureSchema
        """
        return cls(**data)


@dataclass
class TrainingMetrics:
    """Regression scores recorded when fitting :class:`~geecs_data_utils.modeling.ml.models.ModelArtifact`.

    Attributes
    ----------
    r2 : float or None
        In-sample coefficient of determination on the training batch.
    mae : float or None
        In-sample mean absolute error.
    rmse : float or None
        In-sample root mean squared error.
    cv_r2_mean : float or None
        Mean R² across folds when CV is used during fit.
    cv_r2_std : float or None
        Standard deviation of fold R² when CV is used during fit.
    extra : dict
        Extension payload without changing the schema file layout.
    """

    r2: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    cv_r2_mean: Optional[float] = None
    cv_r2_std: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all fields using :func:`dataclasses.asdict`.

        Returns
        -------
        dict
            JSON-friendly mapping.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingMetrics:
        """Instantiate from decoded JSON or compatible mapping.

        Parameters
        ----------
        data : dict
            Keys must match dataclass fields.

        Returns
        -------
        TrainingMetrics
        """
        return cls(**data)


@dataclass
class ModelMetadata:
    """Training provenance stored next to the pickled pipeline.

    Attributes
    ----------
    model_type : str
        Estimator key (for example ``ridge``).
    created_at : str
        ISO timestamp string.
    feature_count : int
        Number of training features.
    target_name : str
        Documentary target label.
    training_rows : int or None
        Sample count used for fitting.
    package_version : str
        ``geecs_data_utils`` version string when available.
    scan_info : dict
        Optional experiment provenance.
    training_params : dict
        Estimator hyperparameters passed at construct time.
    metrics : dict
        Snapshot of metrics as plain dict (often mirrors ``TrainingMetrics``).
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
        """Serialize all fields using :func:`dataclasses.asdict`.

        Returns
        -------
        dict
            JSON-friendly mapping.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelMetadata:
        """Instantiate from decoded JSON or compatible mapping.

        Parameters
        ----------
        data : dict
            Keys must match dataclass fields.

        Returns
        -------
        ModelMetadata
        """
        return cls(**data)

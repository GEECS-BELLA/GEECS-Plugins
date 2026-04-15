"""Inference utilities for loaded model artifacts.

The primary inference interface is :meth:`ModelArtifact.predict`, which is
available directly on artifacts returned by
:func:`~geecs_data_utils.ml.persistence.load_model_artifact`.

This module provides additional helpers for common inference patterns.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from geecs_data_utils.ml.models import ModelArtifact


def predict_from_scan(
    artifact: ModelArtifact,
    scan: Any,
    *,
    feature_specs: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Run predictions using a loaded artifact on data from a :class:`~geecs_data_utils.ScanData`.

    Parameters
    ----------
    artifact : ModelArtifact
        A loaded (or freshly trained) model artifact.
    scan : ScanData
        A loaded scan data object.
    feature_specs : sequence of str, optional
        Search terms to locate the required feature columns via
        ``scan.find_cols``.  If ``None``, the feature names from
        the artifact schema are used directly.

    Returns
    -------
    ndarray
        Predicted values for each row in the scan data.

    Raises
    ------
    ValueError
        If required feature columns are missing from the scan data.
    """
    df = scan.data_frame
    if df is None:
        raise ValueError("ScanData has no loaded data_frame.")

    expected = artifact.feature_schema.feature_names

    if feature_specs is not None:
        # Remap feature specs to actual column names
        col_map = _resolve_feature_map(scan, feature_specs, expected)
        sub = df[list(col_map.values())].rename(
            columns={v: k for k, v in col_map.items()}
        )
    else:
        # Try direct column match
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"Scan data is missing required feature columns: {missing}"
            )
        sub = df[expected]

    return artifact.predict(sub)


def _resolve_feature_map(
    scan: Any,
    specs: Sequence[str],
    expected: list[str],
) -> dict[str, str]:
    """Map expected feature names to actual scan column names via specs."""
    available: list[str] = []
    for spec in specs:
        if hasattr(scan, "find_cols"):
            available.extend(scan.find_cols(spec))
        elif spec in scan.data_frame.columns:
            available.append(spec)

    mapping: dict[str, str] = {}
    for feat in expected:
        if feat in available:
            mapping[feat] = feat
        else:
            raise ValueError(
                f"Feature '{feat}' not found in scan columns resolved from specs."
            )
    return mapping

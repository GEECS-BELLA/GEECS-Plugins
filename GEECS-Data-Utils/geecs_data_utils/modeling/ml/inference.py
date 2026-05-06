"""Prediction helpers for :class:`~geecs_data_utils.modeling.ml.models.ModelArtifact`.

Notes
-----
Training-time column picking in :mod:`~geecs_data_utils.modeling.ml.dataset` uses
:func:`~geecs_data_utils.data.columns.resolve_col` (one best match per spec).

Inference may pass ``feature_specs`` through
:func:`~geecs_data_utils.data.columns.find_cols`, union the hits, and require
every name in ``artifact.feature_schema.feature_names`` to appear in that pool.
If ``feature_specs`` is omitted, scan columns must match the schema exactly.

See Also
--------
geecs_data_utils.modeling.ml.dataset :
    Dataset assembly and ``resolve_col``-based specs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from geecs_data_utils.data.columns import find_cols, flatten_columns
from geecs_data_utils.modeling.ml.models import ModelArtifact


def predict_from_scan(
    artifact: ModelArtifact,
    scan: Any,
    *,
    feature_specs: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Predict from ``scan.data_frame`` aligned to the artifact feature schema.

    Parameters
    ----------
    artifact : ModelArtifact
        Loaded model (fitted pipeline and :class:`~geecs_data_utils.modeling.ml.schemas.FeatureSchema`).
    scan : object
        Must expose ``data_frame`` (e.g. :class:`~geecs_data_utils.scan_data.ScanData`).
    feature_specs : sequence of str, optional
        If ``None``, ``df`` must contain columns named exactly like the trained
        schema. If given, each spec is passed to ``find_cols``; combined hits
        must cover every schema feature name.

    Returns
    -------
    numpy.ndarray
        One prediction per row of the selected feature block.

    Raises
    ------
    ValueError
        If ``data_frame`` is missing, required columns are absent, or columns
        do not validate against the schema after selection/rename.
    """
    df = scan.data_frame
    if df is None:
        raise ValueError("ScanData has no loaded data_frame.")

    expected = artifact.feature_schema.feature_names

    if feature_specs is not None:
        col_map = _resolve_feature_map(df, feature_specs, tuple(expected))
        sub = df[list(col_map.values())].rename(
            columns={v: k for k, v in col_map.items()}
        )
    else:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"Scan data is missing required feature columns: {missing}"
            )
        sub = df[expected]

    return artifact.predict(sub)


def _resolve_feature_map(
    df: pd.DataFrame,
    specs: Sequence[str],
    expected: Sequence[str],
) -> Dict[str, str]:
    """Map schema feature names to physical columns for extraction.

    Builds the candidate set from ``find_cols(df, spec)`` for each ``spec``,
    plus literal ``spec`` when it names a column or flattened multi-index label.
    Every entry in ``expected`` must appear in that set.

    Parameters
    ----------
    df : pandas.DataFrame
        Scan or batch table.
    specs : sequence of str
        Search strings forwarded to ``find_cols``.
    expected : sequence of str
        Names from ``FeatureSchema.feature_names`` that must be covered.

    Returns
    -------
    dict[str, str]
        Identity map ``{name: name}`` when successful (documents rename hook).

    Raises
    ------
    ValueError
        If any ``expected`` name is missing from the candidate pool.
    """
    flat_set = set(flatten_columns(df))
    available: list[str] = []
    for spec in specs:
        matches = find_cols(df, spec)
        if matches:
            available.extend(matches)
            continue
        if spec in df.columns:
            available.append(spec)
        elif spec in flat_set:
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


__all__ = ["predict_from_scan", "_resolve_feature_map"]

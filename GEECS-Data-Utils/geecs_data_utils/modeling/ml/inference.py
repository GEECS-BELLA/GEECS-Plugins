"""Prediction helpers for :class:`~geecs_data_utils.modeling.ml.models.ModelArtifact`.

Notes
-----
Inference requires the scan's ``data_frame`` to contain columns whose names
match the trained artifact's ``feature_schema.feature_names`` exactly. The
training-time builder records the resolved physical column names in the
schema (not the spec strings), so as long as the upstream column conventions
haven't drifted between training and inference, exact match Just Works.

If new column names appear at inference time (a renamed diagnostic, a
provenance refactor), retrain the model — don't paper over the rename in
the predict path. That kept-it-honest constraint is why this module no
longer carries a column-rename machinery.

See Also
--------
geecs_data_utils.modeling.ml.dataset :
    Training-time dataset assembly with ``resolve_col``-based specs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geecs_data_utils.modeling.ml.models import ModelArtifact


def predict_from_scan(artifact: ModelArtifact, scan: Any) -> np.ndarray:
    """Predict from ``scan.data_frame`` aligned to the artifact feature schema.

    Parameters
    ----------
    artifact : ModelArtifact
        Loaded model (fitted pipeline and
        :class:`~geecs_data_utils.modeling.ml.schemas.FeatureSchema`).
    scan : object
        Must expose ``data_frame`` (e.g.
        :class:`~geecs_data_utils.scan_data.ScanData`).

    Returns
    -------
    numpy.ndarray
        One prediction per row of the selected feature block.

    Raises
    ------
    ValueError
        If ``data_frame`` is missing or any feature column from the trained
        schema is absent from the scan.
    """
    df = scan.data_frame
    if df is None:
        raise ValueError("ScanData has no loaded data_frame.")

    expected = artifact.feature_schema.feature_names
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Scan data is missing required feature columns: {missing}")

    return artifact.predict(df[expected])


__all__ = ["predict_from_scan"]

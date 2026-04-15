"""Preprocessing pipeline construction for regression workflows."""

from __future__ import annotations

from typing import List

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessing_pipeline(
    *,
    impute: bool = True,
    impute_strategy: str = "median",
    scale: bool = True,
) -> Pipeline:
    """Build a preprocessing :class:`~sklearn.pipeline.Pipeline`.

    Parameters
    ----------
    impute : bool
        If ``True``, add a :class:`~sklearn.impute.SimpleImputer` step.
    impute_strategy : str
        Strategy for the imputer (``"mean"``, ``"median"``, ``"most_frequent"``).
    scale : bool
        If ``True``, add a :class:`~sklearn.preprocessing.StandardScaler` step.

    Returns
    -------
    Pipeline
        A scikit-learn pipeline with the requested preprocessing steps.
    """
    steps: List[tuple] = []
    if impute:
        steps.append(("imputer", SimpleImputer(strategy=impute_strategy)))
    if scale:
        steps.append(("scaler", StandardScaler()))
    if not steps:
        steps.append(("passthrough", "passthrough"))
    return Pipeline(steps)

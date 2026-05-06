"""Sklearn preprocessing stages prepended to regressors in ``RegressionTrainer``.

See Also
--------
geecs_data_utils.modeling.ml.models.RegressionTrainer :
    Concatenates these steps before the estimator.
"""

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
    """Build imputer and/or scaler steps for tabular regression.

    Parameters
    ----------
    impute : bool, default True
        When True, prepend ``SimpleImputer(strategy=impute_strategy)``.
    impute_strategy : str, default 'median'
        Forwarded to :class:`~sklearn.impute.SimpleImputer`.
    scale : bool, default True
        When True, append ``StandardScaler`` after imputation.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Single-branch pipeline, or ``Pipeline([('passthrough','passthrough')])``
        when both ``impute`` and ``scale`` are False.

    Notes
    -----
    Step order is always imputer (if any) then scaler (if any).
    """
    steps: List[tuple] = []
    if impute:
        steps.append(("imputer", SimpleImputer(strategy=impute_strategy)))
    if scale:
        steps.append(("scaler", StandardScaler()))
    if not steps:
        steps.append(("passthrough", "passthrough"))
    return Pipeline(steps)

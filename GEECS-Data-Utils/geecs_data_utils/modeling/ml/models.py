"""Sklearn regression training wrapped as :class:`ModelArtifact`.

The fitted estimator is always ``Pipeline(preprocessing + regressor)``.
Preprocessing defaults match serialized artifacts in
:mod:`~geecs_data_utils.modeling.ml.persistence`.

See Also
--------
geecs_data_utils.modeling.ml.persistence :
    Save/load ``ModelArtifact`` and sidecar JSON.
geecs_data_utils.modeling.ml.preprocessing :
    Default imputer/scaler pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from geecs_data_utils.modeling.ml.preprocessing import build_preprocessing_pipeline
from geecs_data_utils.modeling.ml.schemas import (
    FeatureSchema,
    ModelMetadata,
    TrainingMetrics,
)

ModelName = Literal["linear", "ridge", "elasticnet"]

_ESTIMATORS: Dict[str, type] = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "elasticnet": ElasticNet,
}


class ModelArtifact:
    """Fitted pipeline plus schema and metrics for I/O and inference.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Preprocessing steps followed by the regression estimator.
    feature_schema : FeatureSchema
        Ordered feature names required by :meth:`predict`.
    metadata : ModelMetadata
        Training provenance and hyperparameters.
    metrics : TrainingMetrics
        Scores computed during :meth:`RegressionTrainer.fit`.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        feature_schema: FeatureSchema,
        metadata: ModelMetadata,
        metrics: TrainingMetrics,
    ) -> None:
        """Attach fitted pipeline and metadata containers.

        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            Fitted preprocess + estimator pipeline.
        feature_schema : FeatureSchema
            Feature column contract for :meth:`predict`.
        metadata : ModelMetadata
            Training summary for persistence.
        metrics : TrainingMetrics
            Fit-time scores for persistence.
        """
        self.pipeline = pipeline
        self.feature_schema = feature_schema
        self.metadata = metadata
        self.metrics = metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Run the fitted pipeline on rows of ``df``.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain exactly the columns in ``feature_schema.feature_names``.

        Returns
        -------
        numpy.ndarray
            Predicted target values, shape ``(n_samples,)``.

        Raises
        ------
        ValueError
            If columns do not match the schema (including unexpected extras).
        """
        self.feature_schema.validate_columns(df.columns.tolist())
        X = df[self.feature_schema.feature_names].values
        return self.pipeline.predict(X)


class RegressionTrainer:
    """Fit linear, ridge, or elastic-net regressors with optional CV metrics.

    Notes
    -----
    Use :meth:`fit` to obtain a :class:`ModelArtifact` suitable for
    :func:`~geecs_data_utils.modeling.ml.persistence.save_model_artifact`.

    See Also
    --------
    geecs_data_utils.modeling.ml.persistence.save_model_artifact
    """

    def __init__(
        self,
        model: ModelName = "ridge",
        model_params: Optional[Dict[str, Any]] = None,
        impute: bool = True,
        scale: bool = True,
    ) -> None:
        """Configure estimator kind and preprocessing flags.

        Parameters
        ----------
        model : {'linear', 'ridge', 'elasticnet'}, default 'ridge'
            Which sklearn linear model to wrap in the pipeline.
        model_params : dict, optional
            Keyword arguments passed to the sklearn estimator constructor.
        impute : bool, default True
            Include ``SimpleImputer`` before scaling/regression when True.
        scale : bool, default True
            Include ``StandardScaler`` after imputation when True.

        Raises
        ------
        ValueError
            If ``model`` is not one of the supported names.
        """
        if model not in _ESTIMATORS:
            raise ValueError(
                f"Unknown model '{model}'. Choose from: {list(_ESTIMATORS)}"
            )
        self.model_name = model
        self.model_params = model_params or {}
        self.impute = impute
        self.scale = scale

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        *,
        feature_names: Optional[List[str]] = None,
        target_name: str = "target",
        cv: Optional[int] = None,
        scan_info: Optional[Dict[str, Any]] = None,
    ) -> ModelArtifact:
        """Fit preprocessing + estimator and return a :class:`ModelArtifact`.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Training features.
        y : pandas.Series or numpy.ndarray
            Training targets.
        feature_names : list of str, optional
            Stored in ``FeatureSchema``. Defaults to DataFrame columns or
            ``feature_0``, ``feature_1``, ... for ndarray ``X``.
        target_name : str, default 'target'
            Recorded in schema/metadata only.
        cv : int, optional
            If greater than 1, runs ``cross_val_score`` on the full pipeline
            (including preprocessing) and fills ``TrainingMetrics.cv_r2_*``.
        scan_info : dict, optional
            Arbitrary JSON-serializable metadata stored in ``ModelMetadata``.

        Returns
        -------
        ModelArtifact

        Notes
        -----
        ``TrainingMetrics.r2``, ``mae``, and ``rmse`` are **in-sample** on the
        training data; interpret ``cv_r2_mean`` / ``cv_r2_std`` separately when
        ``cv`` is set.
        """
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

        preprocess = build_preprocessing_pipeline(impute=self.impute, scale=self.scale)
        estimator = _ESTIMATORS[self.model_name](**self.model_params)
        pipeline = Pipeline(preprocess.steps + [("estimator", estimator)])

        pipeline.fit(X_arr, y_arr)

        y_pred = pipeline.predict(X_arr)
        metrics = TrainingMetrics(
            r2=float(r2_score(y_arr, y_pred)),
            mae=float(mean_absolute_error(y_arr, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_arr, y_pred))),
        )

        if cv is not None and cv > 1:
            cv_scores = cross_val_score(pipeline, X_arr, y_arr, cv=cv, scoring="r2")
            metrics.cv_r2_mean = float(cv_scores.mean())
            metrics.cv_r2_std = float(cv_scores.std())

        schema = FeatureSchema(
            feature_names=feature_names,
            target_name=target_name,
        )

        try:
            from geecs_data_utils import __version__ as pkg_version
        except (ImportError, AttributeError):
            pkg_version = "unknown"

        metadata = ModelMetadata(
            model_type=self.model_name,
            feature_count=len(feature_names),
            target_name=target_name,
            training_rows=int(X_arr.shape[0]),
            package_version=str(pkg_version),
            scan_info=scan_info or {},
            training_params=self.model_params,
            metrics=metrics.to_dict(),
        )

        return ModelArtifact(
            pipeline=pipeline,
            feature_schema=schema,
            metadata=metadata,
            metrics=metrics,
        )

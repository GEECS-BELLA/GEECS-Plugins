"""Regression model training with sklearn pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from geecs_data_utils.ml.preprocessing import build_preprocessing_pipeline
from geecs_data_utils.ml.schemas import FeatureSchema, ModelMetadata, TrainingMetrics

ModelName = Literal["linear", "ridge", "elasticnet"]

_ESTIMATORS: Dict[str, type] = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "elasticnet": ElasticNet,
}


class ModelArtifact:
    """A trained model artifact containing the pipeline, schema, and metadata.

    This is the primary object returned by :meth:`RegressionTrainer.fit` and
    by :func:`~geecs_data_utils.ml.persistence.load_model_artifact`.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted sklearn pipeline (preprocessing + estimator).
    feature_schema : FeatureSchema
        Feature contract for inference.
    metadata : ModelMetadata
        Training metadata.
    metrics : TrainingMetrics
        Evaluation metrics from training.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        feature_schema: FeatureSchema,
        metadata: ModelMetadata,
        metrics: TrainingMetrics,
    ) -> None:
        self.pipeline = pipeline
        self.feature_schema = feature_schema
        self.metadata = metadata
        self.metrics = metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Run predictions on new data with schema validation.

        Parameters
        ----------
        df : DataFrame
            Input data.  Must contain exactly the columns listed in
            :attr:`feature_schema`.

        Returns
        -------
        ndarray
            Predicted values.

        Raises
        ------
        ValueError
            If the input columns don't match the feature schema.
        """
        self.feature_schema.validate_columns(df.columns.tolist())
        X = df[self.feature_schema.feature_names].values
        return self.pipeline.predict(X)


class RegressionTrainer:
    """Train sklearn-pipeline regression models.

    Parameters
    ----------
    model : ``"linear"`` | ``"ridge"`` | ``"elasticnet"``
        Estimator to use.
    model_params : dict, optional
        Keyword arguments forwarded to the estimator constructor.
    impute : bool
        Include imputation in the preprocessing pipeline.
    scale : bool
        Include standard scaling in the preprocessing pipeline.

    Examples
    --------
    >>> trainer = RegressionTrainer(model="ridge", model_params={"alpha": 1.0})
    >>> artifact = trainer.fit(X_train, y_train, feature_names=X_train.columns.tolist(),
    ...                        target_name="charge")
    >>> artifact.metrics.r2
    0.87
    """

    def __init__(
        self,
        model: ModelName = "ridge",
        model_params: Optional[Dict[str, Any]] = None,
        impute: bool = True,
        scale: bool = True,
    ) -> None:
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
        """Fit a regression pipeline and return a :class:`ModelArtifact`.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.
        y : Series or ndarray
            Target values.
        feature_names : list of str, optional
            Feature column names.  Inferred from *X* if it is a DataFrame.
        target_name : str
            Name of the target variable.
        cv : int, optional
            If given, run cross-validation with this many folds and store
            the results in the returned metrics.
        scan_info : dict, optional
            Optional source scan metadata to store in the artifact.

        Returns
        -------
        ModelArtifact
        """
        # Resolve feature names
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

        # Build pipeline
        preprocess = build_preprocessing_pipeline(
            impute=self.impute, scale=self.scale
        )
        estimator = _ESTIMATORS[self.model_name](**self.model_params)
        pipeline = Pipeline(
            preprocess.steps + [("estimator", estimator)]
        )

        # Fit
        pipeline.fit(X_arr, y_arr)

        # Evaluate on training data
        y_pred = pipeline.predict(X_arr)
        metrics = TrainingMetrics(
            r2=float(r2_score(y_arr, y_pred)),
            mae=float(mean_absolute_error(y_arr, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_arr, y_pred))),
        )

        # Optional cross-validation
        if cv is not None and cv > 1:
            cv_scores = cross_val_score(
                pipeline, X_arr, y_arr, cv=cv, scoring="r2"
            )
            metrics.cv_r2_mean = float(cv_scores.mean())
            metrics.cv_r2_std = float(cv_scores.std())

        # Build schema and metadata
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

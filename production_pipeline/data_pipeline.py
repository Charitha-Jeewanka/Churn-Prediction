"""End-to-end preprocessing pipeline for churn prediction.

This module defines a `DataPipeline` class that orchestrates all data
processing steps: handling missing values, optional outlier removal,
feature binning, encoding of categorical variables, and scaling of
numeric features. The pipeline follows scikit-learn's fit/transform
interface so it can be integrated seamlessly with training and
inference workflows. The pipeline can be saved and loaded via joblib
for deployment.

Example
-------
>>> from production_pipeline.config import PreprocessingConfig
>>> from production_pipeline.data_pipeline import DataPipeline
>>> config = PreprocessingConfig(missing_strategy='median', encoding_strategy='onehot', scaling_strategy='standard')
>>> pipeline = DataPipeline(config, target_column='Churn', handle_outliers=True)
>>> X_processed, y = pipeline.fit_transform(raw_df)
>>> pipeline.save('pipeline.joblib')
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import joblib
import pandas as pd

from .config import PreprocessingConfig
from .handle_missing_values import MissingValueHandler
from .outlier_detection import OutlierDetector
from .feature_binning import bin_tenure
from .feature_encoding import CategoricalEncoder
from .feature_scaling import FeatureScaler

logger = logging.getLogger(__name__)


class DataPipeline:
    """Orchestrates data preprocessing steps for churn prediction."""

    def __init__(
        self,
        config: PreprocessingConfig,
        target_column: str = "Churn",
        handle_outliers: bool = False,
        outlier_method: str = "iqr",
    ) -> None:
        """Initialise the pipeline with preprocessing strategies.

        Parameters
        ----------
        config : PreprocessingConfig
            Configuration specifying missing value, encoding, scaling and binning strategies.
        target_column : str, default='Churn'
            Name of the target column to separate from features.
        handle_outliers : bool, default=False
            Whether to detect and remove outliers from numeric features.
        outlier_method : str, default='iqr'
            Method for outlier detection when `handle_outliers` is True. See
            `OutlierDetector` for supported methods.
        """
        self.config = config
        self.target_column = target_column
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method

        # Internal state set during fitting
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.encoder: Optional[CategoricalEncoder] = None
        self.scaler: Optional[FeatureScaler] = None
        self.feature_names_: Optional[List[str]] = None
        self._fitted: bool = False

        # cached processed data after fit_transform
        self.X_: Optional[pd.DataFrame] = None
        self.y_: Optional[pd.Series] = None

    def _separate_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split a DataFrame into features and target. Raises if target missing."""
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in input data")

        # Drop identifier columns (e.g. customerID) before processing
        drop_cols = [self.target_column]
        for col in ["customerID", "CustomerID", "customer_id"]:
            if col in df.columns:
                drop_cols.append(col)

        X = df.drop(columns=drop_cols)
        y = df[self.target_column].copy()
        # Convert target to numeric: map 'Yes'/'No' to 1/0 if necessary
        if y.dtype == object:
            y = y.map({"Yes": 1, "No": 0}).astype(int)
        return X, y

    def _identify_feature_types(self, X: pd.DataFrame) -> None:
        """Determine categorical and numeric columns based on dtypes."""
        categorical_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "category"]
        numeric_cols = [c for c in X.columns if c not in categorical_cols]
        self.categorical_columns = categorical_cols
        self.numeric_columns = numeric_cols
        logger.debug("Identified %d categorical and %d numeric cols", len(categorical_cols), len(numeric_cols))

    def fit(self, df: pd.DataFrame) -> "DataPipeline":
        """Fit the preprocessing pipeline to the provided DataFrame."""
        logger.info("Starting pipeline fit on data with shape %s", df.shape)

        # 1) Missing values (safe even if your dataset is already cleaned)
        missing_handler = MissingValueHandler(self.config.missing_strategy)
        df_imputed = missing_handler.impute(df)

        # 2) Separate features and target
        X, y = self._separate_features_target(df_imputed)

        # 3) Optional binning
        if self.config.binning_strategy == "tenure" and "tenure" in X.columns:
            X = bin_tenure(X)

        # 4) Identify types
        self._identify_feature_types(X)

        # 5) Optional outlier removal (numeric only)
        if self.handle_outliers and self.numeric_columns:
            detector = OutlierDetector(method=self.outlier_method)
            outlier_indices: set[int] = set()
            for col in self.numeric_columns:
                outlier_indices.update(detector.detect(X[col]))
            if outlier_indices:
                logger.info("Removing %d outlier rows from training data", len(outlier_indices))
                mask = ~X.index.isin(outlier_indices)
                X = X[mask]
                y = y[mask]

        # 6) Fit + apply encoder
        self.encoder = CategoricalEncoder(self.config.encoding_strategy)
        if self.categorical_columns:
            self.encoder.fit(X, self.categorical_columns)
            X = self.encoder.transform(X, self.categorical_columns)

        # 7) Fit + apply scaler
        self.scaler = FeatureScaler(self.config.scaling_strategy)
        if self.numeric_columns:
            self.scaler.fit(X, self.numeric_columns)
            X = self.scaler.transform(X, self.numeric_columns)

        # 8) Save feature order and mark as fitted
        self.feature_names_ = list(X.columns)
        self.X_ = X
        self.y_ = y
        self._fitted = True
        logger.info("Pipeline fitting complete. Processed data shape: %s", X.shape)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted preprocessing pipeline."""
        if not self._fitted:
            raise RuntimeError("The pipeline has not been fitted yet. Call fit() first.")

        # Drop target if present
        if self.target_column in df.columns:
            df = df.drop(columns=[self.target_column])

        # Binning if applicable
        if self.config.binning_strategy == "tenure" and "tenure" in df.columns:
            df = bin_tenure(df)

        X = df.copy()

        # Verify required columns exist
        missing_cols = [c for c in self.categorical_columns if c not in X.columns]
        missing_num = [c for c in self.numeric_columns if c not in X.columns]
        if missing_cols or missing_num:
            raise ValueError(f"Input data is missing required columns: {missing_cols + missing_num}")

        # Apply fitted encoder/scaler
        if self.categorical_columns and self.encoder is not None:
            X = self.encoder.transform(X, self.categorical_columns)
        if self.numeric_columns and self.scaler is not None:
            X = self.scaler.transform(X, self.numeric_columns)

        # Ensure column order matches training
        X = X.reindex(columns=self.feature_names_, fill_value=0)
        return X

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit the pipeline to the data and return the transformed features and target."""
        self.fit(df)
        return self.X_, self.y_

    def save(self, path: str) -> None:
        """Serialize the fitted pipeline to the given path using joblib."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted pipeline")
        joblib.dump(self, path)
        logger.info("Saved preprocessing pipeline to %s", path)

    @staticmethod
    def load(path: str) -> "DataPipeline":
        """Deserialize a previously saved pipeline from disk."""
        pipeline = joblib.load(path)
        if not isinstance(pipeline, DataPipeline):
            raise TypeError("Loaded object is not a DataPipeline instance")
        return pipeline

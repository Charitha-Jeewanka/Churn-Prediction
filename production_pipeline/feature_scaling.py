"""Numerical feature scaling utilities.

This module defines a `FeatureScaler` class that applies different
scaling strategies to numerical features. Scaling ensures that
numerical variables share a common scale, improving the performance of
certain machine learning algorithms.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


logger = logging.getLogger(__name__)


class FeatureScaler:
    """Scale numerical features using a specified strategy."""

    def __init__(self, strategy: str = 'standard') -> None:
        """Initialize the scaler.

        Parameters
        ----------
        strategy : str
            The scaling strategy to use ('standard', 'minmax', or 'none').
        """
        supported = {'standard', 'minmax', 'none'}
        if strategy not in supported:
            raise ValueError(f"Unsupported scaling strategy '{strategy}'. Supported: {supported}")
        self.strategy = strategy
        self.scaler = None

    def fit(self, X: pd.DataFrame, numeric_columns: List[str]) -> 'FeatureScaler':
        """Fit the scaler to the numerical columns.

        Parameters
        ----------
        X : pd.DataFrame
            The training data containing numerical features.
        numeric_columns : list[str]
            Names of the numeric columns to scale.

        Returns
        -------
        FeatureScaler
            The fitted scaler instance.
        """
        if self.strategy == 'standard':
            self.scaler = StandardScaler().fit(X[numeric_columns])
        elif self.strategy == 'minmax':
            self.scaler = MinMaxScaler().fit(X[numeric_columns])
        else:
            self.scaler = None
        return self

    def transform(self, X: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """Scale the numerical columns of a DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input data containing numerical features.
        numeric_columns : list[str]
            Names of the numeric columns to transform.

        Returns
        -------
        pd.DataFrame
            A DataFrame with scaled numerical features.
        """
        df = X.copy()
        if self.strategy == 'none' or self.scaler is None:
            return df
        scaled_values = self.scaler.transform(df[numeric_columns])
        df[numeric_columns] = scaled_values
        return df
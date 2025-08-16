"""Missing value handling for the churn prediction pipeline.

This module defines strategies for detecting and imputing missing values
in a pandas DataFrame. The `MissingValueHandler` class encapsulates
different imputation methods and can be easily extended to support
custom strategies.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class MissingValueHandler:
    """Handle missing values in a dataset according to a specified strategy."""

    def __init__(self, strategy: str = 'median') -> None:
        """Initialize the handler with a given strategy.

        Parameters
        ----------
        strategy : str
            The imputation strategy to use. Supported values are:
            - 'median': Replace missing numeric values with the median of the column.
            - 'mean': Replace missing numeric values with the mean of the column.
            - 'mode': Replace missing categorical values with the mode of the column.
            - 'drop': Drop rows containing any missing values.
        """
        self.strategy = strategy
        supported = {'median', 'mean', 'mode', 'drop'}
        if strategy not in supported:
            raise ValueError(f"Unsupported missing value strategy '{strategy}'. Supported: {supported}")

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured missing value strategy to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame in which to impute missing values.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with missing values imputed or rows dropped.
        """
        logger.info("Imputing missing values using strategy: %s", self.strategy)
        df = df.copy()

        if self.strategy == 'drop':
            before_shape = df.shape
            df = df.dropna()
            logger.info("Dropped %d rows containing missing values", before_shape[0] - df.shape[0])
            return df

        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype.kind in 'biufc':  # Numeric types
                    if self.strategy == 'median':
                        value = df[col].median()
                    elif self.strategy == 'mean':
                        value = df[col].mean()
                    else:
                        value = df[col].median()
                    df[col].fillna(value, inplace=True)
                    logger.debug("Filled missing numeric values in %s with %s", col, value)
                else:
                    # Categorical
                    if self.strategy == 'mode':
                        value = df[col].mode().iloc[0]
                        df[col].fillna(value, inplace=True)
                        logger.debug("Filled missing categorical values in %s with '%s'", col, value)
        return df
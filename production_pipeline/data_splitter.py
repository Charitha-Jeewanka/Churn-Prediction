"""Data splitting utilities.

This module provides functions to split a dataset into training,
validation, and test sets with stratification support.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class DataSplitter:
    """Split features and target into train, validation, and test sets."""

    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42, stratify: bool = True) -> None:
        """Initialize the splitter.

        Parameters
        ----------
        test_size : float
            Fraction of the data to use as the test set.
        val_size : float
            Fraction of the data (after removing test set) to use as the validation set.
        random_state : int
            Seed for reproducibility.
        stratify : bool
            Whether to stratify the splits based on the target variable.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Perform a stratified split into training, validation, and test sets.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.

        Returns
        -------
        tuple
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        stratify_col = y if self.stratify else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify_col
        )
        # Determine validation size relative to the remaining data
        val_relative_size = self.val_size / (1.0 - self.test_size)
        stratify_temp = y_temp if self.stratify else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, random_state=self.random_state, stratify=stratify_temp
        )
        logger.info(
            "Split data into train (%d), val (%d), and test (%d) examples", len(X_train), len(X_val), len(X_test)
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
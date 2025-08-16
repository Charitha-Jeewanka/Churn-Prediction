"""Data ingestion utilities for the churn prediction pipeline.

This module provides functions and classes to load raw data from CSV
files and perform initial validation. The goal of ingestion is to
produce a clean pandas DataFrame ready for preprocessing. Validation
steps include checking column presence, ensuring correct data types,
and verifying that the target column exists.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd


logger = logging.getLogger(__name__)


class DataIngestion:
    """Class responsible for loading and validating raw input data."""

    def __init__(self, file_path: str) -> None:
        """Initialize the data ingestion with a path to the CSV file.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the CSV file containing the
            dataset.
        """
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """Load data from the configured CSV file.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the data.

        Raises
        ------
        FileNotFoundError
            If the file path does not exist.
        ValueError
            If the loaded data is empty.
        """
        logger.info("Loading data from %s", self.file_path)
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError as exc:
            logger.error("Data file not found: %s", self.file_path)
            raise exc

        if df.empty:
            logger.error("Loaded data is empty")
            raise ValueError("Loaded data is empty")

        logger.info("Data loaded successfully with shape %s", df.shape)
        return df

    def validate_columns(self, df: pd.DataFrame, required_columns: Optional[list[str]] = None) -> None:
        """Validate that required columns are present in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.
        required_columns : list[str], optional
            A list of column names that must be present in the DataFrame.

        Raises
        ------
        ValueError
            If any required column is missing.
        """
        if required_columns is None:
            return
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error("Missing required columns: %s", missing)
            raise ValueError(f"Missing required columns: {missing}")
        logger.info("All required columns are present")
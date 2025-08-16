"""Outlier detection utilities.

This module offers basic methods for identifying outliers in numeric data
using either the interquartile range (IQR) or Z‑score techniques. The
outlier detector returns indices of rows considered outliers to enable
downstream processing (e.g. removal or capping).
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np
import pandas as pd
from scipy import stats


logger = logging.getLogger(__name__)


class OutlierDetector:
    """Detect outliers in numeric features using IQR or Z‑score methods."""

    def __init__(self, method: str = 'iqr', threshold: float = 3.0) -> None:
        """Initialize the detector.

        Parameters
        ----------
        method : str
            Outlier detection method ('iqr' or 'zscore').
        threshold : float
            Threshold for the Z‑score method (ignored when using IQR).
        """
        if method not in {'iqr', 'zscore'}:
            raise ValueError("method must be 'iqr' or 'zscore'")
        self.method = method
        self.threshold = threshold

    def detect(self, series: pd.Series) -> List[int]:
        """Return indices of outliers in a numeric series.

        Parameters
        ----------
        series : pd.Series
            Numerical series from which to detect outliers.

        Returns
        -------
        List[int]
            Indices in the series considered outliers.
        """
        if self.method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)].index.tolist()
            logger.debug("IQR outliers detected: %d", len(outliers))
            return outliers
        # Z‑score method
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        outliers = series[(z_scores > self.threshold)].index.tolist()
        logger.debug("Z‑score outliers detected: %d", len(outliers))
        return outliers
"""Feature binning utilities.

This module provides simple functions to discretize continuous variables.
Binning can help capture non‑linear relationships and reduce the impact
of outliers.
"""

from __future__ import annotations

import pandas as pd


def bin_tenure(df: pd.DataFrame, tenure_col: str = 'tenure', new_col: str = 'tenure_group') -> pd.DataFrame:
    """Bin the tenure feature into categorical groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a `tenure` column.
    tenure_col : str
        Name of the column containing tenure values.
    new_col : str
        Name of the new categorical column to create.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column representing tenure groups.
    """
    bins = [0, 12, 48, float('inf')]
    labels = ['New', 'Established', 'Loyal']
    df[new_col] = pd.cut(df[tenure_col], bins=bins, labels=labels, right=True, include_lowest=True)
    return df
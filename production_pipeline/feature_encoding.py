"""Categorical encoding strategies.

This module defines a `CategoricalEncoder` class that applies different
encoding techniques to categorical features. It uses the strategy
pattern to select the appropriate encoder at runtime based on the
configured strategy. Supported strategies include one-hot encoding,
label encoding, and ordinal encoding.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)


class CategoricalEncoder:
    """Encode categorical variables using a specified strategy."""

    def __init__(self, strategy: str = "onehot") -> None:
        """Initialize the encoder.

        Parameters
        ----------
        strategy : str
            The encoding strategy to use ('onehot', 'label', or 'ordinal').
        """
        supported = {"onehot", "label", "ordinal"}
        if strategy not in supported:
            raise ValueError(f"Unsupported encoding strategy '{strategy}'. Supported: {supported}")
        self.strategy = strategy
        self.encoder: Optional[object] = None
        self._cols: List[str] = []  # columns captured at fit-time

    def fit(self, X: pd.DataFrame, categorical_columns: List[str]) -> "CategoricalEncoder":
        """Fit the encoder to the categorical columns.

        Parameters
        ----------
        X : pd.DataFrame
            The training data containing categorical features.
        categorical_columns : list[str]
            Names of the categorical columns to encode.

        Returns
        -------
        CategoricalEncoder
            The fitted encoder instance.
        """
        # Remember the columns used at fit time and ignore any not in X
        self._cols = [c for c in categorical_columns if c in X.columns]

        # No categorical columns -> no-op
        if not self._cols:
            self.encoder = None
            return self

        if self.strategy == "onehot":
            # sklearn >= 1.2 uses sparse_output; older versions use sparse
            try:
                enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            except TypeError:
                enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
            enc.fit(X[self._cols])
            self.encoder = enc

        elif self.strategy == "label":
            self.encoder = {col: LabelEncoder().fit(X[col]) for col in self._cols}

        elif self.strategy == "ordinal":
            enc = OrdinalEncoder()
            enc.fit(X[self._cols])
            self.encoder = enc

        return self

    def transform(self, X: pd.DataFrame, categorical_columns: List[str] | None = None) -> pd.DataFrame:
        """Transform categorical columns according to the fitted encoder.

        Parameters
        ----------
        X : pd.DataFrame
            Input data containing categorical features.
        categorical_columns : list[str] | None
            Ignored at transform-time; we always use the columns learned at fit-time.

        Returns
        -------
        pd.DataFrame
            The DataFrame with encoded categorical features. Non-categorical
            columns are preserved.
        """
        df = X.copy()
        cols = self._cols  # always use the fit-time columns

        # No categorical columns -> no-op
        if not cols:
            return df

        if self.strategy == "onehot":
            if self.encoder is None:
                raise RuntimeError("CategoricalEncoder not fitted. Call fit() first.")
            encoded = self.encoder.transform(df[cols])
            # Older sklearn may return sparse matrix
            if hasattr(encoded, "toarray"):
                encoded = encoded.toarray()
            encoded_cols = self.encoder.get_feature_names_out(cols)
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            df = df.drop(columns=cols)
            df = pd.concat([df, encoded_df], axis=1)

        elif self.strategy == "label":
            for col in cols:
                df[col] = self.encoder[col].transform(df[col])

        elif self.strategy == "ordinal":
            encoded = self.encoder.transform(df[cols])
            encoded_df = pd.DataFrame(encoded, columns=cols, index=df.index)
            df[cols] = encoded_df

        return df

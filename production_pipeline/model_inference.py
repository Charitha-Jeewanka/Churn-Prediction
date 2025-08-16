"""Inference utilities for deployed churn prediction models.

This module defines an `InferenceEngine` class that wraps a
preprocessing pipeline and a trained classifier to provide both
probability and class predictions on new data. It supports batch
predictions for multiple samples as well as single‑sample inference.
Input data is validated against the expected feature schema, and
probabilities are returned to facilitate custom thresholding.

Example
-------
>>> from production_pipeline.model_inference import InferenceEngine
>>> engine = InferenceEngine.load('pipeline.joblib', 'model.joblib', threshold=0.6)
>>> results = engine.predict_batch(new_customers_df)
>>> # results is a DataFrame with probability and predicted label
>>> single_result = engine.predict_single({
...     'gender': 'Female',
...     'SeniorCitizen': 0,
...     ...
... })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for holding prediction outputs for a single sample.

    Attributes
    ----------
    probability : float
        Estimated probability of the positive class (churn).
    prediction : int
        Binary prediction (1 for churn, 0 for non‑churn) based on the
        configured threshold.
    """
    probability: float
    prediction: int


class InferenceEngine:
    """Perform inference using a fitted preprocessing pipeline and classifier."""

    def __init__(self, pipeline: Any, model: Any, threshold: float = 0.5) -> None:
        """Initialise the inference engine.

        Parameters
        ----------
        pipeline : object
            Preprocessing pipeline implementing a `transform` method. It must
            be fitted on training data.
        model : object
            Fitted classifier implementing `predict_proba`.
        threshold : float, default=0.5
            Probability threshold for converting probabilities to class labels.
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.pipeline = pipeline
        self.model = model
        self.threshold = threshold

    @staticmethod
    def load(pipeline_path: str, model_path: str, threshold: float = 0.5) -> 'InferenceEngine':
        """Load a saved preprocessing pipeline and model from disk.

        Parameters
        ----------
        pipeline_path : str
            Path to the serialized preprocessing pipeline (joblib format).
        model_path : str
            Path to the serialized model (joblib format).
        threshold : float, default=0.5
            Probability threshold for classification.

        Returns
        -------
        InferenceEngine
            An instantiated inference engine ready for predictions.
        """
        logger.info("Loading pipeline from %s and model from %s", pipeline_path, model_path)
        pipeline = joblib.load(pipeline_path)
        model = joblib.load(model_path)
        return InferenceEngine(pipeline, model, threshold)

    def _validate_input(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """Validate and coerce input data into a DataFrame.

        Parameters
        ----------
        data : pd.DataFrame or dict
            Input data representing one or more samples.

        Returns
        -------
        pd.DataFrame
            DataFrame with consistent columns for the pipeline.
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, dict):
            # Assume single sample; convert to one‑row DataFrame
            return pd.DataFrame([data])
        raise ValueError("data must be a pandas DataFrame or a dict representing a single sample")

    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict churn probabilities and labels for a batch of samples.

        Parameters
        ----------
        data : pd.DataFrame
            Input features for multiple samples. Must contain the same
            columns as used during training (order does not matter).

        Returns
        -------
        pd.DataFrame
            DataFrame with two columns: `probability` and `prediction`,
            indexed by the input data's index.
        """
        df = self._validate_input(data)
        logger.debug("Predicting batch of size %d", len(df))
        # Transform using preprocessing pipeline
        try:
            processed = self.pipeline.transform(df)
        except Exception as exc:
            logger.error("Error during preprocessing of input data: %s", exc)
            raise
        # Compute probabilities
        try:
            probas = self.model.predict_proba(processed)[:, 1]
        except AttributeError:
            logger.warning("Model does not support predict_proba; using decision_function with sigmoid")
            decision_scores = self.model.decision_function(processed)
            probas = 1 / (1 + np.exp(-decision_scores))
        preds = (probas >= self.threshold).astype(int)
        return pd.DataFrame({'probability': probas, 'prediction': preds}, index=df.index)

    def predict_single(self, sample: Dict[str, Any]) -> PredictionResult:
        """Predict the churn probability and label for a single sample.

        Parameters
        ----------
        sample : dict
            Mapping of feature names to values for a single customer.

        Returns
        -------
        PredictionResult
            Object containing the probability and binary prediction.
        """
        df = self._validate_input(sample)
        result_df = self.predict_batch(df)
        prob = float(result_df['probability'].iloc[0])
        pred = int(result_df['prediction'].iloc[0])
        return PredictionResult(probability=prob, prediction=pred)

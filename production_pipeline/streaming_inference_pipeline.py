"""Real‑time streaming inference pipeline.

This module defines a `StreamingInferencePipeline` class that wraps
the `InferenceEngine` for use in streaming or near‑real‑time settings.
It allows loading a preprocessing pipeline and trained model from disk
and performing predictions on individual records or small batches as
they arrive. The pipeline validates input, computes churn
probabilities and binary predictions, and logs results for
monitoring.

Example
-------
>>> from production_pipeline.streaming_inference_pipeline import StreamingInferencePipeline
>>> sip = StreamingInferencePipeline('pipeline.joblib', 'model.joblib', threshold=0.6)
>>> for sample in stream_of_new_customers:
...     result = sip.predict(sample)
...     handle_result(result)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Union

import pandas as pd

from .model_inference import InferenceEngine, PredictionResult

logger = logging.getLogger(__name__)


class StreamingInferencePipeline:
    """Provide real‑time inference on streaming data using a saved model."""

    def __init__(self, pipeline_path: str, model_path: str, threshold: float = 0.5) -> None:
        """Load the preprocessing pipeline and model.

        Parameters
        ----------
        pipeline_path : str
            Path to the persisted preprocessing pipeline (joblib file).
        model_path : str
            Path to the persisted trained model (joblib file).
        threshold : float, default=0.5
            Probability threshold for classification decisions.
        """
        self.engine = InferenceEngine.load(pipeline_path, model_path, threshold=threshold)

    def predict(self, sample: Dict[str, Any]) -> PredictionResult:
        """Predict for a single sample represented as a dictionary.

        Parameters
        ----------
        sample : dict
            Mapping of feature names to values.

        Returns
        -------
        PredictionResult
            Probability and binary prediction for the sample.
        """
        logger.debug("Performing streaming prediction for a single sample")
        return self.engine.predict_single(sample)

    def predict_batch(self, batch: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Predict probabilities and labels for a batch of samples.

        Parameters
        ----------
        batch : pd.DataFrame or list[dict]
            Collection of samples to process. If a list of dictionaries is
            provided, it will be converted to a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing probabilities and predictions for each sample.
        """
        if isinstance(batch, list):
            batch_df = pd.DataFrame(batch)
        else:
            batch_df = batch
        logger.debug("Performing streaming prediction for batch of size %d", len(batch_df))
        return self.engine.predict_batch(batch_df)

    def process_stream(self, samples: Iterable[Dict[str, Any]]) -> Iterable[PredictionResult]:
        """Process an iterable of samples and yield prediction results one by one.

        Parameters
        ----------
        samples : iterable of dict
            Streaming iterable where each element is a mapping of feature
            names to values.

        Yields
        ------
        PredictionResult
            Prediction result for each sample in the stream.
        """
        for sample in samples:
            result = self.predict(sample)
            yield result

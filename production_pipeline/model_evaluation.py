"""Model evaluation utilities for imbalanced classification.

This module provides a `ModelEvaluator` class that computes a variety
of classification metrics on held‑out data. It supports threshold
adjustment for converting predicted probabilities into class labels and
returns a comprehensive dictionary of evaluation results including
accuracy, precision, recall, F1 score, ROC AUC, PR AUC, and the
confusion matrix. The evaluator is designed for use both in the
training pipeline and post‑deployment monitoring.

Example
-------
>>> from production_pipeline.model_evaluation import ModelEvaluator
>>> evaluator = ModelEvaluator(threshold=0.6)
>>> results = evaluator.evaluate(model, X_test, y_test)
>>> print(results['f1'])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for storing evaluation metrics.

    Attributes
    ----------
    accuracy : float
        Overall classification accuracy.
    precision : float
        Precision of positive class predictions.
    recall : float
        Recall (sensitivity) of positive class predictions.
    f1 : float
        F1 score.
    roc_auc : float
        Area under the ROC curve.
    pr_auc : float
        Area under the precision–recall curve (average precision).
    confusion_matrix : np.ndarray
        2×2 array representing the confusion matrix as
        [[TN, FP], [FN, TP]].
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: np.ndarray


class ModelEvaluator:
    """Evaluate classification models on test data."""

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialise the evaluator with a decision threshold.

        Parameters
        ----------
        threshold : float, default=0.5
            Probability threshold used to convert predicted
            probabilities into class labels. Adjusting this value can
            trade off precision and recall, which in turn affects the
            F1 score.
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1 (exclusive)")
        self.threshold = threshold

    def evaluate(self, model: Any, X: pd.DataFrame, y_true: pd.Series) -> EvaluationResults:
        """Compute evaluation metrics on the provided dataset.

        Parameters
        ----------
        model : estimator
            Fitted classifier with a `predict_proba` method.
        X : pd.DataFrame
            Features for evaluation.
        y_true : pd.Series
            True binary labels.

        Returns
        -------
        EvaluationResults
            Dataclass containing various evaluation metrics.
        """
        logger.info("Evaluating model on %d samples", len(X))
        # Predict probabilities for the positive class
        try:
            probas: np.ndarray = model.predict_proba(X)[:, 1]
        except AttributeError:
            # Some classifiers (e.g. SVM with linear kernel) may not implement predict_proba
            # In such cases we fallback to decision_function and apply a sigmoid transformation
            logger.warning("Model does not implement predict_proba; using decision_function with sigmoid")
            decision_scores = model.decision_function(X)
            probas = 1 / (1 + np.exp(-decision_scores))
        # Convert probabilities to binary predictions using threshold
        y_pred = (probas >= self.threshold).astype(int)

        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        # Use roc_auc and average precision on probabilities
        try:
            roc_auc = roc_auc_score(y_true, probas)
        except ValueError:
            roc_auc = float('nan')
        try:
            pr_auc = average_precision_score(y_true, probas)
        except ValueError:
            pr_auc = float('nan')
        cm = confusion_matrix(y_true, y_pred)
        return EvaluationResults(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            confusion_matrix=cm,
        )

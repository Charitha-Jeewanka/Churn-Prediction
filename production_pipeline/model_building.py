"""Model factory for churn prediction.

This module defines a `ModelFactory` class that constructs machine
learning models based on configuration. It supports a variety of
classification algorithms commonly used for churn prediction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating machine learning model instances."""

    @staticmethod
    def create_model(model_type: str, params: Dict[str, Any] | None = None) -> Any:
        """Create a model instance of the specified type.

        Parameters
        ----------
        model_type : str
            Identifier of the model type to create ('logistic', 'decision_tree',
            'random_forest', 'xgboost', 'catboost').
        params : dict, optional
            Hyperparameters to pass to the model constructor.

        Returns
        -------
        Any
            Instantiated model object.

        Raises
        ------
        ValueError
            If an unsupported model type is requested.
        """
        params = params or {}
        model_type = model_type.lower()
        if model_type == 'logistic':
            logger.info("Creating LogisticRegression with params: %s", params)
            return LogisticRegression(**params)
        if model_type == 'decision_tree':
            logger.info("Creating DecisionTreeClassifier with params: %s", params)
            return DecisionTreeClassifier(**params)
        if model_type == 'random_forest':
            logger.info("Creating RandomForestClassifier with params: %s", params)
            return RandomForestClassifier(**params)
        if model_type == 'xgboost':
            logger.info("Creating XGBClassifier with params: %s", params)
            # Provide sensible defaults for imbalanced classification
            return XGBClassifier(**params)
        if model_type == 'catboost':
            logger.info("Creating CatBoostClassifier with params: %s", params)
            return CatBoostClassifier(**params)
        raise ValueError(f"Unsupported model type: {model_type}")
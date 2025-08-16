"""Model training utilities.

This module defines a `ModelTrainer` class that encapsulates common
training routines for churn prediction models. It supports fitting a
model on provided data, performing cross‑validation to estimate
generalisation performance, and optionally tuning hyperparameters via
grid search. The class returns both the trained estimator and a
detailed summary of cross‑validation scores. All operations use
stratified folds by default to respect class imbalance.

Example
-------
>>> from production_pipeline.model_building import ModelFactory
>>> from production_pipeline.model_training import ModelTrainer
>>> X_train, y_train = ...  # load your training data
>>> model = ModelFactory.create_model('random_forest', params={'n_estimators': 100})
>>> trainer = ModelTrainer(model, cv=5, scoring=['f1', 'precision', 'recall'])
>>> best_model, cv_results = trainer.train(X_train, y_train)
>>> print(cv_results['mean_test_f1'])

Notes
-----
The `scoring` argument follows scikit‑learn's scoring API. For
imbalanced classification tasks, F1 and PR‑AUC (average precision)
are often more informative than accuracy. When hyperparameter grids
are provided, the trainer will perform exhaustive grid search and
select the best model according to the first scoring metric.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Encapsulates model training routines with optional hyperparameter search.

    Parameters
    ----------
    model : BaseEstimator
        An instantiated scikit‑learn style classifier. This object will
        be cloned internally when performing grid search to avoid
        modifying the original instance.
    cv : int, default=5
        Number of folds to use for stratified cross‑validation. More
        folds provide a better estimate of generalisation at the cost
        of longer training time.
    scoring : list[str] | str, optional
        One or more scoring metrics to evaluate during cross‑validation.
        See scikit‑learn's metrics module for valid metric names. If
        multiple scores are provided, the first metric will be used to
        select the best estimator during hyperparameter search.
    n_jobs : int, default=-1
        Number of parallel jobs to run for cross‑validation and grid
        search. A value of -1 uses all available cores.
    """

    def __init__(
        self,
        model: BaseEstimator,
        cv: int = 5,
        scoring: Optional[Iterable[str]] = None,
        n_jobs: int = -1,
    ) -> None:
        self.base_model = model
        self.cv = cv
        if scoring is None:
            # Default to metrics suitable for imbalanced classification
            self.scoring: List[str] = ['f1', 'precision', 'recall', 'roc_auc', 'average_precision']
        elif isinstance(scoring, str):
            self.scoring = [scoring]
        else:
            self.scoring = list(scoring)
        self.n_jobs = n_jobs
        self.best_model_: Optional[BaseEstimator] = None
        self.cv_results_: Optional[Dict[str, Any]] = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Fit the model to the provided data, optionally performing hyperparameter tuning.

        When a `param_grid` is supplied, an exhaustive grid search is
        executed over the specified hyperparameter combinations using
        stratified cross‑validation. The best model according to the
        first metric in `self.scoring` is retained. If no grid is
        provided, the base model is simply fitted and cross‑validated.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Training target labels.
        param_grid : dict[str, list[Any]], optional
            Hyperparameter grid to search. Keys correspond to parameter
            names (with their respective estimator prefix, e.g.
            `'n_estimators'` for random forests) and values are lists
            of parameter settings to try. If `None`, no search is
            performed.

        Returns
        -------
        best_model : BaseEstimator
            The fitted estimator achieving the highest mean cross‑validation
            score according to the primary metric.
        cv_results : dict[str, Any]
            A dictionary of cross‑validation results. When grid search
            is performed, this corresponds to the `cv_results_` attribute
            of `GridSearchCV`; otherwise it includes keys `test_<metric>`
            and `train_<metric>` containing arrays of scores per fold.
        """
        logger.info("Starting training with cv=%d folds", self.cv)
        stratified_cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        if param_grid:
            logger.info("Performing grid search over %d parameter combinations", np.prod([len(v) for v in param_grid.values()]))
            grid_search = GridSearchCV(
                estimator=self.base_model,
                param_grid=param_grid,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                cv=stratified_cv,
                refit=self.scoring[0],
                return_train_score=True,
            )
            grid_search.fit(X, y)
            self.best_model_ = grid_search.best_estimator_
            self.cv_results_ = grid_search.cv_results_
            logger.info("Grid search completed. Best params: %s", grid_search.best_params_)
            return self.best_model_, self.cv_results_

        # Without hyperparameter search: evaluate cross‑validation performance
        logger.info("Fitting base model and performing cross‑validation")
        cv_scores = cross_validate(
            estimator=self.base_model,
            X=X,
            y=y,
            cv=stratified_cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            return_train_score=True,
        )
        # Fit on full training data
        self.base_model.fit(X, y)
        self.best_model_ = self.base_model
        self.cv_results_ = cv_scores
        return self.best_model_, self.cv_results_

    @property
    def best_estimator_(self) -> BaseEstimator:
        """Alias for `best_model_` to mirror scikit‑learn's interface."""
        if self.best_model_ is None:
            raise RuntimeError("The model has not been trained yet.")
        return self.best_model_

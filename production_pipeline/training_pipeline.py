"""Comprehensive training pipeline orchestrator.

This module defines a `TrainingPipeline` class that ties together
ingestion, preprocessing, model selection, training, evaluation and
persistence into a single workflow. Configurations for data,
preprocessing and the model are supplied via dataclasses, enabling
reproducible experiments. The pipeline supports optional
hyperparameter search and outputs both cross‑validation results and
test set evaluation metrics.

Example
-------
>>> from production_pipeline.config import DataConfig, PreprocessingConfig, ModelConfig
>>> from production_pipeline.training_pipeline import TrainingPipeline
>>> data_cfg = DataConfig(file_path='data.csv', test_size=0.2, val_size=0.1)
>>> prep_cfg = PreprocessingConfig(missing_strategy='median', encoding_strategy='onehot', scaling_strategy='standard')
>>> model_cfg = ModelConfig(model_type='random_forest', params={'n_estimators': [100, 200]})
>>> pipeline = TrainingPipeline(data_cfg, prep_cfg, model_cfg, hyperparams={'n_estimators': [100, 200]}, save_dir='models')
>>> results = pipeline.run()
>>> print(results['evaluation'])
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from .config import DataConfig, ModelConfig, PreprocessingConfig
from .data_ingestion import DataIngestion
from .data_pipeline import DataPipeline
from .data_splitter import DataSplitter
from .model_building import ModelFactory
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End‑to‑end pipeline for training and evaluating churn models."""

    def __init__(
        self,
        data_config: DataConfig,
        preprocessing_config: PreprocessingConfig,
        model_config: ModelConfig,
        hyperparams: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
        save_dir: str = 'artifacts',
    ) -> None:
        """Initialise the training pipeline.

        Parameters
        ----------
        data_config : DataConfig
            Configuration for data ingestion and splitting.
        preprocessing_config : PreprocessingConfig
            Configuration for preprocessing strategies.
        model_config : ModelConfig
            Configuration specifying the model type and base parameters. If a
            hyperparameter grid is passed via `hyperparams`, the values in
            `model_config.params` serve as defaults for the estimator.
        hyperparams : dict[str, list[Any]], optional
            Grid of hyperparameters to search over. The keys should match
            the arguments of the estimator specified by `model_config`.
        threshold : float, default=0.5
            Probability threshold for evaluating the final model.
        save_dir : str, default='artifacts'
            Directory in which to persist the preprocessing pipeline and
            trained model. Will be created if it does not exist.
        """
        self.data_config = data_config
        self.preprocessing_config = preprocessing_config
        self.model_config = model_config
        self.hyperparams = hyperparams
        self.threshold = threshold
        self.save_dir = save_dir

    def run(self) -> Dict[str, Any]:
        """Execute the training pipeline from start to finish.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the cross‑validation results and
            evaluation metrics on the test set. It also includes the
            paths to the saved pipeline and model.
        """
        logger.info("Starting training pipeline")
        # 1. Data ingestion
        ingest = DataIngestion(self.data_config.file_path)
        df = ingest.load_data()
        # Ensure that the target column exists in the raw dataset
        ingest.validate_columns(df, required_columns=['Churn'])
        # 2. Preprocessing
        pipeline = DataPipeline(self.preprocessing_config, target_column='Churn')
        X_processed, y = pipeline.fit_transform(df)
        # 3. Data splitting
        splitter = DataSplitter(
            test_size=self.data_config.test_size,
            val_size=self.data_config.val_size,
            random_state=self.data_config.random_state,
            stratify=self.data_config.stratify,
        )
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X_processed, y)
        # 4. Model construction
        # `model_config.params` may contain default values; for hyperparameter search
        base_params = self.model_config.params.copy()
        model = ModelFactory.create_model(self.model_config.model_type, params=base_params)
        # 5. Training
        trainer = ModelTrainer(model)
        best_model, cv_results = trainer.train(X_train, y_train, param_grid=self.hyperparams)
        # 6. Evaluation on validation and test sets
        evaluator = ModelEvaluator(threshold=self.threshold)
        val_results = evaluator.evaluate(best_model, X_val, y_val)
        test_results = evaluator.evaluate(best_model, X_test, y_test)
        # 7. Persistence
        os.makedirs(self.save_dir, exist_ok=True)
        # Use timestamp to avoid name collisions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pipeline_path = os.path.join(self.save_dir, f'pipeline_{timestamp}.joblib')
        model_path = os.path.join(self.save_dir, f'model_{timestamp}.joblib')
        pipeline.save(pipeline_path)
        # Save the best model using joblib
        import joblib as _joblib  # import inside to avoid top‑level dependency confusion
        _joblib.dump(best_model, model_path)
        logger.info("Saved model to %s", model_path)
        # 8. Return summary
        return {
            'pipeline_path': pipeline_path,
            'model_path': model_path,
            'cv_results': cv_results,
            'validation_evaluation': val_results,
            'test_evaluation': test_results,
        }

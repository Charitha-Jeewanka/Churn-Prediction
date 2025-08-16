"""Production-ready pipeline package for churn prediction.

This package organizes various data processing and modelling components into
modular units. Each submodule encapsulates a specific task, allowing
flexible composition and reuse across different workflows. The package
includes data ingestion, preprocessing, feature engineering, model
construction, training, evaluation, and inference utilities.

Modules
-------
- `config`: Data classes for configuring the pipeline.
- `data_ingestion`: Loading and basic validation of raw data.
- `handle_missing_values`: Strategies for detecting and imputing missing values.
- `outlier_detection`: Methods for identifying outliers in numerical features.
- `feature_encoding`: Encoding strategies for categorical variables.
- `feature_scaling`: Scaling strategies for numerical features.
- `feature_binning`: Utility functions for binning continuous variables.
- `data_splitter`: Tools for splitting data into train/validation/test sets.
- `model_building`: Factory for creating machine learning models.
    - `model_training`: Training routines with hyperparameter tuning support.
    - `model_evaluation`: Evaluation utilities for imbalanced classification.
    - `model_inference`: Simplified inference wrapper for deployed models.
    - `data_pipeline`: End‑to‑end preprocessing pipeline orchestrator.
    - `training_pipeline`: Full training pipeline orchestrator.
    - `streaming_inference_pipeline`: Real‑time inference orchestrator for incoming samples.
"""
"""Configuration data structures for the churn prediction pipeline.

This module defines dataclasses that capture configuration for data
processing, model creation, and pipeline orchestration. Centralising
configuration simplifies experimentation and ensures reproducibility.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DataConfig:
    """Configuration for data ingestion and splitting.

    Attributes
    ----------
    file_path: str
        Path to the input CSV file containing the raw dataset.
    test_size: float
        Proportion of the dataset to include in the test split.
    val_size: float
        Proportion of the training set to include in the validation split.
    random_state: int
        Seed for random number generators to ensure reproducibility.
    stratify: bool
        Whether to stratify the split based on the target variable.
    """

    file_path: str
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    stratify: bool = True


@dataclass
class ModelConfig:
    """Configuration for model construction.

    Attributes
    ----------
    model_type: str
        Identifier for the model type (e.g. 'logistic', 'decision_tree',
        'random_forest', 'xgboost', 'catboost').
    params: Dict[str, Any]
        Hyperparameters to pass to the model constructor. The keys and
        values depend on the specific model type.
    """

    model_type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing strategies.

    Attributes
    ----------
    missing_strategy: str
        Strategy to handle missing values (e.g. 'median', 'mean', 'drop').
    encoding_strategy: str
        Encoding method for categorical variables ('onehot', 'label', 'ordinal').
    scaling_strategy: str
        Scaling method for numeric variables ('standard', 'minmax', 'none').
    binning_strategy: str
        Binning scheme for continuous variables ('tenure', 'none').
    """

    missing_strategy: str = 'median'
    encoding_strategy: str = 'onehot'
    scaling_strategy: str = 'standard'
    binning_strategy: str = 'tenure'
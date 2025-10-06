"""
LightGBM Model Configuration for StocketAI

Implements LightGBM baseline model configuration optimized for VN30 stock prediction.
"""

import logging
from typing import Dict, Any, List

from .base_config import BaseModelConfig

logger = logging.getLogger(__name__)


class LightGBMConfig(BaseModelConfig):
    """
    LightGBM model configuration.

    Based on qlib's LGBModel implementation with VN30-specific optimizations.
    """

    @property
    def model_type(self) -> str:
        return "lightgbm"

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default LightGBM configuration parameters."""
        return {
            # Core parameters
            "objective": "regression",
            "boosting": "gbdt",
            "loss": "mse",

            # Hyperparameters (optimized for VN30)
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,

            # Training parameters
            "early_stopping_rounds": 50,
            "num_boost_round": 1000,
            "verbose": -1,  # Suppress output

            # Additional parameters
            "min_child_samples": 20,
            "min_child_weight": 0.001,
            "subsample_freq": 1,
            "feature_fraction_seed": 42,
            "bagging_seed": 42,
        }

    def _get_required_keys(self) -> List[str]:
        """Get required configuration keys."""
        return [
            "objective",
            "boosting",
            "loss",
            "learning_rate",
            "max_depth",
            "num_leaves",
            "early_stopping_rounds",
            "num_boost_round"
        ]

    def _validate_parameter_ranges(self) -> None:
        """Validate LightGBM-specific parameter ranges."""
        super()._validate_parameter_ranges()

        # LightGBM-specific validations
        if not (0.0 < self.config['learning_rate'] <= 1.0):
            raise ValueError("Learning rate must be between 0.0 and 1.0")

        if self.config['max_depth'] < 1:
            raise ValueError("Max depth must be positive")

        if self.config['num_leaves'] < 2:
            raise ValueError("Number of leaves must be at least 2")

        if not (0.0 < self.config['colsample_bytree'] <= 1.0):
            raise ValueError("Column sample by tree must be between 0.0 and 1.0")

        if not (0.0 < self.config['subsample'] <= 1.0):
            raise ValueError("Subsample must be between 0.0 and 1.0")

        if self.config['lambda_l1'] < 0:
            raise ValueError("L1 regularization must be non-negative")

        if self.config['lambda_l2'] < 0:
            raise ValueError("L2 regularization must be non-negative")

        if self.config['early_stopping_rounds'] < 1:
            raise ValueError("Early stopping rounds must be positive")

        if self.config['num_boost_round'] < 1:
            raise ValueError("Number of boost rounds must be positive")

    def _get_qlib_model_class(self) -> str:
        """Get qlib model class name for LightGBM."""
        return "LGBModel"

    def _get_qlib_module_path(self) -> str:
        """Get qlib module path for LightGBM."""
        return "qlib.contrib.model.gbdt"

    def get_lightgbm_params(self) -> Dict[str, Any]:
        """
        Get LightGBM parameters dictionary.

        Returns:
            Parameters dictionary for lightgbm.train()
        """
        # Exclude qlib-specific parameters
        exclude_keys = ['loss']  # loss is handled separately in qlib

        params = {}
        for key, value in self.config.items():
            if key not in exclude_keys:
                params[key] = value

        return params

    def get_qlib_specific_params(self) -> Dict[str, Any]:
        """
        Get qlib-specific parameters.

        Returns:
            Qlib-specific parameters dictionary
        """
        return {
            "loss": self.config["loss"],
            "early_stopping_rounds": self.config["early_stopping_rounds"],
            "num_boost_round": self.config["num_boost_round"],
        }

    def optimize_for_memory(self) -> None:
        """
        Optimize configuration for memory efficiency.

        Adjusts parameters to reduce memory usage while maintaining performance.
        """
        logger.info("Optimizing LightGBM configuration for memory efficiency")

        # Reduce memory-intensive parameters
        self.config.update({
            "max_depth": min(self.config["max_depth"], 6),
            "num_leaves": min(self.config["num_leaves"], 64),
            "colsample_bytree": min(self.config["colsample_bytree"], 0.8),
            "subsample": min(self.config["subsample"], 0.8),
        })

        logger.info("Memory optimization applied")

    def optimize_for_speed(self) -> None:
        """
        Optimize configuration for training speed.

        Adjusts parameters to improve training speed while maintaining accuracy.
        """
        logger.info("Optimizing LightGBM configuration for training speed")

        # Speed optimization parameters
        self.config.update({
            "max_depth": min(self.config["max_depth"], 5),
            "num_leaves": min(self.config["num_leaves"], 32),
            "learning_rate": max(self.config["learning_rate"], 0.1),
            "colsample_bytree": 0.8,
            "subsample": 0.8,
        })

        logger.info("Speed optimization applied")

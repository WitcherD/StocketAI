"""
TFT (Temporal Fusion Transformer) Model Configuration for StocketAI

Implements TFT model configuration optimized for VN30 stock prediction.
"""

import logging
from typing import Dict, Any, List

from .base_config import BaseModelConfig

logger = logging.getLogger(__name__)


class TFTConfig(BaseModelConfig):
    """
    TFT (Temporal Fusion Transformer) model configuration.

    Based on qlib's TFT implementation with VN30-specific optimizations.
    """

    @property
    def model_type(self) -> str:
        return "tft"

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default TFT configuration parameters."""
        return {
            # Model architecture
            "input_dim": 158,  # Alpha158 features from Task 04
            "hidden_dim": 64,
            "attention_dim": 32,
            "num_heads": 4,
            "dropout": 0.1,

            # Training parameters
            "lr": 0.001,
            "batch_size": 1024,
            "epochs": 50,

            # TFT-specific parameters
            "num_quantiles": 3,
            "valid_quantiles": [0.1, 0.5, 0.9],
            "total_time_steps": 6 + 6,  # encoder + decoder steps
            "num_encoder_steps": 6,
            "num_epochs": 100,
            "early_stopping_patience": 10,
            "multiprocessing_workers": 5,

            # Model hyperparameters
            "dropout_rate": 0.4,
            "hidden_layer_size": 160,
            "learning_rate": 0.0001,
            "minibatch_size": 128,
            "max_gradient_norm": 0.0135,
            "stack_size": 1,

            # Data processing
            "label_shift": 5,  # Prediction horizon
            "step_len": 20,  # Sequence length for time series
        }

    def _get_required_keys(self) -> List[str]:
        """Get required configuration keys."""
        return [
            "input_dim",
            "hidden_dim",
            "attention_dim",
            "num_heads",
            "dropout",
            "lr",
            "batch_size",
            "epochs",
            "num_quantiles",
            "valid_quantiles"
        ]

    def _validate_parameter_ranges(self) -> None:
        """Validate TFT-specific parameter ranges."""
        super()._validate_parameter_ranges()

        # TFT-specific validations
        if self.config['input_dim'] < 1:
            raise ValueError("Input dimension must be positive")

        if self.config['hidden_dim'] < 1:
            raise ValueError("Hidden dimension must be positive")

        if self.config['attention_dim'] < 1:
            raise ValueError("Attention dimension must be positive")

        if not (1 <= self.config['num_heads'] <= 16):
            raise ValueError("Number of heads must be between 1 and 16")

        if not (0.0 <= self.config['dropout'] <= 1.0):
            raise ValueError("Dropout must be between 0.0 and 1.0")

        if self.config['num_quantiles'] < 1:
            raise ValueError("Number of quantiles must be positive")

        if not all(0.0 <= q <= 1.0 for q in self.config['valid_quantiles']):
            raise ValueError("Quantiles must be between 0.0 and 1.0")

    def _get_qlib_model_class(self) -> str:
        """Get qlib model class name for TFT."""
        return "TFTModel"

    def _get_qlib_module_path(self) -> str:
        """Get qlib module path for TFT."""
        return "tft"  # Custom module path as per qlib examples

    def get_tft_experiment_params(self) -> Dict[str, Any]:
        """
        Get TFT experiment parameters for qlib workflow.

        Returns:
            Experiment parameters dictionary
        """
        return {
            "total_time_steps": self.config["total_time_steps"],
            "num_encoder_steps": self.config["num_encoder_steps"],
            "num_epochs": self.config["num_epochs"],
            "early_stopping_patience": self.config["early_stopping_patience"],
            "multiprocessing_workers": self.config["multiprocessing_workers"],
        }

    def get_tft_model_params(self) -> Dict[str, Any]:
        """
        Get TFT model hyperparameters.

        Returns:
            Model parameters dictionary
        """
        return {
            "dropout_rate": self.config["dropout_rate"],
            "hidden_layer_size": self.config["hidden_layer_size"],
            "learning_rate": self.config["learning_rate"],
            "minibatch_size": self.config["minibatch_size"],
            "max_gradient_norm": self.config["max_gradient_norm"],
            "num_heads": self.config["num_heads"],
            "stack_size": self.config["stack_size"],
        }

    def get_data_formatter_config(self) -> Dict[str, Any]:
        """
        Get data formatter configuration for TFT.

        Returns:
            Data formatter configuration
        """
        return {
            "total_time_steps": self.config["total_time_steps"],
            "num_encoder_steps": self.config["num_encoder_steps"],
            "num_epochs": self.config["num_epochs"],
            "early_stopping_patience": self.config["early_stopping_patience"],
            "multiprocessing_workers": self.config["multiprocessing_workers"],
            "dropout_rate": self.config["dropout_rate"],
            "hidden_layer_size": self.config["hidden_layer_size"],
            "learning_rate": self.config["learning_rate"],
            "minibatch_size": self.config["minibatch_size"],
            "max_gradient_norm": self.config["max_gradient_norm"],
            "num_heads": self.config["num_heads"],
            "stack_size": self.config["stack_size"],
        }

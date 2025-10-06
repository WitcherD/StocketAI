"""
LSTM Model Configuration for StocketAI

Implements LSTM baseline model configuration optimized for VN30 stock prediction.
"""

import logging
from typing import Dict, Any, List

from .base_config import BaseModelConfig

logger = logging.getLogger(__name__)


class LSTMConfig(BaseModelConfig):
    """
    LSTM model configuration.

    Based on qlib's PyTorch LSTM implementation with VN30-specific optimizations.
    """

    @property
    def model_type(self) -> str:
        return "lstm"

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default LSTM configuration parameters."""
        return {
            # Model architecture
            "input_dim": 158,  # Alpha158 features from Task 04
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,

            # Training parameters
            "lr": 0.001,
            "batch_size": 2048,
            "epochs": 100,

            # LSTM-specific parameters
            "step_len": 20,  # Sequence length for time series
            "bidirectional": False,

            # Loss and metrics
            "loss": "mse",
            "metric": "loss",

            # Training control
            "early_stop": 10,
            "GPU": 0,  # GPU device ID, -1 for CPU

            # Additional parameters
            "n_jobs": 20,  # Number of parallel jobs
            "seed": 42,  # Random seed
        }

    def _get_required_keys(self) -> List[str]:
        """Get required configuration keys."""
        return [
            "input_dim",
            "hidden_dim",
            "num_layers",
            "dropout",
            "lr",
            "batch_size",
            "epochs",
            "step_len"
        ]

    def _validate_parameter_ranges(self) -> None:
        """Validate LSTM-specific parameter ranges."""
        super()._validate_parameter_ranges()

        # LSTM-specific validations
        if self.config['input_dim'] < 1:
            raise ValueError("Input dimension must be positive")

        if self.config['hidden_dim'] < 1:
            raise ValueError("Hidden dimension must be positive")

        if self.config['num_layers'] < 1:
            raise ValueError("Number of layers must be positive")

        if not (0.0 <= self.config['dropout'] <= 1.0):
            raise ValueError("Dropout must be between 0.0 and 1.0")

        if self.config['step_len'] < 1:
            raise ValueError("Step length (sequence length) must be positive")

        if self.config['early_stop'] < 1:
            raise ValueError("Early stop patience must be positive")

        if self.config['GPU'] < -1:
            raise ValueError("GPU device ID must be -1 (CPU) or non-negative")

    def _get_qlib_model_class(self) -> str:
        """Get qlib model class name for LSTM."""
        return "LSTM"

    def _get_qlib_module_path(self) -> str:
        """Get qlib module path for LSTM."""
        return "qlib.contrib.model.pytorch_lstm_ts"

    def get_lstm_architecture_params(self) -> Dict[str, Any]:
        """
        Get LSTM architecture parameters.

        Returns:
            Architecture parameters dictionary
        """
        return {
            "d_feat": self.config["input_dim"],
            "hidden_size": self.config["hidden_dim"],
            "num_layers": self.config["num_layers"],
            "dropout": self.config["dropout"],
        }

    def get_lstm_training_params(self) -> Dict[str, Any]:
        """
        Get LSTM training parameters.

        Returns:
            Training parameters dictionary
        """
        return {
            "n_epochs": self.config["epochs"],
            "lr": self.config["lr"],
            "early_stop": self.config["early_stop"],
            "batch_size": self.config["batch_size"],
            "metric": self.config["metric"],
            "loss": self.config["loss"],
            "GPU": self.config["GPU"],
            "n_jobs": self.config["n_jobs"],
        }

    def enable_bidirectional(self) -> None:
        """
        Enable bidirectional LSTM.

        Sets bidirectional flag to True for bidirectional processing.
        """
        logger.info("Enabling bidirectional LSTM")
        self.config["bidirectional"] = True

    def disable_bidirectional(self) -> None:
        """
        Disable bidirectional LSTM.

        Sets bidirectional flag to False for unidirectional processing.
        """
        logger.info("Disabling bidirectional LSTM")
        self.config["bidirectional"] = False

    def optimize_for_memory(self) -> None:
        """
        Optimize configuration for memory efficiency.

        Adjusts parameters to reduce memory usage while maintaining performance.
        """
        logger.info("Optimizing LSTM configuration for memory efficiency")

        # Reduce memory-intensive parameters
        self.config.update({
            "hidden_dim": min(self.config["hidden_dim"], 32),
            "num_layers": min(self.config["num_layers"], 1),
            "batch_size": min(self.config["batch_size"], 1024),
            "step_len": min(self.config["step_len"], 10),
        })

        logger.info("Memory optimization applied")

    def optimize_for_speed(self) -> None:
        """
        Optimize configuration for training speed.

        Adjusts parameters to improve training speed while maintaining accuracy.
        """
        logger.info("Optimizing LSTM configuration for training speed")

        # Speed optimization parameters
        self.config.update({
            "hidden_dim": min(self.config["hidden_dim"], 32),
            "num_layers": 1,
            "batch_size": max(self.config["batch_size"], 4096),
            "lr": max(self.config["lr"], 0.01),
        })

        logger.info("Speed optimization applied")

    def set_gpu_device(self, gpu_id: int) -> None:
        """
        Set GPU device for training.

        Args:
            gpu_id: GPU device ID (-1 for CPU, 0+ for GPU)
        """
        if gpu_id < -1:
            raise ValueError("GPU device ID must be -1 (CPU) or non-negative")

        self.config["GPU"] = gpu_id
        logger.info(f"Set GPU device to {gpu_id}")

    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data configuration for LSTM.

        Returns:
            Data configuration dictionary for qlib workflow
        """
        return {
            "step_len": self.config["step_len"],
            # Additional data processing parameters can be added here
        }

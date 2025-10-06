"""
Base Model Configuration for StocketAI

Provides common configuration and utilities for all baseline models.
"""

import os
import yaml
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata for tracking and versioning."""
    symbol: str
    model_type: str
    version: str = "v1.0"
    training_date: Optional[str] = None
    config_hash: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None


class BaseModelConfig(ABC):
    """
    Base configuration class for all baseline models.

    Provides common functionality for model configuration, validation,
    and metadata management.
    """

    def __init__(self, symbol: str, **kwargs):
        """
        Initialize base model configuration.

        Args:
            symbol: VN30 stock symbol
            **kwargs: Additional configuration parameters
        """
        self.symbol = symbol
        self.config = self._get_default_config()
        self.config.update(kwargs)
        self.metadata = ModelMetadata(symbol=symbol, model_type=self.model_type)

        # Validate configuration
        self._validate_config()

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Model type identifier."""
        pass

    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        pass

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_keys = self._get_required_keys()
        missing_keys = [key for key in required_keys if key not in self.config]

        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        # Validate parameter types and ranges
        self._validate_parameter_types()
        self._validate_parameter_ranges()

    @abstractmethod
    def _get_required_keys(self) -> List[str]:
        """Get list of required configuration keys."""
        pass

    def _validate_parameter_types(self) -> None:
        """Validate parameter types."""
        # Common type validations
        if 'lr' in self.config and not isinstance(self.config['lr'], (int, float)):
            raise TypeError("Learning rate (lr) must be numeric")

        if 'batch_size' in self.config and not isinstance(self.config['batch_size'], int):
            raise TypeError("Batch size must be integer")

        if 'epochs' in self.config and not isinstance(self.config['epochs'], int):
            raise TypeError("Epochs must be integer")

    def _validate_parameter_ranges(self) -> None:
        """Validate parameter ranges."""
        # Common range validations
        if 'lr' in self.config and not (1e-6 <= self.config['lr'] <= 1.0):
            raise ValueError("Learning rate must be between 1e-6 and 1.0")

        if 'batch_size' in self.config and self.config['batch_size'] < 1:
            raise ValueError("Batch size must be positive")

        if 'epochs' in self.config and self.config['epochs'] < 1:
            raise ValueError("Epochs must be positive")

    def get_model_path(self, base_dir: str = "models") -> str:
        """
        Get model storage path.

        Args:
            base_dir: Base models directory

        Returns:
            Full path to model directory
        """
        return os.path.join(base_dir, "symbols", self.symbol, self.model_type)

    def save_config(self, path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save configuration
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        config_data = {
            'model_type': self.model_type,
            'symbol': self.symbol,
            'config': self.config,
            'metadata': asdict(self.metadata)
        }

        with open(path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {path}")

    def load_config(self, path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            path: Path to load configuration from
        """
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)

        self.config = config_data.get('config', {})
        metadata_dict = config_data.get('metadata', {})
        self.metadata = ModelMetadata(**metadata_dict)

        logger.info(f"Configuration loaded from {path}")

    def update_metadata(self, **kwargs) -> None:
        """
        Update model metadata.

        Args:
            **kwargs: Metadata fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
            else:
                logger.warning(f"Unknown metadata field: {key}")

    def get_qlib_config(self) -> Dict[str, Any]:
        """
        Get qlib-compatible configuration.

        Returns:
            Configuration dictionary for qlib workflow
        """
        return {
            "class": self._get_qlib_model_class(),
            "module_path": self._get_qlib_module_path(),
            "kwargs": self.config.copy()
        }

    @abstractmethod
    def _get_qlib_model_class(self) -> str:
        """Get qlib model class name."""
        pass

    @abstractmethod
    def _get_qlib_module_path(self) -> str:
        """Get qlib module path."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(symbol={self.symbol}, model_type={self.model_type})"

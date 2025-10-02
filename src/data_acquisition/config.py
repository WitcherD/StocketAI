"""
Configuration module for data acquisition components.

This module provides configuration management for data acquisition,
including API settings, rate limiting, caching, and validation parameters.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yaml


@dataclass
class DataAcquisitionConfig:
    """Configuration for data acquisition operations."""

    # API settings
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    timeout: int = 30

    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limits: Dict[str, int] = None  # requests per minute per source
    rate_limit_buffer: float = 0.1  # 10% buffer

    # Caching
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds

    # Data validation
    max_missing_ratio: float = 0.1  # 10% missing values allowed

    def __post_init__(self):
        """Initialize default rate limits if not provided."""
        if self.rate_limits is None:
            self.rate_limits = {
                'VCI': 100,      # Vietstock
                'TCBS': 50,      # Techcom Securities
                'MSN': 200,      # MSN Money
                'FMARKET': 30    # Fmarket
            }


def load_config_from_yaml(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to config/config.yaml relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / 'config' / 'config.yaml'

    if not config_path.exists():
        # Return default configuration if file doesn't exist
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        # Return empty dict on error
        return {}


def get_config() -> DataAcquisitionConfig:
    """Get data acquisition configuration."""
    # Load from YAML if available
    yaml_config = load_config_from_yaml()

    # Extract data acquisition settings
    data_config = yaml_config.get('data', {})
    api_config = yaml_config.get('api', {})

    # Build rate limits from config
    rate_limits = {}
    if 'vnstock' in api_config:
        # Use vnstock specific rate limit if available
        rate_limits['VCI'] = api_config['vnstock'].get('timeout', 100)
        rate_limits['TCBS'] = api_config['vnstock'].get('timeout', 50)
        rate_limits['MSN'] = api_config['vnstock'].get('timeout', 200)
        rate_limits['FMARKET'] = api_config['vnstock'].get('timeout', 30)

    # Create configuration object
    config = DataAcquisitionConfig(
        max_retries=api_config.get('vnstock', {}).get('max_retries', 3),
        retry_delay=1.0,
        backoff_factor=api_config.get('vnstock', {}).get('retry_backoff', 2.0),
        timeout=api_config.get('vnstock', {}).get('timeout', 30),
        enable_rate_limiting=True,
        rate_limits=rate_limits if rate_limits else None,
        rate_limit_buffer=0.1,
        cache_enabled=True,
        cache_ttl=3600,
        max_missing_ratio=data_config.get('validation', {}).get('missing_value_threshold', 0.1)
    )

    return config


def get_project_config() -> Dict[str, Any]:
    """Get the full project configuration."""
    return load_config_from_yaml()


# Global configuration instance
_config_instance: Optional[DataAcquisitionConfig] = None


def get_global_config() -> DataAcquisitionConfig:
    """Get global configuration instance (singleton pattern)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = get_config()
    return _config_instance


def reload_config() -> None:
    """Reload configuration from files."""
    global _config_instance
    _config_instance = get_config()

"""
Data Acquisition Module for StocketAI

This module provides interfaces for acquiring financial data from various sources,
including vnstock, external APIs, and other data providers. It implements
provider abstraction, multi-source fallback mechanisms, and data caching.

Key Components:
- VNStock client wrapper with error handling and caching
- Provider abstraction layer for extensible data sources
- Data validation and quality checks
- Rate limiting and retry mechanisms
- Multi-source data merging and integration

The module follows the project's coding standards and integrates with
the existing data processing pipeline.
"""

from .config import get_config, DataAcquisitionConfig
from .vnstock_client import VNStockClient, DataValidator, DataCache, RateLimiter
from .qlib_converter import QLibConverter

__version__ = "1.0.0"
__all__ = [
    "get_config",
    "DataAcquisitionConfig",
    "VNStockClient",
    "DataValidator",
    "DataCache",
    "RateLimiter",
    "QlibConverter"
]

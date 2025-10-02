"""
Data Processing Module for VN30 Stock Price Prediction System.

This module provides comprehensive data cleaning, integration, and validation
pipelines for processing raw financial data into formats suitable for
feature engineering and model training.

Key Components:
- Data cleaning and imputation
- Outlier detection and treatment
- Data normalization and scaling
- Multi-source data integration
- Data quality validation and monitoring
- Processed data storage and management

The module follows the project's coding standards and integrates with
the existing data acquisition pipeline.
"""

from .cleaning import DataCleaner
from .integration import DataIntegrator
from .validation import DataValidator
from .normalization import DataNormalizer
from .storage import ProcessedDataManager
from .time_series import TimeSeriesProcessor

__version__ = "1.0.0"
__all__ = [
    "DataCleaner",
    "DataIntegrator",
    "DataValidator",
    "DataNormalizer",
    "ProcessedDataManager",
    "TimeSeriesProcessor"
]

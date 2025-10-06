"""
Feature Engineering Module for StocketAI

This module provides comprehensive feature engineering capabilities optimized
for VN30 stock prediction using TFT (Temporal Fusion Transformers) model.
Implements Alpha158-compatible technical indicators and feature categories.

Key Components:
- TFT-compatible feature categories (OBSERVED_INPUT, KNOWN_INPUT, STATIC_INPUT)
- Technical indicators: RESI, WVMA, RSQR, CORR, CORD, ROC, VSTD, STD, KLEN, KLOW
- VN30-specific feature optimization
- Qlib expression engine integration
- Feature validation and quality assessment
- Qlib data management and loading utilities

The module follows the project's coding standards and integrates with
the data processing and model training pipelines.
"""

__version__ = "1.0.0"

from .tft_feature_engineer import VN30TFTFeatureEngineer

__all__ = [
    'VN30TFTFeatureEngineer'
]

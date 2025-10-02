"""
StocketAI - Stock Price Prediction System

A comprehensive framework for quantitative stock price prediction using
machine learning and time series analysis. Built with vnstock, qlib,
and modern Python data science stack.

Main Components:
- data_acquisition: Multi-source data collection and caching
- data_processing: Data cleaning, validation, and preprocessing
- feature_engineering: Feature generation and engineering
- model_training: Model training and hyperparameter optimization
- prediction: Inference and prediction serving
- evaluation: Backtesting and performance evaluation
- reporting: Report generation and visualization

The system follows modular architecture with provider abstraction,
multi-source fallback mechanisms, and production-ready error handling.
"""

from . import data_acquisition
from . import data_processing

__version__ = "1.0.0"
__all__ = [
    "data_acquisition",
    "data_processing"
]

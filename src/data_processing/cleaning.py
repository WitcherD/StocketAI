"""
Data Cleaning Module for VN30 Stock Price Prediction System.

This module provides comprehensive data cleaning functionality including:
- Missing data detection and imputation
- Outlier detection and treatment
- Data normalization and scaling
- Time-series specific cleaning operations

The module is designed to handle financial time-series data with appropriate
strategies for different data types (prices, volumes, fundamentals).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""

    # Missing data imputation
    imputation_method: str = "forward_fill"  # forward_fill, interpolation, mean, median
    max_consecutive_missing: int = 5
    fill_missing_with_zero: bool = False

    # Outlier detection
    outlier_method: str = "iqr"  # iqr, zscore, modified_zscore
    outlier_threshold: float = 3.0
    remove_outliers: bool = False
    winsorize_outliers: bool = True
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)

    # Data validation
    min_price: float = 0.01
    max_price: float = 1e8
    min_volume: int = 0
    max_volume: int = 1e10

    # Time-series specific
    remove_weekends: bool = True
    handle_trading_holidays: bool = True
    ensure_monotonic_index: bool = True


class DataCleaner:
    """
    Comprehensive data cleaning class for financial time-series data.

    Handles missing data, outliers, normalization, and financial data specific
    cleaning operations with configurable strategies.
    """

    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize the DataCleaner with configuration.

        Args:
            config: CleaningConfig object with cleaning parameters
        """
        self.config = config or CleaningConfig()
        self.cleaning_stats = {}

    def clean_constituents_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean VN30 constituents data.

        Args:
            df: Raw constituents DataFrame

        Returns:
            Cleaned constituents DataFrame
        """
        logger.info("Starting constituents data cleaning")

        cleaned_df = df.copy()

        # Standardize column names
        cleaned_df.columns = cleaned_df.columns.str.lower().str.strip()

        # Handle missing values
        cleaned_df = self._handle_missing_constituents_data(cleaned_df)

        # Validate data types
        cleaned_df = self._validate_constituents_dtypes(cleaned_df)

        # Clean string fields
        cleaned_df = self._clean_string_fields(cleaned_df)

        # Validate business rules
        cleaned_df = self._validate_constituents_business_rules(cleaned_df)

        logger.info(f"Constituents cleaning completed. Shape: {cleaned_df.shape}")
        return cleaned_df

    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV price data.

        Args:
            df: Raw price DataFrame with columns like open, high, low, close, volume

        Returns:
            Cleaned price DataFrame
        """
        logger.info("Starting price data cleaning")

        cleaned_df = df.copy()

        # Ensure datetime index
        if not isinstance(cleaned_df.index, pd.DatetimeIndex):
            cleaned_df.index = pd.to_datetime(cleaned_df.index)

        # Sort by date
        cleaned_df = cleaned_df.sort_index()

        # Handle missing values
        cleaned_df = self._handle_missing_price_data(cleaned_df)

        # Detect and handle outliers
        cleaned_df = self._handle_price_outliers(cleaned_df)

        # Validate price relationships
        cleaned_df = self._validate_price_relationships(cleaned_df)

        # Remove non-trading days if configured
        if self.config.remove_weekends:
            cleaned_df = self._remove_weekends(cleaned_df)

        logger.info(f"Price data cleaning completed. Shape: {cleaned_df.shape}")
        return cleaned_df

    def _handle_missing_constituents_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data in constituents DataFrame."""
        missing_stats = {}

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_stats[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            }

            # Handle different column types
            if col in ['sector', 'industry']:
                # Fill with 'Unknown' for categorical fields
                df[col] = df[col].fillna('Unknown')
            elif col in ['shares_outstanding', 'market_cap']:
                # These are often null in raw data, leave as null for now
                pass
            elif col == 'weight':
                # Weights should never be null, use equal weight if missing
                if missing_count > 0:
                    equal_weight = 100.0 / len(df)
                    df[col] = df[col].fillna(equal_weight)
            else:
                # For other fields, use forward fill
                df[col] = df[col].fillna(method='ffill')

        self.cleaning_stats['constituents_missing'] = missing_stats
        return df

    def _validate_constituents_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types for constituents data."""
        # Symbol should be string
        df['symbol'] = df['symbol'].astype(str)

        # Weight should be float
        if 'weight' in df.columns:
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

        # Shares outstanding should be numeric
        if 'shares_outstanding' in df.columns:
            df['shares_outstanding'] = pd.to_numeric(df['shares_outstanding'], errors='coerce')

        # Market cap should be numeric
        if 'market_cap' in df.columns:
            df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce')

        return df

    def _clean_string_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean string fields in constituents data."""
        string_columns = ['symbol', 'name', 'sector', 'industry']

        for col in string_columns:
            if col in df.columns:
                df[col] = (df[col]
                          .astype(str)
                          .str.strip()
                          .str.upper()
                          .str.replace(r'\s+', ' ', regex=True))

        return df

    def _validate_constituents_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate business rules for constituents data."""
        # Weights should sum to approximately 100%
        if 'weight' in df.columns:
            total_weight = df['weight'].sum()
            if abs(total_weight - 100.0) > 0.1:  # Allow 0.1% tolerance
                logger.warning(f"Total weight is {total_weight:.2f}%, expected ~100%")

        # No duplicate symbols
        duplicates = df['symbol'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate symbols")

        # Validate weight ranges
        if 'weight' in df.columns:
            invalid_weights = df[(df['weight'] < 0) | (df['weight'] > 50)].shape[0]
            if invalid_weights > 0:
                logger.warning(f"Found {invalid_weights} stocks with invalid weights")

        return df

    def _handle_missing_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data in price DataFrame."""
        missing_stats = {}

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_stats[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            }

            if missing_count > 0:
                if self.config.imputation_method == "forward_fill":
                    df[col] = df[col].fillna(method='ffill')
                elif self.config.imputation_method == "interpolation":
                    df[col] = df[col].interpolate(method='linear')
                elif self.config.imputation_method == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif self.config.imputation_method == "median":
                    df[col] = df[col].fillna(df[col].median())

        self.cleaning_stats['price_missing'] = missing_stats
        return df

    def _handle_price_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in price data."""
        outlier_stats = {}

        for col in ['high', 'low', 'close', 'open', 'volume']:
            if col not in df.columns:
                continue

            if self.config.outlier_method == "iqr":
                outliers = self._detect_outliers_iqr(df[col])
            elif self.config.outlier_method == "zscore":
                outliers = self._detect_outliers_zscore(df[col])
            else:
                outliers = pd.Series(False, index=df.index)

            outlier_count = outliers.sum()
            outlier_stats[col] = outlier_count

            if outlier_count > 0:
                if self.config.remove_outliers:
                    df = df[~outliers]
                    logger.info(f"Removed {outlier_count} outliers from {col}")
                elif self.config.winsorize_outliers:
                    df[col] = self._winsorize_series(df[col])
                    logger.info(f"Winsorized {outlier_count} outliers in {col}")

        self.cleaning_stats['outliers'] = outlier_stats
        return df

    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return (series < lower_bound) | (series > upper_bound)

    def _detect_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > self.config.outlier_threshold

    def _winsorize_series(self, series: pd.Series) -> pd.Series:
        """Winsorize a series to handle outliers."""
        return series.clip(
            lower=series.quantile(self.config.winsorize_limits[0]),
            upper=series.quantile(self.config.winsorize_limits[1])
        )

    def _validate_price_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate price relationship constraints."""
        validation_errors = {}

        # High >= max(Open, Close)
        if all(col in df.columns for col in ['high', 'open', 'close']):
            high_errors = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
            if high_errors > 0:
                validation_errors['high_vs_oc'] = high_errors

        # Low <= min(Open, Close)
        if all(col in df.columns for col in ['low', 'open', 'close']):
            low_errors = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
            if low_errors > 0:
                validation_errors['low_vs_oc'] = low_errors

        # Open, High, Low, Close should be positive
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    validation_errors[f'{col}_negative'] = negative_count

        # Volume should be non-negative
        if 'volume' in df.columns:
            negative_vol = (df['volume'] < 0).sum()
            if negative_vol > 0:
                validation_errors['volume_negative'] = negative_vol

        self.cleaning_stats['validation_errors'] = validation_errors

        if validation_errors:
            logger.warning(f"Found {len(validation_errors)} types of validation errors")

        return df

    def _remove_weekends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove weekend data from price DataFrame."""
        weekend_days = df.index.weekday >= 5  # Saturday = 5, Sunday = 6
        weekend_count = weekend_days.sum()

        if weekend_count > 0:
            df = df[~weekend_days]
            logger.info(f"Removed {weekend_count} weekend records")

        return df

    def get_cleaning_stats(self) -> Dict:
        """Get comprehensive cleaning statistics."""
        return {
            'config': self.config.__dict__,
            'cleaning_stats': self.cleaning_stats,
            'timestamp': datetime.now().isoformat()
        }

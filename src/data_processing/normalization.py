"""
Data Normalization Module for VN30 Stock Price Prediction System.

This module provides data normalization and scaling functionality for
financial time-series data, including:

- Min-max scaling and standardization
- Robust scaling for financial data
- Log transformations and power transforms
- Feature scaling configuration management
- Time-series specific normalization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Available normalization methods."""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"
    LOG = "log"
    POWER = "power"
    QUANTILE = "quantile"


class ScalingScope(Enum):
    """Scope of scaling operations."""
    GLOBAL = "global"  # Use entire dataset statistics
    ROLLING = "rolling"  # Use rolling window statistics
    GROUP = "group"  # Group by categories
    INDIVIDUAL = "individual"  # Individual scaling per feature


@dataclass
class NormalizationConfig:
    """Configuration for data normalization operations."""
    method: NormalizationMethod = NormalizationMethod.Z_SCORE
    scope: ScalingScope = ScalingScope.GLOBAL
    feature_range: Tuple[float, float] = (0, 1)  # For min-max scaling
    rolling_window: int = 252  # Trading days in a year
    quantile_range: Tuple[float, float] = (0.05, 0.95)  # For quantile scaling
    handle_outliers: bool = True
    preserve_sparsity: bool = False
    copy: bool = True


class DataNormalizer:
    """
    Comprehensive data normalization and scaling system.

    Provides various normalization methods optimized for financial time-series
    data with configurable scaling strategies and outlier handling.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize the DataNormalizer.

        Args:
            config: NormalizationConfig object with normalization parameters
        """
        self.config = config or NormalizationConfig()
        self.scalers = {}
        self.normalization_stats = {}

    def normalize_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize OHLCV price data.

        Args:
            df: Price DataFrame to normalize

        Returns:
            Normalized price DataFrame
        """
        logger.info("Starting price data normalization")

        normalized_df = df.copy() if self.config.copy else df

        # Identify price and volume columns
        price_columns = ['open', 'high', 'low', 'close']
        volume_columns = ['volume']
        other_columns = [col for col in df.columns
                        if col not in price_columns + volume_columns]

        # Normalize price columns
        for col in price_columns:
            if col in normalized_df.columns:
                normalized_df[col] = self._normalize_series(
                    normalized_df[col], f"price_{col}"
                )

        # Normalize volume columns (often needs different treatment)
        for col in volume_columns:
            if col in normalized_df.columns:
                normalized_df[col] = self._normalize_volume_series(
                    normalized_df[col], f"volume_{col}"
                )

        # Normalize other numeric columns
        for col in other_columns:
            if pd.api.types.is_numeric_dtype(normalized_df[col]):
                normalized_df[col] = self._normalize_series(
                    normalized_df[col], f"other_{col}"
                )

        logger.info(f"Price data normalization completed. Shape: {normalized_df.shape}")
        return normalized_df

    def normalize_constituents_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize constituents data.

        Args:
            df: Constituents DataFrame to normalize

        Returns:
            Normalized constituents DataFrame
        """
        logger.info("Starting constituents data normalization")

        normalized_df = df.copy() if self.config.copy else df

        # Identify different column types
        numeric_columns = []
        categorical_columns = []

        for col in normalized_df.columns:
            if pd.api.types.is_numeric_dtype(normalized_df[col]):
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)

        # Normalize numeric columns
        for col in numeric_columns:
            if col not in ['symbol']:  # Skip symbol column
                normalized_df[col] = self._normalize_series(
                    normalized_df[col], f"constituents_{col}"
                )

        # Encode categorical columns if needed
        for col in categorical_columns:
            if col not in ['symbol']:  # Skip symbol column
                normalized_df[col] = self._encode_categorical(
                    normalized_df[col], f"constituents_{col}"
                )

        logger.info(f"Constituents normalization completed. Shape: {normalized_df.shape}")
        return normalized_df

    def _normalize_series(self, series: pd.Series, feature_name: str) -> pd.Series:
        """Normalize a single series using configured method."""
        if series.isnull().all():
            logger.warning(f"Series {feature_name} is all null, skipping normalization")
            return series

        # Handle outliers if configured
        if self.config.handle_outliers:
            series = self._handle_outliers_in_series(series)

        # Apply normalization method
        if self.config.method == NormalizationMethod.MIN_MAX:
            return self._min_max_normalize(series, feature_name)
        elif self.config.method == NormalizationMethod.Z_SCORE:
            return self._z_score_normalize(series, feature_name)
        elif self.config.method == NormalizationMethod.ROBUST:
            return self._robust_normalize(series, feature_name)
        elif self.config.method == NormalizationMethod.LOG:
            return self._log_normalize(series, feature_name)
        elif self.config.method == NormalizationMethod.POWER:
            return self._power_normalize(series, feature_name)
        elif self.config.method == NormalizationMethod.QUANTILE:
            return self._quantile_normalize(series, feature_name)
        else:
            logger.warning(f"Unknown normalization method: {self.config.method}")
            return series

    def _normalize_volume_series(self, series: pd.Series, feature_name: str) -> pd.Series:
        """Normalize volume series with volume-specific handling."""
        # Volume often benefits from log transformation
        if self.config.method == NormalizationMethod.LOG:
            # Use log normalization for volume
            return self._log_normalize(series, feature_name)
        else:
            # Use standard normalization for volume
            return self._normalize_series(series, feature_name)

    def _min_max_normalize(self, series: pd.Series, feature_name: str) -> pd.Series:
        """Apply min-max normalization."""
        if self.config.scope == ScalingScope.GLOBAL:
            min_val = series.min()
            max_val = series.max()

            if max_val == min_val:
                logger.warning(f"Constant series {feature_name}, returning zeros")
                return pd.Series(0, index=series.index)

            normalized = (series - min_val) / (max_val - min_val)

            # Scale to feature range if specified
            if self.config.feature_range != (0, 1):
                min_range, max_range = self.config.feature_range
                normalized = normalized * (max_range - min_range) + min_range

            return normalized

        else:
            # For rolling or group scaling, use sklearn
            scaler = MinMaxScaler(feature_range=self.config.feature_range)
            values = series.values.reshape(-1, 1)
            normalized = scaler.fit_transform(values).flatten()

            # Store scaler for inverse transform if needed
            self.scalers[feature_name] = scaler

            return pd.Series(normalized, index=series.index)

    def _z_score_normalize(self, series: pd.Series, feature_name: str) -> pd.Series:
        """Apply z-score normalization."""
        if self.config.scope == ScalingScope.GLOBAL:
            mean_val = series.mean()
            std_val = series.std()

            if std_val == 0:
                logger.warning(f"Zero standard deviation for {feature_name}, returning zeros")
                return pd.Series(0, index=series.index)

            return (series - mean_val) / std_val

        else:
            # Use sklearn for rolling/group scaling
            scaler = StandardScaler()
            values = series.values.reshape(-1, 1)
            normalized = scaler.fit_transform(values).flatten()

            self.scalers[feature_name] = scaler
            return pd.Series(normalized, index=series.index)

    def _robust_normalize(self, series: pd.Series, feature_name: str) -> pd.Series:
        """Apply robust normalization using median and IQR."""
        if self.config.scope == ScalingScope.GLOBAL:
            median_val = series.median()
            q75, q25 = series.quantile([0.75, 0.25])
            iqr_val = q75 - q25

            if iqr_val == 0:
                logger.warning(f"Zero IQR for {feature_name}, using z-score normalization")
                return self._z_score_normalize(series, feature_name)

            return (series - median_val) / iqr_val

        else:
            # Use sklearn RobustScaler
            scaler = RobustScaler(quantile_range=self.config.quantile_range)
            values = series.values.reshape(-1, 1)
            normalized = scaler.fit_transform(values).flatten()

            self.scalers[feature_name] = scaler
            return pd.Series(normalized, index=series.index)

    def _log_normalize(self, series: pd.Series, feature_name: str) -> pd.Series:
        """Apply log normalization."""
        # Handle zeros and negative values
        min_positive = series[series > 0].min() if (series > 0).any() else 1

        # Shift series to make all values positive
        shifted_series = series + abs(series.min()) + min_positive

        # Apply log transformation
        log_series = np.log(shifted_series)

        # Apply standard normalization to log-transformed data
        return self._z_score_normalize(log_series, f"log_{feature_name}")

    def _power_normalize(self, series: pd.Series, feature_name: str) -> pd.Series:
        """Apply power transformation."""
        # Use Yeo-Johnson or Box-Cox transformation
        from sklearn.preprocessing import PowerTransformer

        # Handle non-positive values
        if (series <= 0).any():
            # Use Yeo-Johnson which can handle negative values
            transformer = PowerTransformer(method='yeo-johnson')
        else:
            # Use Box-Cox for positive values
            transformer = PowerTransformer(method='box-cox')

        values = series.values.reshape(-1, 1)
        transformed = transformer.fit_transform(values).flatten()

        self.scalers[feature_name] = transformer
        return pd.Series(transformed, index=series.index)

    def _quantile_normalize(self, series: pd.Series, feature_name: str) -> pd.Series:
        """Apply quantile normalization."""
        lower_q, upper_q = self.config.quantile_range

        # Calculate quantiles
        lower_bound = series.quantile(lower_q)
        upper_bound = series.quantile(upper_q)

        # Clip outliers
        clipped_series = series.clip(lower=lower_bound, upper=upper_bound)

        # Apply robust normalization
        return self._robust_normalize(clipped_series, feature_name)

    def _handle_outliers_in_series(self, series: pd.Series) -> pd.Series:
        """Handle outliers in a series before normalization."""
        # Use IQR method to detect and handle outliers
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Clip outliers
        return series.clip(lower=lower_bound, upper=upper_bound)

    def _encode_categorical(self, series: pd.Series, feature_name: str) -> pd.Series:
        """Encode categorical variables."""
        # Simple label encoding for categorical variables
        # For more sophisticated encoding, consider one-hot or target encoding
        unique_values = series.unique()
        encoding_map = {val: idx for idx, val in enumerate(unique_values)}

        return series.map(encoding_map)

    def inverse_normalize(self, normalized_df: pd.DataFrame, original_stats: Dict) -> pd.DataFrame:
        """
        Inverse normalize data back to original scale.

        Args:
            normalized_df: Normalized DataFrame
            original_stats: Statistics from original data

        Returns:
            DataFrame in original scale
        """
        logger.info("Starting inverse normalization")

        inversed_df = normalized_df.copy()

        for col in normalized_df.columns:
            feature_name = f"price_{col}" if col in ['open', 'high', 'low', 'close'] else f"other_{col}"

            if feature_name in self.scalers:
                scaler = self.scalers[feature_name]
                values = normalized_df[col].values.reshape(-1, 1)

                try:
                    inversed_values = scaler.inverse_transform(values).flatten()
                    inversed_df[col] = pd.Series(inversed_values, index=normalized_df.index)
                except Exception as e:
                    logger.warning(f"Could not inverse transform {col}: {e}")

        logger.info("Inverse normalization completed")
        return inversed_df

    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get comprehensive normalization statistics."""
        stats = {
            'config': self.config.__dict__,
            'scalers_count': len(self.scalers),
            'normalization_stats': self.normalization_stats,
            'timestamp': datetime.now().isoformat()
        }

        # Add scaler information
        scaler_info = {}
        for name, scaler in self.scalers.items():
            scaler_info[name] = {
                'type': type(scaler).__name__,
                'parameters': getattr(scaler, 'get_params', lambda: {})()
            }

        stats['scaler_info'] = scaler_info
        return stats

    def fit_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit normalizer on data and transform it.

        Args:
            df: DataFrame to fit and transform
            columns: Specific columns to normalize (default: all numeric)

        Returns:
            Normalized DataFrame
        """
        if columns is None:
            columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

        normalized_df = df.copy()

        for col in columns:
            if col in df.columns:
                feature_name = f"fit_{col}"
                normalized_df[col] = self._normalize_series(df[col], feature_name)

        return normalized_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted normalizers.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        transformed_df = df.copy()

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_name = f"fit_{col}"

                if feature_name in self.scalers:
                    scaler = self.scalers[feature_name]
                    values = df[col].values.reshape(-1, 1)
                    transformed_values = scaler.transform(values).flatten()
                    transformed_df[col] = pd.Series(transformed_values, index=df.index)

        return transformed_df

"""
Time Series Operations Module for VN30 Stock Price Prediction System.

This module provides time-series specific operations including:
- Time alignment utilities for multiple frequencies
- Resampling functions (daily, weekly, monthly)
- Forward-fill and backward-fill strategies
- Aggregation methods for different timeframes
- Time-series consistency validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ResamplingMethod(Enum):
    """Resampling methods for time series data."""
    LAST = "last"  # Use last value in period
    FIRST = "first"  # Use first value in period
    MEAN = "mean"  # Average of values in period
    SUM = "sum"  # Sum of values in period
    MAX = "max"  # Maximum value in period
    MIN = "min"  # Minimum value in period
    MEDIAN = "median"  # Median value in period
    OHLC = "ohlc"  # Open, High, Low, Close for price data


class FillMethod(Enum):
    """Methods for filling missing values in time series."""
    FORWARD_FILL = "ffill"  # Forward fill
    BACKWARD_FILL = "bfill"  # Backward fill
    INTERPOLATE = "interpolate"  # Linear interpolation
    NEAREST = "nearest"  # Nearest neighbor
    ZERO = "zero"  # Fill with zeros
    MEAN = "mean"  # Fill with rolling mean


@dataclass
class TimeSeriesConfig:
    """Configuration for time series operations."""
    base_frequency: str = "D"  # Base frequency for operations
    target_frequencies: List[str] = field(default_factory=lambda: ["D", "W", "M"])
    fill_method: FillMethod = FillMethod.FORWARD_FILL
    fill_limit: Optional[int] = None
    handle_holidays: bool = True
    ensure_monotonic: bool = True
    validate_completeness: bool = True


class TimeSeriesProcessor:
    """
    Comprehensive time series processing for financial data.

    Handles time alignment, resampling, and consistency validation
    for financial time series data with trading calendar awareness.
    """

    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        """
        Initialize the TimeSeriesProcessor.

        Args:
            config: TimeSeriesConfig object with processing parameters
        """
        self.config = config or TimeSeriesConfig()
        self.processing_stats = {}

    def align_time_series(self, df: pd.DataFrame, target_frequency: str = None) -> pd.DataFrame:
        """
        Align time series to specified frequency.

        Args:
            df: DataFrame with datetime index
            target_frequency: Target frequency ('D', 'W', 'M', etc.)

        Returns:
            Time-aligned DataFrame
        """
        logger.info(f"Aligning time series to frequency: {target_frequency or self.config.base_frequency}")

        aligned_df = df.copy()

        # Ensure datetime index
        if not isinstance(aligned_df.index, pd.DatetimeIndex):
            aligned_df.index = pd.to_datetime(aligned_df.index)

        # Sort by time
        aligned_df = aligned_df.sort_index()

        # Set target frequency
        freq = target_frequency or self.config.base_frequency

        # Create complete date range
        if len(aligned_df) > 0:
            start_date = aligned_df.index.min()
            end_date = aligned_df.index.max()

            # Create complete date range for the frequency
            complete_range = pd.date_range(
                start=start_date,
                end=end_date,
                freq=freq,
                tz=aligned_df.index.tz
            )

            # Handle duplicate indices before reindexing
            if aligned_df.index.duplicated().any():
                logger.warning("Found duplicate timestamps, aggregating before alignment")
                # Aggregate duplicate timestamps, handling mixed data types
                aligned_df = self._aggregate_duplicate_timestamps(aligned_df)

            # Reindex to complete range
            aligned_df = aligned_df.reindex(complete_range)

            # Apply filling method
            aligned_df = self._apply_fill_method(aligned_df)

        logger.info(f"Time alignment completed. New shape: {aligned_df.shape}")
        return aligned_df

    def resample_time_series(self,
                           df: pd.DataFrame,
                           target_frequency: str,
                           resampling_methods: Dict[str, ResamplingMethod] = None) -> pd.DataFrame:
        """
        Resample time series to different frequency.

        Args:
            df: DataFrame to resample
            target_frequency: Target frequency ('W', 'M', etc.)
            resampling_methods: Methods for each column

        Returns:
            Resampled DataFrame
        """
        logger.info(f"Resampling time series to frequency: {target_frequency}")

        if df.empty:
            logger.warning("Empty DataFrame provided for resampling")
            return df

        # Default resampling methods for different column types
        if resampling_methods is None:
            resampling_methods = self._get_default_resampling_methods(df)

        resampled_df = pd.DataFrame(index=pd.DatetimeIndex([]))

        # Group by symbol if present
        if 'symbol' in df.columns:
            grouped = df.groupby('symbol')
        else:
            # Treat as single time series
            grouped = [('data', df)]

        for group_name, group_df in grouped:
            # Remove symbol column for resampling
            if 'symbol' in group_df.columns:
                group_df = group_df.drop('symbol', axis=1)

            # Set datetime index if not already
            if not isinstance(group_df.index, pd.DatetimeIndex):
                group_df.index = pd.to_datetime(group_df.index)

            # Resample each column according to its method
            resampled_group = self._resample_group(
                group_df, target_frequency, resampling_methods
            )

            # Add symbol column back if it was present
            if 'symbol' in df.columns:
                resampled_group['symbol'] = group_name

            # Combine with main result
            if resampled_df.empty:
                resampled_df = resampled_group
            else:
                resampled_df = pd.concat([resampled_df, resampled_group])

        logger.info(f"Resampling completed. New shape: {resampled_df.shape}")
        return resampled_df

    def validate_time_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate time series consistency.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating time series consistency")

        validation_results = {
            'monotonic_index': True,
            'duplicate_timestamps': 0,
            'missing_timestamps': 0,
            'gaps_detected': [],
            'frequency_consistent': True,
            'inferred_frequency': None,
            'recommendations': []
        }

        if df.empty:
            return validation_results

        # Check for monotonic index
        if self.config.ensure_monotonic:
            validation_results['monotonic_index'] = df.index.is_monotonic_increasing

        # Check for duplicate timestamps
        duplicate_timestamps = df.index.duplicated().sum()
        validation_results['duplicate_timestamps'] = duplicate_timestamps

        # Infer frequency and check consistency
        if len(df) > 1:
            try:
                inferred_freq = pd.infer_freq(df.index)
                validation_results['inferred_frequency'] = inferred_freq

                # Check for gaps in regular frequency
                expected_index = pd.date_range(
                    start=df.index.min(),
                    end=df.index.max(),
                    freq=inferred_freq
                )

                missing_timestamps = len(expected_index) - len(df)
                validation_results['missing_timestamps'] = missing_timestamps

                if missing_timestamps > 0:
                    validation_results['recommendations'].append(
                        f"Consider forward/backward filling {missing_timestamps} missing timestamps"
                    )

            except Exception as e:
                validation_results['frequency_consistent'] = False
                validation_results['recommendations'].append(
                    f"Could not infer frequency: {e}"
                )

        # Generate recommendations
        if not validation_results['monotonic_index']:
            validation_results['recommendations'].append(
                "Index is not monotonic. Consider sorting by timestamp."
            )

        if validation_results['duplicate_timestamps'] > 0:
            validation_results['recommendations'].append(
                f"Found {duplicate_timestamps} duplicate timestamps. Consider aggregation."
            )

        logger.info(f"Time consistency validation completed. Monotonic: {validation_results['monotonic_index']}")
        return validation_results

    def _aggregate_duplicate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate duplicate timestamps, handling mixed data types."""
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

        aggregated_df = pd.DataFrame()

        # Handle numeric columns with mean
        if len(numeric_cols) > 0:
            numeric_df = df[numeric_cols].groupby(level=0).mean()
            aggregated_df = pd.concat([aggregated_df, numeric_df], axis=1)

        # Handle non-numeric columns with last value
        if len(non_numeric_cols) > 0:
            non_numeric_df = df[non_numeric_cols].groupby(level=0).last()
            aggregated_df = pd.concat([aggregated_df, non_numeric_df], axis=1)

        return aggregated_df

    def _apply_fill_method(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply configured fill method to DataFrame."""
        if self.config.fill_method == FillMethod.FORWARD_FILL:
            return df.fillna(method='ffill', limit=self.config.fill_limit)
        elif self.config.fill_method == FillMethod.BACKWARD_FILL:
            return df.fillna(method='bfill', limit=self.config.fill_limit)
        elif self.config.fill_method == FillMethod.INTERPOLATE:
            return df.interpolate(method='linear', limit=self.config.fill_limit)
        elif self.config.fill_method == FillMethod.ZERO:
            return df.fillna(0)
        elif self.config.fill_method == FillMethod.MEAN:
            return df.fillna(df.mean())
        else:
            return df

    def _get_default_resampling_methods(self, df: pd.DataFrame) -> Dict[str, ResamplingMethod]:
        """Get default resampling methods based on column types."""
        methods = {}

        for col in df.columns:
            if col in ['open', 'high', 'low', 'close']:
                methods[col] = ResamplingMethod.LAST  # Use last price for OHLC
            elif col == 'volume':
                methods[col] = ResamplingMethod.SUM  # Sum volumes
            elif col in ['symbol']:
                methods[col] = ResamplingMethod.LAST  # Keep last symbol
            else:
                methods[col] = ResamplingMethod.MEAN  # Default to mean

        return methods

    def _resample_group(self,
                       group_df: pd.DataFrame,
                       target_frequency: str,
                       methods: Dict[str, ResamplingMethod]) -> pd.DataFrame:
        """Resample a single group of data."""
        resampled_data = {}

        for col, method in methods.items():
            if col not in group_df.columns:
                continue

            try:
                if method == ResamplingMethod.LAST:
                    resampled_data[col] = group_df[col].resample(target_frequency).last()
                elif method == ResamplingMethod.FIRST:
                    resampled_data[col] = group_df[col].resample(target_frequency).first()
                elif method == ResamplingMethod.MEAN:
                    resampled_data[col] = group_df[col].resample(target_frequency).mean()
                elif method == ResamplingMethod.SUM:
                    resampled_data[col] = group_df[col].resample(target_frequency).sum()
                elif method == ResamplingMethod.MAX:
                    resampled_data[col] = group_df[col].resample(target_frequency).max()
                elif method == ResamplingMethod.MIN:
                    resampled_data[col] = group_df[col].resample(target_frequency).min()
                elif method == ResamplingMethod.MEDIAN:
                    resampled_data[col] = group_df[col].resample(target_frequency).median()
                elif method == ResamplingMethod.OHLC:
                    # Special handling for OHLC resampling
                    ohlc_data = group_df[col].resample(target_frequency).ohlc()
                    # For simplicity, use close price
                    resampled_data[col] = ohlc_data['close']

            except Exception as e:
                logger.warning(f"Error resampling column {col} with method {method}: {e}")
                # Fallback to forward fill
                resampled_data[col] = group_df[col].resample(target_frequency).last()

        # Combine all resampled columns
        if resampled_data:
            result_df = pd.DataFrame(resampled_data)

            # Apply filling for any remaining missing values
            result_df = self._apply_fill_method(result_df)

            return result_df
        else:
            return pd.DataFrame()

    def create_multi_frequency_dataset(self,
                                     df: pd.DataFrame,
                                     frequencies: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Create dataset with multiple time frequencies.

        Args:
            df: Base DataFrame
            frequencies: List of target frequencies

        Returns:
            Dictionary mapping frequency to DataFrame
        """
        logger.info("Creating multi-frequency dataset")

        if frequencies is None:
            frequencies = self.config.target_frequencies

        multi_freq_data = {}

        for freq in frequencies:
            try:
                if freq == self.config.base_frequency:
                    # Use original frequency
                    multi_freq_data[freq] = df.copy()
                else:
                    # Resample to target frequency
                    resampled = self.resample_time_series(df, freq)
                    multi_freq_data[freq] = resampled

            except Exception as e:
                logger.error(f"Error creating {freq} frequency data: {e}")
                multi_freq_data[freq] = pd.DataFrame()

        logger.info(f"Multi-frequency dataset created with {len([f for f in multi_freq_data.values() if not f.empty])} frequencies")
        return multi_freq_data

    def align_multiple_series(self, series_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align multiple time series to common timeline.

        Args:
            series_dict: Dictionary of DataFrames to align

        Returns:
            Dictionary of aligned DataFrames
        """
        logger.info(f"Aligning {len(series_dict)} time series")

        if not series_dict:
            return {}

        # Find common date range
        all_dates = set()
        for name, df in series_dict.items():
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                all_dates.update(df.index)

        if not all_dates:
            logger.warning("No valid dates found in series")
            return series_dict

        # Create common date range
        common_start = min(all_dates)
        common_end = max(all_dates)
        common_index = pd.date_range(start=common_start, end=common_end, freq=self.config.base_frequency)

        aligned_series = {}

        for name, df in series_dict.items():
            try:
                if df.empty:
                    aligned_series[name] = df
                    continue

                # Reindex to common timeline
                aligned_df = df.reindex(common_index)

                # Apply filling
                aligned_df = self._apply_fill_method(aligned_df)

                aligned_series[name] = aligned_df

            except Exception as e:
                logger.error(f"Error aligning series {name}: {e}")
                aligned_series[name] = df

        logger.info(f"Multiple series alignment completed. Common range: {common_start} to {common_end}")
        return aligned_series

    def get_time_series_stats(self) -> Dict[str, Any]:
        """Get comprehensive time series processing statistics."""
        return {
            'config': self.config.__dict__,
            'processing_stats': self.processing_stats,
            'timestamp': datetime.now().isoformat()
        }

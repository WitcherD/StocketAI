"""
Qlib Format Conversion Module

This module provides functionality for converting vnstock data to qlib-compatible format
using dump_bin.py equivalent functionality for high-performance processing.
"""

import logging
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import h5py
import pickle

from .config import get_config, DataAcquisitionConfig
from .vnstock_client import DataValidator, DataCache, timing_decorator


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class QLibDataFormat:
    """Container for qlib format data structure."""

    instruments: List[str]  # List of stock symbols
    features: pd.DataFrame  # Feature data
    labels: pd.DataFrame    # Label data (future prices)
    market_data: pd.DataFrame  # Market data (OHLCV)
    feature_columns: List[str]
    label_columns: List[str]
    frequency: str = "day"  # "day", "hour", "minute"

    def validate(self) -> Dict[str, Any]:
        """Validate the qlib data format."""
        errors = []

        if not self.instruments:
            errors.append("No instruments specified")

        if self.features.empty:
            errors.append("Features DataFrame is empty")

        if self.market_data.empty:
            errors.append("Market data DataFrame is empty")

        # Check for required columns - market data should have symbol-prefixed columns
        # For each instrument, we should have open, high, low, close, volume columns
        required_base_cols = ['open', 'high', 'low', 'close', 'volume']
        found_cols = set()
        for instrument in self.instruments:
            for base_col in required_base_cols:
                col_name = f"{instrument}_{base_col}"
                if col_name in self.market_data.columns:
                    found_cols.add(base_col)

        missing_market_cols = [col for col in required_base_cols if col not in found_cols]
        if missing_market_cols:
            errors.append(f"Missing market data columns: {missing_market_cols}")

        # Check date index alignment
        if not isinstance(self.features.index, pd.DatetimeIndex):
            errors.append("Features must have DatetimeIndex")

        if not isinstance(self.market_data.index, pd.DatetimeIndex):
            errors.append("Market data must have DatetimeIndex")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'feature_shape': self.features.shape,
            'market_shape': self.market_data.shape,
            'instruments_count': len(self.instruments)
        }


class QLibConverter:
    """Main class for converting vnstock data to qlib format."""

    def __init__(self, config: Optional[DataAcquisitionConfig] = None):
        """Initialize the converter with configuration."""
        self.config = config or get_config()
        self.validator = DataValidator(self.config)
        self.cache = DataCache()

        logger.info("QLibConverter initialized")

    @timing_decorator
    def convert_price_data_to_qlib(
        self,
        price_data: Dict[str, Any],
        output_dir: Optional[Path] = None,
        frequency: str = "day"
    ) -> Path:
        """
        Convert price data to qlib format.

        Args:
            price_data: Dictionary of price data collections by symbol
            output_dir: Output directory for qlib files
            frequency: Data frequency ("day", "hour", "minute")

        Returns:
            Path to the created qlib data directory
        """
        if output_dir is None:
            output_dir = self.config.qlib_data_path / f"vn30_data_{datetime.now().strftime('%Y%m%d')}"

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting price data to qlib format in {output_dir}")

        try:
            # Process each symbol's data
            all_features = []
            all_market_data = []
            instruments = []

            for symbol, collection in price_data.items():
                if collection is None or not collection.data:
                    logger.warning(f"No data available for {symbol}, skipping")
                    continue

                try:
                    # Convert to DataFrame
                    df = collection.to_dataframe()

                    if df.empty:
                        continue

                    # Generate features and labels
                    symbol_features, symbol_market = self._process_symbol_data(df, symbol, frequency)

                    if symbol_features is not None and symbol_market is not None:
                        all_features.append(symbol_features)
                        all_market_data.append(symbol_market)
                        instruments.append(symbol)

                except Exception as e:
                    logger.error(f"Error processing data for {symbol}: {e}")
                    continue

            if not all_features:
                raise ValueError("No valid data found to convert")

            # Combine all data
            combined_features = pd.concat(all_features, axis=1) if len(all_features) > 1 else all_features[0]
            combined_market = pd.concat(all_market_data, axis=1) if len(all_market_data) > 1 else all_market_data[0]

            # Create qlib format data
            qlib_data = QLibDataFormat(
                instruments=instruments,
                features=combined_features,
                labels=combined_features.copy(),  # For now, use features as labels
                market_data=combined_market,
                feature_columns=combined_features.columns.tolist(),
                label_columns=combined_features.columns.tolist(),
                frequency=frequency
            )

            # Validate the data
            validation = qlib_data.validate()
            if not validation['is_valid']:
                logger.error(f"Qlib data validation failed: {validation['errors']}")
                raise ValueError(f"Data validation failed: {validation['errors']}")

            # Save to qlib format
            self._save_qlib_format(qlib_data, output_dir)

            logger.info(f"Successfully converted data for {len(instruments)} instruments")
            logger.info(f"Features shape: {combined_features.shape}")
            logger.info(f"Market data shape: {combined_market.shape}")

            return output_dir

        except Exception as e:
            logger.error(f"Error converting price data to qlib format: {e}")
            raise

    def _process_symbol_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        frequency: str
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Process data for a single symbol.

        Args:
            df: DataFrame with price data for the symbol
            symbol: Stock symbol
            frequency: Data frequency

        Returns:
            Tuple of (features DataFrame, market DataFrame)
        """
        try:
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns for {symbol}")
                return None, None

            # Generate features using qlib-style expressions
            features_df = self._generate_features(df, symbol)

            # Market data (OHLCV)
            market_df = df[required_cols].copy()
            market_df.columns = [f'{symbol}_{col}' for col in market_df.columns]

            return features_df, market_df

        except Exception as e:
            logger.error(f"Error processing symbol data for {symbol}: {e}")
            return None, None

    def _generate_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate features from price data using qlib-style expressions.

        Args:
            df: DataFrame with price data
            symbol: Stock symbol

        Returns:
            DataFrame with generated features
        """
        try:
            # Basic price features
            features = {}

            # Price-based features
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']

            # Moving averages
            features[f'{symbol}_MA5'] = close.rolling(window=5).mean()
            features[f'{symbol}_MA10'] = close.rolling(window=10).mean()
            features[f'{symbol}_MA20'] = close.rolling(window=20).mean()

            # RSI (Relative Strength Index)
            features[f'{symbol}_RSI'] = self._calculate_rsi(close, 14)

            # MACD (Moving Average Convergence Divergence)
            macd, signal = self._calculate_macd(close)
            features[f'{symbol}_MACD'] = macd
            features[f'{symbol}_MACD_SIGNAL'] = signal

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close)
            features[f'{symbol}_BB_UPPER'] = bb_upper
            features[f'{symbol}_BB_MIDDLE'] = bb_middle
            features[f'{symbol}_BB_LOWER'] = bb_lower

            # Volume features
            features[f'{symbol}_VOLUME_MA5'] = volume.rolling(window=5).mean()
            features[f'{symbol}_VOLUME_RATIO'] = volume / features[f'{symbol}_VOLUME_MA5']

            # Price momentum
            features[f'{symbol}_MOMENTUM_5'] = (close / close.shift(5) - 1) * 100
            features[f'{symbol}_MOMENTUM_10'] = (close / close.shift(10) - 1) * 100

            # Volatility
            features[f'{symbol}_VOLATILITY'] = close.rolling(window=20).std()

            # Return features
            features[f'{symbol}_RETURN'] = close.pct_change()
            features[f'{symbol}_LOG_RETURN'] = np.log(close / close.shift(1))

            return pd.DataFrame(features)

        except Exception as e:
            logger.error(f"Error generating features for {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.fillna(50)  # Fill NaN with neutral value

        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()

            return macd, macd_signal

        except Exception:
            return pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index)

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            middle = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper = middle + (std * num_std)
            lower = middle - (std * num_std)

            # Fill NaN values with the first valid value to avoid comparison issues
            upper = upper.fillna(method='bfill').fillna(upper.mean())
            middle = middle.fillna(method='bfill').fillna(middle.mean())
            lower = lower.fillna(method='bfill').fillna(lower.mean())

            return upper, middle, lower

        except Exception:
            return pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index)

    def _save_qlib_format(self, qlib_data: QLibDataFormat, output_dir: Path) -> None:
        """
        Save data in qlib format.

        Args:
            qlib_data: QLibDataFormat object to save
            output_dir: Output directory
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create instruments file
            instruments_file = output_dir / "instruments.pkl"
            with open(instruments_file, 'wb') as f:
                pickle.dump(qlib_data.instruments, f)

            # Create features file (HDF5 format)
            features_file = output_dir / "features.h5"
            with h5py.File(features_file, 'w') as f:
                # Store features data
                f.create_dataset('features', data=qlib_data.features.values)
                # Convert datetime index to timestamps (seconds since epoch)
                dates_ts = qlib_data.features.index.astype('int64') // 10**9
                f.create_dataset('dates', data=dates_ts.values)
                f.create_dataset('symbols', data=np.array(qlib_data.instruments, dtype='S10'))

            # Create market data file
            market_file = output_dir / "market_data.h5"
            with h5py.File(market_file, 'w') as f:
                f.create_dataset('market_data', data=qlib_data.market_data.values)
                # Convert datetime index to timestamps (seconds since epoch)
                dates_ts = qlib_data.market_data.index.astype('int64') // 10**9
                f.create_dataset('dates', data=dates_ts.values)
                f.create_dataset('symbols', data=np.array(qlib_data.instruments, dtype='S10'))

            # Create metadata file
            metadata = {
                'feature_columns': qlib_data.features.columns.tolist(),
                'label_columns': qlib_data.labels.columns.tolist(),
                'frequency': qlib_data.frequency,
                'created_at': datetime.now().isoformat(),
                'data_shape': qlib_data.features.shape,
                'instruments_count': len(qlib_data.instruments)
            }

            metadata_file = output_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved qlib format data to {output_dir}")

        except Exception as e:
            logger.error(f"Error saving qlib format: {e}")
            raise

    def load_qlib_format(self, data_dir: Path) -> QLibDataFormat:
        """
        Load data from qlib format.

        Args:
            data_dir: Directory containing qlib format files

        Returns:
            QLibDataFormat object
        """
        try:
            # Load instruments
            instruments_file = data_dir / "instruments.pkl"
            with open(instruments_file, 'rb') as f:
                instruments = pickle.load(f)

            # Load metadata
            metadata_file = data_dir / "metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Load features
            features_file = data_dir / "features.h5"
            with h5py.File(features_file, 'r') as f:
                features_data = f['features'][:]
                dates_data = f['dates'][:]

            # Load market data
            market_file = data_dir / "market_data.h5"
            with h5py.File(market_file, 'r') as f:
                market_data = f['market_data'][:]

            # Create DataFrames - convert timestamps back to datetime
            dates = pd.DatetimeIndex(pd.to_datetime(dates_data, unit='s'))
            features_df = pd.DataFrame(
                features_data,
                index=dates,
                columns=metadata['feature_columns']
            )

            # Create expected column names for all instruments
            expected_columns = []
            for inst in instruments:
                expected_columns.extend([f'{inst}_open', f'{inst}_high', f'{inst}_low', f'{inst}_close', f'{inst}_volume'])

            # Create market DataFrame
            # The saved data should have columns for all instruments, so use expected_columns
            market_df = pd.DataFrame(
                market_data,
                index=dates,
                columns=expected_columns[:market_data.shape[1]]  # Slice to match actual data shape
            )

            return QLibDataFormat(
                instruments=instruments,
                features=features_df,
                labels=features_df.copy(),  # For now
                market_data=market_df,
                feature_columns=metadata['feature_columns'],
                label_columns=metadata['label_columns'],
                frequency=metadata['frequency']
            )

        except Exception as e:
            logger.error(f"Error loading qlib format: {e}")
            raise

    def validate_qlib_format(self, data_dir: Path) -> Dict[str, Any]:
        """
        Validate qlib format data.

        Args:
            data_dir: Directory containing qlib format files

        Returns:
            Validation results
        """
        try:
            qlib_data = self.load_qlib_format(data_dir)
            return qlib_data.validate()

        except Exception as e:
            return {
                'is_valid': False,
                'errors': [f'Validation failed: {str(e)}']
            }

    def create_data_calendar(self, start_date: str, end_date: str, frequency: str = "day") -> pd.DataFrame:
        """
        Create a trading calendar for qlib.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency

        Returns:
            DataFrame with trading calendar
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            if frequency == "day":
                # Create daily calendar (excluding weekends)
                calendar_dates = pd.date_range(start=start, end=end, freq='D')
                calendar_df = pd.DataFrame(index=calendar_dates)

                # Mark weekdays as trading days
                calendar_df['is_trading_day'] = calendar_df.index.weekday < 5

            elif frequency == "hour":
                # Create hourly calendar for trading hours
                # For a single day, create 24 hours from 00:00 to 23:00
                if start.date() == end.date():
                    calendar_dates = pd.date_range(start=start, periods=24, freq='h')
                else:
                    calendar_dates = pd.date_range(start=start, end=end, freq='h')
                calendar_df = pd.DataFrame(index=calendar_dates)

                # Mark trading hours (9 AM - 3 PM on weekdays)
                calendar_df['is_trading_hour'] = (
                    (calendar_df.index.weekday < 5) &
                    (calendar_df.index.hour >= 9) &
                    (calendar_df.index.hour <= 15)
                )

            else:
                raise ValueError(f"Unsupported frequency: {frequency}")

            return calendar_df

        except Exception as e:
            logger.error(f"Error creating data calendar: {e}")
            raise

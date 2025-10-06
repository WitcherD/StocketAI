"""
VN30 Data Handler and Converter Module

Comprehensive data management and conversion utilities for VN30 stocks.
Handles raw data validation, cleaning, qlib format conversion, and data loading
for high-performance processing and TFT model training.

Key Components:
- Raw data validation and cleaning
- Qlib binary format conversion
- Data loading and management
- TFT training data preparation
- Data quality validation and reporting
- Multi-symbol processing capabilities

Author: StocketAI
Created: 2025
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import qlib
from qlib.data import D
from qlib.utils import fname_to_code
import qlib.config as qconfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VN30DataHandler:
    """
    VN30 Data Handler for TFT Feature Engineering Compatibility.

    Transforms raw VN30 CSV data to TFT feature engineer compatible format.
    Handles data validation, cleaning, qlib format conversion, and provides
    standardized OHLCV data ready for feature engineering and TFT model training.

    Primary goal: Convert raw data to format expected by tft_feature_engineer.py
    Output format: DataFrame with columns [time, symbol, open, high, low, close, volume]
    """

    def __init__(self, symbols_dir: str = None):
        """
        Initialize VN30 data handler.

        Args:
            symbols_dir: Directory containing VN30 symbol folders (for raw data, processed data, and qlib data)
        """
        self.symbols_dir = Path(symbols_dir) if symbols_dir else Path("data/symbols")

        # VN30 symbols list
        self.vn30_symbols = self._load_vn30_symbols()

        # Data validation rules
        self.validation_rules = {
            'required_columns': ['time', 'open', 'high', 'low', 'close', 'volume'],
            'date_format': '%Y-%m-%d',
            'min_data_points': 100,
            'max_missing_ratio': 0.05
        }

        logger.info(f"Initialized VN30DataHandler with symbols_dir: {self.symbols_dir}")
        logger.info(f"VN30 symbols loaded: {len(self.vn30_symbols)} symbols")

    def validate_raw_data(self, symbol: str) -> Tuple[bool, str]:
        """
        Validate raw CSV data for a symbol.

        Args:
            symbol: Stock symbol to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        csv_file = self.symbols_dir / symbol / 'raw' / 'historical_price.csv'

        if not csv_file.exists():
            return False, f"CSV file not found: {csv_file}"

        try:
            # Read CSV file
            df = pd.read_csv(csv_file)

            # Check required columns
            missing_cols = set(self.validation_rules['required_columns']) - set(df.columns)
            if missing_cols:
                return False, f"Missing columns: {missing_cols}"

            # Check minimum data points
            if len(df) < self.validation_rules['min_data_points']:
                return False, f"Insufficient data points: {len(df)} < {self.validation_rules['min_data_points']}"

            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if missing_ratio > self.validation_rules['max_missing_ratio']:
                return False, f"Too many missing values: {missing_ratio:.2%} > {self.validation_rules['max_missing_ratio']:.2%}"

            # Check date format (but don't require sorting - we'll sort during cleaning)
            try:
                df['time'] = pd.to_datetime(df['time'])
                # Check if we can parse all dates
                if df['time'].isnull().any():
                    return False, "Some dates could not be parsed"
            except Exception as e:
                return False, f"Date format error: {e}"

            logger.info(f"Data validation passed for {symbol}: {len(df)} records")
            return True, ""

        except Exception as e:
            return False, f"Error reading CSV file: {e}"

    def clean_raw_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Clean and preprocess raw data for a symbol.

        Args:
            symbol: Stock symbol to clean

        Returns:
            Cleaned DataFrame or None if cleaning fails
        """
        csv_file = self.symbols_dir / symbol / 'raw' / 'historical_price.csv'

        try:
            df = pd.read_csv(csv_file)

            # Convert time column
            df['time'] = pd.to_datetime(df['time'])

            # Remove duplicates
            df = df.drop_duplicates(subset=['time'])

            # Sort by time
            df = df.sort_values('time').reset_index(drop=True)

            # Handle missing values (forward fill for small gaps)
            df = df.set_index('time').asfreq('D')  # Daily frequency
            df = df.ffill(limit=5)  # Fill up to 5 consecutive missing days

            # Reset index and drop rows still containing NaN
            df = df.reset_index()
            df = df.dropna()

            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove any remaining NaN values
            df = df.dropna()

            logger.info(f"Cleaned data for {symbol}: {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error cleaning data for {symbol}: {e}")
            return None

    def prepare_qlib_format(self, symbol: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare data in qlib format.

        Args:
            symbol: Stock symbol
            df: Cleaned DataFrame

        Returns:
            DataFrame in qlib format or None if preparation fails
        """
        try:
            # Create qlib format DataFrame with correct column order
            qlib_df = pd.DataFrame({
                'time': df['time'],
                'symbol': fname_to_code(symbol),
                'open': df['open'],
                'high': df['high'],
                'low': df['low'],
                'close': df['close'],
                'volume': df['volume']
            })

            logger.info(f"Prepared qlib format for {symbol}: {len(qlib_df)} records")
            return qlib_df

        except Exception as e:
            logger.error(f"Error preparing qlib format for {symbol}: {e}")
            return None

    def convert_symbol_to_qlib(self, symbol: str) -> bool:
        """
        Convert single symbol data to cleaned qlib format.

        Args:
            symbol: Stock symbol to convert

        Returns:
            True if conversion successful, False otherwise
        """
        logger.info(f"Converting {symbol} to cleaned qlib format...")

        # Validate raw data
        is_valid, error_msg = self.validate_raw_data(symbol)
        if not is_valid:
            logger.error(f"Data validation failed for {symbol}: {error_msg}")
            return False

        # Clean raw data
        cleaned_df = self.clean_raw_data(symbol)
        if cleaned_df is None:
            logger.error(f"Data cleaning failed for {symbol}")
            return False

        # Prepare qlib format with cleaned data
        qlib_df = self.prepare_qlib_format(symbol, cleaned_df)
        if qlib_df is None:
            logger.error(f"Qlib format preparation failed for {symbol}")
            return False

        # Save cleaned data as pickle for reliable storage and loading
        try:
            # Create symbol-specific processed directory
            processed_dir = self.symbols_dir / symbol / 'processed'
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Save cleaned data as pickle for reliable loading
            pickle_file = processed_dir / f"{symbol.lower()}_cleaned.pkl"
            qlib_df.to_pickle(pickle_file)

            logger.info(f"Saved cleaned data for {symbol} to {pickle_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving cleaned data for {symbol}: {e}")
            return False

    def convert_all_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Convert all specified symbols to qlib format.

        Args:
            symbols: List of stock symbols to convert

        Returns:
            Dictionary mapping symbols to conversion success status
        """
        logger.info(f"Starting conversion of {len(symbols)} symbols to qlib format...")

        results = {}
        successful = 0
        failed = 0

        for symbol in symbols:
            success = self.convert_symbol_to_qlib(symbol)
            results[symbol] = success

            if success:
                successful += 1
            else:
                failed += 1

        logger.info(f"Conversion completed: {successful} successful, {failed} failed")
        return results

    def _load_vn30_symbols(self) -> List[str]:
        """Load VN30 constituent symbols."""
        symbols_file = Path("data/symbols/vn30_constituents.csv")

        if not symbols_file.exists():
            logger.warning(f"VN30 symbols file not found: {symbols_file}")
            return []

        try:
            symbols_df = pd.read_csv(symbols_file)
            symbols = symbols_df['symbol'].tolist()
            logger.info(f"Loaded {len(symbols)} VN30 symbols")
            return symbols
        except Exception as e:
            logger.error(f"Error loading VN30 symbols: {e}")
            return []

    def initialize_qlib(self) -> bool:
        """
        Initialize qlib with VN30 data configuration.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize qlib with default configuration
            # Qlib data is stored per symbol in symbol directories
            qlib.init()

            logger.info("Qlib initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing qlib: {e}")
            return False

    def load_symbol_data(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Load processed data for a specific VN30 symbol.

        Args:
            symbol: Stock symbol to load
            start_date: Start date for data loading (YYYY-MM-DD)
            end_date: End date for data loading (YYYY-MM-DD)

        Returns:
            DataFrame with symbol data or None if loading fails
        """
        try:
            # Convert symbol to qlib format
            qlib_symbol = fname_to_code(symbol)

            # Load data using qlib
            data = D.features([qlib_symbol], ['open', 'high', 'low', 'close', 'volume'])

            if data is None or data.empty:
                logger.warning(f"No data found for symbol: {symbol}")
                return None

            # Convert to standard format
            df = data.reset_index()
            df['symbol'] = symbol
            df = df.rename(columns={'datetime': 'time'})

            # Filter by date range if specified
            if start_date:
                df = df[df['time'] >= start_date]
            if end_date:
                df = df[df['time'] <= end_date]

            logger.info(f"Loaded data for {symbol}: {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None

    def prepare_symbol_training_data(self, symbol: str, start_date: str = None, end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for a single symbol's TFT model training with proper train/validation/test splits.

        Args:
            symbol: Stock symbol to prepare data for
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)

        Returns:
            Tuple of (train_df, valid_df, test_df) for the specific symbol
        """
        # Load symbol data
        symbol_df = self.load_symbol_data(symbol, start_date, end_date)

        if symbol_df is None or symbol_df.empty:
            logger.error(f"No data available for {symbol} TFT preparation")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Create train/validation/test splits based on time
        symbol_df['time'] = pd.to_datetime(symbol_df['time'])
        symbol_df = symbol_df.sort_values('time')

        # Define split boundaries
        end_date = pd.to_datetime(end_date) if end_date else symbol_df['time'].max()
        start_date = pd.to_datetime(start_date) if start_date else symbol_df['time'].min()

        # Calculate split points
        total_days = (end_date - start_date).days

        # Use 70% for training, 15% for validation, 15% for testing
        train_end = start_date + pd.Timedelta(days=int(total_days * 0.7))
        valid_end = train_end + pd.Timedelta(days=int(total_days * 0.15))

        # Split data
        train_df = symbol_df[symbol_df['time'] <= train_end]
        valid_df = symbol_df[(symbol_df['time'] > train_end) & (symbol_df['time'] <= valid_end)]
        test_df = symbol_df[symbol_df['time'] > valid_end]

        logger.info(f"Data splits created for {symbol}:")
        logger.info(f"  Train: {len(train_df)} records ({train_df['time'].min()} to {train_df['time'].max()})")
        logger.info(f"  Valid: {len(valid_df)} records ({valid_df['time'].min()} to {valid_df['time'].max()})")
        logger.info(f"  Test: {len(test_df)} records ({test_df['time'].min()} to {test_df['time'].max()})")

        return train_df, valid_df, test_df

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Union[bool, float, int]]:
        """
        Validate data quality for TFT model training.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation metrics
        """
        validation_results = {}

        if df.empty:
            logger.error("Cannot validate empty DataFrame")
            return {'valid': False, 'error': 'Empty DataFrame'}

        # Basic data quality checks
        validation_results['total_records'] = len(df)
        validation_results['unique_symbols'] = df['symbol'].nunique()
        validation_results['date_range'] = f"{df['time'].min()} to {df['time'].max()}"

        # Check for missing values
        missing_values = df.isnull().sum().sum()
        missing_ratio = missing_values / (len(df) * len(df.columns))
        validation_results['missing_ratio'] = missing_ratio
        validation_results['missing_values'] = missing_values

        # Check for sufficient data per symbol
        symbols_data_count = df.groupby('symbol').size()
        min_records_per_symbol = symbols_data_count.min()
        max_records_per_symbol = symbols_data_count.max()
        avg_records_per_symbol = symbols_data_count.mean()

        validation_results['min_records_per_symbol'] = min_records_per_symbol
        validation_results['max_records_per_symbol'] = max_records_per_symbol
        validation_results['avg_records_per_symbol'] = avg_records_per_symbol

        # Check price data validity
        price_columns = ['open', 'high', 'low', 'close']
        price_issues = {}

        for col in price_columns:
            if col in df.columns:
                negative_prices = (df[col] <= 0).sum()
                price_issues[f'{col}_negative'] = negative_prices

                # Check for reasonable price ranges (VN30 stocks typically 10k-500k VND)
                price_range = df[col].max() - df[col].min()
                price_issues[f'{col}_range'] = price_range

        validation_results['price_issues'] = price_issues

        # Overall validity assessment
        validation_results['valid'] = (
            missing_ratio < 0.05 and  # Less than 5% missing
            min_records_per_symbol >= 100 and  # At least 100 records per symbol
            all(df[col].min() > 0 for col in price_columns if col in df.columns)  # No negative prices
        )

        # Log validation summary
        logger.info(f"Data validation completed:")
        logger.info(f"  Valid: {validation_results['valid']}")
        logger.info(f"  Total records: {validation_results['total_records']}")
        logger.info(f"  Missing ratio: {validation_results['missing_ratio']:.3%}")
        logger.info(f"  Records per symbol: {avg_records_per_symbol:.0f} avg")

        return validation_results


def main():
    """Main function for command-line usage."""
    # Default paths
    symbols_dir = "data/symbols"
    symbols_file = "data/symbols/vn30_constituents.csv"

    # Load VN30 symbols
    try:
        symbols_df = pd.read_csv(symbols_file)
        symbols = symbols_df['symbol'].tolist()
        logger.info(f"Loaded {len(symbols)} VN30 symbols from {symbols_file}")
    except Exception as e:
        logger.error(f"Error loading VN30 symbols: {e}")
        sys.exit(1)

    # Initialize converter
    converter = VN30DataHandler(symbols_dir)

    # Convert all symbols
    results = converter.convert_all_symbols(symbols)

    logger.info("Conversion process completed")

    # Exit with appropriate code
    if sum(results.values()) == len(results):
        logger.info("All conversions completed successfully")
        sys.exit(0)
    else:
        logger.warning("Some conversions failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

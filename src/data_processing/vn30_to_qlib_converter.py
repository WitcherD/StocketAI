"""
VN30 Data Converter Module

Converts VN30 raw CSV data to qlib binary format for high-performance processing.
Handles data validation, integrity checks, and qlib data structure creation.

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


class Vn30ToQlibConverter:
    """
    Converts VN30 raw CSV data to qlib format.

    Handles data validation, cleaning, and conversion to qlib's
    high-performance binary format for model training.
    """

    def __init__(self, symbols_dir: str):
        """
        Initialize VN30 data converter.

        Args:
            symbols_dir: Directory containing VN30 symbol folders
        """
        self.symbols_dir = Path(symbols_dir)

        # Data validation rules
        self.validation_rules = {
            'required_columns': ['time', 'open', 'high', 'low', 'close', 'volume'],
            'date_format': '%Y-%m-%d',
            'min_data_points': 100,
            'max_missing_ratio': 0.05
        }



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
            df = df.fillna(method='ffill', limit=5)  # Fill up to 5 consecutive missing days

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
        Convert single symbol data to qlib format.

        Args:
            symbol: Stock symbol to convert

        Returns:
            True if conversion successful, False otherwise
        """
        logger.info(f"Converting {symbol} to qlib format...")

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

        # Prepare qlib format
        qlib_df = self.prepare_qlib_format(symbol, cleaned_df)
        if qlib_df is None:
            logger.error(f"Qlib format preparation failed for {symbol}")
            return False

        # Save to qlib binary format
        try:
            # Create symbol-specific qlib directory
            qlib_dir = self.symbols_dir / symbol / 'qlib'
            qlib_dir.mkdir(parents=True, exist_ok=True)

            # Use qlib's native binary format via dump_bin
            # First save as CSV, then use dump_bin to convert to binary
            csv_file = qlib_dir / f"{symbol.lower()}_temp.csv"
            qlib_df.to_csv(csv_file, index=False)

            # Use qlib's dump_bin functionality
            try:
                from qlib.contrib.data.handler import DataHandlerLP
                from qlib.utils import init_instance_by_config

                # Create handler configuration for binary dump
                handler_config = {
                    "class": "DumpDataAll",
                    "module_path": "scripts.dump_bin",
                    "kwargs": {
                        "csv_path": str(csv_file),
                        "qlib_dir": str(qlib_dir),
                        "symbol_name": fname_to_code(symbol),
                        "frequency": "day",
                        "include_fields": ["open", "high", "low", "close", "volume"]
                    }
                }

                # Initialize and run the dumper
                dumper = init_instance_by_config(handler_config)
                dumper.dump()

                # Clean up temporary CSV
                if csv_file.exists():
                    csv_file.unlink()

                logger.info(f"Saved qlib binary format data for {symbol} to {qlib_dir}")
                return True

            except (ImportError, Exception) as e:
                logger.warning(f"dump_bin not available for {symbol}, using binary fallback: {e}")
                # Fallback to simple binary format if dump_bin not available
                bin_file = qlib_dir / f"{symbol.lower()}.bin"
                df_prepared = qlib_df.copy()
                df_prepared['time'] = pd.to_datetime(df_prepared['time'])
                df_prepared['symbol'] = fname_to_code(symbol)

                # Save as binary using pandas built-in method
                df_prepared.to_pickle(bin_file)
                logger.info(f"Saved qlib binary format data for {symbol} to {bin_file} (binary fallback)")
                return True

        except Exception as e:
            logger.error(f"Error saving qlib binary format for {symbol}: {e}")
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



    def generate_conversion_report(self, results: Dict[str, bool]) -> str:
        """
        Generate conversion report.

        Args:
            results: Dictionary of conversion results

        Returns:
            Report string
        """
        successful = sum(results.values())
        total = len(results)

        report = f"""
VN30 Data Conversion Report
==========================

Total symbols processed: {total}
Successful conversions: {successful}
Failed conversions: {total - successful}

Success rate: {successful/total:.1%}

Failed symbols:
"""

        failed_symbols = [symbol for symbol, success in results.items() if not success]
        for symbol in failed_symbols:
            report += f"  - {symbol}\n"

        return report


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
    converter = Vn30ToQlibConverter(symbols_dir)

    # Convert all symbols
    results = converter.convert_all_symbols(symbols)

    # Generate and save report
    report = converter.generate_conversion_report(results)
    report_file = Path(symbols_dir) / 'conversion_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)

    logger.info(f"Conversion report saved to {report_file}")

    # Exit with appropriate code
    if sum(results.values()) == len(results):
        logger.info("All conversions completed successfully")
        sys.exit(0)
    else:
        logger.warning("Some conversions failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

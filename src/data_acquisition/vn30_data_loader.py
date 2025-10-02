"""
VN30 Data Loader Module

This module provides functionality for loading comprehensive data for VN30 constituents
from vnstock API with caching to avoid re-downloading existing data.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .vnstock_client import VNStockClient


logger = logging.getLogger(__name__)


class VN30DataLoader:
    """Loader for VN30 constituent data with file existence checking."""

    def __init__(self, client: Optional[VNStockClient] = None):
        """Initialize the data loader.

        Args:
            client: VNStockClient instance. If None, creates a new one.
        """
        self.client = client or VNStockClient()

    def create_symbol_directory(self, symbol: str, base_path: Optional[Path] = None) -> Path:
        """Create directory structure for symbol data.

        Args:
            symbol: Stock symbol
            base_path: Base path for data directory. If None, uses project root.

        Returns:
            Path to the symbol's raw data directory
        """
        if base_path is None:
            # Assume we're in the project root or can find it
            base_path = Path.cwd()
            if not (base_path / 'data').exists():
                # Try parent directory
                base_path = base_path.parent
                if not (base_path / 'data').exists():
                    raise FileNotFoundError("Could not find data directory")

        symbol_dir = base_path / 'data' / 'symbols' / symbol / 'raw'
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir

    def save_data_to_csv(self, data: pd.DataFrame, file_path: Path, data_type: str, symbol: str) -> bool:
        """Save DataFrame to CSV with error handling.

        Args:
            data: DataFrame to save
            file_path: Path to save the file
            data_type: Description of the data type for logging
            symbol: Stock symbol for logging

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if not data.empty:
                data.to_csv(file_path, index=False)
                logger.info(f"Saved {data_type} for {symbol} to {file_path}")
                return True
            else:
                logger.warning(f"No {data_type} data for {symbol}")
                return False
        except Exception as e:
            logger.error(f"Error saving {data_type} for {symbol}: {e}")
            return False

    def file_exists_and_not_empty(self, file_path: Path) -> bool:
        """Check if file exists and is not empty.

        Args:
            file_path: Path to check

        Returns:
            True if file exists and has content, False otherwise
        """
        if not file_path.exists():
            return False

        try:
            # Check if file has content
            if file_path.stat().st_size == 0:
                return False

            # Try to read the file to ensure it's valid
            df = pd.read_csv(file_path)
            return not df.empty
        except Exception:
            # If we can't read it, consider it invalid
            logger.warning(f"Existing file {file_path} appears to be corrupted, will re-download")
            return False

    def load_symbol_data(self, symbol: str, base_path: Optional[Path] = None, force_reload: bool = False) -> bool:
        """Load all available data for a symbol, skipping existing files unless force_reload is True.

        Args:
            symbol: Stock symbol to load data for
            base_path: Base path for data directory
            force_reload: If True, reload all data even if files exist

        Returns:
            True if any data was loaded, False if all data was already cached
        """
        logger.info(f"Starting data loading for {symbol}")

        # Create directory
        symbol_dir = self.create_symbol_directory(symbol, base_path)

        data_loaded = False

        try:
            # Define data loading tasks with file paths and loading functions
            data_tasks = [
                # Historical price data (10 years)
                {
                    'file': symbol_dir / 'historical_price.csv',
                    'type': 'historical price',
                    'loader': lambda: self.client.get_merged_historical_data(
                        symbol,
                        (pd.Timestamp.now() - pd.DateOffset(years=10)).strftime('%Y-%m-%d'),
                        pd.Timestamp.now().strftime('%Y-%m-%d'),
                        resolution='1D'
                    )
                },
                # Financial ratios (yearly)
                {
                    'file': symbol_dir / 'financial_ratios_yearly.csv',
                    'type': 'financial ratios (yearly)',
                    'loader': lambda: self.client.get_merged_financial_ratios(symbol, period='yearly')
                },
                # Financial ratios (quarterly)
                {
                    'file': symbol_dir / 'financial_ratios_quarterly.csv',
                    'type': 'financial ratios (quarterly)',
                    'loader': lambda: self.client.get_merged_financial_ratios(symbol, period='quarterly')
                },
                # Balance sheet (yearly)
                {
                    'file': symbol_dir / 'balance_sheet_yearly.csv',
                    'type': 'balance sheet (yearly)',
                    'loader': lambda: self.client.get_merged_balance_sheet(symbol, period='yearly')
                },
                # Balance sheet (quarterly)
                {
                    'file': symbol_dir / 'balance_sheet_quarterly.csv',
                    'type': 'balance sheet (quarterly)',
                    'loader': lambda: self.client.get_merged_balance_sheet(symbol, period='quarterly')
                },
                # Income statement (yearly)
                {
                    'file': symbol_dir / 'income_statement_yearly.csv',
                    'type': 'income statement (yearly)',
                    'loader': lambda: self.client.get_merged_income_statement(symbol, period='yearly')
                },
                # Income statement (quarterly)
                {
                    'file': symbol_dir / 'income_statement_quarterly.csv',
                    'type': 'income statement (quarterly)',
                    'loader': lambda: self.client.get_merged_income_statement(symbol, period='quarterly')
                },
                # Cash flow statement (yearly)
                {
                    'file': symbol_dir / 'cash_flow_yearly.csv',
                    'type': 'cash flow (yearly)',
                    'loader': lambda: self.client.get_merged_cash_flow(symbol, period='yearly')
                },
                # Cash flow statement (quarterly)
                {
                    'file': symbol_dir / 'cash_flow_quarterly.csv',
                    'type': 'cash flow (quarterly)',
                    'loader': lambda: self.client.get_merged_cash_flow(symbol, period='quarterly')
                },
                # Company profile
                {
                    'file': symbol_dir / 'company_profile.csv',
                    'type': 'company profile',
                    'loader': lambda: self.client.get_company_profile(symbol, source='TCBS')
                },
                # Company shareholders
                {
                    'file': symbol_dir / 'company_shareholders.csv',
                    'type': 'company shareholders',
                    'loader': lambda: self.client.get_company_shareholders(symbol, source='TCBS')
                },
                # Company officers
                {
                    'file': symbol_dir / 'company_officers.csv',
                    'type': 'company officers',
                    'loader': lambda: self.client.get_company_officers(symbol, source='TCBS')
                },
                # Company events
                {
                    'file': symbol_dir / 'company_events.csv',
                    'type': 'company events',
                    'loader': lambda: self.client.get_company_events(symbol, source='TCBS', limit=50)
                },
                # Company news
                {
                    'file': symbol_dir / 'company_news.csv',
                    'type': 'company news',
                    'loader': lambda: self.client.get_company_news(symbol, source='TCBS', limit=50)
                },
                # Dividend history
                {
                    'file': symbol_dir / 'company_dividends.csv',
                    'type': 'company dividends',
                    'loader': lambda: self.client.get_company_dividends(symbol, source='TCBS')
                },
                # Intraday data (recent)
                {
                    'file': symbol_dir / 'intraday_data.csv',
                    'type': 'intraday data',
                    'loader': lambda: self.client.get_intraday_data(symbol, page_size=100, source='VCI')
                },
                # Price depth
                {
                    'file': symbol_dir / 'price_depth.csv',
                    'type': 'price depth',
                    'loader': lambda: self.client.get_price_depth(symbol, source='VCI')
                },
                # Stock info (overview)
                {
                    'file': symbol_dir / 'stock_info.csv',
                    'type': 'stock info',
                    'loader': lambda: self._load_stock_info(symbol)
                }
            ]

            # Process each data task
            for task in data_tasks:
                file_path = task['file']
                data_type = task['type']

                # Check if file already exists and is valid
                if not force_reload and self.file_exists_and_not_empty(file_path):
                    logger.info(f"Skipping {data_type} for {symbol} - file already exists: {file_path}")
                    continue

                # Load and save the data
                try:
                    logger.info(f"Loading {data_type} for {symbol}")
                    data = task['loader']()
                    if self.save_data_to_csv(data, file_path, data_type, symbol):
                        data_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load {data_type} for {symbol}: {e}")

            logger.info(f"Completed data loading for {symbol}")
            return data_loaded

        except Exception as e:
            logger.error(f"Critical error loading data for {symbol}: {e}")
            return False

    def _load_stock_info(self, symbol: str) -> pd.DataFrame:
        """Load stock info and convert to DataFrame."""
        try:
            stock_info = self.client.get_stock_info(symbol, source='VCI')
            # Convert dict to DataFrame
            return pd.DataFrame([stock_info])
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            return pd.DataFrame()

    def load_multiple_symbols(
        self,
        symbols: List[str],
        base_path: Optional[Path] = None,
        force_reload: bool = False,
        delay_between_symbols: float = 1.0
    ) -> dict:
        """Load data for multiple symbols with progress tracking.

        Args:
            symbols: List of stock symbols
            base_path: Base path for data directory
            force_reload: If True, reload all data even if files exist
            delay_between_symbols: Delay in seconds between symbol processing

        Returns:
            Dictionary with loading statistics
        """
        logger.info(f"Starting batch data loading for {len(symbols)} symbols")

        successful_loads = 0
        skipped_loads = 0
        failed_loads = 0

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Processing {symbol} ({i}/{len(symbols)})")

            try:
                data_loaded = self.load_symbol_data(symbol, base_path, force_reload)
                if data_loaded:
                    successful_loads += 1
                    logger.info(f"✓ Successfully loaded new data for {symbol}")
                else:
                    skipped_loads += 1
                    logger.info(f"✓ Skipped {symbol} - all data already exists")
            except Exception as e:
                failed_loads += 1
                logger.error(f"✗ Failed to load data for {symbol}: {e}")

            # Add delay between symbols to be respectful to APIs
            if delay_between_symbols > 0 and i < len(symbols):
                time.sleep(delay_between_symbols)

        results = {
            'total_symbols': len(symbols),
            'successful_loads': successful_loads,
            'skipped_loads': skipped_loads,
            'failed_loads': failed_loads,
            'success_rate': (successful_loads + skipped_loads) / len(symbols) * 100 if symbols else 0
        }

        logger.info(f"Batch loading completed: {results}")
        return results

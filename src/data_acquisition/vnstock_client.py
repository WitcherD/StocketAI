"""
VNStock Client Module

This module provides the VNStockClient class and supporting utilities for
interacting with the vnstock library with error handling and caching.
"""

import hashlib
import json
import logging
import pickle
import time
import threading
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from vnstock import Vnstock
    from vnstock.vnstock import Listing, Quote, Company, Finance, Trading, Screener, Fund
except ImportError:
    # Fallback for different vnstock versions
    try:
        from vnstock.vnstock import Vnstock, Listing, Quote, Company, Finance, Trading, Screener, Fund
    except ImportError:
        # Try direct imports
        from vnstock import Vnstock
        from vnstock.api.quote import Quote
        from vnstock.api.company import Company
        from vnstock.api.financial import Finance
        from vnstock.api.listing import Listing
        from vnstock.api.trading import Trading
        from vnstock.api.screener import Screener
        from vnstock.explorer.fmarket import Fund

from .config import get_config, DataAcquisitionConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataValidationResult:
    """Result of data validation operation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class DataValidator:
    """Utility class for data validation operations."""

    def __init__(self, config: Optional[DataAcquisitionConfig] = None):
        """Initialize validator with configuration."""
        self.config = config or get_config()

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        max_missing_ratio: Optional[float] = None
    ) -> DataValidationResult:
        """
        Validate a pandas DataFrame.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            max_missing_ratio: Maximum allowed missing data ratio

        Returns:
            DataValidationResult with validation details
        """
        errors = []
        warnings = []
        metadata = {}

        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            metadata.update({
                'total_rows': 0,
                'total_columns': 0,
                'missing_ratios': {},
                'numeric_columns': [],
                'datetime_columns': []
            })
            return DataValidationResult(False, errors, warnings, metadata)

        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Check for missing data
        max_ratio = max_missing_ratio or self.config.max_missing_ratio
        missing_ratios = df.isnull().mean()

        high_missing = missing_ratios[missing_ratios > max_ratio]
        if not high_missing.empty:
            warnings.append(f"Columns with high missing ratios: {high_missing.to_dict()}")

        # Check for outliers using IQR method (more robust than z-score for small datasets)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if len(df[col].dropna()) >= 4:  # Need at least 4 values for meaningful IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    warnings.append(f"Column '{col}' has {len(outliers)} potential outliers")

        # Metadata
        metadata.update({
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_ratios': missing_ratios.to_dict(),
            'numeric_columns': list(numeric_columns),
            'datetime_columns': list(df.select_dtypes(include=['datetime']).columns)
        })

        is_valid = len(errors) == 0
        return DataValidationResult(is_valid, errors, warnings, metadata)

    def validate_stock_data(self, df: pd.DataFrame) -> DataValidationResult:
        """
        Validate stock price data specifically.

        Args:
            df: Stock price DataFrame with OHLCV columns

        Returns:
            DataValidationResult with validation details
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return self.validate_dataframe(df, required_columns)

    def validate_financial_data(self, df: pd.DataFrame) -> DataValidationResult:
        """
        Validate financial statement data.

        Args:
            df: Financial data DataFrame

        Returns:
            DataValidationResult with validation details
        """
        # Financial data can have various column structures
        # Basic validation for now
        return self.validate_dataframe(df, [])


class DataCache:
    """Simple file-based cache for API responses."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl: Optional[int] = None, config: Optional[DataAcquisitionConfig] = None):
        """Initialize cache with directory and TTL."""
        self.config = config or get_config()
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl or self.config.cache_ttl

    def _get_cache_key(self, data: Any) -> str:
        """Generate cache key from data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{key}.pkl"

    def get(self, key_data: Any) -> Optional[Any]:
        """Get cached data if available and not expired."""
        if not self.config.cache_enabled:
            return None

        key = self._get_cache_key(key_data)
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            # Check if cache is expired
            mtime = cache_path.stat().st_mtime
            if time.time() - mtime > self.ttl:
                cache_path.unlink()  # Remove expired cache
                return None

            # Load cached data
            with open(cache_path, 'rb') as f:
                cached_item = pickle.load(f)

            logger.debug(f"Cache hit for key: {key}")
            return cached_item['data']

        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            return None

    def set(self, key_data: Any, data: Any) -> None:
        """Cache data with key."""
        if not self.config.cache_enabled:
            return

        key = self._get_cache_key(key_data)
        cache_path = self._get_cache_path(key)

        try:
            cached_item = {
                'data': data,
                'timestamp': time.time(),
                'ttl': self.ttl
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cached_item, f)

            logger.debug(f"Cached data for key: {key}")

        except Exception as e:
            logger.warning(f"Error writing cache: {e}")

    def clear(self) -> None:
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")

    def cleanup(self) -> None:
        """Remove expired cache files."""
        try:
            current_time = time.time()
            expired_files = []

            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    mtime = cache_file.stat().st_mtime
                    if current_time - mtime > self.ttl:
                        cache_file.unlink()
                        expired_files.append(cache_file.name)
                except Exception:
                    continue

            if expired_files:
                logger.info(f"Cleaned up {len(expired_files)} expired cache files")

        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")


class RateLimiter:
    """Thread-safe rate limiter for API calls."""

    def __init__(self, config: Optional[DataAcquisitionConfig] = None):
        """Initialize rate limiter with configuration."""
        self.config = config or get_config()
        self._lock = threading.Lock()
        self._request_times = {}  # source -> list of request timestamps

        # Initialize request tracking for each source
        for source in self.config.rate_limits.keys():
            self._request_times[source] = []

    def _cleanup_old_requests(self, source: str) -> None:
        """Remove requests older than 1 minute."""
        current_time = time.time()
        cutoff_time = current_time - 60  # 1 minute window

        with self._lock:
            self._request_times[source] = [
                req_time for req_time in self._request_times[source]
                if req_time > cutoff_time
            ]

    def _calculate_delay(self, source: str) -> float:
        """Calculate delay needed to respect rate limit."""
        if not self.config.enable_rate_limiting:
            return 0.0

        rate_limit = self.config.rate_limits.get(source, 100)  # Default to 100 if not specified
        effective_limit = int(rate_limit * (1 - self.config.rate_limit_buffer))  # Apply buffer

        self._cleanup_old_requests(source)

        with self._lock:
            request_count = len(self._request_times[source])

            if request_count < effective_limit:
                return 0.0  # No delay needed

            # Calculate delay based on oldest request in the window
            if self._request_times[source]:
                oldest_request = min(self._request_times[source])
                # Time until oldest request expires from the 1-minute window
                delay = 60 - (time.time() - oldest_request)
                return max(0.0, delay)
            else:
                return 0.0

    def wait_if_needed(self, source: str) -> None:
        """Wait if necessary to respect rate limits."""
        if not self.config.enable_rate_limiting:
            return

        delay = self._calculate_delay(source)
        if delay > 0:
            logger.debug(f"Rate limiting: waiting {delay:.2f}s for {source}")
            time.sleep(delay)

        # Record this request
        with self._lock:
            self._request_times[source].append(time.time())

    def get_rate_limit_info(self, source: str) -> Dict[str, Any]:
        """Get current rate limiting information for a source."""
        self._cleanup_old_requests(source)

        with self._lock:
            request_count = len(self._request_times[source])
            rate_limit = self.config.rate_limits.get(source, 100)
            effective_limit = int(rate_limit * (1 - self.config.rate_limit_buffer))

            return {
                'source': source,
                'current_requests': request_count,
                'rate_limit': rate_limit,
                'effective_limit': effective_limit,
                'buffer_percent': self.config.rate_limit_buffer * 100,
                'can_make_request': request_count < effective_limit
            }


class VNStockClient:
    """Wrapper for vnstock library with error handling and caching."""

    def __init__(self, config: Optional[DataAcquisitionConfig] = None):
        """Initialize VNStock client."""
        self.config = config or get_config()
        self.cache = DataCache()
        self.rate_limiter = RateLimiter(self.config)
        self._vnstock = None  # Lazy loading
        self._listing = None  # Lazy loading for listing
        self._stock_instances = {}  # Cache for stock instances

    @property
    def vnstock(self):
        """Get vnstock instance with lazy loading."""
        if self._vnstock is None:
            try:
                self._vnstock = Vnstock()
                logger.info("VNStock client initialized")
            except ImportError as e:
                raise ImportError("VNStock library not installed. Please install from source.") from e
        return self._vnstock

    @property
    def listing(self):
        """Get listing instance with lazy loading."""
        if self._listing is None:
            try:
                # Import here to avoid circular imports
                self._listing = Listing()
                logger.info("VNStock Listing client initialized")
            except ImportError as e:
                raise ImportError("VNStock library not installed. Please install from source.") from e
        return self._listing

    def _get_stock_instance(self, symbol: str, source: str = 'VCI'):
        """Get or create stock instance for symbol and source."""
        key = f"{symbol}_{source}"
        if key not in self._stock_instances:
            try:
                self._stock_instances[key] = self.vnstock.stock(symbol=symbol, source=source)
                logger.info(f"Stock instance created for {symbol} with source {source}")
            except ImportError as e:
                raise ImportError("VNStock library not installed. Please install from source.") from e
        return self._stock_instances[key]

    def _retry_request(self, func: Callable, *args, **kwargs) -> Any:
        """Retry a function call with exponential backoff."""
        max_retries = kwargs.pop('max_retries', self.config.max_retries)
        retry_delay = kwargs.pop('retry_delay', self.config.retry_delay)
        backoff_factor = kwargs.pop('backoff_factor', self.config.backoff_factor)

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    delay = retry_delay * (backoff_factor ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")

        raise last_exception

    def get_stock_info(self, symbol: str, source: str = 'VCI') -> Dict[str, Any]:
        """Get stock information with caching."""
        cache_key = f"stock_info_{symbol}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            # Use company.overview() method which exists according to API docs
            stock_info = self.vnstock.stock(symbol=symbol, source=source).company.overview()

            # Convert to dictionary
            result = {}
            for col in stock_info.columns:
                result[col] = stock_info[col].iloc[0] if len(stock_info) > 0 else None

            # Cache the result
            self.cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            raise

    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        resolution: str = "1D",
        source: str = 'VCI',
        throw_errors: bool = True
    ) -> pd.DataFrame:
        """Get historical price data with caching and rate limiting."""
        cache_key = f"historical_{symbol}_{start_date}_{end_date}_{resolution}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Apply rate limiting
        self.rate_limiter.wait_if_needed(source)

        # Fetch from API
        try:
            # Use Quote class directly instead of through stock instance
            quote = Quote(symbol=symbol, source=source)
            data = quote.history(
                symbol=symbol,
                start=start_date,
                end=end_date,
                resolution=resolution
            )

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            if throw_errors:
                logger.error(f"Error fetching historical data for {symbol} from '{source}' source: {e}")
                raise
            else:
                return pd.DataFrame()

    def get_financial_ratios(self, symbol: str, period: str = "yearly", source: str = 'VCI') -> pd.DataFrame:
        """Get financial ratios with caching."""
        cache_key = f"ratios_{symbol}_{period}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            ratios = self.vnstock.stock(symbol=symbol, source=source).finance.ratio(period=period)

            # Cache the result
            self.cache.set(cache_key, ratios)
            return ratios

        except Exception as e:
            logger.error(f"Error fetching financial ratios for {symbol}: {e}")
            raise

    def get_balance_sheet(self, symbol: str, period: str = "yearly", source: str = 'VCI') -> pd.DataFrame:
        """Get balance sheet data with caching."""
        cache_key = f"balance_sheet_{symbol}_{period}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            balance_sheet = self.vnstock.stock(symbol=symbol, source=source).finance.balance_sheet(period=period, source=source)

            # Cache the result
            self.cache.set(cache_key, balance_sheet)
            return balance_sheet

        except Exception as e:
            logger.error(f"Error fetching balance sheet for {symbol}: {e}")
            raise

    def get_income_statement(self, symbol: str, period: str = "yearly", source: str = 'VCI') -> pd.DataFrame:
        """Get income statement data with caching."""
        cache_key = f"income_statement_{symbol}_{period}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            income_stmt = self.vnstock.stock(symbol=symbol, source=source).finance.income_statement(period=period, source=source)

            # Cache the result
            self.cache.set(cache_key, income_stmt)
            return income_stmt

        except Exception as e:
            logger.error(f"Error fetching income statement for {symbol}: {e}")
            raise

    def get_cash_flow(self, symbol: str, period: str = "yearly", source: str = 'VCI') -> pd.DataFrame:
        """Get cash flow statement data with caching."""
        cache_key = f"cash_flow_{symbol}_{period}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            cash_flow = self.vnstock.stock(symbol=symbol, source=source).finance.cash_flow(period=period, source=source)

            # Cache the result
            self.cache.set(cache_key, cash_flow)
            return cash_flow

        except Exception as e:
            logger.error(f"Error fetching cash flow for {symbol}: {e}")
            raise

    def _get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        # Common sources available in vnstock
        return ['VCI', 'TCBS', 'MSN', 'FMARKET']

    def _merge_financial_data(self, data_frames: List[pd.DataFrame], merge_strategy: str = 'concatenate') -> pd.DataFrame:
        """
        Merge financial data from multiple sources.

        Args:
            data_frames: List of DataFrames from different sources
            merge_strategy: Strategy for merging ('concatenate', 'prioritize', 'average')

        Returns:
            Merged DataFrame
        """
        if not data_frames:
            return pd.DataFrame()

        if len(data_frames) == 1:
            return data_frames[0]

        # Filter out empty DataFrames
        valid_dfs = [df for df in data_frames if not df.empty]

        if not valid_dfs:
            return pd.DataFrame()

        if merge_strategy == 'concatenate':
            # Concatenate along rows (time periods)
            try:
                merged = pd.concat(valid_dfs, ignore_index=True)

                # Remove duplicates based on key columns (symbol, year, quarter, etc.)
                key_columns = []
                for col in ['symbol', 'year', 'quarter', 'period']:
                    if col in merged.columns:
                        key_columns.append(col)

                if key_columns:
                    merged = merged.drop_duplicates(subset=key_columns, keep='first')

                return merged

            except Exception as e:
                logger.warning(f"Error concatenating data: {e}. Falling back to prioritize strategy.")

        # Default strategy: prioritize sources in order
        # Use the first non-empty DataFrame as base
        result = valid_dfs[0].copy()

        # For each subsequent DataFrame, merge in missing data
        for df in valid_dfs[1:]:
            if df.empty:
                continue

            # Find common columns
            common_columns = list(set(result.columns) & set(df.columns))

            if common_columns:
                # For each common column, fill missing values in result with values from df
                for col in common_columns:
                    if col in ['symbol', 'year', 'quarter', 'period']:
                        continue  # Skip key columns

                    # Fill NaN values in result with values from df
                    mask = result[col].isna()
                    if not df.empty and col in df.columns:
                        result.loc[mask, col] = df.loc[:, col].values[:len(result)]

        return result

    def _merge_historical_data(self, data_frames: List[pd.DataFrame], merge_strategy: str = 'prioritize') -> pd.DataFrame:
        """
        Merge historical price data from multiple sources.

        Args:
            data_frames: List of DataFrames from different sources
            merge_strategy: Strategy for merging ('concatenate', 'prioritize', 'average')

        Returns:
            Merged DataFrame
        """
        if not data_frames:
            return pd.DataFrame()

        if len(data_frames) == 1:
            return data_frames[0]

        # Filter out empty DataFrames
        valid_dfs = [df for df in data_frames if not df.empty]

        if not valid_dfs:
            return pd.DataFrame()

        if merge_strategy == 'concatenate':
            # Concatenate along time axis
            try:
                merged = pd.concat(valid_dfs, ignore_index=True)

                # Remove duplicates based on date/time column
                date_columns = []
                for col in ['time', 'date', 'Date', 'Time']:
                    if col in merged.columns:
                        date_columns.append(col)
                        break

                if date_columns:
                    # Sort by date and remove duplicates, keeping first occurrence
                    merged = merged.sort_values(date_columns[0])
                    merged = merged.drop_duplicates(subset=date_columns, keep='first')

                return merged

            except Exception as e:
                logger.warning(f"Error concatenating historical data: {e}. Falling back to prioritize strategy.")

        # Default strategy: prioritize sources in order
        # Use the first non-empty DataFrame as base
        result = valid_dfs[0].copy()

        # For each subsequent DataFrame, merge in missing data
        for df in valid_dfs[1:]:
            if df.empty:
                continue

            # Find date column for merging
            date_col = None
            for col in ['time', 'date', 'Date', 'Time']:
                if col in result.columns and col in df.columns:
                    date_col = col
                    break

            if date_col is None:
                logger.warning("No common date column found for merging historical data")
                continue

            # Merge on date column, filling missing values
            try:
                # Set date column as index for both DataFrames
                result_indexed = result.set_index(date_col)
                df_indexed = df.set_index(date_col)

                # Find common columns (excluding data_source)
                common_columns = [col for col in result.columns if col in df.columns and col != 'data_source' and col != date_col]

                if common_columns:
                    # For each common column, fill missing values in result with values from df
                    for col in common_columns:
                        # Fill NaN values in result with values from df
                        mask = result_indexed[col].isna()
                        if col in df_indexed.columns:
                            result_indexed.loc[mask, col] = df_indexed.loc[:, col]

                # Reset index
                result = result_indexed.reset_index()

            except Exception as e:
                logger.warning(f"Error merging historical data on date column: {e}")
                continue

        return result

    def get_merged_financial_ratios(
        self,
        symbol: str,
        period: str = "yearly",
        merge_strategy: str = 'prioritize'
    ) -> pd.DataFrame:
        """
        Get financial ratios from multiple sources for the specified period and merge the data.

        Args:
            symbol: Stock symbol
            period: Period type ('yearly' or 'quarterly')
            merge_strategy: Strategy for merging data ('concatenate', 'prioritize', 'average')

        Returns:
            Merged DataFrame with financial ratios from multiple sources
        """
        available_sources = ["TCBS", "VCI"]
        all_data = []

        # Fetch data for the specified period from all sources
        for source in available_sources:
            try:
                logger.info(f"Fetching financial ratios for {symbol} from {source} ({period})")
                data = self.get_financial_ratios(symbol, period, source)

                if not data.empty:
                    # Add source and period columns to track data origin
                    data = data.copy()
                    data['data_source'] = source
                    data['period_type'] = period
                    all_data.append(data)

            except Exception as e:
                logger.warning(f"Failed to fetch financial ratios for {symbol} from {source} ({period}): {e}")
                continue

        # Merge all collected data
        merged_data = self._merge_financial_data(all_data, merge_strategy)

        if merged_data.empty:
            logger.warning(f"No financial ratios data found for {symbol} from any source")
            return pd.DataFrame()

        logger.info(f"Successfully merged financial ratios for {symbol} from {len(all_data)} sources")
        return merged_data

    def get_merged_balance_sheet(
        self,
        symbol: str,
        period: str = "yearly",
        merge_strategy: str = 'prioritize'
    ) -> pd.DataFrame:
        """
        Get balance sheet data from multiple sources for the specified period and merge the data.

        Args:
            symbol: Stock symbol
            period: Period type ('yearly' or 'quarterly')
            merge_strategy: Strategy for merging data ('concatenate', 'prioritize', 'average')

        Returns:
            Merged DataFrame with balance sheet data from multiple sources
        """
        available_sources = ["TCBS", "VCI"]
        all_data = []

        # Fetch data for the specified period from all sources
        for source in available_sources:
            try:
                logger.info(f"Fetching balance sheet for {symbol} from {source} ({period})")
                data = self.get_balance_sheet(symbol, period, source)

                if not data.empty:
                    # Add source and period columns to track data origin
                    data = data.copy()
                    data['data_source'] = source
                    data['period_type'] = period
                    all_data.append(data)

            except Exception as e:
                logger.warning(f"Failed to fetch balance sheet for {symbol} from {source} ({period}): {e}")
                continue

        # Merge all collected data
        merged_data = self._merge_financial_data(all_data, merge_strategy)

        if merged_data.empty:
            logger.warning(f"No balance sheet data found for {symbol} from any source")
            return pd.DataFrame()

        logger.info(f"Successfully merged balance sheet for {symbol} from {len(all_data)} sources")
        return merged_data

    def get_merged_income_statement(
        self,
        symbol: str,
        period: str = "yearly",
        merge_strategy: str = 'prioritize'
    ) -> pd.DataFrame:
        """
        Get income statement data from multiple sources for the specified period and merge the data.

        Args:
            symbol: Stock symbol
            period: Period type ('yearly' or 'quarterly')
            merge_strategy: Strategy for merging data ('concatenate', 'prioritize', 'average')

        Returns:
            Merged DataFrame with income statement data from multiple sources
        """
        available_sources = ["TCBS", "VCI"]
        all_data = []

        # Fetch data for the specified period from all sources
        for source in available_sources:
            try:
                logger.info(f"Fetching income statement for {symbol} from {source} ({period})")
                data = self.get_income_statement(symbol, period, source)

                if not data.empty:
                    # Add source and period columns to track data origin
                    data = data.copy()
                    data['data_source'] = source
                    data['period_type'] = period
                    all_data.append(data)

            except Exception as e:
                logger.warning(f"Failed to fetch income statement for {symbol} from {source} ({period}): {e}")
                continue

        # Merge all collected data
        merged_data = self._merge_financial_data(all_data, merge_strategy)

        if merged_data.empty:
            logger.warning(f"No income statement data found for {symbol} from any source")
            return pd.DataFrame()

        logger.info(f"Successfully merged income statement for {symbol} from {len(all_data)} sources")
        return merged_data

    def get_merged_cash_flow(
        self,
        symbol: str,
        period: str = "yearly",
        merge_strategy: str = 'prioritize'
    ) -> pd.DataFrame:
        """
        Get cash flow statement data from multiple sources for the specified period and merge the data.

        Args:
            symbol: Stock symbol
            period: Period type ('yearly' or 'quarterly')
            merge_strategy: Strategy for merging data ('concatenate', 'prioritize', 'average')

        Returns:
            Merged DataFrame with cash flow data from multiple sources
        """
        available_sources = ["TCBS", "VCI"]
        all_data = []

        # Fetch data for the specified period from all sources
        for source in available_sources:
            try:
                logger.info(f"Fetching cash flow for {symbol} from {source} ({period})")
                data = self.get_cash_flow(symbol, period, source)

                if not data.empty:
                    # Add source and period columns to track data origin
                    data = data.copy()
                    data['data_source'] = source
                    data['period_type'] = period
                    all_data.append(data)

            except Exception as e:
                logger.warning(f"Failed to fetch cash flow for {symbol} from {source} ({period}): {e}")
                continue

        # Merge all collected data
        merged_data = self._merge_financial_data(all_data, merge_strategy)

        if merged_data.empty:
            logger.warning(f"No cash flow data found for {symbol} from any source")
            return pd.DataFrame()

        logger.info(f"Successfully merged cash flow for {symbol} from {len(all_data)} sources")
        return merged_data

    def get_merged_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        resolution: str = "1D",
        merge_strategy: str = 'prioritize'
    ) -> pd.DataFrame:
        """
        Get historical price data from multiple sources for the specified date range and merge the data.

        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            resolution: Data resolution ('1D', '1W', '1M', etc.)
            merge_strategy: Strategy for merging data ('concatenate', 'prioritize', 'average')

        Returns:
            Merged DataFrame with historical price data from multiple sources
        """
        available_sources = ["TCBS", "VCI", "MSN"]
        all_data = []

        # Fetch data for the specified date range from all sources
        for source in available_sources:
            try:
                logger.info(f"Fetching historical data for {symbol} from {source} ({start_date} to {end_date}, {resolution})")
                data = self.get_historical_data(symbol, start_date, end_date, resolution, source, False)

                if not data.empty:
                    # Add source column to track data origin
                    data = data.copy()
                    data['data_source'] = source
                    all_data.append(data)

            except Exception as e:
                logger.warning(f"Failed to fetch historical data for {symbol} from {source} ({start_date} to {end_date}, {resolution}): {e}")
                continue

        # Merge all collected data
        merged_data = self._merge_historical_data(all_data, merge_strategy)

        if merged_data.empty:
            logger.error(f"No historical data found for {symbol} from any source")
            return pd.DataFrame()

        logger.info(f"Successfully merged historical data for {symbol} from {len(all_data)} sources")
        return merged_data
    

    def get_vn30_constituents(self, source: str = 'VCI') -> pd.DataFrame:
        """Get VN30 constituents data with caching."""
        cache_key = f"vn30_constituents_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            # Use vnstock listing API to get VN30 constituents
            # According to API docs, listing.symbols_by_group returns a pandas Series
            symbols_series = self.listing.symbols_by_group('VN30')

            # Convert Series to DataFrame - handle both Series and DataFrame responses
            if isinstance(symbols_series, pd.Series):
                constituents = pd.DataFrame({
                    'symbol': symbols_series.values
                })
            else:
                # If it's already a DataFrame, use it as-is
                constituents = symbols_series.copy()

            # Cache the result
            self.cache.set(cache_key, constituents)
            return constituents

        except Exception as e:
            logger.error(f"Error fetching VN30 constituents: {e}")
            raise

    def get_intraday_data(self, symbol: str, page_size: int = 100, source: str = 'VCI') -> pd.DataFrame:
        """Get intraday tick data with caching."""
        cache_key = f"intraday_{symbol}_{page_size}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            quote = Quote(symbol=symbol, source=source)
            data = quote.intraday(symbol=symbol, page_size=page_size, show_log=False)

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            raise

    def get_price_depth(self, symbol: str, source: str = 'VCI') -> pd.DataFrame:
        """Get market depth data with caching."""
        cache_key = f"price_depth_{symbol}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            quote = Quote(symbol=symbol, source=source)
            data = quote.price_depth(symbol=symbol)

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching price depth for {symbol}: {e}")
            raise

    def get_company_profile(self, symbol: str, source: str = 'TCBS') -> pd.DataFrame:
        """Get detailed company profile with caching."""
        cache_key = f"company_profile_{symbol}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            company = Company(symbol=symbol, source=source)
            data = company.profile()

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol}: {e}")
            raise

    def get_company_shareholders(self, symbol: str, source: str = 'TCBS') -> pd.DataFrame:
        """Get company shareholders data with caching."""
        cache_key = f"company_shareholders_{symbol}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            company = Company(symbol=symbol, source=source)
            data = company.shareholders()

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching shareholders for {symbol}: {e}")
            raise

    def get_company_officers(self, symbol: str, source: str = 'TCBS') -> pd.DataFrame:
        """Get company officers and management data with caching."""
        cache_key = f"company_officers_{symbol}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            company = Company(symbol=symbol, source=source)
            data = company.officers()

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching officers for {symbol}: {e}")
            raise

    def get_company_events(self, symbol: str, source: str = 'TCBS', event_type: str = 'all', limit: int = 10) -> pd.DataFrame:
        """Get company events and announcements with caching."""
        cache_key = f"company_events_{symbol}_{source}_{event_type}_{limit}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            company = Company(symbol=symbol, source=source)
            data = company.events(event_type=event_type, limit=limit)

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching events for {symbol}: {e}")
            raise

    def get_company_news(self, symbol: str, source: str = 'TCBS', limit: int = 10, start_date: str = None) -> pd.DataFrame:
        """Get company news and press releases with caching."""
        cache_key = f"company_news_{symbol}_{source}_{limit}_{start_date or 'none'}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            company = Company(symbol=symbol, source=source)
            data = company.news(limit=limit, start_date=start_date)

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            raise

    def get_company_dividends(self, symbol: str, source: str = 'TCBS') -> pd.DataFrame:
        """Get dividend history with caching."""
        cache_key = f"company_dividends_{symbol}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            company = Company(symbol=symbol, source=source)
            data = company.dividends()

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching dividends for {symbol}: {e}")
            raise

    def get_all_symbols(self, source: str = 'VCI') -> pd.DataFrame:
        """Get all listed symbols with caching."""
        cache_key = f"all_symbols_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            listing = Listing(source=source)
            data = listing.all_symbols()

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching all symbols from {source}: {e}")
            raise

    def get_symbols_by_industries(self, source: str = 'VCI') -> pd.DataFrame:
        """Get symbols grouped by industry with caching."""
        cache_key = f"symbols_by_industries_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            listing = Listing(source=source)
            data = listing.symbols_by_industries()

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching symbols by industries from {source}: {e}")
            raise

    def get_symbols_by_exchange(self, source: str = 'VCI') -> pd.DataFrame:
        """Get symbols grouped by exchange with caching."""
        cache_key = f"symbols_by_exchange_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            from vnstock import Listing
            listing = Listing(source=source)
            data = listing.symbols_by_exchange()

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching symbols by exchange from {source}: {e}")
            raise

    def get_symbols_by_group(self, group: str, source: str = 'VCI') -> pd.Series:
        """Get symbols by index group with caching."""
        cache_key = f"symbols_by_group_{group}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            listing = Listing(source=source)
            data = listing.symbols_by_group(group)

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching symbols by group {group} from {source}: {e}")
            raise

    def get_price_board(self, symbols: List[str], source: str = 'VCI') -> pd.DataFrame:
        """Get real-time price board for multiple symbols with caching."""
        symbols_str = '_'.join(sorted(symbols))
        cache_key = f"price_board_{symbols_str}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            trading = Trading(source=source)
            data = trading.price_board(symbols)

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching price board for {symbols} from {source}: {e}")
            raise

    def get_stock_screening(self, params: Dict[str, Any], limit: int = 50, source: str = 'TCBS') -> pd.DataFrame:
        """Get stock screening results with caching."""
        params_str = '_'.join(f"{k}_{v}" for k, v in sorted(params.items()))
        cache_key = f"stock_screening_{params_str}_{limit}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            screener = Screener(source=source)
            data = screener.stock(params=params, limit=limit)

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching stock screening with params {params} from {source}: {e}")
            raise

    def get_fund_listing(self, source: str = 'FMARKET') -> pd.DataFrame:
        """Get mutual fund listing with caching."""
        cache_key = f"fund_listing_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            fund = Fund(source=source)
            data = fund.listing()

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching fund listing from {source}: {e}")
            raise

    def get_fund_top_holdings(self, fund_symbol: str, source: str = 'FMARKET') -> pd.DataFrame:
        """Get top holdings for a mutual fund with caching."""
        cache_key = f"fund_holdings_{fund_symbol}_{source}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from API
        try:
            fund = Fund(source=source)
            data = fund.details.top_holding(fund_symbol)

            # Cache the result
            self.cache.set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching fund holdings for {fund_symbol} from {source}: {e}")
            raise


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate date range format and logic."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if start >= end:
            raise ValueError("Start date must be before end date")

        # Check if date range is reasonable (not more than 20 years)
        if end - start > timedelta(days=20*365):
            logger.warning("Date range is very large (>20 years)")

        return True
    except ValueError as e:
        logger.error(f"Invalid date range: {e}")
        return False


def format_date(date: Union[str, datetime]) -> str:
    """Format date to string."""
    if isinstance(date, datetime):
        return date.strftime("%Y-%m-%d")
    return str(date)

"""
Feature Engineering Module for VN30 with TFT Compatibility

This module provides comprehensive feature engineering capabilities optimized
for VN30 stock prediction using TFT (Temporal Fusion Transformers) model.
Implements Alpha158-compatible technical indicators and feature categories.

Key Components:
- TFT-compatible feature categories (OBSERVED_INPUT, KNOWN_INPUT, STATIC_INPUT)
- Technical indicators: RESI, WVMA, RSQR, CORR, CORD, ROC, VSTD, STD, KLEN, KLOW
- VN30-specific feature optimization
- Pandas-based technical indicator calculations
- Feature validation and quality assessment

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
import warnings



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class VN30TFTFeatureEngineer:
    """
    Feature engineering class optimized for VN30 stocks and TFT model compatibility.

    Implements technical indicators compatible with TFT's Alpha158 feature set:
    - OBSERVED_INPUT: Time-varying technical indicators
    - KNOWN_INPUT: Temporal features (month, day_of_week, year)
    - STATIC_INPUT: Static features (const, market indicators)
    """

    def __init__(self, symbols_directory_path: str = None, lookback_periods: Dict[str, int] = None):
        """
        Initialize VN30 TFT feature engineer.

        Args:
            symbols_directory_path: Directory path containing VN30 symbol folders with processed data
            lookback_periods: Dictionary of lookback periods for indicators
        """
        self.symbols_directory_path = Path(symbols_directory_path) if symbols_directory_path else Path("data/symbols")

        # Default lookback periods optimized for VN30
        self.lookback_periods = lookback_periods or {
            'short': 5,
            'medium': 10,
            'long': 20,
            'very_long': 60
        }

        # TFT-compatible feature configuration
        self.feature_config = {
            'OBSERVED_INPUT': [
                'RESI5', 'RESI10',        # Residuals
                'WVMA5', 'WVMA60',        # Weighted moving averages
                'RSQR5', 'RSQR10', 'RSQR20', 'RSQR60',  # R-squared
                'CORR5', 'CORR10', 'CORR20', 'CORR60',  # Correlations
                'CORD5', 'CORD10', 'CORD60',            # Correlation differences
                'ROC60',                  # Rate of change
                'VSTD5',                  # Volatility
                'STD5',                   # Standard deviation
                'KLEN', 'KLOW'            # Momentum indicators
            ],
            'KNOWN_INPUT': [
                'month', 'day_of_week', 'year'
            ],
            'STATIC_INPUT': [
                'const'
            ]
        }

        logger.info(f"Initialized VN30TFTFeatureEngineer with symbols_directory_path: {self.symbols_directory_path}")
        logger.info(f"Initialized VN30TFTFeatureEngineer with lookback periods: {self.lookback_periods}")

    def load_cleaned_data(self, symbol_name: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Load cleaned data for a specific symbol from pickle file.

        Args:
            symbol_name: Stock symbol name to load cleaned data for
            start_date: Start date for data loading (YYYY-MM-DD)
            end_date: End date for data loading (YYYY-MM-DD)

        Returns:
            DataFrame with cleaned OHLCV data or None if not found
        """
        pickle_file = self.symbols_directory_path / symbol_name / 'processed' / f"{symbol_name.lower()}_cleaned.pkl"

        if not pickle_file.exists():
            logger.warning(f"Cleaned data file not found: {pickle_file}")
            return None

        try:
            df = pd.read_pickle(pickle_file)

            # Filter by date range if specified
            if start_date:
                df = df[df['time'] >= start_date]
            if end_date:
                df = df[df['time'] <= end_date]

            logger.info(f"Loaded cleaned data for {symbol_name}: {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error loading cleaned data for {symbol_name}: {e}")
            return None

    def engineer_features_for_symbol(self, symbol_name: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Load cleaned data for a symbol and engineer all TFT features.

        Args:
            symbol_name: Stock symbol name to process
            start_date: Start date for data loading (YYYY-MM-DD)
            end_date: End date for data loading (YYYY-MM-DD)

        Returns:
            DataFrame with engineered features or None if processing fails
        """
        # Load cleaned data from pickle file
        df = self.load_cleaned_data(symbol_name, start_date, end_date)
        if df is None:
            logger.error(f"Failed to load cleaned data for {symbol_name}")
            return None

        # Engineer features
        try:
            engineered_df = self.engineer_all_features(df)
            logger.info(f"Successfully engineered features for {symbol_name}: {engineered_df.shape}")
            return engineered_df
        except Exception as e:
            logger.error(f"Error engineering features for {symbol_name}: {e}")
            return None

    def save_engineered_features(self, df: pd.DataFrame, symbol_name: str, output_directory_path: str = None) -> bool:
        """
        Save engineered features for a symbol.

        Args:
            df: DataFrame with engineered features
            symbol_name: Stock symbol name
            output_directory_path: Output directory path (default: symbol's features directory)

        Returns:
            True if save successful, False otherwise
        """
        try:
            if output_directory_path is None:
                output_directory_path = self.symbols_directory_path / symbol_name / 'features'
            else:
                output_directory_path = Path(output_directory_path)

            output_directory_path.mkdir(parents=True, exist_ok=True)
            output_file = output_directory_path / f'{symbol_name}_tft_features.csv'

            df.to_csv(output_file, index=False)
            logger.info(f"Saved engineered features for {symbol_name} to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving features for {symbol_name}: {e}")
            return False

    def process_symbol_complete(self, symbol_name: str, save_features: bool = True) -> Optional[pd.DataFrame]:
        """
        Complete processing pipeline for a single symbol: load cleaned data, engineer features, and optionally save.

        Args:
            symbol_name: Stock symbol name to process
            save_features: Whether to save engineered features to disk

        Returns:
            DataFrame with engineered features or None if processing fails
        """
        logger.info(f"Starting complete processing for symbol: {symbol_name}")

        # Engineer features
        features_df = self.engineer_features_for_symbol(symbol_name)
        if features_df is None:
            logger.error(f"Failed to engineer features for {symbol_name}")
            return None

        # Save features if requested
        if save_features:
            success = self.save_engineered_features(features_df, symbol_name)
            if not success:
                logger.warning(f"Failed to save features for {symbol_name}, but processing completed")

        logger.info(f"Completed processing for symbol: {symbol_name}")
        return features_df

    def calculate_residuals(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate residuals (RESI) using pandas operations matching qlib formulas.

        Formula: RESI_period = close - MA(close, period)

        Args:
            df: DataFrame with OHLCV data
            periods: List of periods for residual calculation

        Returns:
            DataFrame with residual features
        """
        result_df = df.copy()

        for period in periods:
            resi_col = f'RESI{period}'

            # Calculate residuals: close - MA(close, period)
            ma = df['close'].rolling(window=period, min_periods=1).mean()
            result_df[resi_col] = df['close'] - ma

        logger.info(f"Calculated residuals for periods: {periods}")
        return result_df

    def calculate_weighted_moving_averages(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate weighted moving averages (WVMA) using pandas operations matching qlib WMA.

        Formula: Weighted moving average with linear decay weights

        Args:
            df: DataFrame with OHLCV data
            periods: List of periods for WVMA calculation

        Returns:
            DataFrame with WVMA features
        """
        result_df = df.copy()

        for period in periods:
            wvma_col = f'WVMA{period}'

            # Calculate weighted moving average with linear decay weights
            weights = np.arange(1, period + 1)
            result_df[wvma_col] = df['close'].rolling(window=period, min_periods=1).apply(
                lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=False
            )

        logger.info(f"Calculated weighted moving averages for periods: {periods}")
        return result_df

    def calculate_r_squared(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate R-squared values (RSQR) using pandas operations matching qlib Rsquare.

        Formula: R-squared of linear regression over the period

        Args:
            df: DataFrame with OHLCV data
            periods: List of periods for R-squared calculation

        Returns:
            DataFrame with R-squared features
        """
        result_df = df.copy()

        for period in periods:
            rsqr_col = f'RSQR{period}'

            # Calculate R-squared using rolling linear regression
            def rolling_r_squared(series):
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                y = series.values
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

            result_df[rsqr_col] = df['close'].rolling(window=period, min_periods=2).apply(rolling_r_squared, raw=False)

        logger.info(f"Calculated R-squared for periods: {periods}")
        return result_df

    def calculate_correlations(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate correlations (CORR) between close price and volume using pandas operations matching qlib Corr.

        Formula: Pearson correlation between close price and volume

        Args:
            df: DataFrame with OHLCV data
            periods: List of periods for correlation calculation

        Returns:
            DataFrame with correlation features
        """
        result_df = df.copy()

        for period in periods:
            corr_col = f'CORR{period}'

            # Calculate rolling correlation between close and volume
            result_df[corr_col] = df['close'].rolling(window=period, min_periods=2).corr(df['volume'])

        logger.info(f"Calculated correlations for periods: {periods}")
        return result_df

    def calculate_correlation_differences(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate correlation differences (CORD) using pandas operations matching qlib formulas.

        Formula: CORD_period = CORR_period - CORR_prev_period

        Args:
            df: DataFrame with OHLCV data
            periods: List of periods for correlation difference calculation

        Returns:
            DataFrame with correlation difference features
        """
        result_df = df.copy()

        # Define mapping of periods to their previous periods
        period_mapping = {5: None, 10: 5, 20: 10, 60: 20}

        for period in periods:
            cord_col = f'CORD{period}'

            # Get the previous period for this CORD calculation
            prev_period = period_mapping.get(period)

            if prev_period is not None:
                curr_corr_col = f'CORR{period}'
                prev_corr_col = f'CORR{prev_period}'

                if curr_corr_col in result_df.columns and prev_corr_col in result_df.columns:
                    # Calculate CORD as difference between correlations
                    result_df[cord_col] = result_df[curr_corr_col] - result_df[prev_corr_col]
                else:
                    logger.warning(f"Cannot calculate {cord_col}: missing correlation columns {curr_corr_col} or {prev_corr_col}")
                    result_df[cord_col] = 0.0
            else:
                # For the first period (5), CORD5 = CORR5 - 0 (or use CORR5 itself as baseline)
                curr_corr_col = f'CORR{period}'
                if curr_corr_col in result_df.columns:
                    result_df[cord_col] = result_df[curr_corr_col] - 0.0
                else:
                    result_df[cord_col] = 0.0

        logger.info(f"Calculated correlation differences for periods: {periods}")
        return result_df

    def calculate_rate_of_change(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate rate of change (ROC) using pandas operations matching qlib Ref formula.

        Formula: ROC_period = (close / close_period_ago - 1) * 100

        Args:
            df: DataFrame with OHLCV data
            periods: List of periods for ROC calculation

        Returns:
            DataFrame with ROC features
        """
        result_df = df.copy()

        for period in periods:
            roc_col = f'ROC{period}'

            # Calculate ROC: (close / close_period_ago - 1) * 100
            result_df[roc_col] = (df['close'] / df['close'].shift(period) - 1) * 100

        logger.info(f"Calculated rate of change for periods: {periods}")
        return result_df

    def calculate_volatility_std(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate volatility and standard deviation features using pandas operations matching qlib Std.

        Formula:
        - STD_period: Standard deviation of close prices
        - VSTD_period: Standard deviation of returns (volatility)

        Args:
            df: DataFrame with OHLCV data
            periods: List of periods for volatility calculation

        Returns:
            DataFrame with volatility and standard deviation features
        """
        result_df = df.copy()

        for period in periods:
            vstd_col = f'VSTD{period}'
            std_col = f'STD{period}'

            # Calculate standard deviation of close prices
            result_df[std_col] = df['close'].rolling(window=period, min_periods=2).std()

            # Calculate volatility (standard deviation of returns)
            returns = df['close'].pct_change()
            result_df[vstd_col] = returns.rolling(window=period, min_periods=2).std()

        logger.info(f"Calculated volatility and standard deviation for periods: {periods}")
        return result_df

    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators (KLEN, KLOW).

        KLEN: Length of consecutive up/down trend
        KLOW: Lowest price in 20-day rolling window

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with momentum indicator features
        """
        result_df = df.copy()

        # KLEN: Length of consecutive up/down movements
        # Calculate price direction changes (1 for up, 0 for down, handling ties)
        price_changes = df['close'].diff()
        direction = (price_changes > 0).astype(int)

        # Handle zero changes (ties) by maintaining previous direction
        direction = direction.replace(0, np.nan).ffill().fillna(0).astype(int)

        # Calculate run lengths using vectorized operations
        # Find where direction changes
        direction_changes = direction.diff().fillna(0).abs()
        change_indices = direction_changes[direction_changes == 1].index

        # Initialize KLEN array
        klen_values = np.zeros(len(df))

        # Calculate run lengths for each segment
        start_idx = 0
        for change_idx in change_indices:
            end_idx = df.index.get_loc(change_idx)
            run_length = end_idx - start_idx + 1
            klen_values[start_idx:end_idx + 1] = np.arange(run_length, 0, -1)
            start_idx = end_idx + 1

        # Handle the last segment
        if start_idx < len(df):
            run_length = len(df) - start_idx
            klen_values[start_idx:] = np.arange(run_length, 0, -1)

        result_df['KLEN'] = klen_values

        # KLOW: Lowest price in recent period (using 20-day lookback)
        # Use rolling min for efficiency
        result_df['KLOW'] = df['low'].rolling(window=20, min_periods=1).min()

        logger.info("Calculated momentum indicators (KLEN, KLOW)")
        return result_df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features (KNOWN_INPUT for TFT).

        Args:
            df: DataFrame with time column

        Returns:
            DataFrame with temporal features
        """
        result_df = df.copy()

        # Ensure time column is datetime
        if 'time' in result_df.columns:
            result_df['time'] = pd.to_datetime(result_df['time'])
            result_df['date'] = result_df['time']
        elif 'date' in result_df.columns:
            result_df['date'] = pd.to_datetime(result_df['date'])
        else:
            raise ValueError("No time or date column found")

        # Add temporal features
        result_df['day_of_week'] = result_df['date'].dt.dayofweek
        result_df['month'] = result_df['date'].dt.month
        result_df['year'] = result_df['date'].dt.year

        logger.info("Added temporal features (day_of_week, month, year)")
        return result_df

    def add_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add static features (STATIC_INPUT for TFT).

        Args:
            df: DataFrame to add static features to

        Returns:
            DataFrame with static features
        """
        result_df = df.copy()

        # Constant feature (required by TFT)
        result_df['const'] = 1.0

        logger.info("Added static features (const)")
        return result_df

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all TFT-compatible features for VN30 data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting comprehensive feature engineering for VN30...")

        result_df = df.copy()

        # 1. Add temporal features (KNOWN_INPUT)
        result_df = self.add_temporal_features(result_df)

        # 2. Calculate residuals (RESI)
        periods_resi = [5, 10]
        result_df = self.calculate_residuals(result_df, periods_resi)

        # 3. Calculate weighted moving averages (WVMA)
        periods_wvma = [5, 60]
        result_df = self.calculate_weighted_moving_averages(result_df, periods_wvma)

        # 4. Calculate R-squared (RSQR)
        periods_rsqr = [5, 10, 20, 60]
        result_df = self.calculate_r_squared(result_df, periods_rsqr)

        # 5. Calculate correlations (CORR)
        periods_corr = [5, 10, 20, 60]
        result_df = self.calculate_correlations(result_df, periods_corr)

        # 6. Calculate correlation differences (CORD)
        periods_cord = [5, 10, 60]
        result_df = self.calculate_correlation_differences(result_df, periods_cord)

        # 7. Calculate rate of change (ROC)
        periods_roc = [60]
        result_df = self.calculate_rate_of_change(result_df, periods_roc)

        # 8. Calculate volatility and standard deviation (VSTD, STD)
        periods_vol = [5]
        result_df = self.calculate_volatility_std(result_df, periods_vol)

        # 9. Calculate momentum indicators (KLEN, KLOW)
        result_df = self.calculate_momentum_indicators(result_df)

        # 10. Add static features (STATIC_INPUT)
        result_df = self.add_static_features(result_df)

        logger.info(f"Feature engineering completed. Final shape: {result_df.shape}")
        logger.info(f"Features created: {len(result_df.columns)} columns")

        return result_df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean engineered features by handling infinite and NaN values.

        Args:
            df: DataFrame with engineered features

        Returns:
            DataFrame with cleaned features
        """
        result_df = df.copy()

        # Replace infinite values with NaN
        result_df = result_df.replace([np.inf, -np.inf], np.nan)

        # Forward fill NaN values for up to 5 consecutive missing days
        result_df = result_df.ffill(limit=5)

        # Drop any remaining rows with NaN values
        initial_rows = len(result_df)
        result_df = result_df.dropna()
        dropped_rows = initial_rows - len(result_df)

        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows due to NaN values after cleaning")

        return result_df

    def validate_features(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate engineered features for TFT compatibility.

        Args:
            df: DataFrame with engineered features

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        # Check required columns for each category
        for category, features in self.feature_config.items():
            missing_features = set(features) - set(df.columns)
            validation_results[f"{category}_complete"] = len(missing_features) == 0

            if missing_features:
                logger.warning(f"Missing {category} features: {missing_features}")

        # Check for sufficient data points
        validation_results['sufficient_data'] = len(df) >= 100

        # Check for reasonable feature value ranges
        numeric_features = [col for col in df.columns if col not in ['time', 'date', 'const']]
        feature_ranges_valid = True

        for feature in numeric_features:
            if df[feature].dtype in [np.float64, np.int64]:
                feature_range = df[feature].max() - df[feature].min()
                if not (0.001 <= feature_range <= 1e6):  # Reasonable range check
                    feature_ranges_valid = False
                    logger.warning(f"Feature {feature} has suspicious range: {feature_range}")
                    break

        validation_results['reasonable_ranges'] = feature_ranges_valid

        # Log validation results
        passed_checks = sum(validation_results.values())
        total_checks = len(validation_results)

        logger.info(f"Feature validation: {passed_checks}/{total_checks} checks passed")

        return validation_results

    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for all engineered features.

        Args:
            df: DataFrame with engineered features

        Returns:
            DataFrame with feature summary statistics
        """
        # Select only numeric features for summary
        numeric_features = [col for col in df.columns if col not in ['time', 'date']]

        summary_data = []
        for feature in numeric_features:
            if df[feature].dtype in [np.float64, np.int64]:
                summary_data.append({
                    'feature': feature,
                    'category': self._categorize_feature(feature),
                    'count': df[feature].count(),
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'missing_ratio': df[feature].isnull().sum() / len(df)
                })

        summary_df = pd.DataFrame(summary_data)
        return summary_df

    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize feature based on name."""
        for category, features in self.feature_config.items():
            if feature_name in features:
                return category
        return 'UNKNOWN'


def main():
    """Main function for testing feature engineering."""
    logger.info("Testing VN30TFTFeatureEngineer...")

    # Create sample data for testing
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    sample_data = {
        'time': dates,
        'open': np.random.uniform(100, 200, len(dates)),
        'high': np.random.uniform(150, 250, len(dates)),
        'low': np.random.uniform(50, 150, len(dates)),
        'close': np.random.uniform(100, 200, len(dates)),
        'volume': np.random.uniform(100000, 1000000, len(dates))
    }

    df = pd.DataFrame(sample_data)

    # Initialize feature engineer
    engineer = VN30TFTFeatureEngineer()

    # Engineer features
    engineered_df = engineer.engineer_all_features(df)

    # Validate features
    validation_results = engineer.validate_features(engineered_df)

    # Get feature summary
    summary_df = engineer.get_feature_summary(engineered_df)

    print("Feature engineering test completed!")
    print(f"Original shape: {df.shape}")
    print(f"Engineered shape: {engineered_df.shape}")
    print(f"Validation results: {validation_results}")
    print(f"Feature summary shape: {summary_df.shape}")

    return engineered_df, validation_results, summary_df


if __name__ == "__main__":
    main()

"""
Comprehensive VN30 TFT Features Visualizer

This module combines all individual visualizers into a single, well-structured
HTML report containing visualizations and explanations for all VN30 TFT features.

Author: StocketAI
Created: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

# Note: Individual visualizers are integrated into this comprehensive visualizer
# No external imports needed for the core functionality

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveVisualizer:
    """
    Master visualizer that combines all individual feature visualizers
    into a single comprehensive HTML report.
    """

    def __init__(self):
        """Initialize comprehensive visualizer."""
        self.title = "VN30 TFT Features - Comprehensive Analysis Report"
        # Define feature configurations for each category
        self.feature_configs = {
            'RESI': {
                'title': 'RESI (Residuals) Analysis',
                'features': ['RESI5', 'RESI10'],
                'explanation': """# RESI (Residuals) Analysis

Residuals represent the difference between actual price and moving average values.

## Key Features:
- **RESI5**: 5-day residual (close - MA5)
- **RESI10**: 10-day residual (close - MA10)

## Interpretation:
- **Positive values**: Price above moving average (bullish)
- **Negative values**: Price below moving average (bearish)
- **Zero crossings**: Potential trend changes
- **Magnitude**: Strength of deviation from trend

## Trading Applications:
- Mean reversion signals
- Trend strength confirmation
- Support/resistance level identification"""
            },
            'WVMA': {
                'title': 'WVMA (Weighted Moving Average) Analysis',
                'features': ['WVMA5', 'WVMA60'],
                'explanation': """# WVMA (Weighted Moving Average) Analysis

Weighted moving averages give more importance to recent prices using linear decay weights.

## Key Features:
- **WVMA5**: Fast 5-day weighted moving average
- **WVMA60**: Slow 60-day weighted moving average

## Interpretation:
- **Crossovers**: Bullish/bearish signals
- **Slope**: Trend direction and strength
- **Separation**: Trend momentum

## Trading Applications:
- Trend following strategies
- Dynamic support/resistance levels
- Momentum confirmation"""
            },
            'RSQR': {
                'title': 'RSQR (R-squared) Analysis',
                'features': ['RSQR5', 'RSQR10', 'RSQR20', 'RSQR60'],
                'explanation': """# RSQR (R-squared) Analysis

R-squared measures the strength of linear trend over different periods.

## Key Features:
- **RSQR5**: 5-day trend strength
- **RSQR10**: 10-day trend strength
- **RSQR20**: 20-day trend strength
- **RSQR60**: 60-day trend strength

## Interpretation:
- **>0.7**: Strong trend
- **0.3-0.7**: Moderate trend
- **<0.3**: Weak/no trend
- **Declining values**: Trend weakening

## Trading Applications:
- Trend confirmation
- Position sizing based on trend strength
- Mean reversion vs trend following"""
            },
            'CORR': {
                'title': 'CORR (Price-Volume Correlation) Analysis',
                'features': ['CORR5', 'CORR10', 'CORR20', 'CORR60'],
                'explanation': """# CORR (Price-Volume Correlation) Analysis

Correlation between price movements and volume provides insight into market conviction.

## Key Features:
- **CORR5**: Short-term price-volume correlation
- **CORR10**: Medium-term price-volume correlation
- **CORR20**: Long-term price-volume correlation
- **CORR60**: Very long-term price-volume correlation

## Interpretation:
- **Positive correlation**: Volume supports price direction
- **Negative correlation**: Volume contradicts price movement
- **High correlation**: Strong market conviction
- **Low correlation**: Weak market participation

## Trading Applications:
- Signal confirmation
- Volume analysis
- Market sentiment assessment"""
            },
            'ROC': {
                'title': 'ROC (Rate of Change) Analysis',
                'features': ['ROC60'],
                'explanation': """# ROC (Rate of Change) Analysis

Rate of change measures price momentum over a specified period.

## Key Features:
- **ROC60**: 60-day price momentum

## Interpretation:
- **Positive ROC**: Upward momentum
- **Negative ROC**: Downward momentum
- **Extreme values**: Overbought (>30%) or oversold (<-30%)
- **Zero crossings**: Momentum shifts

## Trading Applications:
- Momentum trading strategies
- Overbought/oversold identification
- Trend acceleration/deceleration"""
            },
            'VOLATILITY': {
                'title': 'Volatility Analysis (VSTD & STD)',
                'features': ['VSTD5', 'STD5'],
                'explanation': """# Volatility Analysis

Volatility measures price variability and risk.

## Key Features:
- **VSTD5**: 5-day return volatility (standard deviation of returns)
- **STD5**: 5-day price volatility (standard deviation of prices)

## Interpretation:
- **High volatility**: Increased uncertainty and risk
- **Low volatility**: Stable price movement
- **Volatility spikes**: Market events or news impact
- **Volatility cycles**: Market regime changes

## Trading Applications:
- Risk management
- Position sizing
- Option pricing
- Market regime identification"""
            },
            'CORD': {
                'title': 'CORD (Correlation Differences) Analysis',
                'features': ['CORD5', 'CORD10', 'CORD60'],
                'explanation': """# CORD (Correlation Differences) Analysis

Correlation differences measure changes in price-volume relationships over time.

## Key Features:
- **CORD5**: Short-term correlation change
- **CORD10**: Medium-term correlation change
- **CORD60**: Long-term correlation change

## Interpretation:
- **Positive values**: Improving price-volume agreement
- **Negative values**: Deteriorating price-volume agreement
- **Large changes**: Significant shifts in market behavior

## Trading Applications:
- Market regime detection
- Signal quality assessment
- Volume confirmation analysis"""
            },
            'MOMENTUM': {
                'title': 'Momentum Indicators (KLEN & KLOW) Analysis',
                'features': ['KLEN', 'KLOW'],
                'explanation': """# Momentum Indicators Analysis

Momentum indicators measure trend persistence and support levels.

## Key Features:
- **KLEN**: Length of consecutive up/down trend periods
- **KLOW**: Lowest price in 20-day rolling window

## Interpretation:
- **High KLEN**: Strong, persistent trend
- **Low KLEN**: Choppy, directionless market
- **KLOW proximity**: Near-term support levels
- **KLEN > 10**: Strong trending conditions

## Trading Applications:
- Trend strength assessment
- Support/resistance identification
- Breakout confirmation
- Position management"""
            },
            'TEMPORAL': {
                'title': 'Temporal Features Analysis',
                'features': ['month', 'day_of_week', 'year'],
                'explanation': """# Temporal Features Analysis

Temporal features capture time-based patterns in market behavior.

## Key Features:
- **month**: Monthly seasonal patterns
- **day_of_week**: Weekly cyclical patterns
- **year**: Long-term yearly trends

## Interpretation:
- **Monthly patterns**: End-of-month effects, earnings seasons
- **Weekly patterns**: Monday/Tuesday effects, weekend gaps
- **Yearly trends**: Long-term growth or cyclical patterns

## Trading Applications:
- Seasonal trading strategies
- Calendar-based anomalies
- Risk management timing
- Portfolio rebalancing"""
            },
            'STATIC': {
                'title': 'Static Features Analysis',
                'features': ['const'],
                'explanation': """# Static Features Analysis

Static features provide constant reference points for model training.

## Key Features:
- **const**: Constant value (1.0) for all observations

## Interpretation:
- **Baseline**: Reference point for feature scaling
- **Numerical stability**: Prevents division by zero
- **Model architecture**: Required TFT input structure

## Trading Applications:
- Feature normalization baseline
- Model input standardization
- Architecture compliance"""
            }
        }

    def generate_comprehensive_report(self, data: pd.DataFrame, output_path: str = "visualizers", symbol_name: str = "VN30") -> str:
        """
        Generate a comprehensive HTML report with all feature visualizations.

        Args:
            data: DataFrame with TFT-engineered features (required)
            output_path: Directory to save the HTML report

        Returns:
            Path to the generated HTML file
        """
        try:
            logger.info("Starting comprehensive visualization report generation...")

            # Validate input data
            if data is None or data.empty:
                logger.error("No data provided for visualization")
                return None

            # Create output directory
            Path(output_path).mkdir(parents=True, exist_ok=True)

            # Generate all visualizations
            feature_sections = []

            for name, config in self.feature_configs.items():
                logger.info(f"Generating visualization for {name}...")

                try:
                    # Create the plot figure directly using provided data
                    fig = self._create_plot_for_visualizer(name, None, data)

                    # Convert plot to base64
                    plot_base64 = self._figure_to_base64(fig)

                    # Get explanation from config
                    explanation = config['explanation']

                    # Create section HTML
                    section_html = self._create_feature_section(
                        name, config['title'], plot_base64, explanation, config['features']
                    )

                    feature_sections.append(section_html)
                    plt.close(fig)

                except Exception as e:
                    logger.error(f"Error generating {name} visualization: {e}")
                    # Create error section
                    error_section = self._create_error_section(name, str(e))
                    feature_sections.append(error_section)

            # Generate comprehensive HTML
            html_content = self._generate_main_html(feature_sections)

            # Save HTML file with symbol-specific name
            output_file = Path(output_path) / f"{symbol_name}_features_report.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Comprehensive report generated: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return None



    def _create_plot_for_visualizer(self, name: str, visualizer, data: pd.DataFrame) -> plt.Figure:
        """Create plot figure for a specific visualizer using pre-engineered features."""
        try:
            # Validate that required features exist in the data
            if not self._validate_features_for_visualization(name, data):
                return self._create_missing_features_plot(name)

            # Route to appropriate visualization method
            if name == 'RESI':
                return self._create_resi_plot(data)
            elif name == 'WVMA':
                return self._create_wvma_plot(data)
            elif name == 'RSQR':
                return self._create_rsqr_plot(data)
            elif name == 'CORR':
                return self._create_corr_plot(data)
            elif name == 'ROC':
                return self._create_roc_plot(data)
            elif name == 'VOLATILITY':
                return self._create_volatility_plot(data)
            elif name == 'CORD':
                return self._create_cord_plot(data)
            elif name == 'MOMENTUM':
                return self._create_momentum_plot(data)
            elif name == 'TEMPORAL':
                return self._create_temporal_plot(data)
            elif name == 'STATIC':
                return self._create_static_plot(data)
            else:
                return self._create_unsupported_visualization_plot(name)

        except Exception as e:
            logger.error(f"Error creating plot for {name}: {e}")
            return self._create_error_plot(name, str(e))

    def _validate_features_for_visualization(self, name: str, data: pd.DataFrame) -> bool:
        """Validate that required features exist for the visualization."""
        if data is None or data.empty:
            logger.error(f"No data provided for {name} visualization")
            return False

        # Define required features for each visualization type
        required_features_map = {
            'RESI': ['RESI5', 'RESI10', 'close'],
            'WVMA': ['WVMA5', 'WVMA60', 'close'],
            'RSQR': ['RSQR5', 'RSQR10', 'RSQR20', 'RSQR60', 'close'],
            'CORR': ['CORR5', 'CORR10', 'CORR20', 'CORR60', 'close', 'volume'],
            'ROC': ['ROC60', 'close'],
            'VOLATILITY': ['VSTD5', 'STD5', 'close'],
            'CORD': ['CORD5', 'CORD10', 'CORD60', 'close', 'volume'],
            'MOMENTUM': ['KLEN', 'KLOW', 'close'],
            'TEMPORAL': ['month', 'day_of_week', 'year', 'close'],
            'STATIC': ['const']
        }

        required_features = required_features_map.get(name, [])
        missing_features = set(required_features) - set(data.columns)

        if missing_features:
            logger.error(f"Missing required features for {name} visualization: {missing_features}")
            return False

        return True

    def _create_missing_features_plot(self, name: str) -> plt.Figure:
        """Create a plot indicating missing required features."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5,
               f'Missing required features\nfor {name} visualization\n\nPlease ensure features are properly engineered.',
               ha='center', va='center', fontsize=12, color='orange')
        ax.set_title(f'{name} - Missing Features')
        return fig

    def _create_unsupported_visualization_plot(self, name: str) -> plt.Figure:
        """Create a plot for unsupported visualization types."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'{name}\nVisualization\nNot Supported',
               ha='center', va='center', fontsize=14)
        ax.set_title(f'{name} - Unsupported Visualization')
        return fig

    def _create_error_plot(self, name: str, error_message: str) -> plt.Figure:
        """Create a plot indicating an error occurred."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error creating\n{name} plot\n\n{error_message}',
               ha='center', va='center', fontsize=12, color='red')
        ax.set_title(f'{name} - Plot Error')
        return fig

    def _create_resi_plot(self, data: pd.DataFrame) -> plt.Figure:
        """Create RESI visualization."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('RESI (Residuals) Analysis', fontsize=16, fontweight='bold')

        # Calculate RESI features
        data['RESI5'] = data['close'] - data['close'].rolling(window=5).mean()
        data['RESI10'] = data['close'] - data['close'].rolling(window=10).mean()

        # 1. Price and Moving Averages
        axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8)
        ma5 = data['close'].rolling(window=5).mean()
        ma10 = data['close'].rolling(window=10).mean()
        axes[0].plot(ma5, label='5-day MA', linestyle='--', linewidth=1.5)
        axes[0].plot(ma10, label='10-day MA', linestyle='--', linewidth=1.5)
        axes[0].set_title('Price vs Moving Averages', fontsize=12)
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. RESI Values
        colors = ['#2E86AB', '#A23B72']
        axes[1].plot(data['RESI5'], label='RESI5', color=colors[0], linewidth=2)
        axes[1].plot(data['RESI10'], label='RESI10', color=colors[1], linewidth=2)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        axes[1].set_title('RESI Values (Residuals)', fontsize=12)
        axes[1].set_ylabel('Residual Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. RESI Distribution
        resi_combined = pd.concat([data['RESI5'], data['RESI10']], axis=1)
        sns.histplot(data=resi_combined, ax=axes[2], alpha=0.7)
        axes[2].axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        axes[2].set_title('RESI Distribution Analysis', fontsize=12)
        axes[2].set_xlabel('Residual Value')
        axes[2].set_ylabel('Frequency')

        plt.tight_layout()
        return fig

    def _create_wvma_plot(self, data: pd.DataFrame) -> plt.Figure:
        """Create WVMA visualization."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('WVMA (Weighted Moving Average) Analysis', fontsize=16, fontweight='bold')

        # Calculate WVMA features
        def calculate_wvma(series, period):
            weights = np.arange(1, period + 1)
            return series.rolling(window=period, min_periods=1).apply(
                lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=False
            )

        data['WVMA5'] = calculate_wvma(data['close'], 5)
        data['WVMA60'] = calculate_wvma(data['close'], 60)

        # 1. Price and WVMA Comparison
        axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8)
        axes[0].plot(data['WVMA5'], label='WVMA5', linewidth=2, color='#2E86AB')
        axes[0].plot(data['WVMA60'], label='WVMA60', linewidth=2, color='#A23B72')
        axes[0].set_title('Price vs Weighted Moving Averages', fontsize=12)
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. WVMA Difference
        wvma_diff = data['WVMA5'] - data['WVMA60']
        axes[1].plot(wvma_diff, label='WVMA5 - WVMA60', linewidth=2, color='#F18F01')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        axes[1].set_title('WVMA Momentum (WVMA5 - WVMA60)', fontsize=12)
        axes[1].set_ylabel('Price Difference')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. WVMA Distribution
        wvma_data = pd.DataFrame({
            'WVMA5': data['WVMA5'].fillna(0),
            'WVMA60': data['WVMA60'].fillna(0)
        })
        sns.histplot(data=wvma_data, ax=axes[2], alpha=0.7)
        axes[2].set_title('WVMA Distribution Analysis', fontsize=12)
        axes[2].set_xlabel('WVMA Value')
        axes[2].set_ylabel('Frequency')

        plt.tight_layout()
        return fig

    def _create_rsqr_plot(self, data: pd.DataFrame) -> plt.Figure:
        """Create RSQR visualization."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('RSQR (R-squared) Analysis', fontsize=16, fontweight='bold')

        # Calculate RSQR features
        def calculate_r_squared(series, period):
            def rolling_r_squared(x):
                if len(x) < 3:
                    return np.nan
                x_vals = np.arange(len(x))
                y_vals = x.values
                try:
                    coeffs = np.polyfit(x_vals, y_vals, 1)
                    y_pred = coeffs[0] * x_vals + coeffs[1]
                    ss_res = np.sum((y_vals - y_pred) ** 2)
                    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
                except:
                    return np.nan
            return series.rolling(window=period, min_periods=3).apply(rolling_r_squared, raw=False)

        for period in [5, 10, 20, 60]:
            data[f'RSQR{period}'] = calculate_r_squared(data['close'], period)

        # 1. Price Trend Analysis
        axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8)
        axes[0].set_title('Price with Trend Context', fontsize=12)
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. RSQR Values Over Time
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        rsqr_periods = [5, 10, 20, 60]

        for i, period in enumerate(rsqr_periods):
            if f'RSQR{period}' in data.columns:
                axes[1].plot(data[f'RSQR{period}'],
                           label=f'RSQR{period}',
                           color=colors[i % len(colors)],
                           linewidth=2, alpha=0.8)

        axes[1].axhline(y=0.7, color='green', linestyle='-', alpha=0.8, linewidth=1, label='Strong Trend (0.7)')
        axes[1].axhline(y=0.3, color='red', linestyle='-', alpha=0.8, linewidth=1, label='Weak Trend (0.3)')
        axes[1].set_ylim(0, 1)
        axes[1].set_title('R-squared Values Over Time', fontsize=12)
        axes[1].set_ylabel('R-squared Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. RSQR Distribution
        rsqr_data = []
        for period in rsqr_periods:
            if f'RSQR{period}' in data.columns:
                valid_rsqr = data[f'RSQR{period}'].dropna()
                if len(valid_rsqr) > 0:
                    rsqr_data.append(pd.Series(valid_rsqr, name=f'RSQR{period}'))

        if rsqr_data:
            rsqr_df = pd.concat(rsqr_data, axis=1)
            bp = axes[2].boxplot([rsqr_df[col].dropna() for col in rsqr_df.columns],
                               labels=rsqr_df.columns, patch_artist=True)

            for i, box in enumerate(bp['boxes']):
                box.set(facecolor=colors[i % len(colors)])

            axes[2].axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Strong Trend')
            axes[2].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Weak Trend')
            axes[2].set_title('R-squared Distribution by Period', fontsize=12)
            axes[2].set_ylabel('R-squared Value')
            axes[2].legend()

        plt.tight_layout()
        return fig

    def _create_corr_plot(self, data: pd.DataFrame) -> plt.Figure:
        """Create CORR visualization."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('CORR (Price-Volume Correlation) Analysis', fontsize=16, fontweight='bold')

        # Calculate CORR features
        def calculate_correlation(price_series, volume_series, period):
            def rolling_corr(price_window, volume_window):
                if len(price_window) < 3 or len(volume_window) < 3:
                    return np.nan
                try:
                    return price_window.corr(volume_window)
                except:
                    return np.nan

            correlations = []
            for i in range(len(price_series)):
                if i < period - 1:
                    correlations.append(np.nan)
                else:
                    price_window = price_series.iloc[i-period+1:i+1]
                    volume_window = volume_series.iloc[i-period+1:i+1]
                    correlations.append(rolling_corr(price_window, volume_window))

            return pd.Series(correlations, index=price_series.index)

        for period in [5, 10, 20, 60]:
            data[f'CORR{period}'] = calculate_correlation(data['close'], data['volume'], period)

        # 1. Price and Volume
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        ax1.plot(data['close'], label='Close Price', linewidth=2, color='#2E86AB', alpha=0.8)
        ax1_twin.bar(data.index, data['volume'], label='Volume', alpha=0.3, color='#A23B72', width=0.8)
        ax1.set_title('Price and Volume Relationship', fontsize=12)
        ax1.set_ylabel('Price', color='#2E86AB')
        ax1.tick_params(axis='y', labelcolor='#2E86AB')
        ax1_twin.set_ylabel('Volume', color='#A23B72')
        ax1_twin.tick_params(axis='y', labelcolor='#A23B72')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. CORR Values Over Time
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        corr_periods = [5, 10, 20, 60]

        for i, period in enumerate(corr_periods):
            if f'CORR{period}' in data.columns:
                axes[1].plot(data[f'CORR{period}'],
                           label=f'CORR{period}',
                           color=colors[i % len(colors)],
                           linewidth=2, alpha=0.8)

        axes[1].axhline(y=0.7, color='green', linestyle='-', alpha=0.8, linewidth=1, label='Strong Positive (0.7)')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1, label='No Correlation (0)')
        axes[1].axhline(y=-0.3, color='red', linestyle='-', alpha=0.8, linewidth=1, label='Negative (-0.3)')
        axes[1].set_ylim(-1, 1)
        axes[1].set_title('Price-Volume Correlation Over Time', fontsize=12)
        axes[1].set_ylabel('Correlation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. CORR Distribution
        corr_data = []
        for period in corr_periods:
            if f'CORR{period}' in data.columns:
                valid_corr = data[f'CORR{period}'].dropna()
                if len(valid_corr) > 0:
                    corr_data.append(pd.Series(valid_corr, name=f'CORR{period}'))

        if corr_data:
            corr_df = pd.concat(corr_data, axis=1)
            for i, col in enumerate(corr_df.columns):
                valid_data = corr_df[col].dropna()
                if len(valid_data) > 0:
                    axes[2].hist(valid_data, bins=20, alpha=0.7,
                               label=col, color=colors[i % len(colors)])

            axes[2].axvline(x=0.7, color='green', linestyle='--', alpha=0.7, label='Strong Positive')
            axes[2].axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            axes[2].axvline(x=-0.3, color='red', linestyle='--', alpha=0.7, label='Negative')
            axes[2].set_xlim(-1, 1)
            axes[2].set_title('Correlation Distribution by Period', fontsize=12)
            axes[2].set_xlabel('Correlation Value')
            axes[2].set_ylabel('Frequency')
            axes[2].legend()

        plt.tight_layout()
        return fig

    def _create_roc_plot(self, data: pd.DataFrame) -> plt.Figure:
        """Create ROC visualization."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('ROC (Rate of Change) Analysis', fontsize=16, fontweight='bold')

        # Calculate ROC60
        data['ROC60'] = (data['close'] / data['close'].shift(60) - 1) * 100

        # 1. Price and ROC Comparison
        axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8, color='#2E86AB')
        ax_twin = axes[0].twinx()
        ax_twin.plot(data['ROC60'], label='ROC60', linewidth=2, alpha=0.8, color='#A23B72')
        ax_twin.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        axes[0].set_title('Price vs Rate of Change (ROC60)', fontsize=12)
        axes[0].set_ylabel('Price', color='#2E86AB')
        axes[0].tick_params(axis='y', labelcolor='#2E86AB')
        ax_twin.set_ylabel('ROC60 (%)', color='#A23B72')
        ax_twin.tick_params(axis='y', labelcolor='#A23B72')
        lines1, labels1 = axes[0].get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axes[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        axes[0].grid(True, alpha=0.3)

        # 2. ROC with Signal Levels
        axes[1].plot(data['ROC60'], label='ROC60', linewidth=2, color='#F18F01', alpha=0.8)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Overbought (30%)')
        axes[1].axhline(y=-30, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Oversold (-30%)')
        axes[1].axhline(y=10, color='darkgreen', linestyle=':', alpha=0.7, linewidth=1, label='Strong (10%)')
        axes[1].axhline(y=-10, color='darkred', linestyle=':', alpha=0.7, linewidth=1, label='Weak (-10%)')
        axes[1].set_ylim(-60, 60)
        axes[1].set_title('ROC with Signal Levels', fontsize=12)
        axes[1].set_ylabel('ROC60 (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. ROC Distribution
        roc_data = data['ROC60'].dropna()
        if len(roc_data) > 0:
            axes[2].hist(roc_data, bins=30, alpha=0.7, color='#C73E1D', density=True)
            try:
                sns.kdeplot(data=roc_data.dropna(), ax=axes[2], color='black', linewidth=2)
            except:
                pass
            axes[2].axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
            axes[2].axvline(x=30, color='green', linestyle='--', alpha=0.7, linewidth=1)
            axes[2].axvline(x=-30, color='red', linestyle='--', alpha=0.7, linewidth=1)
            axes[2].set_xlim(-60, 60)
            axes[2].set_title('ROC Distribution Analysis', fontsize=12)
            axes[2].set_xlabel('ROC60 (%)')
            axes[2].set_ylabel('Density')

        plt.tight_layout()
        return fig

    def _create_volatility_plot(self, data: pd.DataFrame) -> plt.Figure:
        """Create volatility visualization."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('Volatility Analysis (VSTD & STD)', fontsize=16, fontweight='bold')

        # Calculate volatility measures
        returns = data['close'].pct_change()
        data['VSTD5'] = returns.rolling(window=5, min_periods=3).std()
        data['STD5'] = data['close'].rolling(window=5, min_periods=3).std()

        # 1. Price with Volatility Overlay
        axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8, color='#2E86AB')
        ax_twin = axes[0].twinx()
        vstd5_pct = data['VSTD5'] * 100  # Convert to percentage
        ax_twin.plot(vstd5_pct, label='VSTD5 (%)', linewidth=2, alpha=0.8, color='#A23B72')
        ax_twin.fill_between(data.index, vstd5_pct, alpha=0.3, color='#A23B72')
        axes[0].set_title('Price with Return Volatility (VSTD5)', fontsize=12)
        axes[0].set_ylabel('Price', color='#2E86AB')
        axes[0].tick_params(axis='y', labelcolor='#2E86AB')
        ax_twin.set_ylabel('VSTD5 (%)', color='#A23B72')
        ax_twin.tick_params(axis='y', labelcolor='#A23B72')
        lines1, labels1 = axes[0].get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axes[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        axes[0].grid(True, alpha=0.3)

        # 2. Price Volatility (STD5)
        ax2_twin = axes[1].twinx()
        ax2_twin.plot(data['STD5'], label='STD5', linewidth=2, alpha=0.8, color='#F18F01')
        ax2_twin.fill_between(data.index, data['STD5'], alpha=0.3, color='#F18F01')
        axes[1].set_title('Price Volatility (STD5)', fontsize=12)
        axes[1].set_ylabel('Price')
        ax2_twin.set_ylabel('STD5', color='#F18F01')
        ax2_twin.tick_params(axis='y', labelcolor='#F18F01')
        ax2_twin.legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # 3. Volatility Statistics
        volatility_data = pd.DataFrame({
            'VSTD5': data['VSTD5'].dropna() * 100,
            'STD5': data['STD5'].dropna()
        })

        vstd5_stats = volatility_data["VSTD5"]
        std5_stats = volatility_data["STD5"]

        stats_text = (
            'VSTD5 Statistics:\n'
            f'Mean: {vstd5_stats.mean():.3f}%\n'
            f'Std: {vstd5_stats.std():.3f}%\n'
            f'Max: {vstd5_stats.max():.3f}%\n\n'
            'STD5 Statistics:\n'
            f'Mean: {std5_stats.mean():.3f}\n'
            f'Std: {std5_stats.std():.3f}\n'
            f'Max: {std5_stats.max():.3f}'
        )

        axes[2].text(0.5, 0.5, stats_text,
                    transform=axes[2].transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                    ha='center', va='center')
        axes[2].set_title('Volatility Statistics Summary', fontsize=12)
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        plt.tight_layout()
        return fig

    def _create_cord_plot(self, data: pd.DataFrame) -> plt.Figure:
        """Create CORD visualization."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('CORD (Correlation Differences) Analysis', fontsize=16, fontweight='bold')

        # Calculate CORD features
        def calculate_correlation_differences(price_series, volume_series):
            def rolling_corr(price_window, volume_window):
                if len(price_window) < 3 or len(volume_window) < 3:
                    return np.nan
                try:
                    return price_window.corr(volume_window)
                except:
                    return np.nan

            # Calculate correlations for different periods
            corr5 = []
            corr10 = []
            corr20 = []
            corr60 = []

            for i in range(len(price_series)):
                if i >= 4:
                    price_window = price_series.iloc[i-4:i+1]
                    volume_window = volume_series.iloc[i-4:i+1]
                    corr5.append(rolling_corr(price_window, volume_window))
                else:
                    corr5.append(np.nan)

                if i >= 9:
                    price_window = price_series.iloc[i-9:i+1]
                    volume_window = volume_series.iloc[i-9:i+1]
                    corr10.append(rolling_corr(price_window, volume_window))
                else:
                    corr10.append(np.nan)

                if i >= 19:
                    price_window = price_series.iloc[i-19:i+1]
                    volume_window = volume_series.iloc[i-19:i+1]
                    corr20.append(rolling_corr(price_window, volume_window))
                else:
                    corr20.append(np.nan)

                if i >= 59:
                    price_window = price_series.iloc[i-59:i+1]
                    volume_window = volume_series.iloc[i-59:i+1]
                    corr60.append(rolling_corr(price_window, volume_window))
                else:
                    corr60.append(np.nan)

            # Calculate differences - ensure 1D arrays
            cord_data = pd.DataFrame(index=price_series.index)

            # Convert to numpy arrays and ensure 1D
            corr5_array = np.array(corr5, dtype=float).flatten()
            corr10_array = np.array(corr10, dtype=float).flatten()
            corr20_array = np.array(corr20, dtype=float).flatten()
            corr60_array = np.array(corr60, dtype=float).flatten()

            cord_data['CORD5'] = corr5_array - 0.0
            cord_data['CORD10'] = corr10_array - corr5_array
            cord_data['CORD60'] = corr60_array - corr20_array

            return cord_data

        cord_df = calculate_correlation_differences(data['close'], data['volume'])
        data = pd.concat([data, cord_df], axis=1)

        # 1. Price and Volume
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        ax1.plot(data['close'], label='Close Price', linewidth=2, color='#2E86AB', alpha=0.8)
        ax1_twin.bar(data.index, data['volume'], label='Volume', alpha=0.3, color='#A23B72', width=0.8)
        ax1.set_title('Price and Volume with Correlation Context', fontsize=12)
        ax1.set_ylabel('Price', color='#2E86AB')
        ax1.tick_params(axis='y', labelcolor='#2E86AB')
        ax1_twin.set_ylabel('Volume', color='#A23B72')
        ax1_twin.tick_params(axis='y', labelcolor='#A23B72')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. CORD Values Over Time
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        cord_periods = [5, 10, 60]

        for i, period in enumerate(cord_periods):
            if f'CORD{period}' in data.columns:
                axes[1].plot(data[f'CORD{period}'],
                           label=f'CORD{period}',
                           color=colors[i % len(colors)],
                           linewidth=2, alpha=0.8)

        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        axes[1].axhline(y=0.3, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Strong Improvement (0.3)')
        axes[1].axhline(y=-0.3, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Strong Deterioration (-0.3)')
        axes[1].set_ylim(-1, 1)
        axes[1].set_title('Correlation Differences Over Time', fontsize=12)
        axes[1].set_ylabel('CORD Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. CORD Distribution
        cord_data = []
        for period in cord_periods:
            if f'CORD{period}' in data.columns:
                valid_cord = data[f'CORD{period}'].dropna()
                if len(valid_cord) > 0:
                    cord_data.append(pd.Series(valid_cord, name=f'CORD{period}'))

        if cord_data:
            cord_df = pd.concat(cord_data, axis=1)

            # Fix: Ensure each column is properly converted to 1D array for boxplot
            box_data = []
            box_labels = []

            for col in cord_df.columns:
                col_data = cord_df[col].dropna()
                if len(col_data) > 0:
                    # Convert to numpy array and ensure it's 1D
                    # Use ravel() instead of flatten() to ensure 1D array
                    box_data.append(np.asarray(col_data).ravel())
                    box_labels.append(col)

            if box_data:
                bp = axes[2].boxplot(box_data, labels=box_labels, patch_artist=True)

                for i, box in enumerate(bp['boxes']):
                    box.set(facecolor=colors[i % len(colors)])

                axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                axes[2].axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Strong Improvement')
                axes[2].axhline(y=-0.3, color='red', linestyle='--', alpha=0.7, label='Strong Deterioration')
                axes[2].set_ylim(-1, 1)
                axes[2].set_title('CORD Distribution by Period', fontsize=12)
                axes[2].set_ylabel('CORD Value')
                axes[2].legend()

        plt.tight_layout()
        return fig

    def _create_momentum_plot(self, data: pd.DataFrame) -> plt.Figure:
        """Create momentum visualization."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('Momentum Indicators (KLEN & KLOW) Analysis', fontsize=16, fontweight='bold')

        # Calculate momentum features
        # KLEN calculation
        price_changes = data['close'].diff()
        direction = (price_changes > 0).astype(int)
        direction = direction.replace(0, np.nan).ffill().fillna(0).astype(int)

        klen_values = np.zeros(len(data))
        start_idx = 0
        current_direction = direction.iloc[0] if len(direction) > 0 else 0

        for i in range(1, len(direction)):
            if direction.iloc[i] != current_direction:
                run_length = i - start_idx
                klen_values[start_idx:i] = np.arange(run_length, 0, -1)
                start_idx = i
                current_direction = direction.iloc[i]

        if start_idx < len(data):
            run_length = len(data) - start_idx
            klen_values[start_idx:] = np.arange(run_length, 0, -1)

        data['KLEN'] = pd.Series(klen_values, index=data.index)
        data['KLOW'] = data['close'].rolling(window=20, min_periods=1).min()

        # 1. Price with KLOW Support Levels
        axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8, color='#2E86AB')
        axes[0].plot(data['KLOW'], label='KLOW (20-day low)', linewidth=2, alpha=0.8, color='#A23B72')
        axes[0].set_title('Price vs KLOW Support Level', fontsize=12)
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. KLEN Trend Persistence
        axes[1].plot(data['KLEN'], label='KLEN', linewidth=2, color='#F18F01', alpha=0.8)
        axes[1].axhline(y=5, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Moderate Trend (5)')
        axes[1].axhline(y=10, color='darkgreen', linestyle='--', alpha=0.7, linewidth=1, label='Strong Trend (10)')
        axes[1].set_title('KLEN - Trend Persistence', fontsize=12)
        axes[1].set_ylabel('Trend Length (days)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. Momentum Statistics
        klen_stats = data['KLEN'].describe()
        klow_stats = data['KLOW'].describe()

        stats_text = (
            'KLEN Statistics:\n'
            f'Mean: {klen_stats["mean"]:.1f} days\n'
            f'Max: {klen_stats["max"]:.0f} days\n'
            f'>10 days: {(data["KLEN"] > 10).mean()*100:.1f}%\n\n'
            'KLOW Statistics:\n'
            f'Mean: {klow_stats["mean"]:.2f}\n'
            f'Min: {klow_stats["min"]:.2f}'
        )

        axes[2].text(0.5, 0.5, stats_text,
                    transform=axes[2].transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                    ha='center', va='center')
        axes[2].set_title('Momentum Statistics Summary', fontsize=12)
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        plt.tight_layout()
        return fig

    def _create_temporal_plot(self, data: pd.DataFrame) -> plt.Figure:
        """Create temporal visualization."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Temporal Features Analysis', fontsize=16, fontweight='bold')

        # Ensure datetime conversion
        if 'time' in data.columns:
            data['date'] = pd.to_datetime(data['time'])
        else:
            data['date'] = pd.date_range('2024-01-01', periods=len(data), freq='D')

        # Add temporal features
        data['month'] = data['date'].dt.month
        data['day_of_week'] = data['date'].dt.dayofweek
        data['year'] = data['date'].dt.year

        # 1. Monthly Distribution
        monthly_data = data.groupby('month')['close'].agg(['mean', 'std', 'count']).reset_index()
        axes[0, 0].bar(monthly_data['month'], monthly_data['mean'],
                      yerr=monthly_data['std'], capsize=5, alpha=0.7, color='#2E86AB')
        axes[0, 0].set_title('Average Price by Month', fontsize=12)
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Average Price')
        axes[0, 0].set_xticks(range(1, 13))
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Monthly Volume Pattern
        if 'volume' in data.columns:
            monthly_vol = data.groupby('month')['volume'].mean().reset_index()
            axes[0, 1].bar(monthly_vol['month'], monthly_vol['volume'],
                          alpha=0.7, color='#A23B72')
            axes[0, 1].set_title('Average Volume by Month', fontsize=12)
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Average Volume')
            axes[0, 1].set_xticks(range(1, 13))
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Day of Week Analysis
        dow_data = data.groupby('day_of_week')['close'].agg(['mean', 'std', 'count']).reset_index()
        axes[1, 0].bar(dow_data['day_of_week'], dow_data['mean'],
                      yerr=dow_data['std'], capsize=5, alpha=0.7, color='#F18F01')
        axes[1, 0].set_title('Average Price by Day of Week', fontsize=12)
        axes[1, 0].set_xlabel('Day of Week (0=Monday)')
        axes[1, 0].set_ylabel('Average Price')
        axes[1, 0].set_xticks(range(7))
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Day of Week Volume
        if 'volume' in data.columns:
            dow_vol = data.groupby('day_of_week')['volume'].mean().reset_index()
            axes[1, 1].bar(dow_vol['day_of_week'], dow_vol['volume'],
                          alpha=0.7, color='#C73E1D')
            axes[1, 1].set_title('Average Volume by Day of Week', fontsize=12)
            axes[1, 1].set_xlabel('Day of Week (0=Monday)')
            axes[1, 1].set_ylabel('Average Volume')
            axes[1, 1].set_xticks(range(7))
            axes[1, 1].grid(True, alpha=0.3)

        # 5. Yearly Trend
        if data['year'].nunique() > 1:
            yearly_data = data.groupby('year')['close'].agg(['mean', 'std', 'count']).reset_index()
            axes[2, 0].plot(yearly_data['year'], yearly_data['mean'],
                           marker='o', linewidth=2, markersize=8, color='#2E86AB')
            axes[2, 0].fill_between(yearly_data['year'],
                                  yearly_data['mean'] - yearly_data['std'],
                                  yearly_data['mean'] + yearly_data['std'],
                                  alpha=0.3, color='#2E86AB')
            axes[2, 0].set_title('Average Price by Year', fontsize=12)
            axes[2, 0].set_xlabel('Year')
            axes[2, 0].set_ylabel('Average Price')
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'Insufficient years for yearly analysis',
                           ha='center', va='center', transform=axes[2, 0].transAxes,
                           fontsize=10, alpha=0.7)
            axes[2, 0].set_title('Yearly Analysis (Insufficient Data)')

        # 6. Temporal Summary
        temporal_summary = (
            'Temporal Analysis Summary:\n\n'
            f'Total Periods: {len(data)}\n'
            f'Months Covered: {data["month"].nunique()}\n'
            f'Days of Week: {data["day_of_week"].nunique()}\n'
            f'Years Covered: {data["year"].nunique()}\n\n'
            'Key Patterns:\n'
            '- Monthly cycles in price and volume\n'
            '- Weekly patterns in trading activity\n'
            '- Long-term yearly trends'
        )

        axes[2, 1].text(0.5, 0.5, temporal_summary,
                       transform=axes[2, 1].transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                       ha='center', va='center')
        axes[2, 1].set_title('Temporal Analysis Summary', fontsize=12)
        axes[2, 1].set_xticks([])
        axes[2, 1].set_yticks([])

        plt.tight_layout()
        return fig

    def _create_static_plot(self, data: pd.DataFrame) -> plt.Figure:
        """Create static visualization."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('Static Features Analysis', fontsize=16, fontweight='bold')

        # Add const feature if not present
        if 'const' not in data.columns:
            data['const'] = 1.0

        # 1. Const Feature Over Time
        axes[0].plot(data['const'], label='const = 1.0', linewidth=2, color='#2E86AB', alpha=0.8)
        axes[0].axhline(y=1.0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Expected Value (1.0)')
        axes[0].set_title('Static Feature: const = 1.0', fontsize=12)
        axes[0].set_ylabel('Value')
        axes[0].set_ylim(0.99, 1.01)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Const Distribution
        axes[1].hist(data['const'], bins=1, alpha=0.7, color='#A23B72', rwidth=0.8)
        axes[1].axvline(x=1.0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Value = 1.0')
        axes[1].set_xlim(0.99, 1.01)
        axes[1].set_title('Const Feature Distribution', fontsize=12)
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. Const Statistics and Information
        const_stats = data['const'].describe()

        info_text = (
            'Static Feature Analysis:\n\n'
            'Feature: const\n'
            f'Value: {const_stats["mean"]:.6f}\n'
            f'Standard Deviation: {const_stats["std"]:.6f}\n'
            f'Minimum: {const_stats["min"]:.6f}\n'
            f'Maximum: {const_stats["max"]:.6f}\n'
            f'Unique Values: {data["const"].nunique()}\n\n'
            'Purpose:\n'
            '- TFT Architecture Requirement\n'
            '- Baseline for Feature Scaling\n'
            '- Model Input Standardization\n'
            '- Numerical Stability'
        )

        axes[2].text(0.5, 0.5, info_text,
                    transform=axes[2].transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                    ha='center', va='center')
        axes[2].set_title('Static Feature Information', fontsize=12)
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        plt.tight_layout()
        return fig

    def _figure_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 encoded string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64

    def _create_feature_section(self, section_id: str, title: str, plot_base64: str,
                              explanation: str, feature_names: List[str]) -> str:
        """Create HTML section for a feature group."""

        # Clean explanation text for HTML
        clean_explanation = explanation.replace('\n', '<br>').replace('#', '')

        section_html = f"""
        <div class="feature-section" id="{section_id.lower()}">
            <div class="section-header">
                <h2>{title}</h2>
                <div class="feature-badges">
                    {''.join(f'<span class="feature-badge">{name}</span>' for name in feature_names)}
                </div>
            </div>

            <div class="section-content">
                <div class="explanation">
                    <h3>Understanding {title}</h3>
                    <div class="explanation-text">
                        {clean_explanation}
                    </div>
                </div>

                <div class="plot-container">
                    <img src="data:image/png;base64,{plot_base64}" alt="{title}">
                </div>
            </div>
        </div>
        """
        return section_html

    def _create_error_section(self, section_id: str, error_message: str) -> str:
        """Create HTML section for errors."""
        error_html = f"""
        <div class="feature-section error" id="{section_id.lower()}">
            <div class="section-header">
                <h2>{section_id} Analysis</h2>
                <div class="error-badge">Error</div>
            </div>

            <div class="section-content">
                <div class="error-message">
                    <p><strong>Error generating visualization:</strong> {error_message}</p>
                    <p>This section could not be generated. Please check the data format and try again.</p>
                </div>
            </div>
        </div>
        """
        return error_html

    def _generate_main_html(self, feature_sections: List[str]) -> str:
        """Generate the main HTML structure."""

        # Create navigation menu
        nav_items = []
        for name in self.feature_configs.keys():
            nav_items.append(f'<a href="#{name.lower()}" class="nav-link">{name}</a>')

        navigation = '\n                    '.join(nav_items)

        # Generate main HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 0;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}

        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        .navigation {{
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        .nav-links {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            justify-content: center;
        }}

        .nav-link {{
            color: #667eea;
            text-decoration: none;
            padding: 0.5rem 1rem;
            border: 2px solid #667eea;
            border-radius: 25px;
            transition: all 0.3s ease;
            font-weight: 500;
        }}

        .nav-link:hover {{
            background-color: #667eea;
            color: white;
            transform: translateY(-2px);
        }}

        .feature-section {{
            background: white;
            margin-bottom: 3rem;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .section-header {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-bottom: 3px solid #667eea;
        }}

        .section-header h2 {{
            color: #2c3e50;
            font-size: 1.8rem;
            margin-bottom: 1rem;
        }}

        .feature-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .feature-badge {{
            background-color: #667eea;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.85rem;
            font-weight: 500;
        }}

        .section-content {{
            padding: 2rem;
        }}

        .plot-container {{
            margin-bottom: 2rem;
            text-align: center;
        }}

        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}

        .explanation {{
            margin-bottom: 2rem;
            padding: 1.5rem;
            background-color: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}

        .explanation h3 {{
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 1.4rem;
        }}

        .explanation-text {{
            color: #555;
            line-height: 1.8;
        }}

        .error-message {{
            background-color: #ffebee;
            border: 1px solid #f44336;
            color: #c62828;
            padding: 1rem;
            border-radius: 5px;
            text-align: center;
        }}

        .footer {{
            text-align: center;
            padding: 2rem;
            color: #666;
            border-top: 1px solid #ddd;
            margin-top: 3rem;
        }}

        .timestamp {{
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}

            .header h1 {{
                font-size: 2rem;
            }}

            .nav-links {{
                flex-direction: column;
                align-items: center;
            }}

            .section-content {{
                padding: 1rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <p>Comprehensive Analysis of VN30 TFT Technical Features</p>
    </div>

    <div class="container">
        <div class="timestamp">
            <strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>

        <div class="navigation">
            <div class="nav-links">
                {navigation}
            </div>
        </div>

        <div class="content">
            {''.join(feature_sections)}
        </div>

        <div class="footer">
            <p><strong>Generated by StocketAI Visualizer Engine</strong></p>
            <p>Comprehensive VN30 TFT Features Analysis Report</p>
        </div>
    </div>

    <script>
        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{
                        behavior: 'smooth',
                        block: 'start'
                    }});
                }}
            }});
        }});
    </script>
</body>
</html>
"""

        return html_content

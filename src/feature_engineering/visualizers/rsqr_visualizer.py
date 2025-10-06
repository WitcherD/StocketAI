"""
RSQR (R-squared) Visualizer for VN30 TFT Features

This module provides visualization and explanation for RSQR features:
- RSQR5, RSQR10, RSQR20, RSQR60: R-squared values for different periods

Author: StocketAI
Created: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional, Dict, Any
from base_visualizer import BaseVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RSQRVisualizer(BaseVisualizer):
    """
    Visualizer for RSQR (R-squared) features.

    R-squared measures how well price movements fit a straight line (trend),
    indicating trend strength and reliability.
    """

    def __init__(self):
        """Initialize RSQR visualizer."""
        super().__init__()
        self.feature_names = ['RSQR5', 'RSQR10', 'RSQR20', 'RSQR60']
        self.title = "RSQR (R-squared) Analysis"

    def generate_explanation(self) -> str:
        """Generate explanation for RSQR features."""
        return """
# RSQR (R-squared) - Technical Indicator Explanation

## What is R-squared?
**R-squared (RSQR)** measures how well a stock's price movements fit a straight line over a specific period. Think of it as a "trend quality score" - higher values mean the price is following a more predictable, linear trend.

## R-squared Scale (0 to 1):
- **RSQR = 1.0**: Perfect straight line trend (very rare)
- **RSQR = 0.7-0.9**: Strong, reliable trend
- **RSQR = 0.4-0.6**: Moderate trend with some noise
- **RSQR = 0.0-0.3**: Weak or no trend (random movement)

## How to Understand Different Periods:

### Short-term RSQR (RSQR5, RSQR10):
- **High Values (0.7+)**: Strong short-term trend, good for momentum trading
- **Low Values (<0.3)**: Choppy, unpredictable price action
- **Rising RSQR**: Trend strengthening
- **Falling RSQR**: Trend weakening, potential reversal

### Long-term RSQR (RSQR20, RSQR60):
- **High Values (0.7+)**: Reliable long-term trend, good for position trading
- **Low Values (<0.3)**: Sideways market, range-bound trading
- **Very High Values (0.9+)**: Strong trending market

## Trading Interpretation:

### Bullish Signals:
- **RSQR > 0.7**: Strong trend, trend-following strategies work well
- **RSQR5 > RSQR20**: Short-term trend stronger than long-term
- **RSQR rising**: Trend gaining strength
- **RSQR > 0.8**: Very reliable trend, low risk of false signals

### Bearish Signals:
- **RSQR < 0.3**: Weak trend, avoid trend-following strategies
- **RSQR5 < RSQR20**: Short-term trend weaker than long-term
- **RSQR falling**: Trend losing strength
- **RSQR < 0.2**: Random price movement, consider mean-reversion

### Trading Strategy Selection:
- **High RSQR (>0.7)**: Use trend-following indicators (moving averages, MACD)
- **Medium RSQR (0.4-0.7)**: Use both trend and oscillator indicators
- **Low RSQR (<0.4)**: Use oscillators and support/resistance levels

## Risk Management:

### High RSQR Environment:
- **Lower Risk**: Trends are more predictable
- **Good for**: Larger position sizes, longer holding periods
- **Stop Losses**: Can be wider due to reliable trends

### Low RSQR Environment:
- **Higher Risk**: Price movements are less predictable
- **Good for**: Smaller position sizes, shorter holding periods
- **Stop Losses**: Should be tighter due to random movements

## Market Regime Identification:

### Trending Markets:
- **RSQR consistently high**: Strong directional movement
- **Multiple timeframes agree**: Very reliable trend environment

### Sideways/Ranging Markets:
- **RSQR consistently low**: No clear direction
- **Choppy price action**: Random fluctuations dominate

### Transitioning Markets:
- **RSQR changing rapidly**: Market regime shifting
- **RSQR divergence**: Potential trend reversal

## Best Practices:
- **Combine timeframes**: Use multiple RSQR periods for confirmation
- **Monitor changes**: Watch for RSQR trend changes, not just absolute values
- **Context matters**: RSQR interpretation depends on market conditions
- **Not a timing tool**: RSQR tells you about trend quality, not when to enter/exit
"""

    def generate_visualization(self, data: pd.DataFrame, output_path: str, **kwargs) -> str:
        """
        Generate RSQR visualization.

        Args:
            data: DataFrame with price and RSQR data
            output_path: Path to save the HTML file
            **kwargs: Additional arguments

        Returns:
            Path to generated HTML file
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=100)
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

            # 1. Price Trend Analysis
            axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8)

            # Calculate trend lines for visualization
            for period in [5, 20, 60]:
                if f'RSQR{period}' in data.columns:
                    # Simple linear regression for trend line
                    x = np.arange(len(data))
                    y = data['close'].values

                    # Calculate trend only for valid RSQR periods
                    valid_data = data[f'RSQR{period}'].notna() & data[f'RSQR{period}'] > 0.5
                    if valid_data.sum() > 10:
                        x_valid = x[valid_data]
                        y_valid = y[valid_data]

                        if len(x_valid) > 1:
                            coeffs = np.polyfit(x_valid, y_valid, 1)
                            trend_line = coeffs[0] * x + coeffs[1]

                            axes[0].plot(data.index, trend_line,
                                       label=f'{period}-day Trend (R²>{0.5})',
                                       linestyle='--', linewidth=1.5, alpha=0.7)

            axes[0].set_title('Price with Trend Lines (High R² Periods)', fontsize=12)
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

            # 3. RSQR Distribution and Statistics
            rsqr_data = []
            for period in rsqr_periods:
                if f'RSQR{period}' in data.columns:
                    valid_rsqr = data[f'RSQR{period}'].dropna()
                    if len(valid_rsqr) > 0:
                        rsqr_data.append(pd.Series(valid_rsqr, name=f'RSQR{period}'))

            if rsqr_data:
                rsqr_df = pd.concat(rsqr_data, axis=1)

                # Box plot for distribution
                bp = axes[2].boxplot([rsqr_df[col].dropna() for col in rsqr_df.columns],
                                   labels=rsqr_df.columns, patch_artist=True)

                # Color the boxes
                for i, box in enumerate(bp['boxes']):
                    box.set(facecolor=colors[i % len(colors)])

                axes[2].axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Strong Trend')
                axes[2].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Weak Trend')
                axes[2].set_title('R-squared Distribution by Period', fontsize=12)
                axes[2].set_ylabel('R-squared Value')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save as HTML
            html_path = self.save_plot_as_html(fig, output_path, "rsqr_analysis.html")

            plt.close(fig)
            logger.info(f"Generated RSQR visualization: {html_path}")
            return html_path

        except Exception as e:
            logger.error(f"Error generating RSQR visualization: {e}")
            return None

    def _calculate_r_squared(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate R-squared for visualization."""
        def rolling_r_squared(x):
            if len(x) < 3:
                return np.nan
            x_vals = np.arange(len(x))
            y_vals = x.values

            # Calculate R-squared
            try:
                coeffs = np.polyfit(x_vals, y_vals, 1)
                y_pred = coeffs[0] * x_vals + coeffs[1]
                ss_res = np.sum((y_vals - y_pred) ** 2)
                ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
            except:
                return np.nan

        return series.rolling(window=period, min_periods=3).apply(rolling_r_squared, raw=False)

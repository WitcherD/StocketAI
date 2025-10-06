"""
WVMA (Weighted Moving Average) Visualizer for VN30 TFT Features

This module provides visualization and explanation for WVMA features:
- WVMA5: 5-day weighted moving average
- WVMA60: 60-day weighted moving average

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


class WVMAVisualizer(BaseVisualizer):
    """
    Visualizer for WVMA (Weighted Moving Average) features.

    Weighted Moving Averages give more importance to recent prices,
    making them more responsive to recent price changes than simple moving averages.
    """

    def __init__(self):
        """Initialize WVMA visualizer."""
        super().__init__()
        self.feature_names = ['WVMA5', 'WVMA60']
        self.title = "WVMA (Weighted Moving Average) Analysis"

    def generate_explanation(self) -> str:
        """Generate explanation for WVMA features."""
        return """
# WVMA (Weighted Moving Average) - Technical Indicator Explanation

## What is a Weighted Moving Average?
A **Weighted Moving Average (WVMA)** is similar to a regular moving average, but it gives **more weight to recent prices** and less weight to older prices. Think of it as a "front-loaded" moving average that responds more quickly to recent price changes.

## How WVMA is Calculated:
- **WVMA5**: Uses last 5 days of prices with weights [1, 2, 3, 4, 5] (most recent = highest weight)
- **WVMA60**: Uses last 60 days of prices with linearly increasing weights

## How to Understand WVMA Values:

### WVMA5 (Short-term, 5-day):
- **Responsive**: Quickly follows recent price movements
- **Sensitive**: Reacts fast to price changes
- **Noisy**: Can give false signals in volatile markets
- **Best for**: Short-term trading, identifying immediate trend changes

### WVMA60 (Long-term, 60-day):
- **Stable**: Smooths out short-term price fluctuations
- **Reliable**: Less affected by daily market noise
- **Slower**: Takes longer to reflect trend changes
- **Best for**: Long-term trend identification, major support/resistance levels

## Trading Interpretation:

### Bullish Signals:
- **Price above WVMA5**: Short-term uptrend
- **WVMA5 above WVMA60**: Short-term strength in longer-term uptrend
- **Both WVMAs rising**: Strong upward momentum across timeframes

### Bearish Signals:
- **Price below WVMA5**: Short-term downtrend
- **WVMA5 below WVMA60**: Short-term weakness in longer-term downtrend
- **Both WVMAs falling**: Strong downward momentum across timeframes

### Support and Resistance:
- **WVMA5**: Acts as dynamic support/resistance for short-term trades
- **WVMA60**: Acts as major support/resistance for long-term positions

## Key Differences from Simple Moving Average (SMA):

### WVMA vs SMA:
- **WVMA responds faster** to price changes than SMA
- **WVMA reduces lag** compared to equal-weighted averages
- **WVMA is more sensitive** to recent price movements
- **WVMA is preferred** for trend-following systems

### When to Use Each:
- **WVMA5**: Best for short-term trading and quick entries/exits
- **WVMA60**: Best for identifying major trends and portfolio allocation
- **Combined**: Use both for multi-timeframe analysis

## Risk Management:
- **WVMA5 Crossovers**: Good for timing entries/exits
- **WVMA60 Trends**: Better for overall market direction
- **Divergences**: When price and WVMA move in opposite directions
"""

    def generate_visualization(self, data: pd.DataFrame, output_path: str, **kwargs) -> str:
        """
        Generate WVMA visualization.

        Args:
            data: DataFrame with price and WVMA data
            output_path: Path to save the HTML file
            **kwargs: Additional arguments

        Returns:
            Path to generated HTML file
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=100)
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

            # 1. Price and WVMA Comparison
            axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8)

            # Calculate WVMAs for visualization
            wvma5 = self._calculate_wvma(data['close'], 5)
            wvma60 = self._calculate_wvma(data['close'], 60)

            axes[0].plot(wvma5, label='WVMA5', linewidth=2, color='#2E86AB')
            axes[0].plot(wvma60, label='WVMA60', linewidth=2, color='#A23B72')

            axes[0].set_title('Price vs Weighted Moving Averages', fontsize=12)
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 2. WVMA Difference (WVMA5 - WVMA60)
            wvma_diff = wvma5 - wvma60
            axes[1].plot(wvma_diff, label='WVMA5 - WVMA60', linewidth=2, color='#F18F01')
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            axes[1].fill_between(data.index, wvma_diff, 0,
                               where=(wvma_diff >= 0), color='#2E86AB', alpha=0.3)
            axes[1].fill_between(data.index, wvma_diff, 0,
                               where=(wvma_diff < 0), color='red', alpha=0.3)

            axes[1].set_title('WVMA Momentum (WVMA5 - WVMA60)', fontsize=12)
            axes[1].set_ylabel('Price Difference')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # 3. WVMA Distribution Analysis
            wvma_data = pd.DataFrame({
                'WVMA5': wvma5.fillna(0),
                'WVMA60': wvma60.fillna(0)
            })

            sns.histplot(data=wvma_data, ax=axes[2], alpha=0.7)
            axes[2].set_title('WVMA Distribution Analysis', fontsize=12)
            axes[2].set_xlabel('WVMA Value')
            axes[2].set_ylabel('Frequency')
            axes[2].legend(['WVMA5', 'WVMA60'])

            plt.tight_layout()

            # Save as HTML
            html_path = self.save_plot_as_html(fig, output_path, "wvma_analysis.html")

            plt.close(fig)
            logger.info(f"Generated WVMA visualization: {html_path}")
            return html_path

        except Exception as e:
            logger.error(f"Error generating WVMA visualization: {e}")
            return None

    def _calculate_wvma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate weighted moving average for visualization."""
        def wma_calc(x):
            if len(x) < period:
                return np.nan
            weights = np.arange(1, len(x) + 1)
            return np.dot(x, weights) / weights.sum()

        return series.rolling(window=period, min_periods=1).apply(wma_calc, raw=False)

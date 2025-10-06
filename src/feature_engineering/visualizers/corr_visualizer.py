"""
CORR (Correlation) Visualizer for VN30 TFT Features

This module provides visualization and explanation for CORR features:
- CORR5, CORR10, CORR20, CORR60: Correlation between price and volume

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


class CORRVisualizer(BaseVisualizer):
    """
    Visualizer for CORR (Correlation) features.

    Correlation measures the relationship between price movements and volume,
    indicating whether price and volume move together or in opposite directions.
    """

    def __init__(self):
        """Initialize CORR visualizer."""
        super().__init__()
        self.feature_names = ['CORR5', 'CORR10', 'CORR20', 'CORR60']
        self.title = "CORR (Price-Volume Correlation) Analysis"

    def generate_explanation(self) -> str:
        """Generate explanation for CORR features."""
        return """
# CORR (Price-Volume Correlation) - Technical Indicator Explanation

## What is Price-Volume Correlation?
**CORR measures the relationship between stock price movements and trading volume.** It tells you whether price and volume are moving in the same direction (positive correlation) or opposite directions (negative correlation).

## Correlation Scale (-1 to +1):
- **CORR = +1.0**: Perfect positive correlation (price and volume always move together)
- **CORR = +0.5 to +1.0**: Strong positive correlation (price up = volume up)
- **CORR = 0**: No correlation (price and volume movements are unrelated)
- **CORR = -0.5 to -1.0**: Negative correlation (price up = volume down)
- **CORR = -1.0**: Perfect negative correlation (opposite movements)

## How to Understand Different Periods:

### Short-term CORR (CORR5, CORR10):
- **High Positive (>0.7)**: Strong conviction in short-term price moves
- **Low/Negative (<0.3)**: Weak conviction, potential manipulation or random trading
- **Erratic**: Frequent changes suggest uncertain market sentiment

### Long-term CORR (CORR20, CORR60):
- **Consistently Positive**: Healthy market participation in trends
- **Consistently Negative**: Unusual market dynamics, potential concern
- **Near Zero**: Volume doesn't confirm price movements

## Trading Interpretation:

### Bullish Signals:
- **High Positive CORR**: Volume supports upward price movements
- **Rising CORR**: Increasing conviction in bullish trend
- **CORR5 > CORR20**: Short-term volume supporting longer-term trend

### Bearish Signals:
- **Negative CORR**: Volume decreases as price rises (bearish divergence)
- **Falling CORR**: Decreasing conviction in trend
- **CORR5 < CORR20**: Short-term volume weakening longer-term trend

### Volume Confirmation:
- **Positive CORR + Rising Price**: Healthy bullish trend
- **Positive CORR + Falling Price**: Healthy bearish trend
- **Negative CORR + Rising Price**: Suspicious rally (low volume support)
- **Negative CORR + Falling Price**: Suspicious decline (low volume pressure)

## Market Psychology:

### High Positive Correlation:
- **Strong Trend**: Market participants agree on direction
- **High Conviction**: Volume supports price movements
- **Healthy Market**: Natural relationship between price and volume

### Low/Negative Correlation:
- **Weak Conviction**: Volume doesn't support price moves
- **Potential Manipulation**: Price moving without volume confirmation
- **Unusual Activity**: Consider investigating further

## Risk Management:

### High CORR Environment:
- **Lower Risk**: Volume confirms price movements
- **Reliable Trends**: Price movements are more trustworthy
- **Better Entries**: Volume-supported breakouts tend to be stronger

### Low CORR Environment:
- **Higher Risk**: Price movements lack volume confirmation
- **False Breakouts**: More likely without volume support
- **Caution Advised**: Consider reducing position sizes

## Best Practices:
- **Combine with Price Analysis**: CORR is most useful with trend analysis
- **Multiple Timeframes**: Compare short-term and long-term correlations
- **Monitor Changes**: Watch for correlation regime changes
- **Not Standalone**: Use with other volume and price indicators
"""

    def generate_visualization(self, data: pd.DataFrame, output_path: str, **kwargs) -> str:
        """
        Generate CORR visualization.

        Args:
            data: DataFrame with price, volume, and CORR data
            output_path: Path to save the HTML file
            **kwargs: Additional arguments

        Returns:
            Path to generated HTML file
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(4, 1, figsize=(14, 16), dpi=100)
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

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

            # Add legend
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
            axes[1].axhline(y=0.3, color='orange', linestyle='-', alpha=0.8, linewidth=1, label='Moderate (0.3)')
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

                # Histogram for distribution
                for i, col in enumerate(corr_df.columns):
                    valid_data = corr_df[col].dropna()
                    if len(valid_data) > 0:
                        axes[2].hist(valid_data, bins=20, alpha=0.7,
                                   label=col, color=colors[i % len(colors)])

                axes[2].axvline(x=0.7, color='green', linestyle='--', alpha=0.7, label='Strong Positive')
                axes[2].axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Moderate')
                axes[2].axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                axes[2].axvline(x=-0.3, color='red', linestyle='--', alpha=0.7, label='Negative')
                axes[2].set_xlim(-1, 1)
                axes[2].set_title('Correlation Distribution by Period', fontsize=12)
                axes[2].set_xlabel('Correlation Value')
                axes[2].set_ylabel('Frequency')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)

            # 4. CORR Heatmap (if multiple periods available)
            if len(corr_data) > 1:
                # Calculate correlation between different CORR periods
                corr_matrix = corr_df.corr()

                sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu', center=0,
                           square=True, ax=axes[3], vmin=-1, vmax=1)

                axes[3].set_title('Correlation Between Different CORR Periods', fontsize=12)

                # Add correlation strength labels
                for i in range(len(corr_matrix)):
                    for j in range(len(corr_matrix)):
                        if i != j:
                            val = corr_matrix.iloc[i, j]
                            if abs(val) > 0.7:
                                axes[3].text(j + 0.5, i + 0.5, 'Strong',
                                           ha='center', va='center', fontsize=8,
                                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[3].text(0.5, 0.5, 'Insufficient data for correlation matrix',
                           ha='center', va='center', transform=axes[3].transAxes,
                           fontsize=10, alpha=0.7)
                axes[3].set_title('Correlation Matrix (Insufficient Data)')

            plt.tight_layout()

            # Save as HTML
            html_path = self.save_plot_as_html(fig, output_path, "corr_analysis.html")

            plt.close(fig)
            logger.info(f"Generated CORR visualization: {html_path}")
            return html_path

        except Exception as e:
            logger.error(f"Error generating CORR visualization: {e}")
            return None

    def _calculate_correlation(self, price_series: pd.Series, volume_series: pd.Series, period: int) -> pd.Series:
        """Calculate correlation between price and volume for visualization."""
        def rolling_corr(price_window, volume_window):
            if len(price_window) < 3 or len(volume_window) < 3:
                return np.nan
            try:
                return price_window.corr(volume_window)
            except:
                return np.nan

        return pd.Series([
            rolling_corr(price_series.iloc[i-period:i] if i >= period else price_series.iloc[:i],
                        volume_series.iloc[i-period:i] if i >= period else volume_series.iloc[:i])
            for i in range(len(price_series))
        ], index=price_series.index)

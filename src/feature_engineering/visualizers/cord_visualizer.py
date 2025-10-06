"""
CORD (Correlation Differences) Visualizer for VN30 TFT Features

This module provides visualization and explanation for CORD features:
- CORD5, CORD10, CORD60: Correlation differences between periods

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


class CORDVisualizer(BaseVisualizer):
    """
    Visualizer for CORD (Correlation Differences) features.

    Correlation differences measure how correlation between price and volume
    changes over different time periods, indicating shifts in market behavior.
    """

    def __init__(self):
        """Initialize CORD visualizer."""
        super().__init__()
        self.feature_names = ['CORD5', 'CORD10', 'CORD60']
        self.title = "CORD (Correlation Differences) Analysis"

    def generate_explanation(self) -> str:
        """Generate explanation for CORD features."""
        return """
# CORD (Correlation Differences) - Technical Indicator Explanation

## What are Correlation Differences?
**CORD measures the CHANGE in correlation between price and volume over different time periods.** It tells you whether the relationship between price movements and volume is strengthening or weakening over time.

## CORD Formula:
**CORD_period = CORR_period - CORR_prev_period**

- **CORD5 = CORR5 - CORR0** (baseline)
- **CORD10 = CORR10 - CORR5**
- **CORD60 = CORR60 - CORR20**

## CORD Scale Interpretation:
- **CORD > 0**: Correlation INCREASING (strengthening relationship)
- **CORD = 0**: Correlation UNCHANGED (stable relationship)
- **CORD < 0**: Correlation DECREASING (weakening relationship)

## How to Understand Different Periods:

### Short-term CORD (CORD5, CORD10):
- **Positive Values**: Short-term price-volume relationship strengthening
- **Negative Values**: Short-term price-volume relationship weakening
- **Large Changes**: Significant shifts in market participant behavior

### Long-term CORD (CORD60):
- **Positive Values**: Long-term price-volume dynamics improving
- **Negative Values**: Long-term price-volume dynamics deteriorating
- **Stable Values**: Consistent market behavior over extended periods

## Trading Interpretation:

### Bullish Signals:
- **Positive CORD**: Volume increasingly supporting price movements
- **CORD10 > CORD5**: Medium-term correlation strengthening
- **Rising CORD trend**: Improving price-volume relationship

### Bearish Signals:
- **Negative CORD**: Volume decreasingly supporting price movements
- **CORD10 < CORD5**: Medium-term correlation weakening
- **Falling CORD trend**: Deteriorating price-volume relationship

### Market Regime Changes:
- **CORD shifting from negative to positive**: Market behavior improving
- **CORD shifting from positive to negative**: Market behavior deteriorating
- **Erratic CORD**: Unstable market conditions

## Risk Management:

### Correlation Quality Assessment:
- **High positive CORD**: Reliable price-volume signals
- **High negative CORD**: Unreliable price-volume signals
- **CORD near zero**: Stable but unchanging correlation

### Position Sizing:
- **Strong positive CORD**: Can rely more on volume-based signals
- **Strong negative CORD**: Should rely less on volume-based signals
- **Erratic CORD**: Consider alternative analysis methods

## Market Psychology:

### Improving Correlation (Positive CORD):
- **Market participants agreeing**: Price and volume moving in harmony
- **Clear market direction**: Conviction supporting price movements
- **Healthy participation**: Natural relationship between price and volume

### Deteriorating Correlation (Negative CORD):
- **Market confusion**: Price and volume moving independently
- **Uncertain sentiment**: Mixed signals from market participants
- **Potential manipulation**: Price moving without volume confirmation

## Best Practices:

### Multiple Timeframes:
- **Compare CORD5 vs CORD10**: Short-term vs medium-term changes
- **CORD60 trends**: Long-term correlation stability
- **Convergence/divergence**: Different periods showing similar/different trends

### Combined with Other Indicators:
- **Use with CORR**: CORD shows change, CORR shows absolute level
- **Volume analysis**: CORD helps interpret volume signal reliability
- **Price patterns**: Correlation changes can confirm or contradict price signals

### Trading Strategy Implications:
- **High CORD**: Volume-based strategies more effective
- **Low CORD**: Price-based strategies may be more reliable
- **Changing CORD**: Consider adjusting trading approach

## VN30 Context:

### Vietnamese Market Correlation:
- **Foreign investor impact**: Can cause unusual price-volume relationships
- **Liquidity effects**: Lower liquidity can lead to erratic correlations
- **Market microstructure**: Different from developed markets
- **Regulatory environment**: Can affect price-volume dynamics

### Trading Implications:
- **Monitor CORD trends**: Important for strategy selection
- **Beware of false signals**: Low correlation periods more common
- **Adapt to correlation regime**: Different approaches for different CORD levels
"""

    def generate_visualization(self, data: pd.DataFrame, output_path: str, **kwargs) -> str:
        """
        Generate CORD visualization.

        Args:
            data: DataFrame with price, volume, and CORD data
            output_path: Path to save the HTML file
            **kwargs: Additional arguments

        Returns:
            Path to generated HTML file
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=100)
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

            # 1. Price and Volume with Correlation Overlay
            axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8, color='#2E86AB')

            # Plot volume on secondary axis
            ax_twin = axes[0].twinx()
            ax_twin.bar(data.index, data['volume'], label='Volume', alpha=0.3, color='#A23B72', width=0.8)

            axes[0].set_title('Price and Volume with Correlation Context', fontsize=12)
            axes[0].set_ylabel('Price', color='#2E86AB')
            axes[0].tick_params(axis='y', labelcolor='#2E86AB')
            ax_twin.set_ylabel('Volume', color='#A23B72')
            ax_twin.tick_params(axis='y', labelcolor='#A23B72')

            # Add legend
            lines1, labels1 = axes[0].get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            axes[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            axes[0].grid(True, alpha=0.3)

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

            # 3. CORD Distribution and Analysis
            cord_data = []
            for period in cord_periods:
                if f'CORD{period}' in data.columns:
                    valid_cord = data[f'CORD{period}'].dropna()
                    if len(valid_cord) > 0:
                        cord_data.append(pd.Series(valid_cord, name=f'CORD{period}'))

            if cord_data:
                cord_df = pd.concat(cord_data, axis=1)

                # Box plot for distribution
                bp = axes[2].boxplot([cord_df[col].dropna() for col in cord_df.columns],
                                   labels=cord_df.columns, patch_artist=True)

                # Color the boxes
                for i, box in enumerate(bp['boxes']):
                    box.set(facecolor=colors[i % len(colors)])

                axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                axes[2].axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Strong Improvement')
                axes[2].axhline(y=-0.3, color='red', linestyle='--', alpha=0.7, label='Strong Deterioration')
                axes[2].set_ylim(-1, 1)
                axes[2].set_title('CORD Distribution by Period', fontsize=12)
                axes[2].set_ylabel('CORD Value')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save as HTML
            html_path = self.save_plot_as_html(fig, output_path, "cord_analysis.html")

            plt.close(fig)
            logger.info(f"Generated CORD visualization: {html_path}")
            return html_path

        except Exception as e:
            logger.error(f"Error generating CORD visualization: {e}")
            return None

    def _calculate_correlation_differences(self, price_series: pd.Series, volume_series: pd.Series) -> pd.DataFrame:
        """Calculate correlation differences for visualization."""
        # Calculate correlations for different periods
        corr5 = self._calculate_rolling_correlation(price_series, volume_series, 5)
        corr10 = self._calculate_rolling_correlation(price_series, volume_series, 10)
        corr20 = self._calculate_rolling_correlation(price_series, volume_series, 20)
        corr60 = self._calculate_rolling_correlation(price_series, volume_series, 60)

        # Calculate differences
        cord_data = pd.DataFrame(index=price_series.index)
        cord_data['CORD5'] = corr5 - 0.0  # Baseline
        cord_data['CORD10'] = corr10 - corr5
        cord_data['CORD60'] = corr60 - corr20

        return cord_data

    def _calculate_rolling_correlation(self, price_series: pd.Series, volume_series: pd.Series, period: int) -> pd.Series:
        """Calculate rolling correlation between price and volume."""
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

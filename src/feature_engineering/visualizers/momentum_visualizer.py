"""
Momentum Indicators Visualizer for VN30 TFT Features

This module provides visualization and explanation for momentum features:
- KLEN: Length of consecutive up/down trend
- KLOW: Lowest price in 20-day rolling window

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


class MomentumVisualizer(BaseVisualizer):
    """
    Visualizer for momentum indicators (KLEN and KLOW).

    These indicators measure trend persistence and support levels,
    helping identify the strength and duration of price movements.
    """

    def __init__(self):
        """Initialize momentum visualizer."""
        super().__init__()
        self.feature_names = ['KLEN', 'KLOW']
        self.title = "Momentum Indicators (KLEN & KLOW) Analysis"

    def generate_explanation(self) -> str:
        """Generate explanation for momentum features."""
        return """
# Momentum Indicators - Technical Indicator Explanation

## What are Momentum Indicators?
**Momentum indicators measure the STRENGTH and PERSISTENCE of price trends.** They help identify how long trends have been in place and where potential support levels might be found.

## Two Key Momentum Indicators:

### KLEN (Trend Length)
- **Measures consecutive up/down movements in price**
- **Shows TREND PERSISTENCE** - how long current direction has lasted
- **Identifies TREND STRENGTH** - longer trends tend to be stronger
- **Helps with TREND EXHAUSTION** - very long trends may be near end

### KLOW (Lowest Low)
- **Lowest price in 20-day rolling window**
- **Identifies SUPPORT LEVELS** - where price might find buying interest
- **Shows RELATIVE LOWS** - current price vs recent range
- **Helps with RISK ASSESSMENT** - closer to lows = higher risk

## KLEN Scale Interpretation:

### Trend Length Categories:
- **KLEN 1-3**: Very short trend, likely noise
- **KLEN 4-7**: Moderate trend, some conviction
- **KLEN 8-15**: Strong trend, good momentum
- **KLEN >15**: Very strong trend, but watch for exhaustion

### Trend Direction:
- **Increasing KLEN**: Trend gaining strength and persistence
- **Decreasing KLEN**: Trend losing strength, potential reversal
- **KLEN resets to 1**: Trend change occurred

## KLOW Scale Interpretation:

### Position Relative to Lows:
- **Price near KLOW**: Near support, potential bounce area
- **Price >> KLOW**: In middle or upper range, less immediate support
- **KLOW rising**: Uptrend with higher lows (bullish)
- **KLOW falling**: Downtrend with lower lows (bearish)

## Trading Interpretation:

### KLEN Signals:

#### Bullish Signals:
- **KLEN > 10 and increasing**: Strong, persistent uptrend
- **KLEN rising after consolidation**: New trend emerging
- **KLEN > previous peaks**: Stronger than recent trends

#### Bearish Signals:
- **KLEN > 15 and stable**: Potential trend exhaustion
- **KLEN decreasing**: Trend losing momentum
- **KLEN drops sharply**: Potential trend reversal

### KLOW Signals:

#### Support Levels:
- **Price approaches KLOW**: Potential buying opportunity
- **KLOW holds multiple times**: Strong support level
- **KLOW breaks**: Bearish, support becomes resistance

#### Trend Strength:
- **Rising KLOW**: Healthy uptrend with higher lows
- **Falling KLOW**: Strong downtrend with lower lows
- **Flat KLOW**: Sideways market, range-bound

## Risk Management:

### Using KLEN for Position Management:
- **High KLEN (>10)**: Can use wider stops, trend is strong
- **Low KLEN (<5)**: Use tighter stops, trend may be weak
- **KLEN decreasing**: Consider reducing position size

### Using KLOW for Stop Placement:
- **Stop below KLOW**: For trend-following positions
- **Stop at KLOW**: For range-bound strategies
- **Multiple KLOW levels**: Use for trailing stops

## Market Psychology:

### KLEN Psychology:
- **Long KLEN**: Market participants committed to direction
- **Short KLEN**: Market uncertain, frequent direction changes
- **Increasing KLEN**: Growing conviction in trend
- **Decreasing KLEN**: Diminishing confidence in trend

### KLOW Psychology:
- **Near KLOW**: Fear and capitulation may create support
- **Far from KLOW**: Complacency, less immediate support
- **KLOW breaks**: Panic selling, bearish sentiment

## Best Practices:

### Combining KLEN and KLOW:
- **High KLEN + Price > KLOW**: Strong trend with good support
- **Low KLEN + Price near KLOW**: Weak trend near support
- **High KLEN + Price near KLOW**: Strong trend testing support

### Multiple Timeframes:
- **Compare different KLEN periods**: Short vs long-term trends
- **KLOW across timeframes**: Support levels at different scales
- **Divergence analysis**: Different timeframes showing different signals

### Entry and Exit Timing:
- **KLEN breakouts**: New trends starting
- **KLEN exhaustion**: Trends potentially ending
- **KLOW bounces**: Support holding
- **KLOW breaks**: Support failing

## VN30 Context:

### Vietnamese Market Momentum:
- **Often faster trends** due to lower liquidity
- **More volatile KLEN** patterns than developed markets
- **KLOW more significant** due to thinner order books
- **Foreign investor impact** can create unusual momentum patterns

### Trading Implications:
- **Faster position management** may be needed
- **KLEN changes more rapidly** - need closer monitoring
- **KLOW breaks more significant** - stronger signals
- **Adapt to market conditions** - momentum behaves differently in different regimes
"""

    def generate_visualization(self, data: pd.DataFrame, output_path: str, **kwargs) -> str:
        """
        Generate momentum visualization.

        Args:
            data: DataFrame with price and momentum data
            output_path: Path to save the HTML file
            **kwargs: Additional arguments

        Returns:
            Path to generated HTML file
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(4, 1, figsize=(14, 16), dpi=100)
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

            # 1. Price with KLOW Support Levels
            axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8, color='#2E86AB')

            # Calculate KLOW for visualization if not present
            if 'KLOW' not in data.columns:
                data = data.copy()
                data['KLOW'] = data['low'].rolling(window=20, min_periods=1).min()

            axes[0].plot(data['KLOW'], label='KLOW (20-day low)', linewidth=2, alpha=0.8, color='#A23B72')

            # Fill area between price and KLOW
            axes[0].fill_between(data.index, data['close'], data['KLOW'],
                               where=(data['close'] >= data['KLOW']),
                               color='#2E86AB', alpha=0.3, label='Above KLOW')
            axes[0].fill_between(data.index, data['close'], data['KLOW'],
                               where=(data['close'] < data['KLOW']),
                               color='red', alpha=0.3, label='Below KLOW')

            axes[0].set_title('Price vs KLOW Support Level', fontsize=12)
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 2. KLEN Trend Persistence
            if 'KLEN' not in data.columns:
                data = data.copy()
                data['KLEN'] = self._calculate_klen(data['close'])

            axes[1].plot(data['KLEN'], label='KLEN', linewidth=2, color='#F18F01', alpha=0.8)
            axes[1].axhline(y=5, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Moderate Trend (5)')
            axes[1].axhline(y=10, color='darkgreen', linestyle='--', alpha=0.7, linewidth=1, label='Strong Trend (10)')
            axes[1].axhline(y=15, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Very Strong (15)')

            axes[1].set_title('KLEN - Trend Persistence', fontsize=12)
            axes[1].set_ylabel('Trend Length (days)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # 3. Combined Momentum Analysis
            # Create momentum score combining KLEN and KLOW
            klow_distance = (data['close'] - data['KLOW']) / data['KLOW'] * 100  # Distance from low in %
            klen_normalized = data['KLEN'] / data['KLEN'].rolling(window=20).max()  # Normalized KLEN

            axes[2].plot(klow_distance, label='Distance from KLOW (%)', linewidth=2, color='#2E86AB', alpha=0.8)
            axes[2].plot(klen_normalized * 20, label='KLEN (Normalized)', linewidth=2, color='#A23B72', alpha=0.8)

            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            axes[2].axhline(y=5, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Near Support (5%)')
            axes[2].set_title('Combined Momentum Analysis', fontsize=12)
            axes[2].set_ylabel('Value')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            # 4. KLEN and KLOW Distribution
            fig, ax_dist = plt.subplots(1, 1, figsize=(10, 6))

            momentum_data = pd.DataFrame({
                'KLEN': data['KLEN'].dropna(),
                'KLOW_Distance_%': klow_distance.dropna()
            })

            if len(momentum_data) > 0:
                # Create distribution plots
                for i, col in enumerate(momentum_data.columns):
                    data_col = momentum_data[col].dropna()
                    if len(data_col) > 0:
                        if col == 'KLEN':
                            # Histogram for KLEN
                            ax_dist.hist(data_col, bins=20, alpha=0.7, label=col,
                                       color=['#2E86AB', '#A23B72'][i])
                        else:
                            # KDE for distance (can be any value)
                            try:
                                sns.kdeplot(data=data_col, ax=ax_dist, alpha=0.7, label=col,
                                          color=['#2E86AB', '#A23B72'][i])
                            except:
                                ax_dist.hist(data_col, bins=20, alpha=0.7, label=col,
                                           color=['#2E86AB', '#A23B72'][i])

                ax_dist.axvline(x=momentum_data['KLEN'].mean(), color='#2E86AB',
                              linestyle='--', alpha=0.8, label='KLEN Mean')
                ax_dist.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)

                ax_dist.set_title('Momentum Indicators Distribution', fontsize=12)
                ax_dist.set_xlabel('Value')
                ax_dist.set_ylabel('Frequency')
                ax_dist.legend()
                ax_dist.grid(True, alpha=0.3)

            # Add distribution plot to the figure
            axes[3].remove()
            axes[3] = fig.add_subplot(4, 1, 4)
            axes[3].figure = fig

            # Add statistics summary
            klen_stats = data['KLEN'].describe()
            klow_stats = data['KLOW'].describe()

            stats_text = (
                'KLEN Statistics:\n'
                f'Mean: {klen_stats["mean"]:.1f} days\n'
                f'Max: {klen_stats["max"]:.0f} days\n'
                f'>10 days: {(data["KLEN"] > 10).mean()*100:.1f}%\n\n'
                'KLOW Statistics:\n'
                f'Mean: {klow_stats["mean"]:.2f}\n'
                f'Min: {klow_stats["min"]:.2f}\n'
                f'Avg Distance: {klow_distance.mean():.2f}%'
            )

            axes[3].text(0.5, 0.5, stats_text,
                        transform=axes[3].transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                        ha='center', va='center')
            axes[3].set_title('Momentum Statistics Summary', fontsize=12)
            axes[3].set_xticks([])
            axes[3].set_yticks([])

            plt.tight_layout()

            # Save as HTML
            html_path = self.save_plot_as_html(fig, output_path, "momentum_analysis.html")

            plt.close(fig)
            if 'ax_dist' in locals():
                plt.close(ax_dist.figure)

            logger.info(f"Generated momentum visualization: {html_path}")
            return html_path

        except Exception as e:
            logger.error(f"Error generating momentum visualization: {e}")
            return None

    def _calculate_klen(self, price_series: pd.Series) -> pd.Series:
        """Calculate KLEN (trend length) for visualization."""
        # Calculate price direction changes (1 for up, 0 for down, handling ties)
        price_changes = price_series.diff()
        direction = (price_changes > 0).astype(int)

        # Handle zero changes (ties) by maintaining previous direction
        direction = direction.replace(0, np.nan).ffill().fillna(0).astype(int)

        # Calculate run lengths
        klen_values = np.zeros(len(price_series))

        # Calculate run lengths for each segment
        start_idx = 0
        current_direction = direction.iloc[0] if len(direction) > 0 else 0

        for i in range(1, len(direction)):
            if direction.iloc[i] != current_direction:
                # Direction changed, calculate run length
                run_length = i - start_idx
                klen_values[start_idx:i] = np.arange(run_length, 0, -1)
                start_idx = i
                current_direction = direction.iloc[i]

        # Handle the last segment
        if start_idx < len(price_series):
            run_length = len(price_series) - start_idx
            klen_values[start_idx:] = np.arange(run_length, 0, -1)

        return pd.Series(klen_values, index=price_series.index)

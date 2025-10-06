"""
Volatility Visualizer for VN30 TFT Features

This module provides visualization and explanation for volatility features:
- VSTD5: 5-day volatility (standard deviation of returns)
- STD5: 5-day standard deviation of prices

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


class VolatilityVisualizer(BaseVisualizer):
    """
    Visualizer for volatility features (VSTD and STD).

    Volatility measures the degree of price variation over time.
    VSTD measures return volatility while STD measures price level volatility.
    """

    def __init__(self):
        """Initialize volatility visualizer."""
        super().__init__()
        self.feature_names = ['VSTD5', 'STD5']
        self.title = "Volatility Analysis (VSTD & STD)"

    def generate_explanation(self) -> str:
        """Generate explanation for volatility features."""
        return """
# Volatility Features - Technical Indicator Explanation

## What is Volatility?
**Volatility measures the degree of variation in stock prices over time.** It quantifies how much and how quickly prices change, which is crucial for risk assessment and trading strategy selection.

## Two Types of Volatility:

### VSTD5 (Return Volatility):
- **Standard deviation of price RETURNS** over 5 days
- **Measures percentage price changes**
- **Unitless (represents relative volatility)**
- **Best for**: Risk assessment, position sizing, stop-loss placement

### STD5 (Price Volatility):
- **Standard deviation of absolute PRICES** over 5 days
- **Measures absolute price level variation**
- **In same units as the stock price**
- **Best for**: Support/resistance levels, price targets

## Volatility Scale Interpretation:

### Low Volatility:
- **VSTD5 < 1%**: Very stable, low risk
- **STD5 < 2% of price**: Price moving in narrow range
- **Characteristics**: Predictable, range-bound trading

### Moderate Volatility:
- **VSTD5 1-3%**: Normal market conditions
- **STD5 2-5% of price**: Reasonable price movement
- **Characteristics**: Balanced risk-reward

### High Volatility:
- **VSTD5 > 3%**: Very volatile, high risk
- **STD5 > 5% of price**: Large price swings
- **Characteristics**: Stressful, requires active management

## Trading Interpretation:

### VSTD5 (Return Volatility) Signals:

#### Bullish Signals:
- **Decreasing VSTD5**: Volatility declining, trend stabilizing
- **VSTD5 < 1%**: Very low volatility, potential breakout setup
- **VSTD5 contraction**: Coiling pattern, potential explosive move

#### Bearish Signals:
- **Increasing VSTD5**: Volatility expanding, uncertainty rising
- **VSTD5 > 4%**: Very high volatility, consider reducing exposure
- **VSTD5 spikes**: Panic selling or euphoric buying

### STD5 (Price Volatility) Signals:

#### Support/Resistance:
- **STD5 levels**: Help identify reasonable support/resistance zones
- **Price ± STD5**: Typical daily trading range
- **Price ± 2×STD5**: Extreme daily movement

## Risk Management:

### Position Sizing Based on Volatility:
- **Low VSTD5 (<1%)**: Can use larger position sizes
- **High VSTD5 (>3%)**: Should use smaller position sizes
- **Extreme VSTD5 (>5%)**: Consider avoiding or using very small positions

### Stop Loss Placement:
- **Normal Volatility**: Stop at Price ± 1×ATR or 1.5×STD5
- **High Volatility**: Stop at Price ± 2×ATR or 2.5×STD5
- **Low Volatility**: Stop at Price ± 0.5×ATR or 1×STD5

### Portfolio Allocation:
- **High Volatility Environment**: Reduce overall portfolio risk
- **Low Volatility Environment**: Can increase portfolio leverage
- **Mixed Volatility**: Balance between growth and stability

## Market Regime Identification:

### Low Volatility Regime:
- **Characteristics**: Stable trends, predictable patterns
- **Good for**: Trend following, carry trades, longer-term positions
- **Risk**: Potential breakout risk if volatility suddenly increases

### High Volatility Regime:
- **Characteristics**: Erratic movements, difficult trends
- **Good for**: Short-term trading, volatility harvesting strategies
- **Risk**: Higher probability of large losses

### Volatility Cycles:
- **Volatility tends to cycle**: High volatility periods alternate with low volatility
- **Mean reversion**: Very high volatility often reverts to normal levels
- **Contagion effect**: Volatility can spread between related assets

## Best Practices:

### Multiple Timeframes:
- **Compare VSTD5 with VSTD20**: Short-term vs longer-term volatility
- **Volatility divergence**: Different timeframes showing different trends

### Volatility Indicators:
- **Bollinger Bands**: Use STD5 for band width calculation
- **ATR (Average True Range)**: Alternative volatility measure
- **VIX Comparison**: Compare individual stock volatility to market volatility

### Risk-Adjusted Returns:
- **Sharpe Ratio**: Return ÷ VSTD5 (risk-adjusted performance)
- **Sortino Ratio**: Return ÷ Downside VSTD5 (downside risk focus)
- **Calmar Ratio**: Return ÷ Maximum Drawdown

## VN30 Context:

### Vietnamese Market Volatility:
- **Generally higher volatility** than developed markets
- **Affected by**: Global events, domestic policy, foreign investor flows
- **Currency impact**: VND/USD movements affect volatility
- **Liquidity effects**: Lower liquidity can amplify volatility

### Trading Implications:
- **Wider stops**: Often needed due to higher volatility
- **Smaller positions**: Consider reducing size in volatile periods
- **Active management**: More frequent monitoring may be required
"""

    def generate_visualization(self, data: pd.DataFrame, output_path: str, **kwargs) -> str:
        """
        Generate volatility visualization.

        Args:
            data: DataFrame with price and volatility data
            output_path: Path to save the HTML file
            **kwargs: Additional arguments

        Returns:
            Path to generated HTML file
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(4, 1, figsize=(14, 16), dpi=100)
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

            # 1. Price with Volatility Overlay
            axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8, color='#2E86AB')

            # Calculate volatility measures for visualization
            returns = data['close'].pct_change()
            vstd5 = returns.rolling(window=5, min_periods=3).std() * 100  # Convert to percentage
            std5 = data['close'].rolling(window=5, min_periods=3).std()

            # Plot volatility on secondary axis
            ax_twin = axes[0].twinx()
            ax_twin.plot(vstd5, label='VSTD5 (%)', linewidth=2, alpha=0.8, color='#A23B72')
            ax_twin.fill_between(data.index, vstd5, alpha=0.3, color='#A23B72')

            axes[0].set_title('Price with Return Volatility (VSTD5)', fontsize=12)
            axes[0].set_ylabel('Price', color='#2E86AB')
            axes[0].tick_params(axis='y', labelcolor='#2E86AB')
            ax_twin.set_ylabel('VSTD5 (%)', color='#A23B72')
            ax_twin.tick_params(axis='y', labelcolor='#A23B72')

            # Add legend
            lines1, labels1 = axes[0].get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            axes[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            axes[0].grid(True, alpha=0.3)

            # 2. Price Volatility (STD5)
            ax2_twin = axes[1].twinx()
            ax2_twin.plot(std5, label='STD5', linewidth=2, alpha=0.8, color='#F18F01')
            ax2_twin.fill_between(data.index, std5, alpha=0.3, color='#F18F01')

            axes[1].set_title('Price Volatility (STD5)', fontsize=12)
            axes[1].set_ylabel('Price')
            ax2_twin.set_ylabel('STD5', color='#F18F01')
            ax2_twin.tick_params(axis='y', labelcolor='#F18F01')
            ax2_twin.legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)

            # 3. Volatility Comparison
            if 'VSTD5' in data.columns and 'STD5' in data.columns:
                # Use actual features if available
                vstd5_actual = data['VSTD5']
                std5_actual = data['STD5']
            else:
                # Calculate for visualization
                vstd5_actual = returns.rolling(window=5, min_periods=3).std()
                std5_actual = data['close'].rolling(window=5, min_periods=3).std()

            # Normalize for comparison (z-score)
            vstd5_norm = (vstd5_actual - vstd5_actual.mean()) / vstd5_actual.std()
            std5_norm = (std5_actual - std5_actual.mean()) / std5_actual.std()

            axes[2].plot(vstd5_norm, label='VSTD5 (Normalized)', linewidth=2, alpha=0.8, color='#2E86AB')
            axes[2].plot(std5_norm, label='STD5 (Normalized)', linewidth=2, alpha=0.8, color='#A23B72')

            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            axes[2].set_title('Normalized Volatility Comparison', fontsize=12)
            axes[2].set_ylabel('Standard Deviations')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            # 4. Volatility Distribution
            fig, ax_dist = plt.subplots(1, 1, figsize=(10, 6))

            volatility_data = pd.DataFrame({
                'VSTD5': vstd5_actual.dropna() * 100,  # Convert to percentage for display
                'STD5': std5_actual.dropna()
            })

            if len(volatility_data) > 0:
                # Create distribution plot
                for i, col in enumerate(volatility_data.columns):
                    data_col = volatility_data[col].dropna()
                    if len(data_col) > 0:
                        sns.histplot(data=data_col, ax=ax_dist, alpha=0.7, label=col,
                                   color=[ '#2E86AB', '#A23B72'][i])

                ax_dist.axvline(x=volatility_data['VSTD5'].mean(), color='#2E86AB',
                              linestyle='--', alpha=0.8, label='VSTD5 Mean')
                ax_dist.axvline(x=volatility_data['STD5'].mean(), color='#A23B72',
                              linestyle='--', alpha=0.8, label='STD5 Mean')

                ax_dist.set_title('Volatility Distribution Analysis', fontsize=12)
                ax_dist.set_xlabel('Volatility Value')
                ax_dist.set_ylabel('Frequency')
                ax_dist.legend()
                ax_dist.grid(True, alpha=0.3)

            # Add distribution plot to the figure
            axes[3].remove()  # Remove empty subplot
            axes[3] = fig.add_subplot(4, 1, 4)
            axes[3].figure = fig  # This is a workaround to add the distribution plot

            # Actually, let's create a simpler approach for subplot 4
            vstd5_stats = volatility_data["VSTD5"]
            std5_stats = volatility_data["STD5"]

            stats_text = (
                'VSTD5 Stats:\n'
                f'Mean: {vstd5_stats.mean():.3f}\n'
                f'Std: {vstd5_stats.std():.3f}\n'
                f'Max: {vstd5_stats.max():.3f}\n\n'
                'STD5 Stats:\n'
                f'Mean: {std5_stats.mean():.3f}\n'
                f'Std: {std5_stats.std():.3f}\n'
                f'Max: {std5_stats.max():.3f}'
            )

            axes[3].text(0.5, 0.5, stats_text,
                        transform=axes[3].transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                        ha='center', va='center')
            axes[3].set_title('Volatility Statistics Summary', fontsize=12)
            axes[3].set_xticks([])
            axes[3].set_yticks([])

            plt.tight_layout()

            # Save as HTML
            html_path = self.save_plot_as_html(fig, output_path, "volatility_analysis.html")

            plt.close(fig)
            if 'ax_dist' in locals():
                plt.close(ax_dist.figure)

            logger.info(f"Generated volatility visualization: {html_path}")
            return html_path

        except Exception as e:
            logger.error(f"Error generating volatility visualization: {e}")
            return None

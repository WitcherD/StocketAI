"""
ROC (Rate of Change) Visualizer for VN30 TFT Features

This module provides visualization and explanation for ROC features:
- ROC60: 60-day rate of change

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


class ROCVisualizer(BaseVisualizer):
    """
    Visualizer for ROC (Rate of Change) features.

    Rate of Change measures the percentage change in price over a specific period,
    helping identify momentum and trend strength.
    """

    def __init__(self):
        """Initialize ROC visualizer."""
        super().__init__()
        self.feature_names = ['ROC60']
        self.title = "ROC (Rate of Change) Analysis"

    def generate_explanation(self) -> str:
        """Generate explanation for ROC features."""
        return """
# ROC (Rate of Change) - Technical Indicator Explanation

## What is Rate of Change?
**ROC measures the percentage change in stock price over a specific period.** It tells you how much the price has changed relative to its value at the beginning of the period, helping identify momentum and the speed of price movements.

## ROC Formula:
**ROC = ((Current Price / Price N periods ago) - 1) × 100**

For ROC60: **ROC60 = ((Today\'s Close / Close 60 days ago) - 1) × 100**

## ROC Scale Interpretation:
- **ROC = +50%**: Price has increased 50% over the period
- **ROC = 0%**: Price unchanged from period start
- **ROC = -30%**: Price has decreased 30% over the period

## How to Understand ROC Values:

### Positive ROC (Bullish Momentum):
- **ROC > 0**: Price is higher than at the start of the period
- **ROC > 10%**: Strong upward momentum
- **ROC > 25%**: Very strong bullish trend
- **ROC > 50%**: Exceptional bullish performance

### Negative ROC (Bearish Momentum):
- **ROC < 0**: Price is lower than at the start of the period
- **ROC < -10%**: Strong downward momentum
- **ROC < -25%**: Very strong bearish trend
- **ROC < -50%**: Exceptional bearish performance

### ROC Trends:
- **Rising ROC**: Accelerating momentum (increasing speed of price change)
- **Falling ROC**: Decelerating momentum (decreasing speed of price change)
- **ROC near zero**: Sideways movement, no clear direction

## Trading Interpretation:

### Bullish Signals:
- **ROC crosses above zero**: Momentum shifting from negative to positive
- **ROC > 10% and rising**: Strong upward momentum
- **ROC divergence**: ROC making higher highs while price makes lower highs

### Bearish Signals:
- **ROC crosses below zero**: Momentum shifting from positive to negative
- **ROC < -10% and falling**: Strong downward momentum
- **ROC divergence**: ROC making lower lows while price makes higher lows

### Overbought/Oversold Conditions:
- **ROC > 30%**: Potentially overbought (but can remain overbought in strong trends)
- **ROC < -30%**: Potentially oversold (but can remain oversold in strong downtrends)
- **Extreme ROC (>50% or <-50%)**: Often precedes reversals or consolidation

## ROC vs Price Movement:

### Trend Identification:
- **Sustained positive ROC**: Confirms uptrend
- **Sustained negative ROC**: Confirms downtrend
- **Oscillating ROC around zero**: Sideways/choppy market

### Momentum Strength:
- **High positive ROC**: Strong buying pressure
- **High negative ROC**: Strong selling pressure
- **ROC near zero**: Weak momentum, potential reversal

## Risk Management:

### Position Sizing:
- **High |ROC|**: Consider larger positions (strong momentum)
- **Low |ROC|**: Consider smaller positions (weak momentum)
- **Extreme ROC**: Be cautious of potential reversals

### Stop Loss Placement:
- **Strong ROC trends**: Can use wider stops
- **Weak ROC trends**: Should use tighter stops
- **ROC reversals**: Good signals for stop adjustment

## ROC Periods and Timeframes:

### ROC60 (60-day):
- **Long-term momentum**: Captures broader trend direction
- **Less sensitive**: Smooths out short-term noise
- **Strategic view**: Good for position trading and investment decisions
- **Less frequent signals**: Fewer but more reliable trading signals

### Comparing Multiple ROC Periods:
- **ROC5 vs ROC60**: Short-term vs long-term momentum
- **Convergence**: Both periods showing similar momentum
- **Divergence**: Different momentum directions (potential reversal signal)

## Best Practices:
- **Combine with price action**: ROC works best with trend confirmation
- **Multiple timeframes**: Use different ROC periods for complete picture
- **Avoid extremes**: Very high/low ROC often precede reversals
- **Context matters**: ROC interpretation depends on overall market conditions
"""

    def generate_visualization(self, data: pd.DataFrame, output_path: str, **kwargs) -> str:
        """
        Generate ROC visualization.

        Args:
            data: DataFrame with price and ROC data
            output_path: Path to save the HTML file
            **kwargs: Additional arguments

        Returns:
            Path to generated HTML file
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=100)
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

            # 1. Price and ROC Comparison
            axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8, color='#2E86AB')

            # Calculate ROC for visualization if not present
            if 'ROC60' not in data.columns:
                data = data.copy()
                data['ROC60'] = (data['close'] / data['close'].shift(60) - 1) * 100

            # Plot ROC on secondary axis
            ax_twin = axes[0].twinx()
            ax_twin.plot(data['ROC60'], label='ROC60', linewidth=2, alpha=0.8, color='#A23B72')
            ax_twin.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)

            axes[0].set_title('Price vs Rate of Change (ROC60)', fontsize=12)
            axes[0].set_ylabel('Price', color='#2E86AB')
            axes[0].tick_params(axis='y', labelcolor='#2E86AB')
            ax_twin.set_ylabel('ROC60 (%)', color='#A23B72')
            ax_twin.tick_params(axis='y', labelcolor='#A23B72')

            # Add legend
            lines1, labels1 = axes[0].get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            axes[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            axes[0].grid(True, alpha=0.3)

            # 2. ROC with Signal Levels
            axes[1].plot(data['ROC60'], label='ROC60', linewidth=2, color='#F18F01', alpha=0.8)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)

            # Add signal level zones
            axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Overbought (30%)')
            axes[1].axhline(y=-30, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Oversold (-30%)')
            axes[1].axhline(y=10, color='darkgreen', linestyle=':', alpha=0.7, linewidth=1, label='Strong (10%)')
            axes[1].axhline(y=-10, color='darkred', linestyle=':', alpha=0.7, linewidth=1, label='Weak (-10%)')

            # Fill zones
            axes[1].fill_between(data.index, data['ROC60'], 30,
                               where=(data['ROC60'] >= 30), color='green', alpha=0.2)
            axes[1].fill_between(data.index, data['ROC60'], -30,
                               where=(data['ROC60'] <= -30), color='red', alpha=0.2)

            axes[1].set_ylim(-60, 60)  # Set reasonable limits
            axes[1].set_title('ROC with Signal Levels', fontsize=12)
            axes[1].set_ylabel('ROC60 (%)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # 3. ROC Distribution and Statistics
            roc_data = data['ROC60'].dropna()

            if len(roc_data) > 0:
                # Histogram
                axes[2].hist(roc_data, bins=30, alpha=0.7, color='#C73E1D', density=True)

                # Add kernel density estimate
                try:
                    roc_clean = roc_data.dropna()
                    if len(roc_clean) > 10:
                        sns.kdeplot(data=roc_clean, ax=axes[2], color='black', linewidth=2)
                except:
                    pass  # KDE failed, continue without it

                # Add vertical lines for key levels
                axes[2].axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
                axes[2].axvline(x=30, color='green', linestyle='--', alpha=0.7, linewidth=1)
                axes[2].axvline(x=-30, color='red', linestyle='--', alpha=0.7, linewidth=1)
                axes[2].axvline(x=10, color='darkgreen', linestyle=':', alpha=0.7, linewidth=1)
                axes[2].axvline(x=-10, color='darkred', linestyle=':', alpha=0.7, linewidth=1)

                axes[2].set_xlim(-60, 60)
                axes[2].set_title('ROC Distribution Analysis', fontsize=12)
                axes[2].set_xlabel('ROC60 (%)')
                axes[2].set_ylabel('Density')
                axes[2].legend(['KDE', 'Zero', 'Overbought', 'Oversold', 'Strong', 'Weak'])
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save as HTML
            html_path = self.save_plot_as_html(fig, output_path, "roc_analysis.html")

            plt.close(fig)
            logger.info(f"Generated ROC visualization: {html_path}")
            return html_path

        except Exception as e:
            logger.error(f"Error generating ROC visualization: {e}")
            return None

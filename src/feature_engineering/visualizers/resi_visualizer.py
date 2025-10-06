"""
RESI (Residuals) Visualizer for VN30 TFT Features

This module provides visualization and explanation for RESI features:
- RESI5: 5-day residual (close - MA(close, 5))
- RESI10: 10-day residual (close - MA(close, 10))

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


class RESIVisualizer(BaseVisualizer):
    """
    Visualizer for RESI (Residuals) features.

    Residuals represent the difference between actual price and moving average,
    helping identify overbought/oversold conditions and trend strength.
    """

    def __init__(self):
        """Initialize RESI visualizer."""
        super().__init__()
        self.feature_names = ['RESI5', 'RESI10']
        self.title = "RESI (Residuals) Analysis"

    def generate_explanation(self) -> str:
        """Generate explanation for RESI features."""
        return """
# RESI (Residuals) - Technical Indicator Explanation

## What are Residuals?
Residuals in stock analysis represent the **difference between the actual stock price and a moving average** of that price. Think of it as measuring how much the current price deviates from its recent trend.

## How to Understand RESI Values:

### RESI5 (5-day Residual):
- **Positive Value (+)**: Stock price is ABOVE its 5-day moving average
- **Negative Value (-)**: Stock price is BELOW its 5-day moving average
- **Large Positive**: Stock is significantly overbought or in a strong uptrend
- **Large Negative**: Stock is significantly oversold or in a strong downtrend

### RESI10 (10-day Residual):
- **Positive Value (+)**: Stock price is ABOVE its 10-day moving average
- **Negative Value (-)**: Stock price is BELOW its 10-day moving average
- **Large Positive**: Longer-term overbought condition
- **Large Negative**: Longer-term oversold condition

## Trading Interpretation:

### Bullish Signals:
- RESI5 crosses above zero (price moves above short-term trend)
- Both RESI5 and RESI10 are positive and rising
- RESI values expanding upward during uptrend

### Bearish Signals:
- RESI5 crosses below zero (price moves below short-term trend)
- Both RESI5 and RESI10 are negative and falling
- RESI values expanding downward during downtrend

### Mean Reversion:
- Large positive RESI often precedes price corrections
- Large negative RESI often precedes price recoveries

## Risk Management:
- **High Risk**: Large positive RESI in overbought territory
- **Low Risk**: RESI near zero (price aligned with trend)
- **Opportunity**: Large negative RESI in oversold conditions
"""

    def generate_visualization(self, data: pd.DataFrame, output_path: str, **kwargs) -> str:
        """
        Generate RESI visualization.

        Args:
            data: DataFrame with price and RESI data
            output_path: Path to save the HTML file
            **kwargs: Additional arguments

        Returns:
            Path to generated HTML file
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=100)
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

            # 1. Price and Moving Averages
            axes[0].plot(data['close'], label='Close Price', linewidth=2, alpha=0.8)

            # Calculate moving averages for visualization
            ma5 = data['close'].rolling(window=5).mean()
            ma10 = data['close'].rolling(window=10).mean()

            axes[0].plot(ma5, label='5-day MA', linestyle='--', linewidth=1.5)
            axes[0].plot(ma10, label='10-day MA', linestyle='--', linewidth=1.5)

            axes[0].set_title('Price vs Moving Averages', fontsize=12)
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 2. RESI5 and RESI10
            colors = ['#2E86AB', '#A23B72']
            axes[1].plot(data['RESI5'], label='RESI5', color=colors[0], linewidth=2)
            axes[1].plot(data['RESI10'], label='RESI10', color=colors[1], linewidth=2)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            axes[1].fill_between(data.index, data['RESI5'], 0,
                               where=(data['RESI5'] >= 0), color=colors[0], alpha=0.3)
            axes[1].fill_between(data.index, data['RESI5'], 0,
                               where=(data['RESI5'] < 0), color='red', alpha=0.3)

            axes[1].set_title('RESI Values (Residuals)', fontsize=12)
            axes[1].set_ylabel('Residual Value')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # 3. RESI Distribution
            resi_combined = pd.concat([data['RESI5'], data['RESI10']], axis=1)
            resi_combined.columns = ['RESI5', 'RESI10']

            sns.histplot(data=resi_combined, ax=axes[2], alpha=0.7)
            axes[2].axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            axes[2].set_title('RESI Distribution Analysis', fontsize=12)
            axes[2].set_xlabel('Residual Value')
            axes[2].set_ylabel('Frequency')
            axes[2].legend(['Zero Line', 'RESI5', 'RESI10'])

            plt.tight_layout()

            # Save as HTML
            html_path = self.save_plot_as_html(fig, output_path, "resi_analysis.html")

            plt.close(fig)
            logger.info(f"Generated RESI visualization: {html_path}")
            return html_path

        except Exception as e:
            logger.error(f"Error generating RESI visualization: {e}")
            return None

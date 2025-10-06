"""
Temporal Features Visualizer for VN30 TFT Features

This module provides visualization and explanation for temporal features:
- month: Calendar month (1-12)
- day_of_week: Day of week (0-6, Monday-Sunday)
- year: Calendar year

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


class TemporalVisualizer(BaseVisualizer):
    """
    Visualizer for temporal features (KNOWN_INPUT for TFT).

    Temporal features capture calendar-based patterns that may influence
    stock price movements due to seasonal effects, day-of-week patterns, etc.
    """

    def __init__(self):
        """Initialize temporal visualizer."""
        super().__init__()
        self.feature_names = ['month', 'day_of_week', 'year']
        self.title = "Temporal Features Analysis"

    def generate_explanation(self) -> str:
        """Generate explanation for temporal features."""
        return """
# Temporal Features - Technical Indicator Explanation

## What are Temporal Features?
**Temporal features capture time-based patterns** in stock data that might influence price movements. These include calendar effects like month of year, day of week, and yearly trends that can reveal seasonal patterns in trading behavior.

## Feature Explanations:

### Month (1-12)
- **Calendar Month**: January = 1, December = 12
- **Seasonal Patterns**: Different months may show different return patterns
- **Quarterly Effects**: End of quarter months (3, 6, 9, 12) often show different behavior
- **Holiday Effects**: Months with major holidays may have unique patterns

### Day of Week (0-6)
- **0 = Monday, 1 = Tuesday, ..., 6 = Sunday**
- **Weekly Patterns**: Different days may show different trading behavior
- **Monday Effect**: Often different from other days due to weekend news
- **Friday Effect**: End-of-week position adjustments

### Year
- **Calendar Year**: Full year number (e.g., 2024)
- **Long-term Trends**: Secular trends over multiple years
- **Economic Cycles**: Business cycle effects across years
- **Market Regime Changes**: Different years may have different characteristics

## Trading Interpretation:

### Monthly Patterns:
- **Month-end Effects**: Often higher volume and price pressure
- **Quarter-end Effects**: Even stronger than month-end (March, June, September, December)
- **Holiday Months**: May show reduced volume or different patterns
- **Earnings Seasons**: Months 1-2, 4-5, 7-8, 10-11 often have higher volatility

### Weekly Patterns:
- **Monday**: Often reflects weekend news, can be volatile
- **Tuesday-Thursday**: Usually most stable trading days
- **Friday**: Often shows end-of-week position adjustments
- **Weekend Effect**: Monday often different from Friday close

### Yearly Patterns:
- **Election Years**: Often show different volatility patterns
- **Economic Cycles**: Different years in business cycle have different characteristics
- **Market Regimes**: Bull vs bear markets show different yearly patterns

## Seasonal Effects:

### Calendar Anomalies:
- **January Effect**: Small stocks often outperform in January
- **December Effect**: Tax-loss harvesting can affect late December
- **Summer Effect**: Often lower volume and different patterns
- **Holiday Effects**: Reduced volume around major holidays

### Business Cycle Effects:
- **Early Cycle**: Different patterns than late cycle
- **Recession Years**: Significantly different behavior
- **Recovery Years**: Unique return patterns

## Risk Management:

### Monthly Considerations:
- **Month-end Risk**: Higher volatility around month/quarter ends
- **Holiday Risk**: Unusual patterns around holidays
- **Earnings Risk**: Higher volatility during earnings seasons

### Weekly Considerations:
- **Monday Risk**: Often higher gap risk due to weekend news
- **Friday Risk**: End-of-week position adjustments can increase volatility

## Best Practices:
- **Long-term Data**: Need multiple years to identify reliable patterns
- **Statistical Significance**: Test patterns for statistical significance
- **Market Conditions**: Patterns may change in different market regimes
- **Risk Management**: Don't rely solely on calendar effects for trading decisions

## Common Patterns in VN30:

### Monthly Patterns:
- **Month-end Pressure**: Last few days of month often show directional moves
- **Quarterly Rebalancing**: End of March, June, September, December
- **Holiday Effects**: Tet holiday in January/February affects Vietnamese markets

### Weekly Patterns:
- **Monday Volatility**: Often gaps reflecting weekend news
- **Friday Settlement**: End-of-week position adjustments
- **Mid-week Stability**: Tuesday-Thursday often most stable

### Yearly Patterns:
- **Economic Cycles**: Different years show different volatility patterns
- **Policy Changes**: Government policy changes can affect yearly patterns
- **Global Events**: International events affect Vietnamese market patterns
"""

    def generate_visualization(self, data: pd.DataFrame, output_path: str, **kwargs) -> str:
        """
        Generate temporal features visualization.

        Args:
            data: DataFrame with temporal features data
            output_path: Path to save the HTML file
            **kwargs: Additional arguments

        Returns:
            Path to generated HTML file
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(3, 2, figsize=(16, 12), dpi=100)
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

            # Ensure datetime conversion
            if 'time' in data.columns:
                data = data.copy()
                data['date'] = pd.to_datetime(data['time'])
            elif 'date' in data.columns:
                data = data.copy()
                data['date'] = pd.to_datetime(data['date'])
            else:
                # Generate sample datetime for visualization
                data['date'] = pd.date_range('2024-01-01', periods=len(data), freq='D')

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

            # 5. Yearly Trend (if multiple years available)
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

            # 6. Temporal Heatmap (Month vs Day of Week)
            if 'volume' in data.columns:
                heatmap_data = data.pivot_table(values='volume', index='day_of_week', columns='month', aggfunc='mean')

                sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
                           square=True, ax=axes[2, 1])
                axes[2, 1].set_title('Volume Heatmap: Day of Week vs Month', fontsize=12)
                axes[2, 1].set_xlabel('Month')
                axes[2, 1].set_ylabel('Day of Week (0=Monday)')
            else:
                axes[2, 1].text(0.5, 0.5, 'Volume data required for heatmap',
                               ha='center', va='center', transform=axes[2, 1].transAxes,
                               fontsize=10, alpha=0.7)
                axes[2, 1].set_title('Volume Heatmap (No Volume Data)')

            plt.tight_layout()

            # Save as HTML
            html_path = self.save_plot_as_html(fig, output_path, "temporal_analysis.html")

            plt.close(fig)
            logger.info(f"Generated temporal visualization: {html_path}")
            return html_path

        except Exception as e:
            logger.error(f"Error generating temporal visualization: {e}")
            return None

"""
Static Features Visualizer for VN30 TFT Features

This module provides visualization and explanation for static features:
- const: Constant value (1.0) for TFT compatibility

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


class StaticVisualizer(BaseVisualizer):
    """
    Visualizer for static features (STATIC_INPUT for TFT).

    Static features are constant values that don't change over time,
    used primarily for TFT model architecture compatibility.
    """

    def __init__(self):
        """Initialize static visualizer."""
        super().__init__()
        self.feature_names = ['const']
        self.title = "Static Features Analysis"

    def generate_explanation(self) -> str:
        """Generate explanation for static features."""
        return """
# Static Features - Technical Indicator Explanation

## What are Static Features?
**Static features are CONSTANT values that don't change over time.** In the context of TFT (Temporal Fusion Transformers), these features serve important architectural purposes but don't provide trading signals themselves.

## The Constant Feature (const):

### Definition:
- **const = 1.0** (constant value for all time periods)
- **Never changes**: Same value throughout entire dataset
- **TFT requirement**: Required by TFT model architecture
- **Baseline feature**: Provides reference point for other features

## Why Static Features are Important:

### TFT Architecture Requirements:
- **Model input structure**: TFT expects static inputs
- **Feature normalization**: Provides baseline for scaling
- **Attention mechanisms**: Helps with temporal attention calculations
- **Model stability**: Improves numerical stability

### Technical Benefits:
- **Consistent input size**: Ensures all samples have same feature dimensions
- **Baseline comparison**: Other features measured relative to constant
- **Model convergence**: Helps with training stability
- **Feature completeness**: Satisfies TFT input requirements

## Trading Interpretation:

### No Direct Trading Signals:
- **Not a trading indicator**: Const doesn't provide buy/sell signals
- **No predictive power**: Constant value has no forecasting ability
- **Supporting role**: Enables TFT model to function properly

### Indirect Benefits:
- **Model quality**: Better model performance due to proper architecture
- **Feature interactions**: May influence how other features are processed
- **Prediction stability**: Helps maintain consistent model outputs

## Risk Management:

### No Risk Management Applications:
- **Not for position sizing**: Const doesn't indicate risk levels
- **Not for stop placement**: No support/resistance information
- **Not for timing**: No entry/exit timing information

### Supporting Role in Risk Management:
- **Model reliability**: Better models make better risk assessments
- **Stable predictions**: More consistent risk calculations
- **Architecture benefits**: Proper model structure improves all outputs

## Best Practices:

### Implementation:
- **Always include**: Required for TFT compatibility
- **Set to 1.0**: Standard value for baseline features
- **No preprocessing**: Don't normalize or transform const
- **Maintain consistency**: Same value across all samples

### Model Training:
- **Feature engineering**: Const helps with feature scaling
- **Attention weights**: Influences temporal attention mechanisms
- **Model interpretation**: Provides baseline for feature importance

## VN30 Context:

### Vietnamese Market Considerations:
- **TFT compatibility**: Essential for VN30 prediction models
- **Model architecture**: Enables advanced deep learning approaches
- **Research applications**: Supports academic and professional research
- **Industry standards**: Follows modern quantitative finance practices

### Technical Implementation:
- **Data pipeline**: Include const in all feature matrices
- **Model input**: Ensure proper formatting for TFT
- **Feature validation**: Verify const = 1.0 throughout dataset
- **Quality assurance**: Maintain data integrity

## Future Applications:

### Advanced Model Development:
- **Enhanced TFT models**: Foundation for more complex architectures
- **Feature research**: Baseline for testing new features
- **Model comparison**: Consistent benchmark across models
- **Production deployment**: Industry-standard implementation

### Research Opportunities:
- **Feature importance**: Study how const affects model behavior
- **Architecture optimization**: Research alternative static features
- **Model interpretability**: Understand TFT internal mechanics
- **Performance benchmarking**: Compare different static feature approaches
"""

    def generate_visualization(self, data: pd.DataFrame, output_path: str, **kwargs) -> str:
        """
        Generate static features visualization.

        Args:
            data: DataFrame with static features data
            output_path: Path to save the HTML file
            **kwargs: Additional arguments

        Returns:
            Path to generated HTML file
        """

        # Add const feature if not present
        if 'const' not in data.columns:
            data = data.copy()
            data['const'] = 1.0

        try:
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=100)
            fig.suptitle(self.title, fontsize=16, fontweight='bold')

            # 1. Const Feature Over Time
            axes[0].plot(data['const'], label='const = 1.0', linewidth=2, color='#2E86AB', alpha=0.8)
            axes[0].axhline(y=1.0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Expected Value (1.0)')

            axes[0].set_title('Static Feature: const = 1.0', fontsize=12)
            axes[0].set_ylabel('Value')
            axes[0].set_ylim(0.99, 1.01)  # Zoom in to show it's constant
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 2. Const Distribution (should be single value)
            axes[1].hist(data['const'], bins=1, alpha=0.7, color='#A23B72', rwidth=0.8)
            axes[1].axvline(x=1.0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Value = 1.0')
            axes[1].set_xlim(0.99, 1.01)  # Zoom in to show single value
            axes[1].set_title('Const Feature Distribution', fontsize=12)
            axes[1].set_xlabel('Value')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # 3. Const Statistics and Information
            const_stats = data['const'].describe()

            # Create informative text
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

            # Save as HTML
            html_path = self.save_plot_as_html(fig, output_path, "static_analysis.html")

            plt.close(fig)
            logger.info(f"Generated static visualization: {html_path}")
            return html_path

        except Exception as e:
            logger.error(f"Error generating static visualization: {e}")
            return None

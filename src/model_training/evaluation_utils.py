"""
Evaluation Utilities for StocketAI

Provides comprehensive evaluation metrics and benchmarking utilities for baseline models.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class EvaluationUtils:
    """
    Utilities for model evaluation and performance benchmarking.
    """

    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of evaluation metrics
        """
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        smape = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

        # Directional accuracy (for financial returns)
        directional_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

        # Information Coefficient (IC) - correlation between predictions and true values
        ic = np.corrcoef(y_true, y_pred)[0, 1]

        # Rank IC - rank correlation
        true_ranks = pd.Series(y_true).rank()
        pred_ranks = pd.Series(y_pred).rank()
        rank_ic = true_ranks.corr(pred_ranks)

        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "smape": float(smape),
            "directional_accuracy": float(directional_accuracy),
            "ic": float(ic),
            "rank_ic": float(rank_ic)
        }

        return metrics

    @staticmethod
    def calculate_quantile_metrics(
        y_true: np.ndarray,
        y_pred_quantiles: Dict[float, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate quantile-specific metrics for TFT model.

        Args:
            y_true: True values
            y_pred_quantiles: Dictionary of quantile predictions

        Returns:
            Dictionary of quantile metrics
        """
        quantile_metrics = {}

        for quantile, y_pred in y_pred_quantiles.items():
            # Quantile loss
            quantile_loss = np.mean(
                np.where(y_true >= y_pred,
                        quantile * (y_true - y_pred),
                        (1 - quantile) * (y_pred - y_true))
            )

            # Quantile MAE
            quantile_mae = np.mean(np.abs(y_true - y_pred))

            quantile_metrics[f"quantile_{quantile}_loss"] = float(quantile_loss)
            quantile_metrics[f"quantile_{quantile}_mae"] = float(quantile_mae)

        return quantile_metrics

    @staticmethod
    def calculate_financial_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        benchmark_return: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate financial-specific metrics.

        Args:
            y_true: True returns
            y_pred: Predicted returns
            benchmark_return: Benchmark return for comparison

        Returns:
            Dictionary of financial metrics
        """
        # Cumulative returns
        cum_true = np.cumprod(1 + y_true)
        cum_pred = np.cumprod(1 + y_pred)

        # Sharpe ratio (assuming daily returns, annualized)
        if len(y_true) > 1:
            sharpe_true = np.sqrt(252) * np.mean(y_true) / np.std(y_true)
            sharpe_pred = np.sqrt(252) * np.mean(y_pred) / np.std(y_pred)
        else:
            sharpe_true = sharpe_pred = 0.0

        # Maximum drawdown
        def max_drawdown(returns):
            cum_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / running_max
            return np.min(drawdown)

        mdd_true = max_drawdown(y_true)
        mdd_pred = max_drawdown(y_pred)

        # Win rate (percentage of correct directional predictions)
        win_rate = np.mean(np.sign(y_true) == np.sign(y_pred))

        # Profit factor
        gains = y_pred[y_true > 0]
        losses = y_pred[y_true < 0]
        profit_factor = (np.sum(gains) / len(gains)) / (np.abs(np.sum(losses)) / len(losses)) if len(losses) > 0 else np.inf

        metrics = {
            "cumulative_return_true": float(cum_true[-1] - 1),
            "cumulative_return_pred": float(cum_pred[-1] - 1),
            "sharpe_ratio_true": float(sharpe_true),
            "sharpe_ratio_pred": float(sharpe_pred),
            "max_drawdown_true": float(mdd_true),
            "max_drawdown_pred": float(mdd_pred),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor)
        }

        # Benchmark comparison
        if benchmark_return is not None:
            metrics["benchmark_alpha"] = float(cum_pred[-1] - 1 - benchmark_return)

        return metrics

    @staticmethod
    def evaluate_model_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_type: str = "regression",
        y_pred_quantiles: Optional[Dict[float, np.ndarray]] = None,
        benchmark_return: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_type: Type of model evaluation
            y_pred_quantiles: Quantile predictions (for TFT)
            benchmark_return: Benchmark return for financial metrics

        Returns:
            Comprehensive evaluation metrics
        """
        # Basic regression metrics
        metrics = EvaluationUtils.calculate_regression_metrics(y_true, y_pred)

        # Quantile metrics for TFT
        if y_pred_quantiles and model_type.lower() == "tft":
            quantile_metrics = EvaluationUtils.calculate_quantile_metrics(y_true, y_pred_quantiles)
            metrics.update(quantile_metrics)

        # Financial metrics
        financial_metrics = EvaluationUtils.calculate_financial_metrics(y_true, y_pred, benchmark_return)
        metrics.update(financial_metrics)

        # Model-specific metrics
        if model_type.lower() == "lightgbm":
            # Feature importance metrics can be added here
            pass
        elif model_type.lower() == "lstm":
            # Sequence-specific metrics can be added here
            pass

        return metrics

    @staticmethod
    def create_performance_report(
        symbol: str,
        model_metrics: Dict[str, Dict[str, float]],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create performance comparison report.

        Args:
            symbol: Stock symbol
            model_metrics: Dictionary of model metrics
            output_path: Optional path to save report

        Returns:
            Performance report DataFrame
        """
        # Create report DataFrame
        report_data = []
        for model_type, metrics in model_metrics.items():
            row = {"symbol": symbol, "model_type": model_type}
            row.update(metrics)
            report_data.append(row)

        report_df = pd.DataFrame(report_data)

        # Save report if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            report_df.to_csv(output_path, index=False)
            logger.info(f"Performance report saved to {output_path}")

        return report_df

    @staticmethod
    def aggregate_symbol_performance(
        symbol_reports: List[pd.DataFrame],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate performance across symbols.

        Args:
            symbol_reports: List of symbol performance reports
            output_path: Optional path to save aggregated report

        Returns:
            Aggregated performance report
        """
        if not symbol_reports:
            return pd.DataFrame()

        # Concatenate all reports
        combined_df = pd.concat(symbol_reports, ignore_index=True)

        # Calculate aggregate statistics
        agg_functions = {
            'mse': ['mean', 'std', 'min', 'max'],
            'rmse': ['mean', 'std', 'min', 'max'],
            'mae': ['mean', 'std', 'min', 'max'],
            'r2': ['mean', 'std', 'min', 'max'],
            'ic': ['mean', 'std', 'min', 'max'],
            'rank_ic': ['mean', 'std', 'min', 'max'],
            'directional_accuracy': ['mean', 'std', 'min', 'max'],
            'win_rate': ['mean', 'std', 'min', 'max']
        }

        # Group by model_type and calculate statistics
        aggregated = combined_df.groupby('model_type').agg(agg_functions)

        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]

        # Save aggregated report
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            aggregated.to_csv(output_path)
            logger.info(f"Aggregated performance report saved to {output_path}")

        return aggregated

    @staticmethod
    def generate_benchmark_comparison(
        baseline_metrics: Dict[str, float],
        enhanced_metrics: Dict[str, float],
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Generate benchmark comparison between baseline and enhanced models.

        Args:
            baseline_metrics: Baseline model metrics
            enhanced_metrics: Enhanced model metrics
            metric_names: Specific metrics to compare

        Returns:
            Comparison metrics dictionary
        """
        if metric_names is None:
            metric_names = ['mse', 'rmse', 'mae', 'r2', 'ic', 'rank_ic', 'directional_accuracy']

        comparison = {}
        for metric in metric_names:
            if metric in baseline_metrics and metric in enhanced_metrics:
                baseline_val = baseline_metrics[metric]
                enhanced_val = enhanced_metrics[metric]

                # Calculate improvement (positive = better for enhanced)
                if metric in ['mse', 'rmse', 'mae']:  # Lower is better
                    improvement = baseline_val - enhanced_val
                else:  # Higher is better
                    improvement = enhanced_val - baseline_val

                improvement_pct = (improvement / abs(baseline_val)) * 100 if baseline_val != 0 else 0

                comparison[f"{metric}_improvement"] = improvement
                comparison[f"{metric}_improvement_pct"] = improvement_pct

        return comparison

    @staticmethod
    def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Validate prediction arrays for evaluation.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Validation results dictionary
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Check array shapes
        if y_true.shape != y_pred.shape:
            validation["valid"] = False
            validation["errors"].append(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
            return validation

        # Check for NaN values
        if np.isnan(y_true).any():
            validation["warnings"].append("NaN values found in true values")

        if np.isnan(y_pred).any():
            validation["warnings"].append("NaN values found in predictions")

        # Check for infinite values
        if np.isinf(y_true).any():
            validation["warnings"].append("Infinite values found in true values")

        if np.isinf(y_pred).any():
            validation["warnings"].append("Infinite values found in predictions")

        # Check data range
        if np.std(y_true) == 0:
            validation["warnings"].append("True values have zero variance")

        if np.std(y_pred) == 0:
            validation["warnings"].append("Predictions have zero variance")

        return validation

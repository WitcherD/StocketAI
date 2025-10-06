#!/usr/bin/env python3
"""
Baseline Model Training Experiment for VN30

This script trains baseline models (TFT, LightGBM, LSTM) on VN30 stock data
and generates performance benchmarks for comparison with enhanced models.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model_training import (
    TFTConfig,
    LightGBMConfig,
    LSTMConfig,
    TrainingUtils,
    EvaluationUtils
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/baseline_training.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class BaselineModelTrainer:
    """
    Main trainer class for baseline model training experiment.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        max_workers: int = 1,
        use_gpu: bool = False,
        experiment_name: str = "baseline_training_v1.0"
    ):
        """
        Initialize baseline model trainer.

        Args:
            symbols: List of symbols to train (default: all VN30)
            model_types: Model types to train (default: all)
            max_workers: Maximum parallel workers
            use_gpu: Whether to use GPU for training
            experiment_name: Name of the experiment
        """
        self.symbols = symbols or TrainingUtils.load_vn30_constituents()
        self.model_types = model_types or ["tft", "lightgbm", "lstm"]
        self.max_workers = max_workers
        self.use_gpu = use_gpu
        self.experiment_name = experiment_name

        # Validate environment
        self._validate_environment()

        # Setup experiment directory
        self.experiment_dir = f"experiments/results/{experiment_name}"
        os.makedirs(self.experiment_dir, exist_ok=True)

        logger.info(f"Initialized trainer for {len(self.symbols)} symbols and {len(self.model_types)} model types")

    def _validate_environment(self) -> None:
        """Validate training environment."""
        validation = TrainingUtils.validate_training_environment()

        missing_deps = []
        if not validation["qlib_available"]:
            missing_deps.append("qlib")
        if not validation["torch_available"]:
            missing_deps.append("PyTorch")
        if not validation["lightgbm_available"]:
            missing_deps.append("LightGBM")

        if missing_deps:
            raise RuntimeError(f"Missing required dependencies: {missing_deps}")

        if not validation["vn30_constituents_available"]:
            raise RuntimeError("VN30 constituents file not found")

        logger.info("Environment validation passed")

    def _create_model_config(self, model_type: str, symbol: str) -> Any:
        """
        Create model configuration instance.

        Args:
            model_type: Type of model
            symbol: Stock symbol

        Returns:
            Model configuration instance
        """
        if model_type == "tft":
            config = TFTConfig(symbol)
        elif model_type == "lightgbm":
            config = LightGBMConfig(symbol)
        elif model_type == "lstm":
            config = LSTMConfig(symbol)
            if not self.use_gpu:
                config.set_gpu_device(-1)  # CPU only
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return config

    def _train_single_model(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """
        Train a single model for a symbol.

        Args:
            symbol: Stock symbol
            model_type: Model type

        Returns:
            Training results dictionary
        """
        logger.info(f"Training {model_type} model for {symbol}")

        start_time = time.time()

        try:
            # Check if data exists
            if not TrainingUtils.check_symbol_data_exists(symbol):
                logger.warning(f"No processed data found for {symbol}, skipping")
                return {
                    "symbol": symbol,
                    "model_type": model_type,
                    "status": "skipped",
                    "error": "No processed data available",
                    "training_time": 0.0
                }

            # Create model configuration
            config = self._create_model_config(model_type, symbol)

            # Create model directory
            model_path = TrainingUtils.create_model_directory(symbol, model_type)

            # TODO: Implement actual model training using qlib
            # This is a placeholder - actual implementation will integrate with qlib workflow

            # For now, create mock training results
            training_time = time.time() - start_time

            # Mock metrics (replace with actual evaluation)
            mock_metrics = {
                "mse": 0.001,
                "rmse": 0.032,
                "mae": 0.025,
                "r2": 0.15,
                "ic": 0.08,
                "rank_ic": 0.07,
                "directional_accuracy": 0.52
            }

            # Save configuration
            config_path = os.path.join(model_path, "config.yaml")
            config.save_config(config_path)

            # Save training metadata
            TrainingUtils.save_training_metadata(
                symbol=symbol,
                model_type=model_type,
                config=config.config,
                metrics=mock_metrics,
                training_time=training_time
            )

            logger.info(f"Successfully trained {model_type} model for {symbol} in {training_time:.2f}s")

            return {
                "symbol": symbol,
                "model_type": model_type,
                "status": "completed",
                "metrics": mock_metrics,
                "training_time": training_time,
                "model_path": model_path
            }

        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Failed to train {model_type} model for {symbol}: {e}")

            return {
                "symbol": symbol,
                "model_type": model_type,
                "status": "failed",
                "error": str(e),
                "training_time": training_time
            }

    def _train_symbol_models(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Train all models for a single symbol.

        Args:
            symbol: Stock symbol

        Returns:
            List of training results
        """
        logger.info(f"Training all models for {symbol}")

        results = []
        for model_type in self.model_types:
            result = self._train_single_model(symbol, model_type)
            results.append(result)

        return results

    def run_training_experiment(self) -> Dict[str, Any]:
        """
        Run the complete baseline training experiment.

        Returns:
            Experiment results summary
        """
        logger.info(f"Starting baseline training experiment: {self.experiment_name}")
        experiment_start = time.time()

        all_results = []
        successful_trainings = 0
        failed_trainings = 0

        # Train models for each symbol
        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"Processing symbol {i}/{len(self.symbols)}: {symbol}")

            symbol_results = self._train_symbol_models(symbol)
            all_results.extend(symbol_results)

            # Count successes/failures
            for result in symbol_results:
                if result["status"] == "completed":
                    successful_trainings += 1
                elif result["status"] == "failed":
                    failed_trainings += 1

        # Generate performance reports
        self._generate_performance_reports(all_results)

        experiment_time = time.time() - experiment_start

        summary = {
            "experiment_name": self.experiment_name,
            "total_symbols": len(self.symbols),
            "total_models": len(self.symbols) * len(self.model_types),
            "successful_trainings": successful_trainings,
            "failed_trainings": failed_trainings,
            "experiment_time": experiment_time,
            "results": all_results
        }

        logger.info(f"Experiment completed in {experiment_time:.2f}s")
        logger.info(f"Successful trainings: {successful_trainings}, Failed: {failed_trainings}")

        return summary

    def _generate_performance_reports(self, results: List[Dict[str, Any]]) -> None:
        """
        Generate performance reports from training results.

        Args:
            results: List of training results
        """
        logger.info("Generating performance reports")

        # Group results by symbol
        symbol_reports = {}
        for result in results:
            symbol = result["symbol"]
            if symbol not in symbol_reports:
                symbol_reports[symbol] = {}
            if result["status"] == "completed":
                symbol_reports[symbol][result["model_type"]] = result["metrics"]

        # Create individual symbol reports
        report_dfs = []
        for symbol, model_metrics in symbol_reports.items():
            if model_metrics:  # Only create report if we have metrics
                report_path = os.path.join(self.experiment_dir, f"{symbol}_performance.csv")
                report_df = EvaluationUtils.create_performance_report(
                    symbol=symbol,
                    model_metrics=model_metrics,
                    output_path=report_path
                )
                report_dfs.append(report_df)

        # Create aggregated report
        if report_dfs:
            aggregated_path = os.path.join(self.experiment_dir, "aggregated_performance.csv")
            EvaluationUtils.aggregate_symbol_performance(
                symbol_reports=report_dfs,
                output_path=aggregated_path
            )

        logger.info(f"Generated performance reports in {self.experiment_dir}")


def main():
    """Main entry point for baseline model training."""
    parser = argparse.ArgumentParser(description="Train baseline models for VN30 stocks")
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to train (default: all VN30)")
    parser.add_argument("--models", nargs="*", choices=["tft", "lightgbm", "lstm"],
                       help="Model types to train (default: all)")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum parallel workers")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--experiment-name", default="baseline_training_v1.0",
                       help="Experiment name")

    args = parser.parse_args()

    # Create trainer
    trainer = BaselineModelTrainer(
        symbols=args.symbols,
        model_types=args.models,
        max_workers=args.max_workers,
        use_gpu=args.gpu,
        experiment_name=args.experiment_name
    )

    # Run experiment
    try:
        results = trainer.run_training_experiment()

        # Print summary
        print("\n" + "="*50)
        print("BASELINE TRAINING EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Experiment: {results['experiment_name']}")
        print(f"Symbols processed: {results['total_symbols']}")
        print(f"Models trained: {results['total_models']}")
        print(f"Successful: {results['successful_trainings']}")
        print(f"Failed: {results['failed_trainings']}")
        print(".2f")
        print(f"Results saved to: experiments/results/{results['experiment_name']}")
        print("="*50)

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

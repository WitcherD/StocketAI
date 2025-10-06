#!/usr/bin/env python3
"""
Integration test for VCB model training pipeline.

Tests the complete workflow of training baseline models (TFT, LightGBM, LSTM)
for VCB symbol using the model training infrastructure.
"""

import sys
import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pytest
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from model_training import (
    TFTConfig,
    LightGBMConfig,
    LSTMConfig,
    TrainingUtils,
    EvaluationUtils
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestVCBModelTrainingIntegration:
    """Integration test class for VCB model training pipeline."""

    @pytest.fixture(scope="class")
    def test_setup(self):
        """Set up test environment and components."""
        test_symbol = "VCB"

        # Create temporary directory for test models
        temp_dir = tempfile.mkdtemp(prefix="stocketai_test_")
        models_dir = os.path.join(temp_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        # Initialize model configurations
        tft_config = TFTConfig(test_symbol)
        lightgbm_config = LightGBMConfig(test_symbol)
        lstm_config = LSTMConfig(test_symbol)

        # Validate environment
        env_validation = TrainingUtils.validate_training_environment()

        # Check if processed data exists for VCB
        data_exists = TrainingUtils.check_symbol_data_exists(test_symbol)

        return {
            'test_symbol': test_symbol,
            'temp_dir': temp_dir,
            'models_dir': models_dir,
            'tft_config': tft_config,
            'lightgbm_config': lightgbm_config,
            'lstm_config': lstm_config,
            'env_validation': env_validation,
            'data_exists': data_exists
        }

    @pytest.fixture(scope="class", autouse=True)
    def cleanup_temp_dir(self, test_setup):
        """Clean up temporary directory after tests."""
        yield
        # NOTE: Keeping temporary directory for inspection of test results
        # Uncomment the following lines to enable cleanup:
        # temp_dir = test_setup['temp_dir']
        # if os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)
        #     logger.info(f"Cleaned up temporary directory: {temp_dir}")

        # Log the location of test results instead
        temp_dir = test_setup['temp_dir']
        if os.path.exists(temp_dir):
            logger.info(f"Test results available at: {temp_dir}")
            # Print directory contents for easy inspection
            print(f"\nTest Results Directory: {temp_dir}")
            for root, dirs, files in os.walk(temp_dir):
                level = root.replace(temp_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
            print()

    def test_environment_validation(self, test_setup):
        """Test training environment validation."""
        env_validation = test_setup['env_validation']

        assert isinstance(env_validation, dict), "Environment validation should return dictionary"

        # Check critical components
        required_keys = [
            'data_directory_exists',
            'models_directory_exists',
            'vn30_constituents_available'
        ]

        for key in required_keys:
            assert key in env_validation, f"Missing validation key: {key}"

        # VN30 constituents should be available
        assert env_validation['vn30_constituents_available'], "VN30 constituents file not found"

        logger.info("✅ Environment validation test passed")

    def test_model_configurations_initialization(self, test_setup):
        """Test model configuration initialization."""
        tft_config = test_setup['tft_config']
        lightgbm_config = test_setup['lightgbm_config']
        lstm_config = test_setup['lstm_config']
        test_symbol = test_setup['test_symbol']

        # Test TFT configuration
        assert tft_config.symbol == test_symbol
        assert tft_config.model_type == "tft"
        assert hasattr(tft_config, 'config')
        assert 'input_dim' in tft_config.config
        assert 'hidden_dim' in tft_config.config

        # Test LightGBM configuration
        assert lightgbm_config.symbol == test_symbol
        assert lightgbm_config.model_type == "lightgbm"
        assert 'learning_rate' in lightgbm_config.config
        assert 'max_depth' in lightgbm_config.config

        # Test LSTM configuration
        assert lstm_config.symbol == test_symbol
        assert lstm_config.model_type == "lstm"
        assert 'hidden_dim' in lstm_config.config
        assert 'num_layers' in lstm_config.config

        logger.info("✅ Model configurations initialization test passed")

    def test_model_config_validation(self, test_setup):
        """Test model configuration validation."""
        tft_config = test_setup['tft_config']
        lightgbm_config = test_setup['lightgbm_config']
        lstm_config = test_setup['lstm_config']

        # Test valid configurations (should not raise exceptions)
        try:
            tft_config._validate_config()
            lightgbm_config._validate_config()
            lstm_config._validate_config()
            logger.info("✅ Model configuration validation passed")
        except Exception as e:
            pytest.fail(f"Model configuration validation failed: {e}")

    def test_invalid_config_handling(self, test_setup):
        """Test handling of invalid configurations."""
        test_symbol = test_setup['test_symbol']

        # Test TFT with invalid parameters
        with pytest.raises(ValueError):
            TFTConfig(test_symbol, input_dim=-1)  # Negative input dimension

        with pytest.raises(ValueError):
            TFTConfig(test_symbol, num_heads=20)  # Too many heads

        # Test LightGBM with invalid parameters
        with pytest.raises(ValueError):
            LightGBMConfig(test_symbol, learning_rate=2.0)  # Learning rate > 1

        with pytest.raises(ValueError):
            LightGBMConfig(test_symbol, max_depth=-1)  # Negative depth

        # Test LSTM with invalid parameters
        with pytest.raises(ValueError):
            LSTMConfig(test_symbol, hidden_dim=0)  # Zero hidden dimension

        with pytest.raises(ValueError):
            LSTMConfig(test_symbol, GPU=-2)  # Invalid GPU ID

        logger.info("✅ Invalid configuration handling test passed")

    def test_model_directory_creation(self, test_setup):
        """Test model directory creation utilities."""
        test_symbol = test_setup['test_symbol']
        models_dir = test_setup['models_dir']

        # Test directory creation for each model type
        for model_type in ["tft", "lightgbm", "lstm"]:
            model_path = TrainingUtils.create_model_directory(test_symbol, model_type, models_dir)
            assert os.path.exists(model_path), f"Model directory not created: {model_path}"

            # Verify directory structure
            assert os.path.basename(model_path) == model_type
            assert test_symbol in model_path

        logger.info("✅ Model directory creation test passed")

    def test_training_utils_data_loading(self, test_setup):
        """Test training utilities data loading functions."""
        # Test VN30 constituents loading
        symbols = TrainingUtils.load_vn30_constituents()
        assert isinstance(symbols, list), "VN30 symbols should be a list"
        assert len(symbols) > 0, "VN30 symbols list should not be empty"
        assert test_setup['test_symbol'] in symbols, f"VCB should be in VN30 symbols"

        # Test data existence checking
        data_exists = TrainingUtils.check_symbol_data_exists(test_setup['test_symbol'])
        # Note: This may be False if data processing hasn't been run, which is expected

        logger.info(f"✅ Training utilities data loading test passed - loaded {len(symbols)} symbols")

    def test_evaluation_utils_metrics(self, test_setup):
        """Test evaluation utilities metrics calculation."""
        # Create mock prediction data
        np.random.seed(42)
        n_samples = 100

        y_true = np.random.randn(n_samples)
        y_pred = y_true + 0.1 * np.random.randn(n_samples)  # Add some noise

        # Test regression metrics
        metrics = EvaluationUtils.calculate_regression_metrics(y_true, y_pred)

        required_metrics = ['mse', 'rmse', 'mae', 'r2', 'ic', 'rank_ic', 'directional_accuracy']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Metric {metric} should be numeric"

        # Test reasonable ranges
        assert metrics['mse'] >= 0, "MSE should be non-negative"
        assert metrics['rmse'] >= 0, "RMSE should be non-negative"
        assert -1 <= metrics['r2'] <= 1, "R2 should be between -1 and 1"
        assert -1 <= metrics['ic'] <= 1, "IC should be between -1 and 1"

        logger.info("✅ Evaluation utilities metrics test passed")

    def test_evaluation_utils_financial_metrics(self, test_setup):
        """Test financial metrics calculation."""
        # Create mock financial return data
        np.random.seed(42)
        n_samples = 50

        y_true = 0.001 + 0.01 * np.random.randn(n_samples)  # Small positive returns
        y_pred = y_true + 0.002 * np.random.randn(n_samples)

        # Test financial metrics
        financial_metrics = EvaluationUtils.calculate_financial_metrics(y_true, y_pred)

        required_financial_metrics = [
            'cumulative_return_true', 'cumulative_return_pred',
            'sharpe_ratio_true', 'sharpe_ratio_pred',
            'win_rate'
        ]

        for metric in required_financial_metrics:
            assert metric in financial_metrics, f"Missing financial metric: {metric}"

        # Test win rate is reasonable
        assert 0 <= financial_metrics['win_rate'] <= 1, "Win rate should be between 0 and 1"

        logger.info("✅ Financial metrics test passed")

    def test_model_config_serialization(self, test_setup):
        """Test model configuration save/load functionality."""
        tft_config = test_setup['tft_config']
        models_dir = test_setup['models_dir']
        test_symbol = test_setup['test_symbol']

        # Create config file path
        config_path = os.path.join(
            models_dir, "symbols", test_symbol, "tft", "config.yaml"
        )

        # Save configuration
        tft_config.save_config(config_path)
        assert os.path.exists(config_path), f"Config file not saved: {config_path}"

        # Load configuration
        loaded_config = TFTConfig(test_symbol)
        loaded_config.load_config(config_path)

        # Verify loaded config matches original
        assert loaded_config.symbol == tft_config.symbol
        assert loaded_config.model_type == tft_config.model_type
        assert loaded_config.config['input_dim'] == tft_config.config['input_dim']

        logger.info("✅ Model configuration serialization test passed")

    def test_training_metadata_management(self, test_setup):
        """Test training metadata save/load functionality."""
        test_symbol = test_setup['test_symbol']
        models_dir = test_setup['models_dir']

        # Mock training results
        mock_metrics = {
            "mse": 0.002,
            "rmse": 0.045,
            "mae": 0.035,
            "r2": 0.12,
            "ic": 0.06,
            "rank_ic": 0.05,
            "directional_accuracy": 0.51
        }
        training_time = 45.2

        # Save metadata
        TrainingUtils.save_training_metadata(
            symbol=test_symbol,
            model_type="tft",
            config={"input_dim": 158, "hidden_dim": 64},
            metrics=mock_metrics,
            training_time=training_time,
            base_dir=models_dir
        )

        # Load metadata
        loaded_metadata = TrainingUtils.load_training_metadata(
            symbol=test_symbol,
            model_type="tft",
            base_dir=models_dir
        )

        assert loaded_metadata is not None, "Failed to load training metadata"
        assert loaded_metadata['symbol'] == test_symbol
        assert loaded_metadata['model_type'] == "tft"
        assert loaded_metadata['metrics']['mse'] == mock_metrics['mse']
        assert loaded_metadata['training_time_seconds'] == training_time

        logger.info("✅ Training metadata management test passed")

    def test_training_status_tracking(self, test_setup):
        """Test training status tracking functionality."""
        test_symbol = test_setup['test_symbol']
        models_dir = test_setup['models_dir']

        # Test status for non-existent model
        status = TrainingUtils.get_training_status(test_symbol, "tft", models_dir)
        assert not status['model_directory_exists']
        assert not status['has_metadata']
        assert not status['has_model_file']

        # Create model directory and test again
        model_path = TrainingUtils.create_model_directory(test_symbol, "tft", models_dir)
        status = TrainingUtils.get_training_status(test_symbol, "tft", models_dir)
        assert status['model_directory_exists']
        assert not status['has_metadata']  # No metadata yet
        assert not status['has_model_file']  # No model file yet

        logger.info("✅ Training status tracking test passed")

    def test_performance_report_generation(self, test_setup):
        """Test performance report generation."""
        test_symbol = test_setup['test_symbol']
        models_dir = test_setup['models_dir']

        # Create mock model metrics
        model_metrics = {
            "tft": {
                "mse": 0.0015, "rmse": 0.039, "mae": 0.030, "r2": 0.18,
                "ic": 0.09, "rank_ic": 0.08, "directional_accuracy": 0.53
            },
            "lightgbm": {
                "mse": 0.0021, "rmse": 0.046, "mae": 0.035, "r2": 0.15,
                "ic": 0.07, "rank_ic": 0.06, "directional_accuracy": 0.51
            },
            "lstm": {
                "mse": 0.0018, "rmse": 0.042, "mae": 0.032, "r2": 0.16,
                "ic": 0.08, "rank_ic": 0.07, "directional_accuracy": 0.52
            }
        }

        # Generate performance report
        report_path = os.path.join(models_dir, "symbols", test_symbol, "performance_report.csv")
        report_df = EvaluationUtils.create_performance_report(
            symbol=test_symbol,
            model_metrics=model_metrics,
            output_path=report_path
        )

        assert report_df is not None, "Performance report generation failed"
        assert not report_df.empty, "Performance report is empty"
        assert len(report_df) == 3, "Report should have 3 model entries"
        assert os.path.exists(report_path), f"Report file not saved: {report_path}"

        # Verify report structure
        assert 'symbol' in report_df.columns
        assert 'model_type' in report_df.columns
        assert 'mse' in report_df.columns

        # Verify all models are present
        model_types = set(report_df['model_type'].tolist())
        expected_models = {"tft", "lightgbm", "lstm"}
        assert model_types == expected_models, f"Missing models in report: {expected_models - model_types}"

        logger.info("✅ Performance report generation test passed")

    def test_batch_processing_utilities(self, test_setup):
        """Test batch processing utilities."""
        symbols = ["VCB", "TCB", "ACB", "BID", "CTG"]

        # Test batch creation
        batches = TrainingUtils.get_batch_symbols(symbols, batch_size=2)
        assert len(batches) == 3, "Should create 3 batches"
        assert len(batches[0]) == 2, "First batch should have 2 symbols"
        assert len(batches[1]) == 2, "Second batch should have 2 symbols"
        assert len(batches[2]) == 1, "Third batch should have 1 symbol"

        # Test with batch_size=3
        batches = TrainingUtils.get_batch_symbols(symbols, batch_size=3)
        assert len(batches) == 2, "Should create 2 batches with batch_size=3"
        assert len(batches[0]) == 3, "First batch should have 3 symbols"
        assert len(batches[1]) == 2, "Second batch should have 2 symbols"

        logger.info("✅ Batch processing utilities test passed")



    def test_evaluation_validation(self, test_setup):
        """Test evaluation input validation."""
        # Test with mismatched array shapes
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])  # Different length

        validation = EvaluationUtils.validate_predictions(y_true, y_pred)
        assert not validation['valid'], "Should detect shape mismatch"
        assert len(validation['errors']) > 0, "Should have error messages"

        # Test with valid data
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 3.1])

        validation = EvaluationUtils.validate_predictions(y_true, y_pred)
        assert validation['valid'], "Valid data should pass validation"

        logger.info("✅ Evaluation validation test passed")

    def test_model_config_optimization(self, test_setup):
        """Test model configuration optimization methods."""
        lightgbm_config = test_setup['lightgbm_config']
        lstm_config = test_setup['lstm_config']

        # Test LightGBM optimization
        original_max_depth = lightgbm_config.config['max_depth']
        lightgbm_config.optimize_for_memory()

        assert lightgbm_config.config['max_depth'] <= original_max_depth, "Memory optimization should reduce max_depth"

        # Test LSTM optimization
        original_hidden_dim = lstm_config.config['hidden_dim']
        lstm_config.optimize_for_memory()

        assert lstm_config.config['hidden_dim'] <= original_hidden_dim, "Memory optimization should reduce hidden_dim"

        logger.info("✅ Model configuration optimization test passed")

    @pytest.mark.slow
    def test_end_to_end_training_simulation(self, test_setup):
        """Test complete end-to-end training simulation (marked as slow)."""
        import time

        test_symbol = test_setup['test_symbol']
        models_dir = test_setup['models_dir']

        # Simulate training all three models
        model_types = ["tft", "lightgbm", "lstm"]
        training_results = []

        start_time = time.time()

        for model_type in model_types:
            # Create model config
            if model_type == "tft":
                config = TFTConfig(test_symbol)
            elif model_type == "lightgbm":
                config = LightGBMConfig(test_symbol)
            else:  # lstm
                config = LSTMConfig(test_symbol)

            # Create model directory
            model_path = TrainingUtils.create_model_directory(test_symbol, model_type, models_dir)

            # Simulate training (in real implementation, this would call actual training)
            training_start = time.time()

            # Mock training process - just sleep briefly
            time.sleep(0.1)

            training_time = time.time() - training_start

            # Mock metrics
            mock_metrics = {
                "mse": 0.001 + 0.0005 * np.random.random(),
                "rmse": 0.032 + 0.005 * np.random.random(),
                "mae": 0.025 + 0.005 * np.random.random(),
                "r2": 0.15 + 0.05 * np.random.random(),
                "ic": 0.08 + 0.03 * np.random.random(),
                "rank_ic": 0.07 + 0.03 * np.random.random(),
                "directional_accuracy": 0.52 + 0.04 * np.random.random()
            }

            # Save configuration and metadata
            config_path = os.path.join(model_path, "config.yaml")
            config.save_config(config_path)

            TrainingUtils.save_training_metadata(
                symbol=test_symbol,
                model_type=model_type,
                config=config.config,
                metrics=mock_metrics,
                training_time=training_time,
                base_dir=models_dir
            )

            training_results.append({
                "model_type": model_type,
                "training_time": training_time,
                "metrics": mock_metrics,
                "success": True
            })

        total_time = time.time() - start_time

        # Verify all models were "trained"
        assert len(training_results) == 3, "Should have trained 3 models"

        for result in training_results:
            assert result['success'], f"Training failed for {result['model_type']}"
            assert result['training_time'] > 0, f"Invalid training time for {result['model_type']}"
            assert 'metrics' in result, f"Missing metrics for {result['model_type']}"

        # Generate performance report
        model_metrics = {r['model_type']: r['metrics'] for r in training_results}
        report_path = os.path.join(models_dir, "symbols", test_symbol, "simulation_performance.csv")
        report_df = EvaluationUtils.create_performance_report(
            symbol=test_symbol,
            model_metrics=model_metrics,
            output_path=report_path
        )

        assert os.path.exists(report_path), "Performance report not generated"

        # Performance check - should complete within reasonable time
        assert total_time < 5.0, f"End-to-end simulation took too long: {total_time:.2f}s"

        logger.info(f"✅ End-to-end training simulation passed: {total_time:.2f}s total time")

    def test_complete_pipeline_validation(self, test_setup):
        """Test complete pipeline validation."""
        test_symbol = test_setup['test_symbol']
        models_dir = test_setup['models_dir']

        # Step 1: Verify environment
        logger.info("Step 1: Validating environment...")
        env_validation = TrainingUtils.validate_training_environment()
        assert env_validation['vn30_constituents_available'], "VN30 constituents not available"

        # Step 2: Test model configurations
        logger.info("Step 2: Testing model configurations...")
        configs = {
            "tft": TFTConfig(test_symbol),
            "lightgbm": LightGBMConfig(test_symbol),
            "lstm": LSTMConfig(test_symbol)
        }

        for model_type, config in configs.items():
            assert config.symbol == test_symbol
            assert config.model_type == model_type
            config._validate_config()  # Should not raise

        # Step 3: Test directory creation
        logger.info("Step 3: Testing directory creation...")
        for model_type in configs.keys():
            model_path = TrainingUtils.create_model_directory(test_symbol, model_type, models_dir)
            assert os.path.exists(model_path)

        # Step 4: Test evaluation utilities
        logger.info("Step 4: Testing evaluation utilities...")
        y_true = np.random.randn(50)
        y_pred = y_true + 0.1 * np.random.randn(50)

        metrics = EvaluationUtils.evaluate_model_predictions(y_true, y_pred)
        assert 'mse' in metrics and 'ic' in metrics

        # Step 5: Test report generation
        logger.info("Step 5: Testing report generation...")
        mock_metrics = {model_type: {"mse": 0.001, "ic": 0.05} for model_type in configs.keys()}
        report_df = EvaluationUtils.create_performance_report(test_symbol, mock_metrics)
        assert not report_df.empty

        logger.info("✅ Complete pipeline validation passed")


def main():
    """Main function to run the VCB model training integration test."""
    print("VCB Model Training Integration Test")
    print("=" * 50)

    # Run pytest programmatically
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--color=yes",
        "-m", "integration"
    ]

    exit_code = pytest.main(pytest_args)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

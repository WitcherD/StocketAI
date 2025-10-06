"""
Training Utilities for StocketAI

Provides utilities for model training, data loading, and experiment management.
"""

import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingUtils:
    """
    Utilities for model training and experiment management.
    """

    @staticmethod
    def load_vn30_constituents(constituents_file: str = "data/symbols/vn30_constituents.csv") -> List[str]:
        """
        Load VN30 constituent symbols.

        Args:
            constituents_file: Path to VN30 constituents CSV file

        Returns:
            List of VN30 symbols
        """
        try:
            df = pd.read_csv(constituents_file)
            symbols = df['symbol'].tolist()
            logger.info(f"Loaded {len(symbols)} VN30 symbols")
            return symbols
        except Exception as e:
            logger.error(f"Failed to load VN30 constituents: {e}")
            raise

    @staticmethod
    def get_symbol_data_path(symbol: str, data_dir: str = "data") -> str:
        """
        Get data path for a specific symbol.

        Args:
            symbol: Stock symbol
            data_dir: Base data directory

        Returns:
            Path to symbol data directory
        """
        return os.path.join(data_dir, "symbols", symbol)

    @staticmethod
    def get_qlib_data_path(symbol: str, data_dir: str = "data") -> str:
        """
        Get qlib format data path for a specific symbol.

        Args:
            symbol: Stock symbol
            data_dir: Base data directory

        Returns:
            Path to symbol qlib data directory
        """
        return os.path.join(TrainingUtils.get_symbol_data_path(symbol, data_dir), "qlib_format")

    @staticmethod
    def check_symbol_data_exists(symbol: str, data_dir: str = "data") -> bool:
        """
        Check if processed data exists for a symbol.

        Args:
            symbol: Stock symbol
            data_dir: Base data directory

        Returns:
            True if data exists, False otherwise
        """
        qlib_path = TrainingUtils.get_qlib_data_path(symbol, data_dir)
        return os.path.exists(qlib_path) and len(os.listdir(qlib_path)) > 0

    @staticmethod
    def get_available_symbols(data_dir: str = "data") -> List[str]:
        """
        Get list of symbols with available processed data.

        Args:
            data_dir: Base data directory

        Returns:
            List of symbols with available data
        """
        symbols_dir = os.path.join(data_dir, "symbols")
        if not os.path.exists(symbols_dir):
            return []

        available_symbols = []
        for symbol in os.listdir(symbols_dir):
            symbol_path = os.path.join(symbols_dir, symbol)
            if os.path.isdir(symbol_path) and TrainingUtils.check_symbol_data_exists(symbol, data_dir):
                available_symbols.append(symbol)

        logger.info(f"Found {len(available_symbols)} symbols with available data")
        return available_symbols

    @staticmethod
    def create_model_directory(symbol: str, model_type: str, base_dir: str = "models") -> str:
        """
        Create model directory structure.

        Args:
            symbol: Stock symbol
            model_type: Model type (tft, lightgbm, lstm)
            base_dir: Base models directory

        Returns:
            Path to model directory
        """
        model_path = os.path.join(base_dir, "symbols", symbol, model_type)
        os.makedirs(model_path, exist_ok=True)
        logger.info(f"Created model directory: {model_path}")
        return model_path

    @staticmethod
    def save_training_metadata(
        symbol: str,
        model_type: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        training_time: float,
        base_dir: str = "models"
    ) -> None:
        """
        Save training metadata.

        Args:
            symbol: Stock symbol
            model_type: Model type
            config: Model configuration
            metrics: Performance metrics
            training_time: Training time in seconds
            base_dir: Base models directory
        """
        model_path = TrainingUtils.create_model_directory(symbol, model_type, base_dir)
        metadata_path = os.path.join(model_path, "metadata.json")

        metadata = {
            "symbol": symbol,
            "model_type": model_type,
            "training_date": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "training_time_seconds": training_time,
            "version": "v1.0"
        }

        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved training metadata to {metadata_path}")

    @staticmethod
    def load_training_metadata(symbol: str, model_type: str, base_dir: str = "models") -> Optional[Dict[str, Any]]:
        """
        Load training metadata.

        Args:
            symbol: Stock symbol
            model_type: Model type
            base_dir: Base models directory

        Returns:
            Metadata dictionary or None if not found
        """
        model_path = os.path.join(base_dir, "symbols", symbol, model_type)
        metadata_path = os.path.join(model_path, "metadata.json")

        if not os.path.exists(metadata_path):
            return None

        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return metadata

    @staticmethod
    def get_training_status(symbol: str, model_type: str, base_dir: str = "models") -> Dict[str, Any]:
        """
        Get training status for a symbol-model combination.

        Args:
            symbol: Stock symbol
            model_type: Model type
            base_dir: Base models directory

        Returns:
            Status dictionary
        """
        model_path = os.path.join(base_dir, "symbols", symbol, model_type)

        status = {
            "symbol": symbol,
            "model_type": model_type,
            "model_directory_exists": os.path.exists(model_path),
            "has_metadata": False,
            "has_model_file": False,
            "training_date": None,
            "metrics": None
        }

        if status["model_directory_exists"]:
            # Check for model files
            model_files = [f for f in os.listdir(model_path) if not f.startswith('.') and f != "metadata.json"]
            status["has_model_file"] = len(model_files) > 0

            # Load metadata if available
            metadata = TrainingUtils.load_training_metadata(symbol, model_type, base_dir)
            if metadata:
                status["has_metadata"] = True
                status["training_date"] = metadata.get("training_date")
                status["metrics"] = metadata.get("metrics")

        return status

    @staticmethod
    def get_batch_symbols(symbols: List[str], batch_size: int = 5) -> List[List[str]]:
        """
        Split symbols into batches for parallel processing.

        Args:
            symbols: List of symbols
            batch_size: Number of symbols per batch

        Returns:
            List of symbol batches
        """
        return [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

    @staticmethod
    def validate_training_environment() -> Dict[str, Any]:
        """
        Validate training environment and dependencies.

        Returns:
            Validation results dictionary
        """
        validation = {
            "qlib_available": False,
            "torch_available": False,
            "lightgbm_available": False,
            "data_directory_exists": False,
            "models_directory_exists": False,
            "vn30_constituents_available": False
        }

        # Check qlib
        try:
            import qlib
            validation["qlib_available"] = True
        except ImportError:
            logger.warning("qlib not available")

        # Check PyTorch
        try:
            import torch
            validation["torch_available"] = True
        except ImportError:
            logger.warning("PyTorch not available")

        # Check LightGBM
        try:
            import lightgbm
            validation["lightgbm_available"] = True
        except ImportError:
            logger.warning("LightGBM not available")

        # Check directories
        validation["data_directory_exists"] = os.path.exists("data")
        validation["models_directory_exists"] = os.path.exists("models")

        # Check VN30 constituents
        constituents_file = "data/symbols/vn30_constituents.csv"
        validation["vn30_constituents_available"] = os.path.exists(constituents_file)

        # Log validation results
        issues = [k for k, v in validation.items() if not v]
        if issues:
            logger.warning(f"Environment validation issues: {issues}")
        else:
            logger.info("Environment validation passed")

        return validation

    @staticmethod
    def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
        """
        Setup logging configuration.

        Args:
            log_level: Logging level
            log_file: Optional log file path
        """
        import logging

        # Create logs directory if needed
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )

        logger.info(f"Logging configured with level {log_level}")

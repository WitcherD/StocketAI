"""
Data Storage Module for VN30 Stock Price Prediction System.

This module provides processed data storage, versioning, and management
functionality including:

- Processed data storage in organized directory structure
- Data versioning and backup strategies
- Data access and retrieval functions
- Data compression for large datasets
- Metadata management and tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
import pickle
from pathlib import Path
import shutil
import gzip
import zipfile
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class StorageFormat(Enum):
    """Supported storage formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    FEATHER = "feather"


class CompressionMethod(Enum):
    """Available compression methods."""
    NONE = "none"
    GZIP = "gzip"
    BZ2 = "bz2"
    XZ = "xz"
    ZIP = "zip"


@dataclass
class StorageConfig:
    """Configuration for data storage operations."""
    base_path: str = "data/processed"
    format: StorageFormat = StorageFormat.PARQUET
    compression: CompressionMethod = CompressionMethod.GZIP
    create_versioned_backups: bool = True
    max_versions: int = 5
    enable_metadata_tracking: bool = True
    chunk_size: int = 100000  # For large file processing
    auto_cleanup: bool = True
    cleanup_threshold_days: int = 30


@dataclass
class DataMetadata:
    """Metadata for stored data."""
    source: str
    processing_timestamp: datetime
    data_shape: Tuple[int, int]
    columns: List[str]
    data_types: Dict[str, str]
    processing_pipeline: List[str]
    quality_score: Optional[float] = None
    version: str = "1.0.0"
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class ProcessedDataManager:
    """
    Manages storage and retrieval of processed data.

    Provides functionality for storing, versioning, and retrieving
    processed financial data with metadata tracking and compression.
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize the ProcessedDataManager.

        Args:
            config: StorageConfig object with storage parameters
        """
        self.config = config or StorageConfig()
        self.metadata_store = {}
        self._ensure_directories()

    def store_constituents_data(self,
                              df: pd.DataFrame,
                              dataset_name: str,
                              metadata: Optional[DataMetadata] = None) -> str:
        """
        Store processed constituents data.

        Args:
            df: Processed constituents DataFrame
            dataset_name: Name for the dataset
            metadata: Optional metadata for the dataset

        Returns:
            Path to stored data
        """
        logger.info(f"Storing constituents data: {dataset_name}")

        # Create metadata if not provided
        if metadata is None:
            metadata = self._create_default_metadata(df, dataset_name, "constituents")

        # Generate storage path
        storage_path = self._generate_storage_path(dataset_name, "constituents")

        # Store data
        file_path = self._store_dataframe(df, storage_path, metadata)

        # Store metadata
        if self.config.enable_metadata_tracking:
            self._store_metadata(dataset_name, metadata, "constituents")

        # Create versioned backup if configured
        if self.config.create_versioned_backups:
            self._create_versioned_backup(file_path, dataset_name, "constituents")

        logger.info(f"Constituents data stored successfully: {file_path}")
        return file_path

    def store_price_data(self,
                        df: pd.DataFrame,
                        dataset_name: str,
                        metadata: Optional[DataMetadata] = None) -> str:
        """
        Store processed price data.

        Args:
            df: Processed price DataFrame
            dataset_name: Name for the dataset
            metadata: Optional metadata for the dataset

        Returns:
            Path to stored data
        """
        logger.info(f"Storing price data: {dataset_name}")

        # Create metadata if not provided
        if metadata is None:
            metadata = self._create_default_metadata(df, dataset_name, "price")

        # Generate storage path
        storage_path = self._generate_storage_path(dataset_name, "price")

        # Store data
        file_path = self._store_dataframe(df, storage_path, metadata)

        # Store metadata
        if self.config.enable_metadata_tracking:
            self._store_metadata(dataset_name, metadata, "price")

        # Create versioned backup if configured
        if self.config.create_versioned_backups:
            self._create_versioned_backup(file_path, dataset_name, "price")

        logger.info(f"Price data stored successfully: {file_path}")
        return file_path

    def retrieve_constituents_data(self, dataset_name: str, version: str = "latest") -> Tuple[pd.DataFrame, DataMetadata]:
        """
        Retrieve stored constituents data.

        Args:
            dataset_name: Name of the dataset to retrieve
            version: Version to retrieve ("latest" or specific version)

        Returns:
            Tuple of (DataFrame, metadata)
        """
        logger.info(f"Retrieving constituents data: {dataset_name}, version: {version}")

        # Find the file path
        if version == "latest":
            file_path = self._find_latest_version(dataset_name, "constituents")
        else:
            file_path = self._find_specific_version(dataset_name, version, "constituents")

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} version {version} not found")

        # Load metadata
        metadata = self._load_metadata(dataset_name, "constituents")

        # Load data
        df = self._load_dataframe(file_path)

        logger.info(f"Constituents data retrieved successfully: {file_path}")
        return df, metadata

    def retrieve_price_data(self, dataset_name: str, version: str = "latest") -> Tuple[pd.DataFrame, DataMetadata]:
        """
        Retrieve stored price data.

        Args:
            dataset_name: Name of the dataset to retrieve
            version: Version to retrieve ("latest" or specific version)

        Returns:
            Tuple of (DataFrame, metadata)
        """
        logger.info(f"Retrieving price data: {dataset_name}, version: {version}")

        # Find the file path
        if version == "latest":
            file_path = self._find_latest_version(dataset_name, "price")
        else:
            file_path = self._find_specific_version(dataset_name, version, "price")

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} version {version} not found")

        # Load metadata
        metadata = self._load_metadata(dataset_name, "price")

        # Load data
        df = self._load_dataframe(file_path)

        logger.info(f"Price data retrieved successfully: {file_path}")
        return df, metadata

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        base_path = Path(self.config.base_path)

        # Create main directories
        directories = [
            base_path / "constituents",
            base_path / "price",
            base_path / "metadata",
            base_path / "backups",
            base_path / "temp"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _create_default_metadata(self, df: pd.DataFrame, dataset_name: str, data_type: str) -> DataMetadata:
        """Create default metadata for a dataset."""
        return DataMetadata(
            source="data_processing_pipeline",
            processing_timestamp=datetime.now(),
            data_shape=df.shape,
            columns=df.columns.tolist(),
            data_types={col: str(df[col].dtype) for col in df.columns},
            processing_pipeline=["data_processing"],
            quality_score=None,
            version="1.0.0",
            description=f"Processed {data_type} data: {dataset_name}",
            tags=[data_type, dataset_name, "processed"]
        )

    def _generate_storage_path(self, dataset_name: str, data_type: str) -> Path:
        """Generate storage path for a dataset."""
        base_path = Path(self.config.base_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{data_type}_{dataset_name}_{timestamp}"

        if self.config.format == StorageFormat.PARQUET:
            filename += ".parquet"
        elif self.config.format == StorageFormat.CSV:
            filename += ".csv"
        elif self.config.format == StorageFormat.JSON:
            filename += ".json"
        elif self.config.format == StorageFormat.HDF5:
            filename += ".h5"
        elif self.config.format == StorageFormat.FEATHER:
            filename += ".feather"
        elif self.config.format == StorageFormat.PICKLE:
            filename += ".pkl"

        return base_path / data_type / filename

    def _store_dataframe(self, df: pd.DataFrame, file_path: Path, metadata: DataMetadata) -> Path:
        """Store DataFrame to file with appropriate format and compression."""
        try:
            # Create parent directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if self.config.format == StorageFormat.PARQUET:
                df.to_parquet(file_path, compression=self.config.compression.value if self.config.compression != CompressionMethod.NONE else None)
            elif self.config.format == StorageFormat.CSV:
                compression = self.config.compression.value if self.config.compression != CompressionMethod.NONE else None
                df.to_csv(file_path, compression=compression, index=True)
            elif self.config.format == StorageFormat.JSON:
                df.to_json(file_path, compression=self.config.compression.value if self.config.compression != CompressionMethod.NONE else None, orient='records')
            elif self.config.format == StorageFormat.HDF5:
                df.to_hdf(file_path, key='data', mode='w', complevel=9 if self.config.compression == CompressionMethod.GZIP else None)
            elif self.config.format == StorageFormat.FEATHER:
                df.to_feather(file_path)
            elif self.config.format == StorageFormat.PICKLE:
                with open(file_path, 'wb') as f:
                    pickle.dump(df, f)

            logger.info(f"DataFrame stored to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error storing DataFrame to {file_path}: {e}")
            raise

    def _load_dataframe(self, file_path: Path) -> pd.DataFrame:
        """Load DataFrame from file."""
        try:
            if self.config.format == StorageFormat.PARQUET:
                return pd.read_parquet(file_path)
            elif self.config.format == StorageFormat.CSV:
                return pd.read_csv(file_path, index_col=0)
            elif self.config.format == StorageFormat.JSON:
                return pd.read_json(file_path, orient='records')
            elif self.config.format == StorageFormat.HDF5:
                return pd.read_hdf(file_path, key='data')
            elif self.config.format == StorageFormat.FEATHER:
                return pd.read_feather(file_path)
            elif self.config.format == StorageFormat.PICKLE:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {self.config.format}")

        except Exception as e:
            logger.error(f"Error loading DataFrame from {file_path}: {e}")
            raise

    def _store_metadata(self, dataset_name: str, metadata: DataMetadata, data_type: str) -> None:
        """Store metadata for a dataset."""
        metadata_path = Path(self.config.base_path) / "metadata" / f"{data_type}_{dataset_name}_metadata.json"

        try:
            metadata_dict = {
                'source': metadata.source,
                'processing_timestamp': metadata.processing_timestamp.isoformat(),
                'data_shape': metadata.data_shape,
                'columns': metadata.columns,
                'data_types': metadata.data_types,
                'processing_pipeline': metadata.processing_pipeline,
                'quality_score': metadata.quality_score,
                'version': metadata.version,
                'description': metadata.description,
                'tags': metadata.tags
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

            self.metadata_store[f"{data_type}_{dataset_name}"] = metadata_dict
            logger.info(f"Metadata stored for {dataset_name}")

        except Exception as e:
            logger.error(f"Error storing metadata for {dataset_name}: {e}")

    def _load_metadata(self, dataset_name: str, data_type: str) -> DataMetadata:
        """Load metadata for a dataset."""
        metadata_path = Path(self.config.base_path) / "metadata" / f"{data_type}_{dataset_name}_metadata.json"

        try:
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)

                return DataMetadata(
                    source=metadata_dict['source'],
                    processing_timestamp=datetime.fromisoformat(metadata_dict['processing_timestamp']),
                    data_shape=tuple(metadata_dict['data_shape']),
                    columns=metadata_dict['columns'],
                    data_types=metadata_dict['data_types'],
                    processing_pipeline=metadata_dict['processing_pipeline'],
                    quality_score=metadata_dict.get('quality_score'),
                    version=metadata_dict.get('version', '1.0.0'),
                    description=metadata_dict.get('description'),
                    tags=metadata_dict.get('tags', [])
                )
            else:
                # Return default metadata if file doesn't exist
                logger.warning(f"Metadata file not found for {dataset_name}, creating default")
                return DataMetadata(
                    source="unknown",
                    processing_timestamp=datetime.now(),
                    data_shape=(0, 0),
                    columns=[],
                    data_types={},
                    processing_pipeline=[],
                    description=f"Metadata not found for {dataset_name}"
                )

        except Exception as e:
            logger.error(f"Error loading metadata for {dataset_name}: {e}")
            raise

    def _create_versioned_backup(self, file_path: Path, dataset_name: str, data_type: str) -> None:
        """Create versioned backup of stored data."""
        try:
            backup_dir = Path(self.config.base_path) / "backups" / data_type / dataset_name
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{data_type}_{dataset_name}_{timestamp}{file_path.suffix}"
            backup_path = backup_dir / backup_filename

            # Copy file to backup location
            shutil.copy2(file_path, backup_path)

            # Clean up old backups if needed
            if self.config.auto_cleanup:
                self._cleanup_old_backups(backup_dir)

            logger.info(f"Backup created: {backup_path}")

        except Exception as e:
            logger.error(f"Error creating backup for {dataset_name}: {e}")

    def _cleanup_old_backups(self, backup_dir: Path) -> None:
        """Clean up old backup files."""
        try:
            if not backup_dir.exists():
                return

            # Get all backup files
            backup_files = list(backup_dir.glob("*"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Keep only the latest max_versions files
            if len(backup_files) > self.config.max_versions:
                files_to_remove = backup_files[self.config.max_versions:]

                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        logger.info(f"Removed old backup: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove old backup {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error during backup cleanup: {e}")

    def _find_latest_version(self, dataset_name: str, data_type: str) -> Path:
        """Find the latest version of a dataset."""
        data_dir = Path(self.config.base_path) / data_type

        if not data_dir.exists():
            raise FileNotFoundError(f"No data directory found for {data_type}")

        # Find all files matching the dataset pattern
        # Pattern: {data_type}_{dataset_name}_{timestamp}.ext
        pattern = f"{data_type}_{dataset_name}_*.parquet" if self.config.format == StorageFormat.PARQUET else f"{data_type}_{dataset_name}_*"
        matching_files = list(data_dir.glob(pattern))

        # If no files found with the expected pattern, try a more flexible search
        if not matching_files:
            # Look for any file containing the dataset name
            all_files = list(data_dir.glob("*"))
            matching_files = [f for f in all_files if dataset_name in f.name and f.is_file()]

        if not matching_files:
            raise FileNotFoundError(f"No files found for dataset {dataset_name}")

        # Return the most recent file
        latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
        return latest_file

    def _find_specific_version(self, dataset_name: str, version: str, data_type: str) -> Path:
        """Find a specific version of a dataset."""
        data_dir = Path(self.config.base_path) / data_type

        if not data_dir.exists():
            raise FileNotFoundError(f"No data directory found for {data_type}")

        # Look for file with specific version
        version_pattern = f"{data_type}_{dataset_name}_{version}"
        matching_files = list(data_dir.glob(f"{version_pattern}.*"))

        if not matching_files:
            raise FileNotFoundError(f"Version {version} not found for dataset {dataset_name}")

        return matching_files[0]

    def list_datasets(self, data_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available datasets.

        Args:
            data_type: Optional data type filter ("constituents" or "price")

        Returns:
            Dictionary with dataset information
        """
        datasets = {
            'constituents': [],
            'price': []
        }

        try:
            base_path = Path(self.config.base_path)

            # List constituents datasets
            constituents_dir = base_path / "constituents"
            if constituents_dir.exists():
                for file_path in constituents_dir.iterdir():
                    if file_path.is_file():
                        dataset_name = self._extract_dataset_name(file_path.name, "constituents")
                        if dataset_name:
                            datasets['constituents'].append(dataset_name)

            # List price datasets
            price_dir = base_path / "price"
            if price_dir.exists():
                for file_path in price_dir.iterdir():
                    if file_path.is_file():
                        dataset_name = self._extract_dataset_name(file_path.name, "price")
                        if dataset_name:
                            datasets['price'].append(dataset_name)

            # Remove duplicates
            datasets['constituents'] = list(set(datasets['constituents']))
            datasets['price'] = list(set(datasets['price']))

            # Filter by data_type if specified
            if data_type:
                return {data_type: datasets.get(data_type, [])}

        except Exception as e:
            logger.error(f"Error listing datasets: {e}")

        return datasets

    def _extract_dataset_name(self, filename: str, data_type: str) -> Optional[str]:
        """Extract dataset name from filename."""
        try:
            # Expected format: {data_type}_{dataset_name}_{timestamp}.ext
            parts = filename.split('_')
            if len(parts) >= 2 and parts[0] == data_type:
                return parts[1]
            return None
        except Exception:
            return None

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        stats = {
            'config': self.config.__dict__,
            'total_datasets': 0,
            'storage_usage': {},
            'metadata_count': len(self.metadata_store),
            'timestamp': datetime.now().isoformat()
        }

        try:
            base_path = Path(self.config.base_path)

            # Calculate storage usage by data type
            for data_type in ['constituents', 'price']:
                data_dir = base_path / data_type
                if data_dir.exists():
                    total_size = 0
                    file_count = 0

                    for file_path in data_dir.rglob('*'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                            file_count += 1

                    stats['storage_usage'][data_type] = {
                        'file_count': file_count,
                        'total_size_bytes': total_size,
                        'total_size_mb': total_size / (1024 * 1024)
                    }

            stats['total_datasets'] = sum(
                usage.get('file_count', 0)
                for usage in stats['storage_usage'].values()
            )

        except Exception as e:
            logger.error(f"Error calculating storage stats: {e}")

        return stats

    def cleanup_old_data(self, days_threshold: Optional[int] = None) -> Dict[str, Any]:
        """
        Clean up old data files.

        Args:
            days_threshold: Days threshold for cleanup (uses config default if None)

        Returns:
            Dictionary with cleanup results
        """
        if not self.config.auto_cleanup:
            return {'message': 'Auto cleanup is disabled'}

        threshold_days = days_threshold or self.config.cleanup_threshold_days
        cutoff_date = datetime.now() - timedelta(days=threshold_days)

        cleanup_results = {
            'files_removed': 0,
            'space_freed_bytes': 0,
            'errors': []
        }

        try:
            base_path = Path(self.config.base_path)

            for data_type in ['constituents', 'price']:
                data_dir = base_path / data_type

                if not data_dir.exists():
                    continue

                for file_path in data_dir.iterdir():
                    if file_path.is_file():
                        file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)

                        if file_modified < cutoff_date:
                            try:
                                file_size = file_path.stat().st_size
                                file_path.unlink()

                                cleanup_results['files_removed'] += 1
                                cleanup_results['space_freed_bytes'] += file_size

                                logger.info(f"Removed old file: {file_path}")

                            except Exception as e:
                                cleanup_results['errors'].append(f"Error removing {file_path}: {e}")

        except Exception as e:
            cleanup_results['errors'].append(f"Cleanup error: {e}")

        logger.info(f"Cleanup completed: {cleanup_results['files_removed']} files removed, {cleanup_results['space_freed_bytes'] / (1024*1024):.2f} MB freed")
        return cleanup_results

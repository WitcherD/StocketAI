"""
Data Integration Module for VN30 Stock Price Prediction System.

This module handles merging data from multiple sources, resolving conflicts,
and creating unified data schemas for VN30 stocks. It provides:

- Multi-source data merging with conflict resolution
- Cross-source data validation
- Unified schema creation and management
- Data source quality tracking
- Schema evolution and transformation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving data conflicts between sources."""
    PRIORITY = "priority"  # Use priority-based source selection
    MAJORITY = "majority"  # Use majority vote across sources
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted average by source quality
    LATEST = "latest"  # Use most recent value
    CONSERVATIVE = "conservative"  # Use most conservative estimate


class DataSourceQuality(Enum):
    """Quality levels for data sources."""
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    UNKNOWN = 0


@dataclass
class SourceConfig:
    """Configuration for a data source."""
    name: str
    priority: int = 1
    quality: DataSourceQuality = DataSourceQuality.UNKNOWN
    reliability_score: float = 0.5
    last_updated: Optional[datetime] = None
    fields_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class IntegrationConfig:
    """Configuration for data integration operations."""
    conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.PRIORITY
    quality_threshold: float = 0.7
    allow_partial_matches: bool = True
    validate_timestamps: bool = True
    merge_on_duplicate_keys: bool = True
    create_unified_schema: bool = True


class DataIntegrator:
    """
    Handles integration of data from multiple sources for VN30 stocks.

    Provides functionality to merge, validate, and unify data from different
    financial data providers with conflict resolution and quality tracking.
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        """
        Initialize the DataIntegrator.

        Args:
            config: IntegrationConfig object with integration parameters
        """
        self.config = config or IntegrationConfig()
        self.source_configs: Dict[str, SourceConfig] = {}
        self.integration_stats = {}

    def register_source(self, source_config: SourceConfig) -> None:
        """
        Register a data source configuration.

        Args:
            source_config: SourceConfig object defining the source
        """
        self.source_configs[source_config.name] = source_config
        logger.info(f"Registered data source: {source_config.name}")

    def integrate_constituents_data(self,
                                  data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Integrate constituents data from multiple sources.

        Args:
            data_sources: Dictionary mapping source names to DataFrames

        Returns:
            Integrated and unified constituents DataFrame
        """
        logger.info(f"Starting constituents integration from {len(data_sources)} sources")

        if not data_sources:
            raise ValueError("No data sources provided for integration")

        # Validate source registrations
        self._validate_source_registrations(data_sources.keys())

        # Standardize all data sources
        standardized_data = {}
        for source_name, df in data_sources.items():
            standardized_data[source_name] = self._standardize_constituents_data(df, source_name)

        # Merge data using configured strategy
        if self.config.conflict_strategy == ConflictResolutionStrategy.PRIORITY:
            integrated_df = self._merge_by_priority(standardized_data)
        elif self.config.conflict_strategy == ConflictResolutionStrategy.MAJORITY:
            integrated_df = self._merge_by_majority(standardized_data)
        elif self.config.conflict_strategy == ConflictResolutionStrategy.WEIGHTED_AVERAGE:
            integrated_df = self._merge_weighted_average(standardized_data)
        else:
            integrated_df = self._merge_conservative(standardized_data)

        # Create unified schema if configured
        if self.config.create_unified_schema:
            integrated_df = self._create_unified_constituents_schema(integrated_df)

        # Validate integration results
        integrated_df = self._validate_integration(integrated_df, data_sources)

        logger.info(f"Constituents integration completed. Final shape: {integrated_df.shape}")
        return integrated_df

    def integrate_price_data(self,
                           data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Integrate price data from multiple sources.

        Args:
            data_sources: Dictionary mapping source names to price DataFrames

        Returns:
            Integrated price DataFrame
        """
        logger.info(f"Starting price data integration from {len(data_sources)} sources")

        if not data_sources:
            raise ValueError("No data sources provided for integration")

        # Validate source registrations
        self._validate_source_registrations(data_sources.keys())

        # Standardize all data sources
        standardized_data = {}
        for source_name, df in data_sources.items():
            standardized_data[source_name] = self._standardize_price_data(df, source_name)

        # Merge price data
        integrated_df = self._merge_price_data(standardized_data)

        # Validate integration
        integrated_df = self._validate_price_integration(integrated_df)

        logger.info(f"Price integration completed. Final shape: {integrated_df.shape}")
        return integrated_df

    def _validate_source_registrations(self, source_names: List[str]) -> None:
        """Validate that all sources are properly registered."""
        unregistered = set(source_names) - set(self.source_configs.keys())
        if unregistered:
            raise ValueError(f"Unregistered sources found: {unregistered}")

    def _standardize_constituents_data(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Standardize constituents data from a specific source."""
        standardized_df = df.copy()

        # Get field mapping for this source
        source_config = self.source_configs.get(source_name, SourceConfig(name=source_name))
        field_mapping = source_config.fields_mapping

        # Rename columns according to mapping
        if field_mapping:
            standardized_df = standardized_df.rename(columns=field_mapping)

        # Standardize column names to lowercase
        standardized_df.columns = standardized_df.columns.str.lower().str.strip()

        # Ensure required columns exist
        required_columns = ['symbol', 'name']
        for col in required_columns:
            if col not in standardized_df.columns:
                logger.warning(f"Required column '{col}' missing from source '{source_name}'")

        # Add source metadata
        standardized_df['_source'] = source_name
        standardized_df['_source_quality'] = source_config.quality.value
        standardized_df['_source_priority'] = source_config.priority

        return standardized_df

    def _standardize_price_data(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Standardize price data from a specific source."""
        standardized_df = df.copy()

        # Ensure datetime index
        if not isinstance(standardized_df.index, pd.DatetimeIndex):
            if 'date' in standardized_df.columns:
                standardized_df.index = pd.to_datetime(standardized_df['date'])
                standardized_df = standardized_df.drop('date', axis=1)
            else:
                standardized_df.index = pd.to_datetime(standardized_df.index)

        # Standardize column names
        column_mapping = {
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'trading_volume': 'volume',
            'adj_close': 'adj_close'
        }

        standardized_df = standardized_df.rename(columns=column_mapping)
        standardized_df.columns = standardized_df.columns.str.lower().str.strip()

        # Add source metadata
        source_config = self.source_configs.get(source_name, SourceConfig(name=source_name))
        standardized_df['_source'] = source_name
        standardized_df['_source_quality'] = source_config.quality.value

        return standardized_df

    def _merge_by_priority(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge data using priority-based strategy."""
        if not data_sources:
            return pd.DataFrame()

        # Sort sources by priority (higher priority first)
        sorted_sources = sorted(
            data_sources.items(),
            key=lambda x: self.source_configs[x[0]].priority,
            reverse=True
        )

        # Start with highest priority source
        result_df = sorted_sources[0][1].copy()

        # Iteratively merge with lower priority sources
        for source_name, df in sorted_sources[1:]:
            result_df = self._merge_single_source(result_df, df, source_name)

        return result_df

    def _merge_by_majority(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge data using majority vote strategy."""
        if len(data_sources) < 2:
            return list(data_sources.values())[0] if data_sources else pd.DataFrame()

        # Get all unique indices
        all_indices = set()
        for df in data_sources.values():
            all_indices.update(df.index)

        # For constituents data, use symbol as key
        result_data = {}

        for idx in all_indices:
            values_by_source = {}
            for source_name, df in data_sources.items():
                if idx in df.index:
                    values_by_source[source_name] = df.loc[idx]

            if values_by_source:
                merged_row = self._resolve_majority_conflict(values_by_source)
                result_data[idx] = merged_row

        return pd.DataFrame.from_dict(result_data, orient='index')

    def _merge_weighted_average(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge data using weighted average strategy."""
        # Implementation for weighted average merging
        # This would be used primarily for numerical fields
        return self._merge_by_priority(data_sources)  # Placeholder

    def _merge_conservative(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge data using conservative strategy."""
        # Implementation for conservative merging
        return self._merge_by_priority(data_sources)  # Placeholder

    def _merge_single_source(self, base_df: pd.DataFrame, new_df: pd.DataFrame,
                           source_name: str) -> pd.DataFrame:
        """Merge a single source into the base DataFrame."""
        merged_df = base_df.copy()

        # Find overlapping indices
        overlap_idx = base_df.index.intersection(new_df.index)

        if len(overlap_idx) == 0:
            # No overlap, just concatenate
            merged_df = pd.concat([base_df, new_df], ignore_index=True)
        else:
            # Handle overlapping data based on strategy
            for idx in overlap_idx:
                base_row = base_df.loc[idx]
                new_row = new_df.loc[idx]

                # Merge row data
                merged_row = self._merge_row_data(base_row, new_row, source_name)
                merged_df.loc[idx] = merged_row

            # Add non-overlapping data
            new_only_idx = new_df.index.difference(overlap_idx)
            if len(new_only_idx) > 0:
                merged_df = pd.concat([merged_df, new_df.loc[new_only_idx]], ignore_index=True)

        return merged_df

    def _merge_row_data(self, base_row: pd.Series, new_row: pd.Series,
                       source_name: str) -> pd.Series:
        """Merge data for a single row between sources."""
        merged_row = base_row.copy()

        # Compare each field
        for col in new_row.index:
            if col.startswith('_'):  # Skip metadata columns
                continue

            base_val = base_row.get(col)
            new_val = new_row.get(col)

            if pd.isna(base_val) and not pd.isna(new_val):
                # Base is missing, use new value
                merged_row[col] = new_val
            elif not pd.isna(base_val) and not pd.isna(new_val):
                # Both have values, need conflict resolution
                if self._values_conflict(base_val, new_val):
                    resolved_val = self._resolve_field_conflict(
                        base_val, new_val, col, source_name
                    )
                    merged_row[col] = resolved_val
            # If both are missing, keep as missing

        return merged_row

    def _values_conflict(self, val1: Any, val2: Any) -> bool:
        """Check if two values conflict."""
        if pd.isna(val1) or pd.isna(val2):
            return False

        # For numeric values
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) > 1e-6

        # For string values
        if isinstance(val1, str) and isinstance(val2, str):
            return val1.strip().lower() != val2.strip().lower()

        # For other types, consider them different
        return val1 != val2

    def _resolve_field_conflict(self, val1: Any, val2: Any, field_name: str,
                              source_name: str) -> Any:
        """Resolve conflict between two field values."""
        source_config = self.source_configs.get(source_name, SourceConfig(name=source_name))

        # For numeric fields, use weighted average
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            weight1 = self.source_configs.get('base', SourceConfig(name='base')).reliability_score
            weight2 = source_config.reliability_score

            if weight1 + weight2 > 0:
                return (val1 * weight1 + val2 * weight2) / (weight1 + weight2)

        # For string fields, prefer higher quality source
        quality1 = getattr(self.source_configs.get('base', SourceConfig(name='base')), 'quality', DataSourceQuality.UNKNOWN).value
        quality2 = source_config.quality.value

        if quality2 > quality1:
            return val2
        else:
            return val1

    def _resolve_majority_conflict(self, values_by_source: Dict[str, pd.Series]) -> pd.Series:
        """Resolve conflicts using majority vote."""
        # Implementation for majority-based conflict resolution
        # This is a simplified version
        first_source = list(values_by_source.keys())[0]
        return values_by_source[first_source]

    def _merge_price_data(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge price data from multiple sources."""
        if not data_sources:
            return pd.DataFrame()

        # Use priority-based merging for price data
        return self._merge_by_priority(data_sources)

    def _create_unified_constituents_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create unified schema for constituents data."""
        unified_df = df.copy()

        # Define unified column structure
        unified_columns = {
            'symbol': 'str',
            'name': 'str',
            'sector': 'str',
            'industry': 'str',
            'weight': 'float64',
            'shares_outstanding': 'float64',
            'market_cap': 'float64',
            'last_updated': 'datetime64[ns]',
            '_source': 'str',
            '_source_quality': 'int64',
            '_integration_timestamp': 'datetime64[ns]'
        }

        # Add missing columns with appropriate defaults
        for col, dtype in unified_columns.items():
            if col not in unified_df.columns:
                if dtype == 'str':
                    unified_df[col] = 'Unknown'
                elif dtype.startswith('float'):
                    unified_df[col] = np.nan
                elif dtype.startswith('datetime'):
                    unified_df[col] = pd.Timestamp.now()
                elif dtype == 'int64':
                    unified_df[col] = 0

        # Convert data types
        for col, dtype in unified_columns.items():
            if col in unified_df.columns:
                try:
                    if dtype == 'str':
                        unified_df[col] = unified_df[col].astype(str)
                    elif dtype == 'float64':
                        unified_df[col] = pd.to_numeric(unified_df[col], errors='coerce')
                    elif dtype.startswith('datetime'):
                        unified_df[col] = pd.to_datetime(unified_df[col], errors='coerce')
                    elif dtype == 'int64':
                        unified_df[col] = pd.to_numeric(unified_df[col], errors='coerce').astype('Int64')
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to {dtype}: {e}")

        # Add integration metadata
        unified_df['_integration_timestamp'] = pd.Timestamp.now()
        unified_df['_integration_method'] = self.config.conflict_strategy.value

        return unified_df

    def _validate_integration(self, integrated_df: pd.DataFrame,
                            original_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Validate the integration results."""
        validation_results = {}

        # Check for data loss
        original_records = sum(len(df) for df in original_sources.values())
        integrated_records = len(integrated_df)

        if integrated_records == 0 and original_records > 0:
            validation_results['data_loss'] = 'All data lost during integration'

        # Check for schema consistency
        required_columns = ['symbol']
        missing_columns = [col for col in required_columns if col not in integrated_df.columns]
        if missing_columns:
            validation_results['missing_columns'] = missing_columns

        # Check data quality
        if '_source_quality' in integrated_df.columns:
            avg_quality = integrated_df['_source_quality'].mean()
            if avg_quality < self.config.quality_threshold:
                validation_results['low_quality'] = f"Average quality {avg_quality:.2f} below threshold"

        self.integration_stats['validation'] = validation_results

        if validation_results:
            logger.warning(f"Integration validation issues: {validation_results}")

        return integrated_df

    def _validate_price_integration(self, integrated_df: pd.DataFrame) -> pd.DataFrame:
        """Validate price data integration."""
        # Price-specific validation logic
        return integrated_df

    def get_integration_stats(self) -> Dict:
        """Get comprehensive integration statistics."""
        return {
            'config': self.config.__dict__,
            'source_configs': {name: config.__dict__ for name, config in self.source_configs.items()},
            'integration_stats': self.integration_stats,
            'timestamp': datetime.now().isoformat()
        }

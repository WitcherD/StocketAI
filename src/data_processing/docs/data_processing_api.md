# Data Processing Module - API Documentation

## Overview

The Data Processing Module provides comprehensive data cleaning, integration, validation, normalization, and storage capabilities for the VN30 Stock Price Prediction System. This module transforms raw financial data into high-quality, analysis-ready datasets.

## Core Modules

### 1. DataCleaner

Handles data cleaning operations including missing data imputation, outlier detection, and data validation.

#### Class: DataCleaner

```python
from data_processing import DataCleaner
from data_processing.cleaning import CleaningConfig

# Initialize with custom configuration
config = CleaningConfig(
    imputation_method="interpolation",
    outlier_method="iqr",
    winsorize_outliers=True
)
cleaner = DataCleaner(config)

# Clean constituents data
cleaned_constituents = cleaner.clean_constituents_data(raw_constituents_df)

# Clean price data
cleaned_price = cleaner.clean_price_data(raw_price_df)

# Get cleaning statistics
stats = cleaner.get_cleaning_stats()
```

#### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `imputation_method` | "forward_fill" | Method for filling missing values |
| `outlier_method` | "iqr" | Outlier detection method |
| `winsorize_outliers` | True | Whether to winsorize outliers |
| `remove_outliers` | False | Whether to remove outliers |
| `outlier_threshold` | 3.0 | Z-score threshold for outliers |

#### Supported Imputation Methods
- `"forward_fill"` - Forward fill missing values
- `"interpolation"` - Linear interpolation
- `"mean"` - Replace with column mean
- `"median"` - Replace with column median

### 2. DataIntegrator

Handles integration of data from multiple sources with conflict resolution.

```python
from data_processing import DataIntegrator
from data_processing.integration import IntegrationConfig, SourceConfig, DataSourceQuality

# Initialize integrator
config = IntegrationConfig(
    conflict_strategy="PRIORITY",
    create_unified_schema=True
)
integrator = DataIntegrator(config)

# Register data sources
integrator.register_source(SourceConfig(
    name="tcbs",
    priority=1,
    quality=DataSourceQuality.HIGH,
    reliability_score=0.95
))

integrator.register_source(SourceConfig(
    name="vndirect",
    priority=2,
    quality=DataSourceQuality.MEDIUM,
    reliability_score=0.85
))

# Integrate data from multiple sources
data_sources = {
    "tcbs": tcbs_constituents_df,
    "vndirect": vndirect_constituents_df
}

integrated_data = integrator.integrate_constituents_data(data_sources)
```

#### Conflict Resolution Strategies
- `"PRIORITY"` - Use priority-based source selection
- `"MAJORITY"` - Use majority vote across sources
- `"WEIGHTED_AVERAGE"` - Weighted average by source quality
- `"LATEST"` - Use most recent value
- `"CONSERVATIVE"` - Use most conservative estimate

### 3. DataValidator

Provides comprehensive data quality validation and health monitoring.

```python
from data_processing import DataValidator
from data_processing.validation import ValidationConfig

# Initialize validator
config = ValidationConfig(
    enable_alerting=True,
    historical_tracking=True
)
validator = DataValidator(config)

# Validate data quality
validation_results = validator.validate_price_data(price_df)
constituents_validation = validator.validate_constituents_data(constituents_df)

# Get quality score
overall_score = validation_results['overall_score']
quality_scores = validation_results['quality_scores']

# Generate quality report
report = validator.generate_quality_report()
```

#### Quality Dimensions
- **Completeness** - Percentage of non-missing values
- **Accuracy** - Validity of data relationships and constraints
- **Consistency** - Data type and format consistency
- **Timeliness** - Data freshness and update frequency
- **Validity** - Format and range validation
- **Uniqueness** - Absence of duplicate records

### 4. DataNormalizer

Provides various normalization and scaling methods for financial data.

```python
from data_processing import DataNormalizer
from data_processing.normalization import NormalizationConfig, NormalizationMethod

# Initialize normalizer
config = NormalizationConfig(
    method=NormalizationMethod.Z_SCORE,
    handle_outliers=True
)
normalizer = DataNormalizer(config)

# Normalize price data
normalized_price = normalizer.normalize_price_data(price_df)

# Normalize constituents data
normalized_constituents = normalizer.normalize_constituents_data(constituents_df)

# Get normalization statistics
stats = normalizer.get_normalization_stats()
```

#### Normalization Methods
- `MIN_MAX` - Min-max scaling to specified range
- `Z_SCORE` - Z-score standardization (mean=0, std=1)
- `ROBUST` - Robust scaling using median and IQR
- `LOG` - Log transformation with standardization
- `POWER` - Power transformation (Yeo-Johnson/Box-Cox)
- `QUANTILE` - Quantile-based normalization

### 5. ProcessedDataManager

Manages storage and retrieval of processed data with versioning.

```python
from data_processing import ProcessedDataManager
from data_processing.storage import StorageConfig, DataMetadata

# Initialize storage manager
config = StorageConfig(
    base_path="data/processed",
    format="PARQUET",
    compression="GZIP",
    create_versioned_backups=True
)
storage_manager = ProcessedDataManager(config)

# Create metadata
metadata = DataMetadata(
    source="data_processing_pipeline",
    processing_timestamp=datetime.now(),
    data_shape=normalized_df.shape,
    columns=normalized_df.columns.tolist(),
    data_types={col: str(normalized_df[col].dtype) for col in normalized_df.columns},
    processing_pipeline=["cleaning", "validation", "normalization"],
    quality_score=0.95,
    version="1.0.0",
    description="Processed VN30 price data",
    tags=["vn30", "price", "normalized"]
)

# Store data
file_path = storage_manager.store_price_data(normalized_df, "vn30_daily", metadata)

# Retrieve data
retrieved_df, retrieved_metadata = storage_manager.retrieve_price_data("vn30_daily")

# List available datasets
datasets = storage_manager.list_datasets()
```

#### Supported Storage Formats
- `PARQUET` - Columnar storage (recommended)
- `CSV` - Comma-separated values
- `JSON` - JavaScript Object Notation
- `HDF5` - Hierarchical Data Format
- `FEATHER` - Fast columnar storage
- `PICKLE` - Python object serialization

### 6. TimeSeriesProcessor

Handles time series operations including alignment and resampling.

```python
from data_processing import TimeSeriesProcessor
from data_processing.time_series import TimeSeriesConfig, FillMethod

# Initialize time series processor
config = TimeSeriesConfig(
    base_frequency="D",
    target_frequencies=["D", "W", "M"],
    fill_method=FillMethod.FORWARD_FILL
)
ts_processor = TimeSeriesProcessor(config)

# Align time series
aligned_df = ts_processor.align_time_series(price_df, "D")

# Resample to weekly frequency
weekly_df = ts_processor.resample_time_series(price_df, "W")

# Create multi-frequency dataset
multi_freq_data = ts_processor.create_multi_frequency_dataset(price_df, ["W", "M"])

# Validate time consistency
validation = ts_processor.validate_time_consistency(price_df)
```

## Complete Pipeline Example

```python
from data_processing import *
from data_processing.cleaning import CleaningConfig
from data_processing.integration import IntegrationConfig, SourceConfig, DataSourceQuality
from data_processing.validation import ValidationConfig
from data_processing.normalization import NormalizationConfig, NormalizationMethod
from data_processing.storage import StorageConfig, DataMetadata
from data_processing.time_series import TimeSeriesConfig

def process_vn30_data(raw_price_data, raw_constituents_data):
    """Complete data processing pipeline for VN30 data."""

    # Step 1: Data Cleaning
    cleaner = DataCleaner(CleaningConfig(
        imputation_method="interpolation",
        outlier_method="iqr",
        winsorize_outliers=True
    ))

    cleaned_price = cleaner.clean_price_data(raw_price_data)
    cleaned_constituents = cleaner.clean_constituents_data(raw_constituents_data)

    # Step 2: Time Series Processing
    ts_processor = TimeSeriesProcessor(TimeSeriesConfig())
    aligned_price = ts_processor.align_time_series(cleaned_price)
    time_validation = ts_processor.validate_time_consistency(aligned_price)

    # Step 3: Data Integration (if multiple sources)
    integrator = DataIntegrator(IntegrationConfig())
    # Register sources and integrate if needed
    integrated_constituents = cleaned_constituents  # Simplified for single source

    # Step 4: Data Validation
    validator = DataValidator(ValidationConfig())
    price_validation = validator.validate_price_data(aligned_price)
    constituents_validation = validator.validate_constituents_data(integrated_constituents)

    # Step 5: Data Normalization
    normalizer = DataNormalizer(NormalizationConfig(
        method=NormalizationMethod.Z_SCORE,
        handle_outliers=True
    ))

    normalized_price = normalizer.normalize_price_data(aligned_price)
    normalized_constituents = normalizer.normalize_constituents_data(integrated_constituents)

    # Step 6: Data Storage
    storage_manager = ProcessedDataManager(StorageConfig())

    # Store with metadata
    price_metadata = DataMetadata(
        source="vn30_pipeline",
        processing_timestamp=datetime.now(),
        data_shape=normalized_price.shape,
        columns=normalized_price.columns.tolist(),
        data_types={col: str(normalized_price[col].dtype) for col in normalized_price.columns},
        processing_pipeline=["cleaning", "validation", "normalization"],
        quality_score=price_validation['overall_score'],
        version="1.0.0",
        description="Processed VN30 daily price data",
        tags=["vn30", "price", "daily", "normalized"]
    )

    storage_manager.store_price_data(normalized_price, "vn30_daily", price_metadata)
    storage_manager.store_constituents_data(normalized_constituents, "vn30_constituents", price_metadata)

    return {
        'cleaned_data': {'price': cleaned_price, 'constituents': cleaned_constituents},
        'validation_scores': {'price': price_validation['overall_score'], 'constituents': constituents_validation['overall_score']},
        'normalized_data': {'price': normalized_price, 'constituents': normalized_constituents},
        'processing_stats': {
            'cleaner': cleaner.get_cleaning_stats(),
            'validator': validator.get_validation_stats(),
            'normalizer': normalizer.get_normalization_stats(),
            'storage': storage_manager.get_storage_stats()
        }
    }

# Usage
results = process_vn30_data(raw_price_data, raw_constituents_data)
print(f"Processing complete! Quality scores: {results['validation_scores']}")
```

## Configuration Management

### Environment-Based Configuration

```python
import os
from data_processing.cleaning import CleaningConfig
from data_processing.validation import ValidationConfig

# Development configuration
if os.getenv('ENVIRONMENT') == 'development':
    cleaning_config = CleaningConfig(
        imputation_method="interpolation",
        outlier_method="iqr",
        remove_outliers=False  # Keep outliers in dev
    )
    validation_config = ValidationConfig(
        enable_alerting=False,  # Disable alerts in dev
        quality_thresholds=QualityThresholds(
            completeness_threshold=0.90,  # Lower threshold for dev
            accuracy_threshold=0.85
        )
    )

# Production configuration
else:
    cleaning_config = CleaningConfig(
        imputation_method="forward_fill",
        outlier_method="iqr",
        remove_outliers=True  # Strict outlier removal
    )
    validation_config = ValidationConfig(
        enable_alerting=True,
        quality_thresholds=QualityThresholds(
            completeness_threshold=0.95,  # Higher threshold for prod
            accuracy_threshold=0.90
        )
    )
```

## Error Handling

All modules include comprehensive error handling:

```python
try:
    cleaned_data = cleaner.clean_price_data(raw_data)
except ValueError as e:
    logger.error(f"Data cleaning failed: {e}")
    # Handle error appropriately
except Exception as e:
    logger.error(f"Unexpected error in data cleaning: {e}")
    # Fallback processing or raise appropriate error
```

## Performance Considerations

### Memory Management
- Use chunked processing for large datasets
- Implement lazy loading where appropriate
- Monitor memory usage in long-running processes

### Processing Optimization
- Configure appropriate chunk sizes for storage operations
- Use vectorized operations in pandas
- Consider parallel processing for independent operations

### Storage Optimization
- Use PARQUET format for large datasets
- Enable compression for storage efficiency
- Implement data partitioning for query performance

## Best Practices

### Data Pipeline Design
1. **Validate early** - Check data quality at each pipeline stage
2. **Preserve metadata** - Track data lineage throughout processing
3. **Handle errors gracefully** - Implement appropriate fallback strategies
4. **Monitor performance** - Track processing times and resource usage
5. **Version outputs** - Maintain versioned backups of processed data

### Configuration Management
1. **Use environment-specific configs** - Different settings for dev/prod
2. **Document configuration changes** - Track rationale for config decisions
3. **Validate configurations** - Ensure config parameters are within valid ranges
4. **Version configurations** - Track configuration changes over time

### Quality Assurance
1. **Set appropriate thresholds** - Balance quality vs. data availability
2. **Monitor quality trends** - Track quality metrics over time
3. **Alert on quality degradation** - Implement alerting for quality issues
4. **Review quality reports** - Regular review of data quality metrics

## Troubleshooting

### Common Issues

#### Missing Data Handling
```python
# Problem: Too many missing values after cleaning
# Solution: Adjust imputation strategy or fill limits
cleaner = DataCleaner(CleaningConfig(
    imputation_method="interpolation",
    fill_limit=10  # Allow larger gaps
))
```

#### Outlier Detection
```python
# Problem: Too many outliers detected
# Solution: Adjust outlier detection parameters
cleaner = DataCleaner(CleaningConfig(
    outlier_method="iqr",
    outlier_threshold=2.0  # Less strict threshold
))
```

#### Performance Issues
```python
# Problem: Slow processing of large datasets
# Solution: Use chunked processing
storage_manager = ProcessedDataManager(StorageConfig(
    chunk_size=50000  # Smaller chunks for memory efficiency
))
```

## API Reference

### Module Imports
```python
# Import all modules
from data_processing import *

# Import specific modules
from data_processing import DataCleaner, DataIntegrator, DataValidator
from data_processing.cleaning import CleaningConfig
from data_processing.integration import IntegrationConfig, SourceConfig
from data_processing.validation import ValidationConfig, QualityThresholds
from data_processing.normalization import NormalizationConfig, NormalizationMethod
from data_processing.storage import StorageConfig, DataMetadata
from data_processing.time_series import TimeSeriesConfig, FillMethod
```

### Key Classes and Functions

#### DataCleaner
- `clean_constituents_data(df)` - Clean VN30 constituents data
- `clean_price_data(df)` - Clean OHLCV price data
- `get_cleaning_stats()` - Get cleaning operation statistics

#### DataIntegrator
- `register_source(config)` - Register data source configuration
- `integrate_constituents_data(sources)` - Integrate constituents from multiple sources
- `integrate_price_data(sources)` - Integrate price data from multiple sources

#### DataValidator
- `validate_constituents_data(df)` - Validate constituents data quality
- `validate_price_data(df)` - Validate price data quality
- `generate_quality_report()` - Generate comprehensive quality report

#### DataNormalizer
- `normalize_price_data(df)` - Normalize OHLCV price data
- `normalize_constituents_data(df)` - Normalize constituents data
- `fit_transform(df)` - Fit normalizer and transform data

#### ProcessedDataManager
- `store_price_data(df, name, metadata)` - Store processed price data
- `store_constituents_data(df, name, metadata)` - Store processed constituents data
- `retrieve_price_data(name, version)` - Retrieve stored price data
- `retrieve_constituents_data(name, version)` - Retrieve stored constituents data

#### TimeSeriesProcessor
- `align_time_series(df, frequency)` - Align time series to frequency
- `resample_time_series(df, frequency, methods)` - Resample to different frequency
- `validate_time_consistency(df)` - Validate time series consistency
- `create_multi_frequency_dataset(df, frequencies)` - Create multi-frequency dataset

This API documentation provides comprehensive guidance for using the Data Processing Module in the VN30 Stock Price Prediction System.

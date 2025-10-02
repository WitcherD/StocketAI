# Data Processing Module - Configuration Guide

## Overview

This guide provides detailed information about configuring the Data Processing Module for different environments and use cases. Proper configuration is essential for optimal performance and data quality.

## Environment-Based Configuration

### Development Environment

```python
# config/development.yaml
data_processing:
  cleaning:
    imputation_method: "interpolation"
    outlier_method: "iqr"
    remove_outliers: false
    outlier_threshold: 2.0

  validation:
    enable_alerting: false
    quality_thresholds:
      completeness_threshold: 0.90
      accuracy_threshold: 0.85

  normalization:
    method: "z_score"
    handle_outliers: true

  storage:
    base_path: "data/processed_dev"
    format: "parquet"
    compression: "gzip"
    create_versioned_backups: false
```

### Production Environment

```python
# config/production.yaml
data_processing:
  cleaning:
    imputation_method: "forward_fill"
    outlier_method: "iqr"
    remove_outliers: true
    outlier_threshold: 3.0

  validation:
    enable_alerting: true
    quality_thresholds:
      completeness_threshold: 0.95
      accuracy_threshold: 0.90

  normalization:
    method: "robust"
    handle_outliers: true

  storage:
    base_path: "data/processed"
    format: "parquet"
    compression: "gzip"
    create_versioned_backups: true
    max_versions: 10
```

## Component Configuration Details

### 1. DataCleaner Configuration

#### CleaningConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `imputation_method` | str | "forward_fill" | Method for filling missing values |
| `max_consecutive_missing` | int | 5 | Max consecutive missing values to fill |
| `outlier_method` | str | "iqr" | Outlier detection method ("iqr" or "zscore") |
| `outlier_threshold` | float | 3.0 | Threshold for outlier detection |
| `remove_outliers` | bool | False | Whether to remove detected outliers |
| `winsorize_outliers` | bool | True | Whether to winsorize outliers |
| `min_price` | float | 0.01 | Minimum valid price value |
| `max_price` | float | 1e8 | Maximum valid price value |
| `remove_weekends` | bool | True | Whether to remove weekend data |

#### Example Configurations

```python
# High-quality cleaning for production
production_cleaning = CleaningConfig(
    imputation_method="interpolation",
    outlier_method="iqr",
    outlier_threshold=3.0,
    remove_outliers=True,
    winsorize_outliers=False,
    max_consecutive_missing=3
)

# Fast cleaning for development
development_cleaning = CleaningConfig(
    imputation_method="forward_fill",
    outlier_method="iqr",
    outlier_threshold=2.0,
    remove_outliers=False,
    winsorize_outliers=True,
    max_consecutive_missing=10
)

# Conservative cleaning for research
research_cleaning = CleaningConfig(
    imputation_method="mean",
    outlier_method="zscore",
    outlier_threshold=2.5,
    remove_outliers=False,
    winsorize_outliers=True,
    max_consecutive_missing=1
)
```

### 2. DataIntegrator Configuration

#### IntegrationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conflict_strategy` | str | "priority" | Strategy for resolving conflicts |
| `quality_threshold` | float | 0.7 | Minimum quality threshold for sources |
| `allow_partial_matches` | bool | True | Allow partial field matching |
| `validate_timestamps` | bool | True | Validate timestamp consistency |
| `merge_on_duplicate_keys` | bool | True | Merge data with duplicate keys |
| `create_unified_schema` | bool | True | Create unified output schema |

#### Source Configuration

```python
# High-quality source configuration
tcbs_config = SourceConfig(
    name="tcbs",
    priority=1,
    quality=DataSourceQuality.HIGH,
    reliability_score=0.95,
    fields_mapping={
        'stock_code': 'symbol',
        'company_name': 'name',
        'sector_name': 'sector',
        'industry_name': 'industry'
    }
)

# Medium-quality source configuration
vndirect_config = SourceConfig(
    name="vndirect",
    priority=2,
    quality=DataSourceQuality.MEDIUM,
    reliability_score=0.85,
    fields_mapping={
        'code': 'symbol',
        'name': 'name',
        'sector': 'sector'
    }
)
```

### 3. DataValidator Configuration

#### QualityThresholds Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `completeness_threshold` | float | 0.95 | Minimum completeness ratio |
| `accuracy_threshold` | float | 0.90 | Minimum accuracy ratio |
| `consistency_threshold` | float | 0.85 | Minimum consistency ratio |
| `timeliness_threshold` | float | 0.80 | Minimum timeliness ratio |
| `validity_threshold` | float | 0.95 | Minimum validity ratio |
| `uniqueness_threshold` | float | 0.99 | Minimum uniqueness ratio |

#### ValidationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_alerting` | bool | True | Enable quality alerts |
| `alert_webhook_url` | str | None | Webhook URL for alerts |
| `quality_report_path` | str | None | Path for quality reports |
| `historical_tracking` | bool | True | Enable quality history tracking |
| `max_history_days` | int | 90 | Days to keep quality history |

### 4. DataNormalizer Configuration

#### NormalizationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | "z_score" | Normalization method |
| `feature_range` | tuple | (0, 1) | Target range for min-max scaling |
| `rolling_window` | int | 252 | Rolling window size |
| `quantile_range` | tuple | (0.05, 0.95) | Quantile range for robust scaling |
| `handle_outliers` | bool | True | Handle outliers before normalization |
| `copy` | bool | True | Copy input data |

### 5. Storage Configuration

#### StorageConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_path` | str | "data/processed" | Base directory for storage |
| `format` | str | "parquet" | Storage format |
| `compression` | str | "gzip" | Compression method |
| `create_versioned_backups` | bool | True | Create versioned backups |
| `max_versions` | int | 5 | Maximum backup versions to keep |
| `enable_metadata_tracking` | bool | True | Track metadata |
| `chunk_size` | int | 100000 | Chunk size for large files |
| `auto_cleanup` | bool | True | Enable automatic cleanup |
| `cleanup_threshold_days` | int | 30 | Days before cleanup |

## Configuration Examples by Use Case

### Use Case 1: High-Frequency Trading Data

```python
# Configuration for high-frequency data processing
hft_config = {
    'cleaning': CleaningConfig(
        imputation_method="interpolation",
        outlier_method="iqr",
        outlier_threshold=2.5,  # Tighter threshold for HFT
        remove_outliers=True,
        max_consecutive_missing=1  # Very strict on missing data
    ),
    'validation': ValidationConfig(
        quality_thresholds=QualityThresholds(
            completeness_threshold=0.99,  # Very high completeness required
            timeliness_threshold=0.95,    # Very fresh data required
            accuracy_threshold=0.95       # High accuracy required
        )
    ),
    'storage': StorageConfig(
        format="parquet",
        compression="gzip",
        chunk_size=50000  # Smaller chunks for frequent updates
    )
}
```

### Use Case 2: Research and Analysis

```python
# Configuration for research and analysis
research_config = {
    'cleaning': CleaningConfig(
        imputation_method="interpolation",
        outlier_method="iqr",
        remove_outliers=False,  # Keep outliers for analysis
        winsorize_outliers=True,
        outlier_threshold=2.0   # More permissive threshold
    ),
    'validation': ValidationConfig(
        enable_alerting=False,  # Disable alerts during research
        historical_tracking=True,
        quality_thresholds=QualityThresholds(
            completeness_threshold=0.85,  # More permissive
            accuracy_threshold=0.80
        )
    ),
    'normalization': NormalizationConfig(
        method="robust",  # Robust to outliers
        handle_outliers=True
    )
}
```

### Use Case 3: Batch Processing

```python
# Configuration for batch processing
batch_config = {
    'storage': StorageConfig(
        base_path="data/processed_batch",
        format="parquet",
        compression="gzip",
        chunk_size=200000,  # Larger chunks for batch processing
        create_versioned_backups=True,
        max_versions=20     # Keep more versions for batch jobs
    ),
    'validation': ValidationConfig(
        enable_alerting=True,
        quality_report_path="reports/batch_quality.json"
    )
}
```

## Configuration Validation

### Validating Configuration Parameters

```python
def validate_configuration(config):
    """Validate configuration parameters."""

    # Validate cleaning config
    if config.imputation_method not in ["forward_fill", "interpolation", "mean", "median"]:
        raise ValueError(f"Invalid imputation method: {config.imputation_method}")

    if not (0 < config.outlier_threshold <= 5):
        raise ValueError(f"Outlier threshold must be between 0 and 5: {config.outlier_threshold}")

    # Validate storage config
    if config.chunk_size <= 0:
        raise ValueError(f"Chunk size must be positive: {config.chunk_size}")

    if config.max_versions < 1:
        raise ValueError(f"Max versions must be at least 1: {config.max_versions}")

    # Validate quality thresholds
    for threshold_name in ['completeness', 'accuracy', 'consistency', 'timeliness', 'validity', 'uniqueness']:
        threshold_value = getattr(config.quality_thresholds, f'{threshold_name}_threshold')
        if not (0 <= threshold_value <= 1):
            raise ValueError(f"Threshold {threshold_name} must be between 0 and 1: {threshold_value}")

    return True
```

### Configuration Templates

#### Template 1: Minimal Configuration

```python
# Minimal configuration for quick testing
minimal_config = {
    'cleaning': CleaningConfig(),
    'validation': ValidationConfig(enable_alerting=False),
    'normalization': NormalizationConfig(method="z_score"),
    'storage': StorageConfig(create_versioned_backups=False)
}
```

#### Template 2: Comprehensive Configuration

```python
# Comprehensive configuration for production
comprehensive_config = {
    'cleaning': CleaningConfig(
        imputation_method="interpolation",
        outlier_method="iqr",
        outlier_threshold=3.0,
        remove_outliers=True,
        winsorize_outliers=False,
        max_consecutive_missing=3
    ),
    'integration': IntegrationConfig(
        conflict_strategy="PRIORITY",
        create_unified_schema=True,
        quality_threshold=0.8
    ),
    'validation': ValidationConfig(
        enable_alerting=True,
        historical_tracking=True,
        quality_thresholds=QualityThresholds(
            completeness_threshold=0.95,
            accuracy_threshold=0.90,
            consistency_threshold=0.85,
            timeliness_threshold=0.80,
            validity_threshold=0.95,
            uniqueness_threshold=0.99
        )
    ),
    'normalization': NormalizationConfig(
        method="robust",
        handle_outliers=True,
        quantile_range=(0.01, 0.99)
    ),
    'storage': StorageConfig(
        base_path="data/processed",
        format="parquet",
        compression="gzip",
        create_versioned_backups=True,
        max_versions=10,
        enable_metadata_tracking=True,
        chunk_size=100000,
        auto_cleanup=True,
        cleanup_threshold_days=30
    )
}
```

## Environment Variables

### Supported Environment Variables

```bash
# General settings
export ENVIRONMENT=production
export LOG_LEVEL=INFO

# Data processing settings
export DATA_PROCESSING_BASE_PATH="data/processed"
export DATA_PROCESSING_FORMAT="parquet"
export DATA_PROCESSING_COMPRESSION="gzip"

# Quality thresholds
export QUALITY_COMPLETENESS_THRESHOLD=0.95
export QUALITY_ACCURACY_THRESHOLD=0.90

# Performance settings
export PROCESSING_CHUNK_SIZE=100000
export ENABLE_BACKUPS=true
export MAX_BACKUP_VERSIONS=10
```

### Loading Configuration from Environment

```python
import os
from data_processing.storage import StorageConfig

def load_config_from_environment():
    """Load configuration from environment variables."""

    storage_config = StorageConfig(
        base_path=os.getenv('DATA_PROCESSING_BASE_PATH', 'data/processed'),
        format=os.getenv('DATA_PROCESSING_FORMAT', 'parquet'),
        compression=os.getenv('DATA_PROCESSING_COMPRESSION', 'gzip'),
        create_versioned_backups=os.getenv('ENABLE_BACKUPS', 'true').lower() == 'true',
        max_versions=int(os.getenv('MAX_BACKUP_VERSIONS', '10')),
        chunk_size=int(os.getenv('PROCESSING_CHUNK_SIZE', '100000'))
    )

    return storage_config
```

## Configuration Files

### YAML Configuration

```yaml
# data_processing_config.yaml
cleaning:
  imputation_method: "interpolation"
  outlier_method: "iqr"
  outlier_threshold: 3.0
  remove_outliers: true
  winsorize_outliers: false

validation:
  enable_alerting: true
  historical_tracking: true
  quality_thresholds:
    completeness_threshold: 0.95
    accuracy_threshold: 0.90
    consistency_threshold: 0.85
    timeliness_threshold: 0.80
    validity_threshold: 0.95
    uniqueness_threshold: 0.99

normalization:
  method: "robust"
  handle_outliers: true
  quantile_range: [0.05, 0.95]

storage:
  base_path: "data/processed"
  format: "parquet"
  compression: "gzip"
  create_versioned_backups: true
  max_versions: 10
  chunk_size: 100000
```

### Loading YAML Configuration

```python
import yaml
from pathlib import Path

def load_yaml_config(config_path: str = "config/data_processing_config.yaml"):
    """Load configuration from YAML file."""

    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # Convert to configuration objects
    cleaning_config = CleaningConfig(**config_dict.get('cleaning', {}))
    validation_config = ValidationConfig(**config_dict.get('validation', {}))
    normalization_config = NormalizationConfig(**config_dict.get('normalization', {}))
    storage_config = StorageConfig(**config_dict.get('storage', {}))

    return {
        'cleaning': cleaning_config,
        'validation': validation_config,
        'normalization': normalization_config,
        'storage': storage_config
    }
```

## Performance Tuning

### Memory Optimization

```python
# For large datasets, optimize memory usage
large_dataset_config = {
    'storage': StorageConfig(
        chunk_size=50000,        # Smaller chunks
        compression="gzip",      # Enable compression
        format="parquet"         # Efficient format
    ),
    'cleaning': CleaningConfig(
        max_consecutive_missing=5  # Limit memory usage for filling
    )
}
```

### Processing Speed Optimization

```python
# For faster processing
fast_processing_config = {
    'cleaning': CleaningConfig(
        imputation_method="forward_fill",  # Fastest method
        outlier_method="iqr",              # Fast outlier detection
        remove_outliers=False,            # Skip removal for speed
        winsorize_outliers=True           # Fast winsorization
    ),
    'storage': StorageConfig(
        format="parquet",        # Fast format
        compression="none"       # Skip compression for speed
    )
}
```

### Storage Optimization

```python
# For optimal storage efficiency
storage_optimized_config = {
    'storage': StorageConfig(
        format="parquet",           # Most efficient format
        compression="gzip",         # Good compression ratio
        chunk_size=100000,          # Balance between speed and memory
        create_versioned_backups=True,
        max_versions=5,             # Reasonable backup limit
        auto_cleanup=True,          # Enable automatic cleanup
        cleanup_threshold_days=30   # Clean old files
    )
}
```

## Monitoring and Alerting

### Quality Monitoring Configuration

```python
# Configuration for quality monitoring
monitoring_config = ValidationConfig(
    enable_alerting=True,
    alert_webhook_url="https://your-webhook-url.com/alerts",
    quality_report_path="reports/quality_report.json",
    historical_tracking=True,
    max_history_days=90
)
```

### Custom Alert Thresholds

```python
# Custom quality thresholds for different data types
price_quality_thresholds = QualityThresholds(
    completeness_threshold=0.98,  # Very high for price data
    accuracy_threshold=0.95,      # Critical for price accuracy
    timeliness_threshold=0.90,    # Fresh price data needed
    validity_threshold=0.99       # Strict validity for prices
)

constituents_quality_thresholds = QualityThresholds(
    completeness_threshold=0.90,  # Lower for constituents
    accuracy_threshold=0.85,      # More permissive
    timeliness_threshold=0.70,    # Less critical
    validity_threshold=0.90       # Moderate validity requirement
)
```

## Troubleshooting Configuration Issues

### Common Configuration Problems

#### Problem 1: Invalid Parameter Values

```python
# Solution: Validate parameters before use
try:
    config = CleaningConfig(
        imputation_method="invalid_method",  # This will cause an error
        outlier_threshold=-1                 # Invalid negative value
    )
except ValueError as e:
    print(f"Configuration error: {e}")
    # Use default configuration
    config = CleaningConfig()
```

#### Problem 2: Missing Required Parameters

```python
# Solution: Use default values or provide fallbacks
def create_safe_config(custom_params=None):
    """Create configuration with safe defaults."""

    default_params = {
        'imputation_method': 'forward_fill',
        'outlier_method': 'iqr',
        'outlier_threshold': 3.0
    }

    if custom_params:
        default_params.update(custom_params)

    return CleaningConfig(**default_params)
```

#### Problem 3: Environment-Specific Issues

```python
# Solution: Environment-aware configuration
def get_environment_config():
    """Get configuration based on environment."""

    env = os.getenv('ENVIRONMENT', 'development')

    configs = {
        'development': {
            'cleaning': CleaningConfig(remove_outliers=False),
            'validation': ValidationConfig(enable_alerting=False),
            'storage': StorageConfig(create_versioned_backups=False)
        },
        'production': {
            'cleaning': CleaningConfig(remove_outliers=True),
            'validation': ValidationConfig(enable_alerting=True),
            'storage': StorageConfig(create_versioned_backups=True)
        }
    }

    return configs.get(env, configs['development'])
```

## Configuration Best Practices

### 1. Version Control
- Store configuration files in version control
- Document configuration changes
- Use semantic versioning for configuration versions

### 2. Environment Separation
- Use separate configurations for dev/staging/prod
- Avoid hard-coded values in source code
- Use environment variables for sensitive settings

### 3. Validation
- Validate configuration parameters on load
- Provide meaningful error messages for invalid configs
- Use type hints and dataclasses for configuration objects

### 4. Documentation
- Document all configuration parameters
- Provide examples for each parameter
- Explain the impact of configuration changes

### 5. Testing
- Test configurations in isolated environments
- Validate configuration impact on data quality
- Monitor configuration performance impact

## Configuration Examples by Data Type

### Price Data Configuration

```python
price_data_config = {
    'cleaning': CleaningConfig(
        imputation_method="interpolation",  # Good for time series
        outlier_method="iqr",
        outlier_threshold=3.0,
        remove_outliers=True,  # Strict for price data
        min_price=0.01,
        max_price=1000000
    ),
    'validation': ValidationConfig(
        quality_thresholds=QualityThresholds(
            completeness_threshold=0.98,  # High completeness for prices
            accuracy_threshold=0.95,      # High accuracy required
            timeliness_threshold=0.90     # Fresh data needed
        )
    )
}
```

### Constituents Data Configuration

```python
constituents_data_config = {
    'cleaning': CleaningConfig(
        imputation_method="forward_fill",  # Simple filling for static data
        outlier_method="iqr",
        outlier_threshold=2.5,
        remove_outliers=False,  # Keep all constituents
        winsorize_outliers=True
    ),
    'validation': ValidationConfig(
        quality_thresholds=QualityThresholds(
            completeness_threshold=0.90,  # Moderate completeness
            accuracy_threshold=0.85,      # Moderate accuracy
            uniqueness_threshold=0.99     # High uniqueness required
        )
    )
}
```

## Summary

Proper configuration is crucial for optimal performance and data quality. This guide provides:

1. **Environment-specific configurations** for development and production
2. **Component-specific settings** for each module
3. **Use case examples** for different scenarios
4. **Validation and troubleshooting** guidance
5. **Best practices** for configuration management

Always validate your configuration in a test environment before deploying to production, and monitor the impact of configuration changes on data quality and processing performance.

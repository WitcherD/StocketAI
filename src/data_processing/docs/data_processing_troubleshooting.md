# Data Processing Module - Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when using the Data Processing Module. It covers error diagnosis, performance problems, and configuration issues.

## Common Issues and Solutions

### 1. Data Cleaning Issues

#### Issue 1.1: High Missing Data Percentage

**Problem**: After cleaning, a large percentage of data is still missing.

**Symptoms**:
- `cleaned_data.isnull().sum().sum()` returns high values
- Data quality scores show low completeness

**Solutions**:

```python
# Solution 1: Adjust imputation strategy
cleaner = DataCleaner(CleaningConfig(
    imputation_method="interpolation",  # Better for time series
    fill_limit=10                       # Allow larger gaps
))

# Solution 2: Use different filling method for specific columns
cleaner = DataCleaner(CleaningConfig(
    imputation_method="interpolation",
    max_consecutive_missing=5  # Limit consecutive missing
))
```

**Verification**:
```python
# Check missing data after cleaning
missing_after = cleaned_data.isnull().sum()
print(f"Missing values after cleaning: {missing_after}")

# Check if missing percentage is acceptable
missing_percentage = (missing_after / len(cleaned_data)) * 100
print(f"Missing percentage: {missing_percentage}")
```

#### Issue 1.2: Too Many Outliers Detected

**Problem**: Outlier detection is removing too much valid data.

**Symptoms**:
- Large amount of data removed as outliers
- Warning messages about outlier removal
- Unexpected data loss

**Solutions**:

```python
# Solution 1: Adjust outlier detection parameters
cleaner = DataCleaner(CleaningConfig(
    outlier_method="iqr",
    outlier_threshold=2.0,  # Less strict (default is 3.0)
    remove_outliers=False,  # Don't remove, just winsorize
    winsorize_outliers=True
))

# Solution 2: Use different method for financial data
cleaner = DataCleaner(CleaningConfig(
    outlier_method="zscore",
    outlier_threshold=2.5,  # Custom threshold
    remove_outliers=False
))
```

**Verification**:
```python
# Check how many outliers were detected
stats = cleaner.get_cleaning_stats()
outlier_info = stats['cleaning_stats'].get('outliers', {})
for column, count in outlier_info.items():
    print(f"Outliers in {column}: {count}")
```

### 2. Data Integration Issues

#### Issue 2.1: Source Registration Errors

**Problem**: Errors when registering data sources.

**Symptoms**:
- `ValueError: Unregistered sources found`
- `AttributeError: 'SourceConfig' object has no attribute 'name'`

**Solutions**:

```python
# Solution 1: Ensure proper source registration
integrator = DataIntegrator()

# Correct way to register source
source_config = SourceConfig(
    name="tcbs",  # Required: unique name
    priority=1,   # Required: priority level
    quality=DataSourceQuality.HIGH,  # Required: quality level
    reliability_score=0.95  # Optional: reliability score
)

integrator.register_source(source_config)

# Solution 2: Check source configuration
try:
    integrator.register_source(source_config)
    print("✅ Source registered successfully")
except Exception as e:
    print(f"❌ Registration failed: {e}")
```

#### Issue 2.2: Integration Conflicts

**Problem**: Too many conflicts when integrating multiple sources.

**Symptoms**:
- Warning messages about conflicting values
- Unexpected data loss during integration
- Poor integration quality scores

**Solutions**:

```python
# Solution 1: Adjust conflict resolution strategy
integrator = DataIntegrator(IntegrationConfig(
    conflict_strategy="WEIGHTED_AVERAGE",  # Use weighted average
    quality_threshold=0.7                  # Lower quality threshold
))

# Solution 2: Register sources with proper quality scores
integrator.register_source(SourceConfig(
    name="primary_source",
    priority=1,
    quality=DataSourceQuality.HIGH,
    reliability_score=0.95  # High reliability
))

integrator.register_source(SourceConfig(
    name="secondary_source",
    priority=2,
    quality=DataSourceQuality.MEDIUM,
    reliability_score=0.75  # Lower reliability
))
```

### 3. Data Validation Issues

#### Issue 3.1: Poor Quality Scores

**Problem**: Data quality scores are consistently low.

**Symptoms**:
- Overall quality score below acceptable thresholds
- Multiple quality alerts generated
- Validation failures

**Solutions**:

```python
# Solution 1: Review and adjust quality thresholds
quality_thresholds = QualityThresholds(
    completeness_threshold=0.90,  # Lower if data has known gaps
    accuracy_threshold=0.85,      # Adjust based on data characteristics
    consistency_threshold=0.80,   # Lower for diverse data sources
    timeliness_threshold=0.70,    # Lower if data is not real-time
    validity_threshold=0.90,      # Adjust based on data format
    uniqueness_threshold=0.95     # Lower if some duplicates are expected
)

validator = DataValidator(ValidationConfig(
    quality_thresholds=quality_thresholds,
    enable_alerting=True
))

# Solution 2: Investigate specific quality issues
validation_results = validator.validate_price_data(your_data)

# Check which dimensions are problematic
for dimension, score in validation_results['quality_scores'].items():
    if score < 0.8:  # Below 80%
        print(f"Low score in {dimension}: {score".2f"}")

# Check rule-specific results
for rule, result in validation_results['rule_results'].items():
    if not result.get('passed', True):
        print(f"Failed rule: {rule} - {result.get('error', 'Unknown error')}")
```

#### Issue 3.2: False Positive Alerts

**Problem**: Quality alerts are generated for acceptable data.

**Symptoms**:
- Alerts for data that appears correct
- Overly sensitive validation rules
- Too many false positive notifications

**Solutions**:

```python
# Solution 1: Adjust alert thresholds
validator = DataValidator(ValidationConfig(
    enable_alerting=True,
    quality_thresholds=QualityThresholds(
        completeness_threshold=0.85,  # More permissive
        accuracy_threshold=0.80,      # More permissive
        timeliness_threshold=0.60     # Less strict on data age
    )
))

# Solution 2: Disable specific problematic rules
# You can modify the validator to disable certain rules
# or adjust their thresholds individually
```

### 4. Data Normalization Issues

#### Issue 4.1: Normalization Errors

**Problem**: Errors during data normalization.

**Symptoms**:
- `ValueError: Input contains NaN` during normalization
- `ZeroDivisionError` during scaling
- Unexpected normalization results

**Solutions**:

```python
# Solution 1: Ensure data is clean before normalization
# Clean data first
cleaner = DataCleaner()
cleaned_data = cleaner.clean_price_data(raw_data)

# Then normalize
normalizer = DataNormalizer(NormalizationConfig(
    method="z_score",
    handle_outliers=True  # Handle outliers before normalization
))
normalized_data = normalizer.normalize_price_data(cleaned_data)

# Solution 2: Handle edge cases
try:
    normalized_data = normalizer.normalize_price_data(data)
except ValueError as e:
    if "NaN" in str(e):
        print("Data contains NaN values. Cleaning first...")
        # Clean the data and retry
    elif "infinity" in str(e):
        print("Data contains infinite values. Removing...")
        data = data.replace([np.inf, -np.inf], np.nan)
        # Clean and retry
```

#### Issue 4.2: Poor Normalization Results

**Problem**: Normalized data doesn't meet expectations.

**Symptoms**:
- Normalized values outside expected range
- Mean not centered at 0 for z-score
- Unexpected variance in normalized data

**Solutions**:

```python
# Solution 1: Check data before normalization
print(f"Data shape: {data.shape}")
print(f"Data types: {data.dtypes}")
print(f"Missing values: {data.isnull().sum()}")

# Check basic statistics
print(f"Original mean: {data['close'].mean()}")
print(f"Original std: {data['close'].std()}")

# Solution 2: Try different normalization methods
methods_to_try = [
    NormalizationMethod.Z_SCORE,
    NormalizationMethod.ROBUST,
    NormalizationMethod.MIN_MAX
]

for method in methods_to_try:
    config = NormalizationConfig(method=method)
    normalizer = DataNormalizer(config)

    try:
        normalized = normalizer.normalize_price_data(data)
        print(f"{method.value}: Mean={normalized['close'].mean()".3f"}, Std={normalized['close'].std()".3f"}")
    except Exception as e:
        print(f"{method.value}: Failed - {e}")
```

### 5. Data Storage Issues

#### Issue 5.1: Storage Format Errors

**Problem**: Errors when storing or retrieving data.

**Symptoms**:
- `FileNotFoundError` when retrieving data
- `ValueError` when storing data
- Corrupted or unreadable stored files

**Solutions**:

```python
# Solution 1: Check storage configuration
storage_manager = ProcessedDataManager(StorageConfig(
    base_path="data/processed",
    format="parquet",        # Ensure format is supported
    compression="gzip",      # Ensure compression is valid
    create_versioned_backups=True
))

# Solution 2: Verify data before storage
print(f"Data shape: {data.shape}")
print(f"Data types: {data.dtypes}")
print(f"Index type: {type(data.index)}")

# Check for common storage issues
if data.empty:
    print("❌ Cannot store empty DataFrame")
if data.index.duplicated().any():
    print("❌ Cannot store DataFrame with duplicate index")
if data.isnull().all().all():
    print("❌ Cannot store DataFrame with all missing values")
```

#### Issue 5.2: Performance Issues

**Problem**: Slow storage or retrieval operations.

**Symptoms**:
- Long time to store large datasets
- Slow data retrieval
- High memory usage during storage

**Solutions**:

```python
# Solution 1: Optimize storage configuration
storage_manager = ProcessedDataManager(StorageConfig(
    format="parquet",           # Most efficient format
    compression="gzip",         # Good compression
    chunk_size=100000,          # Optimize chunk size
    create_versioned_backups=True
))

# Solution 2: Monitor storage performance
import time

start_time = time.time()
storage_manager.store_price_data(data, "test_dataset", metadata)
storage_time = time.time() - start_time

start_time = time.time()
retrieved_data, _ = storage_manager.retrieve_price_data("test_dataset")
retrieval_time = time.time() - start_time

print(f"Storage time: {storage_time".2f"}s")
print(f"Retrieval time: {retrieval_time".2f"}s")
```

### 6. Time Series Issues

#### Issue 6.1: Time Alignment Problems

**Problem**: Issues with time series alignment.

**Symptoms**:
- Missing timestamps after alignment
- Incorrect frequency conversion
- Time zone issues

**Solutions**:

```python
# Solution 1: Check time series consistency first
ts_processor = TimeSeriesProcessor()
consistency_check = ts_processor.validate_time_consistency(data)

print(f"Index is monotonic: {consistency_check['monotonic_index']}")
print(f"Duplicate timestamps: {consistency_check['duplicate_timestamps']}")
print(f"Inferred frequency: {consistency_check['inferred_frequency']}")

# Solution 2: Align with proper frequency
aligned_data = ts_processor.align_time_series(
    data,
    target_frequency="D"  # Specify target frequency
)
```

#### Issue 6.2: Resampling Errors

**Problem**: Errors during time series resampling.

**Symptoms**:
- `ValueError` during resampling
- Unexpected results after resampling
- Loss of data during resampling

**Solutions**:

```python
# Solution 1: Check data before resampling
print(f"Data shape before resampling: {data.shape}")
print(f"Date range: {data.index.min()} to {data.index.max()}")
print(f"Data frequency: {pd.infer_freq(data.index)}")

# Solution 2: Use appropriate resampling methods
resampling_methods = {
    'open': ResamplingMethod.FIRST,
    'high': ResamplingMethod.MAX,
    'low': ResamplingMethod.MIN,
    'close': ResamplingMethod.LAST,
    'volume': ResamplingMethod.SUM
}

resampled_data = ts_processor.resample_time_series(
    data,
    target_frequency="W",  # Weekly resampling
    resampling_methods=resampling_methods
)
```

### 7. Performance Issues

#### Issue 7.1: Slow Processing

**Problem**: Data processing is taking too long.

**Symptoms**:
- Long execution times for large datasets
- High memory usage
- Timeout errors

**Solutions**:

```python
# Solution 1: Optimize chunk size
storage_manager = ProcessedDataManager(StorageConfig(
    chunk_size=50000,  # Smaller chunks for memory efficiency
    compression="gzip"  # Enable compression to reduce I/O
))

# Solution 2: Process in batches
def process_in_batches(data, batch_size=10000):
    """Process data in smaller batches."""

    results = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        # Process batch
        processed_batch = cleaner.clean_price_data(batch)
        results.append(processed_batch)

    return pd.concat(results)

# Solution 3: Monitor memory usage
import psutil
import os

def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory usage: {get_memory_usage()".2f"} MB")
```

#### Issue 7.2: Memory Issues

**Problem**: Out of memory errors or excessive memory usage.

**Symptoms**:
- `MemoryError` exceptions
- System slowdown
- Swap file usage

**Solutions**:

```python
# Solution 1: Process data in chunks
def process_large_dataset_in_chunks(file_path, chunk_size=50000):
    """Process large dataset in chunks."""

    # Read data in chunks
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process each chunk
        processed_chunk = cleaner.clean_price_data(chunk)
        chunks.append(processed_chunk)

        # Clear memory
        del chunk
        import gc
        gc.collect()

    return pd.concat(chunks)

# Solution 2: Use memory-efficient data types
def optimize_memory_usage(df):
    """Optimize DataFrame memory usage."""

    # Convert to more memory-efficient types
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')

    return df
```

### 8. Configuration Issues

#### Issue 8.1: Invalid Configuration

**Problem**: Configuration parameters are invalid or incompatible.

**Symptoms**:
- `ValueError` when creating configuration objects
- Unexpected behavior during processing
- Type errors

**Solutions**:

```python
# Solution 1: Validate configuration before use
def validate_config(config):
    """Validate configuration parameters."""

    try:
        # Test configuration with sample data
        test_data = create_sample_price_data()
        test_cleaner = DataCleaner(config)
        test_cleaned = test_cleaner.clean_price_data(test_data)
        print("✅ Configuration is valid")
        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

# Solution 2: Use safe configuration loading
def load_safe_config(config_path):
    """Load configuration with fallback to defaults."""

    try:
        config = load_yaml_config(config_path)
        if validate_config(config):
            return config
    except Exception as e:
        print(f"Configuration loading failed: {e}")

    # Return default configuration
    print("Using default configuration")
    return {
        'cleaning': CleaningConfig(),
        'validation': ValidationConfig(),
        'normalization': NormalizationConfig(),
        'storage': StorageConfig()
    }
```

### 9. Integration Issues

#### Issue 9.1: Module Import Errors

**Problem**: Cannot import data processing modules.

**Symptoms**:
- `ModuleNotFoundError` when importing
- `ImportError` for specific modules
- Path-related errors

**Solutions**:

```python
# Solution 1: Ensure proper Python path
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Solution 2: Verify installation
try:
    from data_processing import DataCleaner
    print("✅ Data processing modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed")
```

#### Issue 9.2: Dependency Issues

**Problem**: Missing or incompatible dependencies.

**Symptoms**:
- `ImportError` for required packages
- `VersionError` for incompatible versions
- Runtime errors due to missing dependencies

**Solutions**:

```python
# Solution 1: Check required packages
required_packages = [
    'pandas',
    'numpy',
    'scikit-learn'
]

for package in required_packages:
    try:
        __import__(package)
        print(f"✅ {package} is available")
    except ImportError:
        print(f"❌ {package} is not installed")

# Solution 2: Install missing dependencies
# pip install pandas numpy scikit-learn
```

## Debugging Tools

### 1. Data Inspection Tools

```python
def inspect_data_issues(data):
    """Inspect data for common issues."""

    issues = {}

    # Check for missing values
    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        issues['missing_values'] = {
            'count': missing_count,
            'percentage': (missing_count / data.size) * 100
        }

    # Check for duplicates
    if isinstance(data.index, pd.DatetimeIndex):
        duplicate_timestamps = data.index.duplicated().sum()
        if duplicate_timestamps > 0:
            issues['duplicate_timestamps'] = duplicate_timestamps

    # Check for outliers (basic check)
    for col in data.select_dtypes(include=[np.number]).columns:
        if col in ['open', 'high', 'low', 'close']:
            q75, q25 = np.percentile(data[col].dropna(), [75, 25])
            iqr = q75 - q25
            outlier_count = ((data[col] < (q25 - 1.5 * iqr)) |
                           (data[col] > (q75 + 1.5 * iqr))).sum()
            if outlier_count > 0:
                issues[f'outliers_{col}'] = outlier_count

    return issues
```

### 2. Performance Monitoring

```python
def monitor_performance(func):
    """Decorator to monitor function performance."""

    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Function {func.__name__} executed in {execution_time".3f"} seconds")
        return result

    return wrapper

# Usage
@monitor_performance
def process_data(data):
    # Your processing code here
    return processed_data
```

### 3. Memory Usage Monitoring

```python
def monitor_memory_usage():
    """Monitor memory usage during processing."""

    import psutil
    import os

    def get_memory_mb():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    class MemoryMonitor:
        def __init__(self):
            self.start_memory = get_memory_mb()
            self.peak_memory = self.start_memory

        def check_memory(self):
            current = get_memory_mb()
            self.peak_memory = max(self.peak_memory, current)
            return current

        def get_usage_report(self):
            current = self.check_memory()
            return {
                'start_memory': self.start_memory,
                'current_memory': current,
                'peak_memory': self.peak_memory,
                'memory_increase': current - self.start_memory
            }

    return MemoryMonitor()
```

## Error Code Reference

### Common Error Messages

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| `Input contains NaN` | Missing values in data | Clean data before normalization |
| `Zero standard deviation` | Constant values in column | Remove or handle constant columns |
| `Unregistered sources found` | Sources not registered | Register all sources before integration |
| `Outlier threshold must be positive` | Invalid configuration | Check configuration parameters |
| `Chunk size must be positive` | Invalid storage config | Fix storage configuration |
| `Index is not monotonic` | Unsorted time series | Sort data by timestamp |

### Exception Handling Patterns

```python
def safe_data_processing(data, config=None):
    """Safely process data with comprehensive error handling."""

    try:
        # Step 1: Validate input
        if data is None or data.empty:
            raise ValueError("Input data is empty or None")

        # Step 2: Clean data
        cleaner = DataCleaner(config.get('cleaning', CleaningConfig()))
        cleaned_data = cleaner.clean_price_data(data)

        # Step 3: Validate data
        validator = DataValidator(config.get('validation', ValidationConfig()))
        validation_results = validator.validate_price_data(cleaned_data)

        if validation_results['overall_score'] < 0.7:
            print(f"⚠️ Low quality score: {validation_results['overall_score']".2f"}")

        # Step 4: Normalize data
        normalizer = DataNormalizer(config.get('normalization', NormalizationConfig()))
        normalized_data = normalizer.normalize_price_data(cleaned_data)

        # Step 5: Store data
        storage_manager = ProcessedDataManager(config.get('storage', StorageConfig()))
        metadata = create_metadata("processed_data", validation_results['overall_score'])
        storage_path = storage_manager.store_price_data(normalized_data, "processed", metadata)

        return {
            'success': True,
            'data': normalized_data,
            'quality_score': validation_results['overall_score'],
            'storage_path': storage_path
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'data': data  # Return original data if processing failed
        }
```

## Getting Help

### 1. Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create logger for data processing
logger = logging.getLogger('data_processing')
logger.setLevel(logging.DEBUG)
```

### 2. Generate Diagnostic Report

```python
def generate_diagnostic_report(data, processing_results):
    """Generate comprehensive diagnostic report."""

    report = {
        'timestamp': datetime.now().isoformat(),
        'data_info': {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'dtypes': {col: str(data[col].dtype) for col in data.columns},
            'memory_usage': data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        },
        'processing_results': processing_results,
        'recommendations': []
    }

    # Generate recommendations
    if data.isnull().sum().sum() > 0:
        report['recommendations'].append(
            "Consider improving data cleaning strategy for missing values"
        )

    if processing_results.get('quality_score', 1.0) < 0.8:
        report['recommendations'].append(
            "Review quality thresholds or data cleaning parameters"
        )

    return report
```

### 3. Contact Support

When reporting issues, please include:

1. **Error messages** - Complete error traceback
2. **Data characteristics** - Shape, columns, data types
3. **Configuration** - Your configuration settings
4. **Environment** - Python version, package versions
5. **Expected vs actual behavior** - What you expected vs what happened

## Summary

This troubleshooting guide covers the most common issues encountered when using the Data Processing Module:

1. **Data Cleaning Issues** - Missing data, outliers, validation errors
2. **Data Integration Issues** - Source registration, conflict resolution
3. **Data Validation Issues** - Quality scores, false positives
4. **Data Normalization Issues** - Scaling errors, poor results
5. **Data Storage Issues** - Format errors, performance problems
6. **Time Series Issues** - Alignment, resampling, consistency
7. **Performance Issues** - Speed, memory, optimization
8. **Configuration Issues** - Invalid parameters, environment setup
9. **Integration Issues** - Import errors, dependency problems

For additional help, refer to the [API Documentation](data_processing_api.md) and [Usage Guide](data_processing_usage_guide.md), or create an issue in the project repository with detailed information about your problem.

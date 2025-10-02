# Data Processing Module - Usage Guide and Tutorials

## Quick Start Guide

This guide provides step-by-step instructions for using the Data Processing Module to transform raw VN30 data into analysis-ready datasets.

### Prerequisites

```bash
# Ensure required packages are installed
pip install pandas numpy scikit-learn

# For development and testing
pip install pytest jupyter matplotlib
```

### Basic Setup

```python
import pandas as pd
import numpy as np
from datetime import datetime
from data_processing import *

# Load your raw data
raw_price_data = pd.read_csv('data/raw/vn30_prices.csv')
raw_constituents_data = pd.read_csv('data/raw/vn30_constituents.csv')
```

## Tutorial 1: Basic Data Cleaning

### Objective
Clean raw VN30 price and constituents data by handling missing values and outliers.

```python
from data_processing import DataCleaner
from data_processing.cleaning import CleaningConfig

def basic_data_cleaning_tutorial():
    """Tutorial: Basic data cleaning operations."""

    print("🧹 Data Cleaning Tutorial")
    print("=" * 40)

    # Step 1: Initialize the cleaner with custom configuration
    config = CleaningConfig(
        imputation_method="interpolation",  # Use interpolation for missing values
        outlier_method="iqr",               # Use IQR method for outlier detection
        winsorize_outliers=True,           # Winsorize instead of removing outliers
        outlier_threshold=3.0              # 3-sigma rule for outliers
    )

    cleaner = DataCleaner(config)
    print("✅ DataCleaner initialized with custom configuration")

    # Step 2: Load and inspect raw data
    print("\n📊 Loading raw data...")
    # In real usage, load from your data source:
    # raw_data = load_vn30_data()

    # For tutorial, create sample data
    sample_data = create_sample_price_data()
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Missing values: {sample_data.isnull().sum().sum()}")

    # Step 3: Clean the data
    print("\n🧽 Cleaning data...")
    cleaned_data = cleaner.clean_price_data(sample_data)

    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Remaining missing values: {cleaned_data.isnull().sum().sum()}")

    # Step 4: Review cleaning statistics
    stats = cleaner.get_cleaning_stats()
    print("
📈 Cleaning Statistics:"    print(f"   • Operations tracked: {len(stats['cleaning_stats'])}")
    print(f"   • Configuration: {stats['config']['imputation_method']} imputation")

    return cleaned_data, stats

# Run the tutorial
cleaned_data, cleaning_stats = basic_data_cleaning_tutorial()
```

## Tutorial 2: Multi-Source Data Integration

### Objective
Integrate VN30 data from multiple sources with conflict resolution.

```python
from data_processing import DataIntegrator
from data_processing.integration import IntegrationConfig, SourceConfig, DataSourceQuality, ConflictResolutionStrategy

def multi_source_integration_tutorial():
    """Tutorial: Integrating data from multiple sources."""

    print("🔗 Multi-Source Integration Tutorial")
    print("=" * 45)

    # Step 1: Initialize integrator
    config = IntegrationConfig(
        conflict_strategy=ConflictResolutionStrategy.PRIORITY,
        create_unified_schema=True,
        quality_threshold=0.8
    )

    integrator = DataIntegrator(config)
    print("✅ DataIntegrator initialized")

    # Step 2: Register data sources
    print("\n📋 Registering data sources...")

    # Source 1: TCBS (high quality, high priority)
    integrator.register_source(SourceConfig(
        name="tcbs",
        priority=1,  # Highest priority
        quality=DataSourceQuality.HIGH,
        reliability_score=0.95,
        fields_mapping={
            'stock_code': 'symbol',
            'company_name': 'name',
            'sector_name': 'sector'
        }
    ))

    # Source 2: VNDIRECT (medium quality, lower priority)
    integrator.register_source(SourceConfig(
        name="vndirect",
        priority=2,
        quality=DataSourceQuality.MEDIUM,
        reliability_score=0.85,
        fields_mapping={
            'code': 'symbol',
            'name': 'name',
            'industry': 'sector'
        }
    ))

    print("✅ Data sources registered")

    # Step 3: Prepare source data
    print("\n📥 Preparing source data...")

    # Simulate data from different sources
    tcbs_data = create_sample_constituents_data()
    tcbs_data['source'] = 'TCBS'

    # Modify some values to simulate differences
    vndirect_data = tcbs_data.copy()
    vndirect_data['weight'] = vndirect_data['weight'] * 1.05  # Slightly different weights
    vndirect_data['source'] = 'VNDIRECT'

    source_data = {
        "tcbs": tcbs_data,
        "vndirect": vndirect_data
    }

    print(f"   • TCBS data: {tcbs_data.shape}")
    print(f"   • VNDIRECT data: {vndirect_data.shape}")

    # Step 4: Integrate the data
    print("\n🔄 Integrating data...")
    integrated_data = integrator.integrate_constituents_data(source_data)

    print(f"✅ Integration complete: {integrated_data.shape}")

    # Step 5: Review integration results
    integration_stats = integrator.get_integration_stats()
    print("
📊 Integration Results:"    print(f"   • Sources processed: {len(integration_stats['source_configs'])}")
    print(f"   • Conflict strategy: {integration_stats['config']['conflict_strategy']}")

    return integrated_data, integration_stats

# Run the tutorial
integrated_data, integration_stats = multi_source_integration_tutorial()
```

## Tutorial 3: Data Quality Validation

### Objective
Validate data quality and set up health monitoring.

```python
from data_processing import DataValidator
from data_processing.validation import ValidationConfig, QualityThresholds

def data_quality_validation_tutorial():
    """Tutorial: Data quality validation and monitoring."""

    print("✅ Data Quality Validation Tutorial")
    print("=" * 40)

    # Step 1: Initialize validator with strict quality thresholds
    quality_thresholds = QualityThresholds(
        completeness_threshold=0.95,  # 95% completeness required
        accuracy_threshold=0.90,      # 90% accuracy required
        consistency_threshold=0.85,   # 85% consistency required
        timeliness_threshold=0.80,    # 80% timeliness required
        validity_threshold=0.95,      # 95% validity required
        uniqueness_threshold=0.99     # 99% uniqueness required
    )

    config = ValidationConfig(
        quality_thresholds=quality_thresholds,
        enable_alerting=True,
        historical_tracking=True
    )

    validator = DataValidator(config)
    print("✅ DataValidator initialized with strict thresholds")

    # Step 2: Validate sample data
    print("\n🔍 Validating data quality...")

    # Use cleaned data from previous tutorial
    sample_data = create_sample_constituents_data()
    validation_results = validator.validate_constituents_data(sample_data)

    print("📊 Validation Results:"    print(f"   • Overall quality score: {validation_results['overall_score']".2f"}")
    print(f"   • Data shape: {validation_results['data_shape']}")
    print(f"   • Quality dimensions: {len(validation_results['quality_scores'])}")

    # Step 3: Review detailed quality scores
    print("
🎯 Quality Scores by Dimension:"    for dimension, score in validation_results['quality_scores'].items():
        status = "✅" if score >= 0.9 else "⚠️" if score >= 0.7 else "❌"
        print(f"   • {dimension}: {score".2f"} {status}")

    # Step 4: Check for alerts
    alerts = validation_results.get('alerts', [])
    if alerts:
        print(f"\n🚨 Quality Alerts: {len(alerts)} alerts generated")
        for alert in alerts[:3]:  # Show first 3 alerts
            print(f"   • {alert['severity']}: {alert['message']}")
    else:
        print("\n✅ No quality alerts generated")

    # Step 5: Generate quality report
    print("\n📋 Generating quality report...")
    quality_report = validator.generate_quality_report()

    print("📈 Quality Report Summary:"    print(f"   • Total alerts: {quality_report['summary']['total_alerts']}")
    print(f"   • History length: {quality_report['summary']['quality_history_length']}")
    print(f"   • Active rules: {quality_report['summary']['active_rules']}")

    return validation_results, quality_report

# Run the tutorial
validation_results, quality_report = data_quality_validation_tutorial()
```

## Tutorial 4: Data Normalization

### Objective
Normalize financial data for machine learning models.

```python
from data_processing import DataNormalizer
from data_processing.normalization import NormalizationConfig, NormalizationMethod

def data_normalization_tutorial():
    """Tutorial: Data normalization for ML models."""

    print("📏 Data Normalization Tutorial")
    print("=" * 35)

    # Step 1: Initialize normalizer
    config = NormalizationConfig(
        method=NormalizationMethod.Z_SCORE,  # Z-score standardization
        handle_outliers=True,                # Handle outliers before normalization
        feature_range=(0, 1)                 # Optional: scale to specific range
    )

    normalizer = DataNormalizer(config)
    print("✅ DataNormalizer initialized")

    # Step 2: Prepare sample data
    print("\n📊 Preparing sample data...")
    sample_data = create_sample_price_data()
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Data types: {sample_data.dtypes.unique()}")

    # Step 3: Normalize price data
    print("\n🔢 Normalizing price data...")
    normalized_data = normalizer.normalize_price_data(sample_data)

    print(f"✅ Normalization complete: {normalized_data.shape}")

    # Step 4: Compare before and after statistics
    print("
📈 Before vs After Statistics:"
    for col in ['open', 'high', 'low', 'close']:
        if col in sample_data.columns:
            original_mean = sample_data[col].mean()
            original_std = sample_data[col].std()
            normalized_mean = normalized_data[col].mean()
            normalized_std = normalized_data[col].std()

            print(f"   • {col}:")
            print(f"     Original  - Mean: {original_mean".2f"}, Std: {original_std".2f"}")
            print(f"     Normalized - Mean: {normalized_mean".3f"}, Std: {normalized_std".3f"}")

    # Step 5: Test different normalization methods
    print("
🧪 Testing different normalization methods..."
    methods_to_test = [
        NormalizationMethod.MIN_MAX,
        NormalizationMethod.ROBUST,
        NormalizationMethod.Z_SCORE
    ]

    for method in methods_to_test:
        test_config = NormalizationConfig(method=method)
        test_normalizer = DataNormalizer(test_config)

        test_normalized = test_normalizer.normalize_price_data(sample_data)
        test_mean = test_normalized['close'].mean()
        test_std = test_normalized['close'].std()

        print(f"   • {method.value}: Mean={test_mean".3f"}, Std={test_std".3f"}")

    return normalized_data, normalizer

# Run the tutorial
normalized_data, normalizer = data_normalization_tutorial()
```

## Tutorial 5: Complete Pipeline

### Objective
Run the complete data processing pipeline from raw data to storage.

```python
from data_processing import *
from data_processing.cleaning import CleaningConfig
from data_processing.integration import IntegrationConfig, SourceConfig, DataSourceQuality
from data_processing.validation import ValidationConfig, QualityThresholds
from data_processing.normalization import NormalizationConfig, NormalizationMethod
from data_processing.storage import StorageConfig, DataMetadata
from data_processing.time_series import TimeSeriesConfig

def complete_pipeline_tutorial():
    """Tutorial: Complete data processing pipeline."""

    print("🔄 Complete Data Processing Pipeline Tutorial")
    print("=" * 50)

    # Step 1: Initialize all components
    print("🏗️ Initializing pipeline components...")

    # Data cleaning configuration
    cleaning_config = CleaningConfig(
        imputation_method="interpolation",
        outlier_method="iqr",
        winsorize_outliers=True
    )
    cleaner = DataCleaner(cleaning_config)

    # Time series processing configuration
    ts_config = TimeSeriesConfig(
        base_frequency="D",
        target_frequencies=["D", "W"],
        fill_method="FORWARD_FILL"
    )
    ts_processor = TimeSeriesProcessor(ts_config)

    # Data integration configuration
    integration_config = IntegrationConfig(
        conflict_strategy="PRIORITY",
        create_unified_schema=True
    )
    integrator = DataIntegrator(integration_config)

    # Data validation configuration
    quality_thresholds = QualityThresholds(
        completeness_threshold=0.95,
        accuracy_threshold=0.90,
        consistency_threshold=0.85
    )
    validation_config = ValidationConfig(
        quality_thresholds=quality_thresholds,
        enable_alerting=True,
        historical_tracking=True
    )
    validator = DataValidator(validation_config)

    # Data normalization configuration
    normalization_config = NormalizationConfig(
        method=NormalizationMethod.Z_SCORE,
        handle_outliers=True
    )
    normalizer = DataNormalizer(normalization_config)

    # Storage configuration
    storage_config = StorageConfig(
        base_path="data/processed_tutorial",
        format="PARQUET",
        compression="GZIP",
        create_versioned_backups=True
    )
    storage_manager = ProcessedDataManager(storage_config)

    print("✅ All components initialized")

    # Step 2: Load raw data
    print("\n📥 Loading raw data...")
    raw_price_data = create_sample_price_data()
    raw_constituents_data = create_sample_constituents_data()

    print(f"   • Raw price data: {raw_price_data.shape}")
    print(f"   • Raw constituents: {raw_constituents_data.shape}")

    # Step 3: Execute pipeline
    print("\n🚀 Executing data processing pipeline...")

    # 3.1: Data Cleaning
    print("   🧹 Step 1/6: Data Cleaning")
    cleaned_price = cleaner.clean_price_data(raw_price_data)
    cleaned_constituents = cleaner.clean_constituents_data(raw_constituents_data)

    # 3.2: Time Series Processing
    print("   ⏰ Step 2/6: Time Series Processing")
    aligned_price = ts_processor.align_time_series(cleaned_price)
    time_validation = ts_processor.validate_time_consistency(aligned_price)

    # 3.3: Data Integration
    print("   🔗 Step 3/6: Data Integration")
    integrated_constituents = cleaned_constituents  # Simplified for tutorial

    # 3.4: Data Validation
    print("   ✅ Step 4/6: Data Validation")
    price_validation = validator.validate_price_data(aligned_price)
    constituents_validation = validator.validate_constituents_data(integrated_constituents)

    # 3.5: Data Normalization
    print("   📏 Step 5/6: Data Normalization")
    normalized_price = normalizer.normalize_price_data(aligned_price)
    normalized_constituents = normalizer.normalize_constituents_data(integrated_constituents)

    # 3.6: Data Storage
    print("   💾 Step 6/6: Data Storage")

    # Create metadata
    price_metadata = DataMetadata(
        source="tutorial_pipeline",
        processing_timestamp=datetime.now(),
        data_shape=normalized_price.shape,
        columns=normalized_price.columns.tolist(),
        data_types={col: str(normalized_price[col].dtype) for col in normalized_price.columns},
        processing_pipeline=["cleaning", "validation", "normalization"],
        quality_score=price_validation['overall_score'],
        version="1.0.0",
        description="Tutorial processed VN30 price data",
        tags=["tutorial", "vn30", "price", "normalized"]
    )

    # Store data
    price_file = storage_manager.store_price_data(normalized_price, "vn30_daily_tutorial", price_metadata)
    constituents_file = storage_manager.store_constituents_data(normalized_constituents, "vn30_constituents_tutorial", price_metadata)

    print("✅ Pipeline execution complete!"
    # Step 4: Verify results
    print("\n🔍 Verifying pipeline results...")

    # Retrieve stored data
    retrieved_price, retrieved_metadata = storage_manager.retrieve_price_data("vn30_daily_tutorial")

    print("📊 Pipeline Results:"    print(f"   • Original data: {raw_price_data.shape}")
    print(f"   • Processed data: {normalized_price.shape}")
    print(f"   • Data quality score: {price_validation['overall_score']".2f"}")
    print(f"   • Storage path: {price_file}")
    print(f"   • Metadata preserved: {'✅' if retrieved_metadata.quality_score else '❌'}")

    # Step 5: Generate comprehensive report
    print("\n📋 Generating pipeline report...")

    pipeline_report = {
        'input_data': {
            'price_shape': raw_price_data.shape,
            'constituents_shape': raw_constituents_data.shape
        },
        'processing_results': {
            'cleaned_price_shape': cleaned_price.shape,
            'cleaned_constituents_shape': cleaned_constituents.shape,
            'normalized_price_shape': normalized_price.shape,
            'normalized_constituents_shape': normalized_constituents.shape
        },
        'quality_scores': {
            'price': price_validation['overall_score'],
            'constituents': constituents_validation['overall_score']
        },
        'storage_info': {
            'price_file': price_file,
            'constituents_file': constituents_file,
            'total_datasets': len(storage_manager.list_datasets())
        },
        'component_stats': {
            'cleaner': cleaner.get_cleaning_stats(),
            'validator': validator.get_validation_stats(),
            'normalizer': normalizer.get_normalization_stats(),
            'storage': storage_manager.get_storage_stats()
        }
    }

    print("📈 Pipeline Report Summary:"    print(f"   • Data quality: {pipeline_report['quality_scores']['price']".2f"} (price)")
    print(f"   • Processing efficiency: {len(pipeline_report['processing_results'])} steps completed")
    print(f"   • Storage created: {pipeline_report['storage_info']['total_datasets']} datasets")

    return pipeline_report

# Run the complete pipeline tutorial
pipeline_report = complete_pipeline_tutorial()
```

## Advanced Usage Examples

### Custom Configuration Management

```python
def advanced_configuration_example():
    """Example: Advanced configuration management."""

    # Environment-based configuration
    import os

    env = os.getenv('ENVIRONMENT', 'development')

    if env == 'development':
        # Development: More permissive settings
        config = CleaningConfig(
            imputation_method="interpolation",
            outlier_method="iqr",
            remove_outliers=False,  # Keep outliers for analysis
            outlier_threshold=2.0   # Less strict threshold
        )
    elif env == 'production':
        # Production: Strict quality requirements
        config = CleaningConfig(
            imputation_method="forward_fill",
            outlier_method="iqr",
            remove_outliers=True,   # Strict outlier removal
            outlier_threshold=3.0   # Standard 3-sigma rule
        )
    else:
        # Default configuration
        config = CleaningConfig()

    return config
```

### Batch Processing

```python
def batch_processing_example():
    """Example: Processing multiple datasets in batch."""

    # Initialize components
    cleaner = DataCleaner()
    validator = DataValidator()
    storage_manager = ProcessedDataManager()

    # List of datasets to process
    datasets = [
        "vn30_2024_Q1",
        "vn30_2024_Q2",
        "vn30_2024_Q3",
        "vn30_2024_Q4"
    ]

    batch_results = {}

    for dataset_name in datasets:
        try:
            # Load dataset
            raw_data = load_dataset(dataset_name)

            # Process dataset
            cleaned_data = cleaner.clean_price_data(raw_data)
            validation_results = validator.validate_price_data(cleaned_data)

            # Store results
            metadata = create_metadata(dataset_name, validation_results['overall_score'])
            storage_path = storage_manager.store_price_data(cleaned_data, dataset_name, metadata)

            batch_results[dataset_name] = {
                'status': 'success',
                'quality_score': validation_results['overall_score'],
                'storage_path': storage_path
            }

        except Exception as e:
            batch_results[dataset_name] = {
                'status': 'failed',
                'error': str(e)
            }

    return batch_results
```

### Quality Monitoring Dashboard

```python
def quality_monitoring_dashboard():
    """Example: Creating a quality monitoring dashboard."""

    validator = DataValidator(ValidationConfig(
        enable_alerting=True,
        historical_tracking=True
    ))

    # Process multiple datasets
    datasets = ["dataset1", "dataset2", "dataset3"]
    quality_history = []

    for dataset in datasets:
        data = load_dataset(dataset)
        validation_results = validator.validate_price_data(data)
        quality_history.append({
            'dataset': dataset,
            'timestamp': datetime.now(),
            'overall_score': validation_results['overall_score'],
            'quality_scores': validation_results['quality_scores']
        })

    # Generate dashboard data
    dashboard_data = {
        'summary': {
            'total_datasets': len(datasets),
            'average_quality': np.mean([h['overall_score'] for h in quality_history]),
            'quality_trend': 'improving' if len(quality_history) > 1 else 'stable'
        },
        'quality_history': quality_history,
        'alerts': validator.alerts[-10:],  # Last 10 alerts
        'recommendations': generate_recommendations(quality_history)
    }

    return dashboard_data
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: High Missing Data Percentage
```python
# Problem: Too many missing values after processing
# Solution: Adjust imputation strategy

cleaner = DataCleaner(CleaningConfig(
    imputation_method="interpolation",  # Better for time series
    fill_limit=5                       # Limit interpolation distance
))
```

#### Issue 2: Poor Data Quality Scores
```python
# Problem: Low quality scores
# Solution: Review and adjust quality thresholds

validator = DataValidator(ValidationConfig(
    quality_thresholds=QualityThresholds(
        completeness_threshold=0.90,  # Lower threshold if needed
        accuracy_threshold=0.85       # Adjust based on data characteristics
    )
))
```

#### Issue 3: Performance Problems
```python
# Problem: Slow processing of large datasets
# Solution: Optimize configuration

storage_manager = ProcessedDataManager(StorageConfig(
    chunk_size=100000,      # Process in smaller chunks
    compression="GZIP"      # Enable compression
))
```

#### Issue 4: Integration Conflicts
```python
# Problem: Too many conflicts between sources
# Solution: Adjust conflict resolution strategy

integrator = DataIntegrator(IntegrationConfig(
    conflict_strategy="WEIGHTED_AVERAGE",  # Use weighted average
    quality_threshold=0.7                  # Lower quality threshold
))
```

## Best Practices

### 1. Pipeline Design
- **Modular approach**: Use each component independently when possible
- **Error handling**: Implement try-catch blocks for each major operation
- **Logging**: Use appropriate log levels for monitoring
- **Validation**: Validate inputs and outputs at each stage

### 2. Configuration Management
- **Environment-specific**: Different configs for dev/staging/prod
- **Version control**: Track configuration changes
- **Documentation**: Document configuration rationale
- **Validation**: Validate configuration parameters

### 3. Quality Assurance
- **Thresholds**: Set realistic quality thresholds based on data characteristics
- **Monitoring**: Implement continuous quality monitoring
- **Alerting**: Set up alerts for quality degradation
- **Reporting**: Generate regular quality reports

### 4. Performance Optimization
- **Chunking**: Use chunked processing for large datasets
- **Compression**: Enable compression for storage efficiency
- **Indexing**: Create appropriate indexes for fast retrieval
- **Caching**: Cache intermediate results when appropriate

## Next Steps

After completing these tutorials, you should be able to:

1. **🔧 Configure** data processing components for your specific needs
2. **🏗️ Build** custom data processing pipelines
3. **📊 Monitor** data quality and processing performance
4. **🚀 Deploy** the pipeline in production environments
5. **🔍 Troubleshoot** common issues and optimize performance

For more advanced usage, refer to the [API Documentation](data_processing_api.md) and explore the source code in the `src/data_processing/` directory.

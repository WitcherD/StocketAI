# VN30 Data Processing Pipeline

## Overview

The VN30 Data Processing Pipeline is a comprehensive system designed to process raw financial data for VN30 stocks and prepare it for machine learning models, specifically optimized for Temporal Fusion Transformers (TFT). The pipeline handles data validation, cleaning, feature engineering, and storage in an efficient, scalable manner.

## Architecture

### Core Components

#### VN30DataHandler
**Location**: `src/data_processing/vn30_to_qlib_converter.py`

The VN30DataHandler class manages the initial processing of raw VN30 stock data. It performs validation, cleaning, and storage operations.

**Key Responsibilities:**
- Raw data validation and quality checks
- Data cleaning and preprocessing
- Storage of cleaned data in efficient formats
- Individual symbol processing

**Key Methods:**
- `validate_raw_data(symbol: str)` - Validates raw CSV data for a given symbol
- `clean_raw_data(symbol: str)` - Cleans and preprocesses raw data
- `convert_symbol_to_qlib(symbol: str)` - Processes and stores cleaned data
- `prepare_symbol_training_data(symbol: str, start_date, end_date)` - Prepares train/validation/test splits

#### VN30TFTFeatureEngineer
**Location**: `src/feature_engineering/tft_feature_engineer.py`

The VN30TFTFeatureEngineer class handles feature engineering operations, transforming cleaned OHLCV data into TFT-compatible features.

**Key Responsibilities:**
- Loading cleaned data from storage
- Engineering 25+ technical indicators
- Feature validation and quality assessment
- Saving engineered features for model training

**Key Methods:**
- `load_cleaned_data(symbol: str)` - Loads cleaned data for feature engineering
- `engineer_all_features(df: DataFrame)` - Creates all TFT-compatible features
- `save_engineered_features(df: DataFrame, symbol: str)` - Saves features to disk
- `validate_features(df: DataFrame)` - Validates feature quality

## Data Flow

```
Raw CSV Data
    ↓
VN30DataHandler.validate_raw_data()
    ↓
VN30DataHandler.clean_raw_data()
    ↓
VN30DataHandler.convert_symbol_to_qlib()
    ↓
Save to: data/symbols/{symbol}/processed/{symbol}_cleaned.pkl
    ↓
VN30TFTFeatureEngineer.load_cleaned_data()
    ↓
VN30TFTFeatureEngineer.engineer_all_features()
    ↓
Save to: data/symbols/{symbol}/features/{symbol}_tft_features.csv
```

## Directory Structure

```
data/symbols/{symbol}/
├── raw/
│   └── historical_price.csv          # Raw OHLCV data
├── processed/
│   └── {symbol}_cleaned.pkl          # Cleaned data (pickle format)
└── features/
    └── {symbol}_tft_features.csv      # Engineered features
```

## Technical Indicators

The pipeline generates 25 technical indicators organized into TFT-compatible categories:

### OBSERVED_INPUT (13 features)
Time-varying technical indicators that change with market conditions:

- **Residuals**: `RESI5`, `RESI10`
  - Formula: `close - MA(close, period)`

- **Weighted Moving Averages**: `WVMA5`, `WVMA60`
  - Linear decay weighted moving averages

- **R-squared Values**: `RSQR5`, `RSQR10`, `RSQR20`, `RSQR60`
  - Rolling linear regression R-squared

- **Correlations**: `CORR5`, `CORR10`, `CORR20`, `CORR60`
  - Rolling correlation between close price and volume

- **Correlation Differences**: `CORD5`, `CORD10`, `CORD60`
  - Differences between correlation periods

- **Rate of Change**: `ROC60`
  - Percentage change over 60 periods

- **Volatility**: `VSTD5`, `STD5`
  - Standard deviation of returns and prices

- **Momentum**: `KLEN`, `KLOW`
  - Trend length and lowest price indicators

### KNOWN_INPUT (3 features)
Temporal features known in advance:

- `month`: Month of the year (0-11)
- `day_of_week`: Day of week (0-6)
- `year`: Year

### STATIC_INPUT (1 feature)
Static features that don't change over time:

- `const`: Constant value (1.0) required by TFT

## Data Validation

### Raw Data Validation
- **Column presence**: Required OHLCV columns must exist
- **Data completeness**: Minimum 100 records per symbol
- **Missing values**: Maximum 5% missing data allowed
- **Date format**: Valid datetime parsing

### Cleaned Data Validation
- **Data integrity**: No NaN values after cleaning
- **Price validity**: No negative or zero prices
- **Volume validity**: Reasonable volume ranges
- **Date continuity**: Proper time series structure

### Feature Validation
- **Category completeness**: All required TFT features present
- **Value ranges**: Reasonable numerical ranges for each feature
- **Data sufficiency**: Minimum 100 records for training
- **Statistical validity**: Features within expected distributions

## Performance Characteristics

### Processing Speed
- **Feature Engineering**: >100 records/second
- **Data Loading**: Efficient pickle-based storage
- **Memory Usage**: Streaming processing for large datasets

### Scalability
- **Individual Symbol Processing**: Each symbol processed independently
- **Parallel Processing**: Supports concurrent symbol processing
- **Storage Efficiency**: Compressed pickle format

## Usage Examples

### Basic Data Processing
```python
from data_processing.vn30_to_qlib_converter import VN30DataHandler

# Initialize handler
handler = VN30DataHandler()

# Process a single symbol
success = handler.convert_symbol_to_qlib('VCB')
```

### Feature Engineering
```python
from feature_engineering.tft_feature_engineer import VN30TFTFeatureEngineer

# Initialize engineer
engineer = VN30TFTFeatureEngineer()

# Process complete pipeline for a symbol
features_df = engineer.process_symbol_complete('VCB', save_features=True)
```

### Custom Date Ranges
```python
# Load data for specific date range
data = handler.load_symbol_data('VCB', '2023-01-01', '2023-12-31')

# Engineer features for date range
features = engineer.engineer_features_for_symbol('VCB', '2023-01-01', '2023-12-31')
```

## Configuration

### Data Directories
- **Default symbols directory**: `data/symbols/`
- **Raw data location**: `data/symbols/{symbol}/raw/historical_price.csv`
- **Processed data**: `data/symbols/{symbol}/processed/`
- **Features output**: `data/symbols/{symbol}/features/`

### Validation Rules
```python
validation_rules = {
    'required_columns': ['time', 'open', 'high', 'low', 'close', 'volume'],
    'date_format': '%Y-%m-%d',
    'min_data_points': 100,
    'max_missing_ratio': 0.05
}
```

### Lookback Periods
```python
lookback_periods = {
    'short': 5,
    'medium': 10,
    'long': 20,
    'very_long': 60
}
```

## Error Handling

### Data Quality Issues
- **Missing columns**: Clear error messages with required columns
- **Insufficient data**: Minimum record count validation
- **Invalid dates**: Date parsing error handling
- **Corrupted files**: File integrity checks

### Processing Errors
- **Qlib initialization**: Graceful fallback mechanisms
- **File I/O errors**: Comprehensive error logging
- **Memory issues**: Streaming processing for large datasets
- **Feature calculation**: NaN/inf value handling

## Testing

### Integration Tests
**Location**: `tests/integration/test_vcb_feature_engineering_integration.py`

**Test Coverage:**
- Complete pipeline testing (raw data → features)
- Data validation and quality checks
- Feature engineering accuracy
- File I/O operations
- Error handling scenarios

### Test Execution
```bash
# Run specific integration test
pytest tests/integration/test_vcb_feature_engineering_integration.py::TestVCBFeatureEngineeringIntegration::test_complete_pipeline_with_feature_saving -v

# Run all integration tests
pytest tests/integration/ -v
```

## Dependencies

### Core Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **qlib**: Financial data handling (optional, with fallbacks)
- **pytest**: Testing framework

### Optional Dependencies
- **matplotlib/seaborn**: Visualization (for debugging)
- **scikit-learn**: Additional validation metrics

## Maintenance

### Code Quality
- **PEP 8 compliance**: Consistent code formatting
- **Type hints**: Full type annotation coverage
- **Documentation**: Comprehensive docstrings
- **Logging**: Structured logging throughout

### Version Control
- **Semantic versioning**: Major.minor.patch format
- **Changelog**: Detailed change documentation
- **Backwards compatibility**: API stability guarantees

## Troubleshooting

### Common Issues

#### Data Loading Failures
**Symptom**: "Cleaned data file not found"
**Solution**: Ensure raw data exists and processing completed successfully

#### Feature Engineering Errors
**Symptom**: NaN values in features
**Solution**: Check input data quality and validation rules

#### Memory Issues
**Symptom**: Out of memory errors
**Solution**: Process symbols individually or reduce batch sizes

#### Qlib Initialization
**Symptom**: Qlib-related import errors
**Solution**: System falls back to pickle-based storage automatically

## Future Enhancements

### Planned Features
- **GPU acceleration**: CUDA-based feature engineering
- **Distributed processing**: Multi-node symbol processing
- **Real-time processing**: Streaming data ingestion
- **Advanced indicators**: Additional technical indicators
- **Model integration**: Direct TFT model training pipeline

### Performance Optimizations
- **Vectorization**: Further numpy/pandas optimizations
- **Caching**: Intermediate result caching
- **Compression**: Advanced data compression techniques
- **Memory mapping**: Large dataset handling

---

**Version**: 1.0.0
**Date**: October 7, 2025
**Status**: Production Ready

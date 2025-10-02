# Project References and Documentation

This document serves as a central hub for all project documentation, references, and resources related to StocketAI.

### Project Constitution
**Location**: `constitution.md`
**Description**: Comprehensive project constitution defining goals, technologies, architecture, standards, and development practices.

**Key Sections**:
- Project goals and scope
- Core technologies (vnstock, qlib, Python 3.12+)
- System features and architecture
- Technical requirements and standards
- Development workflow and coding standards
- Environment setup and installation guide

## Technical Documentation

### Data Acquisition
**Location**: `src/data_acquisition/`
**Description**: Extensible data acquisition framework supporting multiple data providers and sources.

**Key Components**:
- Company universe definition and management (VN30, custom lists, or any stock universe)
- Historical price data collection from multiple providers
- Financial statement acquisition across different data sources
- Company information gathering with provider abstraction
- Multi-provider fallback mechanisms and data source switching
- Extensible provider interface for adding new data sources (e.g., otherlib/providerA, otherlib/providerB)

### Data Processing
**Location**: `src/data_processing/`
**Description**: Data cleaning, validation, and preprocessing pipeline.

### Feature Engineering
**Location**: `src/feature_engineering/`
**Description**: Feature generation using qlib expression engine.

### Model Training
**Location**: `src/model_training/`
**Description**: Model training infrastructure and algorithms.

### Prediction and Forecasting
**Location**: `src/prediction/`
**Description**: Inference engine and prediction generation.

### Evaluation and Backtesting
**Location**: `src/evaluation/`
**Description**: Performance evaluation and backtesting framework.

### Reporting and Analysis
**Location**: `src/reporting/`
**Description**: Report generation and visualization tools.

## External Library Documentation

### vnstock Documentation
**Location**: `docs/vnstock_api_documentation.md`
**Description**: Comprehensive API documentation for vnstock library integration with StocketAI.

**Key Features**:
- Complete API response formats and usage examples
- Multiple data sources (VCI, TCBS, MSN, FMARKET) integration
- Stock data, financial reports, company information APIs
- Market analysis and screening tools implementation
- International markets and forex data access
- Investment funds and derivatives data structures
- Error handling and troubleshooting guides

**API Reference**:
- `Vnstock` - Main entry point for all data types
- `Quote` - Price and trading data functionality
- `Company` - Corporate information and events
- `Finance` - Financial statements and ratios
- `Listing` - Market listings and symbol information
- `Trading` - Real-time trading data and market depth
- `Screener` - Advanced stock screening capabilities

**Additional Resources**:
- **Complete Source Documentation**: `vnstock/docs/` - Full vnstock project documentation
- **Source Code**: `vnstock/vnstock/` - Main package code and implementation details

### qlib Documentation
**Location**: `docs/qlib_and_vnstock_features.md`
**Description**: Integration guide for qlib quantitative finance library with StocketAI.

**Key Features**:
- Data layer features for high-performance processing
- Model training capabilities (Tree-based, Neural Networks, Transformer-based)
- Workflow automation and experiment management
- Backtesting and performance analysis tools
- Risk management and portfolio optimization
- Feature engineering with expression engine
- Integration points for StocketAI data pipeline

**Model Categories**:
- **Tree-based Models**: LightGBM, XGBoost for tabular data
- **Neural Networks**: MLP, LSTM, GRU, TCN for sequential patterns
- **Transformer Models**: TFT, TRA, ADARNN, HIST, IGMTF for complex relationships
- **Specialized Models**: KRNN, Sandwich, ADD, DDG-DA for specific use cases

**Additional Resources**:
- **Complete Source Documentation**: `qlib/docs/` - Full qlib project documentation including:
  - Component guides (data, model, workflow, strategy)
  - Advanced usage examples and tutorials
  - Developer documentation and API references
  - FAQ and troubleshooting guides
- **Source Code**: `qlib/qlib/` - Main package code and implementation details
- **Examples**: `qlib/examples/` - Comprehensive usage examples and benchmarks

## Development Resources

### Configuration Files
**Location**: `config/

### Scripts and Tools
**Location**: `scripts/`
- Development scripts

### Test Files
**Location**: `tests/`
- Unit tests for all modules
- Integration tests
- Data validation tests
- Performance tests

### Task Documentation
**Location**: `tasks/`
- Task-specific documentation
- Progress tracking
- Deliverables and dependencies

## Research and Analysis

### Notebooks
**Location**: `notebooks/`
**Description**: Jupyter notebooks for research and analysis.

### Model Artifacts
**Location**: `models/`
**Description**: Trained model files and metadata.

### Data Assets
**Location**: `data/`
- `symbols/` - Individual stock symbol organization
  - `{symbol}/` - Each symbol as independent data unit
    - `raw/` - Original data from APIs for this symbol
    - `processed/` - Cleaned and validated data for this symbol
    - `qlib_format/` - qlib-compatible binary format for this symbol
    - `progress/` - Processing progress and status for this symbol
    - `reports/` - Analysis reports and metrics for this symbol
    - `errors/` - Error logs and debugging info for this symbol
- `cache/` - Cached intermediate results

### Logs
**Location**: `logs/`
**Description**: Application logs and debugging information.

## Quick Access URLs

### External Documentation
- **vnstock GitHub**: https://github.com/thinh-vu/vnstock
- **qlib GitHub**: https://github.com/microsoft/qlib
- **vnstock Documentation**: https://vnstocks.com/docs

### Development Tools
- **Python Documentation**: https://docs.python.org/3/
- **pandas Documentation**: https://pandas.pydata.org/docs/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **scikit-learn Documentation**: https://scikit-learn.org/stable/documentation.html

## Getting Started

1. **Read the Constitution** (`constitution.md`) - Understand project goals and standards

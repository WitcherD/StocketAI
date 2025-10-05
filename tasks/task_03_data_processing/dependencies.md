# Task 03A Dependencies - Data Processing Foundation

## Required Previous Tasks
- **Task 02: Data Acquisition** - Must be completed before starting this subtask
  - VN30 constituents data must be available
  - Raw VN30 stock data must be downloaded and organized
  - Data directory structure must be established

## External Dependencies for 03A
- **qlib library** - Must be installed from source code
  - Required for data format conversion and basic model setup
  - Must be properly configured with Python 3.12+ environment
- **Basic Python Packages** - For data processing
  - pandas, numpy for data manipulation
  - scipy for scientific computing
  - scikit-learn for basic ML utilities

## Technical Dependencies for 03A
- **Python Environment**
  - Python 3.12+ with conda environment
  - Basic data processing packages: pandas, numpy, scipy
  - File I/O utilities for CSV and binary data handling

- **Computational Resources for 03A**
  - Sufficient RAM for data processing (8GB+ sufficient)
  - CPU processing adequate (GPU not required for 03A)
  - Storage space for qlib format data (10-20GB estimated)

## Data Dependencies for 03A
- **VN30 Stock Universe**
  - Complete list of 30 VN30 constituent stocks
  - Historical OHLCV data for all constituents (minimum 2-3 years)
  - Basic company information and metadata

- **Data Format Requirements**
  - CSV data files organized by symbol
  - Consistent data structure across all symbols
  - Basic data validation and cleaning

## Configuration Dependencies for 03A
- **Project Configuration**
  - Data paths and directory structure
  - Basic qlib configuration parameters
  - Simple feature engineering settings

## Verification Checklist for 03A
- [ ] Task 02 completion confirmed
- [ ] VN30 data availability verified in CSV format
- [ ] qlib installation and basic configuration tested
- [ ] Required Python packages installed (pandas, numpy, scipy)
- [ ] Sufficient storage space available for processed data
- [ ] Data directory structure properly organized
- [ ] Basic data validation scripts ready

## Success Criteria for 03A Dependencies
- [ ] All required data files accessible and readable
- [ ] qlib properly installed and importable
- [ ] Python environment configured correctly
- [ ] Storage space sufficient for data conversion

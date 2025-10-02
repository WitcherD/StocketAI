# StocketAI - Project Constitution

## 1. Project Goal
Build a research tool for predicting stock prices using extensible data acquisition frameworks and qlib for model training. Forecast stock price movements for 1-month, 3-month, and 6-month horizons for any stock universe. Focus on research flexibility, reproducibility, and quantitative finance methodologies with support for multiple data providers and extensible architecture.

## 2. Core Technologies
- **Python 3.12+** - Primary programming language
- **vnstock** - Data acquisition from Vietnamese financial providers (VCI, TCBS, MSN, FMARKET) - Install from source code
- **qlib** - Quantitative finance modeling framework for feature engineering, training, and evaluation - Install from source code
- **pandas/numpy/scipy** - Data manipulation and scientific computing - Latest compatible versions
- **scikit-learn** - Traditional machine learning algorithms - Latest compatible version
- **PyTorch/TensorFlow** - Deep learning frameworks - Latest compatible versions
- **LightGBM/XGBoost** - Gradient boosting implementations - Latest compatible versions
- **matplotlib/seaborn/plotly** - Data visualization and reporting - Latest compatible versions

## 3. System Features
### Data Pipeline
- Company universe definition and management (VN30, custom lists, or any stock universe)
- Historical OHLCV data collection (minimum 5 years) with multi-provider fallback
- Financial statement acquisition (balance sheets, income statements, cash flow) across different data sources
- Company information gathering (overview, profile, shareholders, corporate events) with provider abstraction
- Financial ratio computation (P/E, P/B, ROE, ROA, etc.)
- Data integrity validation (missing values, outliers, consistency checks)
- Qlib format conversion (.bin files) for high-performance processing
- Extensible provider interface for adding new data sources (e.g., otherlib/providerA, otherlib/providerB)

### Feature Engineering
- Technical indicators via qlib expression engine (moving averages, RSI, MACD, Bollinger Bands)
- Price-based features (returns, volatility measures, momentum indicators)
- Volume-based features and microstructure analysis
- Fundamental features from financial ratios and statements
- Sector and industry classifications
- Multi-horizon labels (1, 3, 6 months) without lookahead bias
- Custom domain-specific feature combinations

### Models Training
- Baseline models: LightGBM (Alpha158/360), XGBoost, MLP, LSTM
- Advanced neural networks: TFT, GRU, TCN, TRA, ADARNN, HIST, IGMTF, KRNN, Sandwich, ADD
- Ensemble methods: Reinforcement Learning, DDG-DA, DoubleEnsemble, meta-learning
- Rolling window retraining and online learning capabilities
- Adaptive feature selection and concept drift handling
- Hyperparameter optimization with automated tuning
- Cross-validation with purged k-fold to prevent data leakage
- TBD

### PredictiSon and Forecasting
- Multi-horizon forecasting (1, 3, 6 months) with confidence intervals
- Uncertainty estimation and prediction reliability metrics
- Trading signal generation (buy/hold/sell) with threshold-based logic

### Evaluation and Backtesting
- Performance metrics: IC, Rank IC, Sharpe/Sortino ratios, max drawdown, MSE/MAE, directional accuracy
- Portfolio backtesting with trading constraints (costs, limits, halts)
- Portfolio strategies: equal weight, market cap weighted, prediction-weighted
- Risk management: position limits, CVaR, stress testing, Monte Carlo simulation
- Walk-forward analysis and purged cross-validation
- Benchmarking against market indices and buy-and-hold strategies

### Reporting and Analysis
- Comparative analysis across models and time periods

## 4. Technical Requirements
### Data Requirements
- Minimum 5 years of historical OHLCV data for any configurable stock universe
- Complete financial statements for feature engineering across multiple data sources
- Data validation pipeline with automated quality checks
- Extensible data source support for adding new providers beyond vnstock

### Performance Requirements
- Efficient data processing for large datasets (30+ stocks × 5+ years)
- Model training optimization for various architectures

### Quality Requirements
- Unit tests
- Comprehensive error handling and logging

## 5. Functional Requirements
### Core Functionality
- End-to-end pipeline from data acquisition to prediction
- Modular design for component testing and replacement
- Configuration-driven execution for research flexibility
- Automated workflow for experiment management
- Integration hooks for external tools and analysis

### Research Capabilities
- Experiment tracking and comparison framework
- Hyperparameter optimization and model selection
- Feature importance analysis and selection
- Model interpretability and explanation tools
- Statistical validation of predictions

### Output Requirements
- Structured prediction outputs with confidence intervals
- Trading signals with risk-adjusted thresholds
- Performance reports with statistical significance tests
- Visualizations for model comparison and analysis
- Export formats for further research and analysis

## 6. Project Structure
```
StocketAI/
├── data/
│   ├── symbols/                # Individual stock symbol organization
│   │   └── {symbol}/           # Each symbol as independent data unit
│   │       ├── raw/            # Raw data from vnstock APIs for this symbol
│   │       ├── processed/      # Cleaned and validated data for this symbol
│   │       ├── qlib_format/    # Qlib .bin format data for this symbol
│   │       ├── progress/       # Processing progress and status for this symbol
│   │       ├── reports/        # Analysis reports and metrics for this symbol
│   │       └── errors/         # Error logs and debugging info for this symbol
│   └── reports/                # Summary and results
├── src/
│   ├── data_acquisition/       # vnstock integration modules
│   ├── data_processing/        # Data cleaning and validation
│   ├── feature_engineering/    # Feature generation and qlib integration
│   ├── model_training/         # Model training and optimization
│   ├── prediction/             # Inference and signal generation
│   ├── evaluation/             # Backtesting and performance analysis
│   └── reporting/              # Report generation and visualization
├── notebooks/                  # Jupyter notebooks for research
├── tests/                      # Unit and integration tests
├── config/                     # Configuration files and parameters
└── docs/                       # Documentation and guides
```

## 7. System Architecture
### Modular Pipeline Design
- **Data Acquisition Module**: Extensible framework supporting multiple data providers (vnstock, otherlib/providerA, otherlib/providerB) with provider abstraction and fallback mechanisms
- **Data Processing Module**: Cleaning, normalization, and validation pipeline
- **Feature Engineering Module**: qlib expression engine for technical and fundamental features
- **Model Training Module**: Multiple model architectures with automated optimization
- **Prediction Module**: Multi-horizon forecasting with uncertainty quantification
- **Evaluation Module**: Backtesting framework with risk management
- **Reporting Module**: Automated report generation with interactive visualizations

### Design Principles
- **Modularity**: Loosely-coupled components with well-defined interfaces
- **Reproducibility**: Versioned data transformations and model configurations
- **Extensibility**: Easy addition of new data sources, features, or models

### Data Flow
1. Data acquisition framework fetches data from multiple providers (vnstock, otherlib/providerA, otherlib/providerB) with fallback mechanisms
2. Company universe definition and stock list management (VN30, custom lists, or any stock universe)
3. Data validation and integrity checks performed across all data sources
4. Raw data converted to qlib .bin format using dump_bin.py
5. qlib handlers and custom expressions generate features and labels
6. Multiple models trained using qlib workflow with cross-validation
7. Trained models generate predictions for multiple horizons
8. Predictions backtested using historical data with trading constraints
9. Performance reports and analysis visualizations generated

## 8. Coding Standards
### Python Standards
- **PEP 8**: Follow official Python style guidelines
- **Type Hints**: Use type annotations for all function parameters and returns
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Imports**: Group imports (standard library, third-party, local) with blank lines
- **No Lazy Imports**: Avoid lazy imports to ensure all dependencies are loaded at startup for better error detection and performance predictability
- **Line Length**: Maximum 88 characters for code, 72 for comments

### Code Organization
- **Functions**: Single responsibility principle, <50 lines preferred
- **Classes**: Cohesive design with clear separation of concerns
- **Modules**: One module per major component with clear interfaces
- **Packages**: Logical grouping with __init__.py files

### Error Handling
- **Exceptions**: Specific exception types with descriptive messages
- **Logging**: Structured logging with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **Validation**: Input validation with clear error messages
- **Recovery**: Graceful degradation with fallback mechanisms

### Testing Standards
- **Unit Tests**: Test individual functions with edge cases
- **Integration Tests**: Test component interactions and data flow
- **Coverage**: 90%+ code coverage with meaningful test cases
- **Naming**: test_function_name for unit tests, test_integration_component for integration tests
- **External APIs**: Do not test external APIs, test only business logic
- **API Wrappers**: Do not test wrappers around API calls, focus on core functionality

### Documentation Standards
- **Inline Comments**: Explain complex algorithms and business logic
- **Function Documentation**: Purpose, parameters, returns, examples, exceptions
- **Module Documentation**: Module purpose, dependencies, usage examples
- **README Files**: Setup instructions, usage examples, API reference

### Code Simplification Rules
- **Keep Code Simple**: Avoid multiple layers and abstractions. Refactor and simplify when possible
- **File Naming Convention**: Do not name any file starting with numbers

### Conciseness Rule
- **Be concise**: Eliminate unnecessary words in code comments and documentation
- **Be direct**: State technical requirements clearly without filler
- **Be brief**: Use short, clear function and variable names
- **Focus on essentials**: Include only relevant information in task definitions and code

## 9. Environment Setup and Installation
### Windows Development Environment
- **Operating System**: Windows 11 with developer tools enabled
- **Shell**: PowerShell (primary) or Git Bash for Unix-like commands
- **Package Manager**: Conda for environment management and package installation (MANDATORY - venv/virtualenv not permitted)
- **IDE**: VSCode with Python extensions and Jupyter notebook support
- **Git**: Git for Windows with proper line ending configuration (checkout as-is, commit Unix-style)
- **Command Chaining**: Windows PowerShell does not support && command chaining. Use separate commands or PowerShell syntax (;) for command chaining
- **Terminal Commands**: Be aware of "The token '&&' is not a valid statement separator in this version" for windows terminal commands

### Conda Installation and Initialization (Required)
**Conda is mandatory for this project. Virtual environments (venv/virtualenv) are not permitted.**

If conda is not installed, install Miniconda or Anaconda first:

```powershell
# Download and install Miniconda (recommended for this project)
# Visit https://docs.conda.io/en/latest/miniconda.html and download the Windows installer
# Run the installer and follow the setup wizard

# Alternative: Install Anaconda (includes more packages)
# Visit https://www.anaconda.com/products/distribution and download the Windows installer
# Run the installer and follow the setup wizard

# Initialize conda for PowerShell (run in new PowerShell session after installation)
conda init powershell

# Verify conda installation
conda --version
conda info
```

### Conda Environment Setup
```powershell
# Create conda environment with Python 3.12
conda create -n StocketAI python=3.12 -y
conda activate StocketAI

# Install pip and core scientific packages
conda install pip pandas numpy scipy matplotlib seaborn plotly -y
conda install scikit-learn lightgbm xgboost -y
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda install tensorflow -c conda-forge -y

# Install development and testing tools
conda install jupyter jupyterlab pytest flake8 black mypy -y
pip install pre-commit
```

### Source Code Installation for Custom Libraries
#### vnstock Installation
```powershell
# Clone vnstock repository
git clone https://github.com/thinh-vu/vnstock.git
cd vnstock

# Install in development mode
pip install -e .
# or for Windows-specific installation
pip install -e . --no-build-isolation

# Verify installation
python -c "import vnstock; print('vnstock installed successfully')"
```

#### qlib Installation
```powershell
# Clone qlib repository
git clone https://github.com/microsoft/qlib.git
cd qlib

# Install system dependencies for Windows
# Note: qlib requires specific build tools and may need Visual Studio Build Tools

# Install in development mode
pip install -e .
# or for Windows-specific installation
pip install -e . --no-build-isolation

# Verify installation
python -c "import qlib; print('qlib installed successfully')"
```

### Package Version Management
- **Core Dependencies**: Use latest stable versions compatible with Python 3.12
- **ML Libraries**: Use latest versions with Windows support
- **Data Processing**: Use pandas>=2.0, numpy>=1.24, scipy>=1.11
- **Visualization**: Use matplotlib>=3.7, seaborn>=0.12, plotly>=5.15
- **Deep Learning**: Use PyTorch>=2.0, TensorFlow>=2.13 for Windows
- **Development Tools**: Use flake8>=6.0, black>=23.0, mypy>=1.5

### Environment Configuration
```powershell
# Set environment variables for Windows
$env:PYTHONPATH = "$PWD/src;$PWD"
$env:QLIB_DATA = "$PWD/data/qlib_format"

# Create .env file for environment configuration
echo "PYTHONPATH=src/" > .env
echo "QLIB_DATA=data/qlib_format" >> .env
echo "LOG_LEVEL=INFO" >> .env
```

### Development Tools Setup
```powershell
# Install pre-commit hooks for code quality
pre-commit install

# Configure Jupyter kernel for the environment
python -m ipykernel install --user --name StocketAI --display-name "StocketAI"

# Set up VSCode workspace settings
# Create .vscode/settings.json with Python interpreter path
```

### Troubleshooting Windows-Specific Issues
- **Line Endings**: Configure Git to handle line endings properly (core.autocrlf=false)
- **Path Length**: Use short paths or enable long path support in Windows
- **Permissions**: Run PowerShell as Administrator for system-wide installations
- **Build Tools**: Install Microsoft Visual C++ Build Tools if compilation issues occur
- **Conda Issues**: Use `conda clean --all` and recreate environment if conflicts arise

## 10. AI Coding Agent Task Development Workflow
### Task Structure
- **Sequential Execution**: Tasks completed in dependency order (01-09)
- **Self-Contained Tasks**: Each task has clear deliverables and completion criteria
- **Progress Tracking**: Checklist-based progress monitoring with status updates
- **Documentation**: Comprehensive task documentation with requirements and dependencies

### Script Organization Requirements
**All commands and scripts must be declarative and organized in specific directories:**

#### Scripts
- **Location**: `/scripts/`
- **Purpose**: All commands needed for ongoing development must be declarative scripts
- **Naming Convention**: Use descriptive names (e.g., `run_data_processing.ps1`, `train_model.ps1`, `evaluate_performance.ps1`)
- **Content**: Each script should be self-contained with proper error handling and logging
- **Execution**: Scripts should be executable with clear entry points and configuration options
- **Development Scripts**: PowerShell scripts with proper encoding (UTF-8 with BOM)
- **Setup Scripts**: Python scripts with proper shebang (`#!/usr/bin/env python3`)
- **Documentation**: Include comments and help documentation explaining purpose, usage, and parameters
- **Error Handling**: Comprehensive error handling with informative messages and proper exit codes
- **Configuration**: Accept configuration via parameters, environment variables, or config files
- **Logging**: Structured logging with appropriate levels using Write-Host, Write-Warning, Write-Error
- **Testing**: Include validation and verification steps for complex scripts

### Development Process
1. **Task Analysis**: Read complete task.md, review dependencies, assess resources
2. **Planning**: Create dependencies.md and deliverables.md files
3. **Implementation**: Develop code in appropriate src/ subdirectories
4. **Testing**: Create unit tests with 90%+ coverage
5. **Validation**: Verify integration with dependent components
6. **Documentation**: Update documentation and mark task complete

### Quality Gates
- **Code Quality**: Passes linting (flake8, black) and type checking (mypy)
- **Functionality**: Meets all specified requirements and use cases
- **Performance**: Satisfies performance criteria for data processing and model training
- **Testing**: Comprehensive test suite with passing tests
- **Documentation**: Complete and accurate documentation

### Artifact Management
- **Code Artifacts**: Python modules in src/ with proper imports and interfaces
- **Data Artifacts**: Raw, processed, and qlib-format data in data/ directories
- **Model Artifacts**: Trained models with metadata in models/ directory
- **Documentation Artifacts**: Updated docs, README files, and usage examples

## 11. Hands-On Best Practices
### Data Handling
- Always validate data inputs with schema checking
- Implement caching for expensive API calls
- Use consistent data formats across all modules
- Document data transformations and assumptions
- Implement automated data quality monitoring

### Model Development
- Use cross-validation for robust performance estimates
- Track experiments with systematic hyperparameter logging
- Save model artifacts with comprehensive metadata
- Implement proper train/validation/test splits
- Document model assumptions and limitations

### Performance Optimization
- Profile code to identify bottlenecks
- Implement lazy loading for large datasets
- Cache intermediate results when appropriate
- Monitor memory usage for large-scale operations

### Error Prevention
- Implement defensive programming with input validation
- Use type hints to catch errors early
- Write comprehensive unit tests for edge cases
- Document failure modes and recovery procedures
- Implement health checks for critical components

### Collaboration Practices
- Use meaningful commit messages with context
- Write self-documenting code with clear variable names
- Maintain consistent code formatting across the team
- Document API changes and breaking changes
- Create reusable components for common patterns

## 12. Research Documentation and Notebook Standards
### Notebook Organization Principles
- **Use Case Specific**: Notebooks are designed for specific use cases and stock universes
- **Atomic and Simple**: Notebooks must be "stupid simple" with no Python business logic
- **No Business Logic**: All Python functions and complex operations must be extracted to separate script modules
- **Clear Sections**: Include Overview, Requirements, Usage, and step-by-step execution sections
- **Progress Tracking**: Implement comprehensive logging and progress monitoring throughout execution
- **Reproducibility**: Ensure notebooks can be run independently with clear dependency management

### Production-Ready Notebook Requirements
- **No Demo Code**: All notebooks must be production-ready with full functionality
- **Jupyter Notebooks Only**: All user-executable scripts must be organized as Jupyter notebooks
- **Full Scope Execution**: Notebooks must be ready for complete workflow execution, not demonstrations
- **No Partial Implementations**: Every notebook must represent complete, working solutions for specific use cases
- **Complete Workflows**: Notebooks must execute end-to-end workflows without requiring external scripts

### Directory Structure and Use Case Organization
```
notebooks/
├── vn30/                    # VN30 specific notebooks only
│   ├── 01_data_acquisition.ipynb     # VN30 data acquisition
│   ├── 02_data_processing.ipynb      # VN30 data processing
│   ├── 03_feature_engineering.ipynb # VN30 feature engineering
│   └── data_acquisition_scripts/     # VN30-specific modules
├── [provider_name]/        # Other provider-specific notebooks
│   ├── 01_data_acquisition.ipynb
│   └── [provider_specific_modules]/
└── common/                # Provider-agnostic notebooks
    ├── 01_model_training.ipynb
    └── [shared_modules]/
```

This constitution establishes the technical foundation, standards, and practices for developing StocketAI focused on research excellence and quantitative finance methodologies.

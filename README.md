# StocketAI

**Predict stock prices with AI - Simple, Research-Focused, Extensible**

---

## What is StocketAI?

StocketAI helps you predict stock price movements using machine learning. It can forecast whether stocks will go up or down in 1, 3, or 6 months. The system is designed for researchers and analysts who want to experiment with different data sources and prediction models.

**Key Benefits:**
- 🎯 **Research-First**: Built for experimentation and scientific validation
- 🔧 **Extensible**: Easy to add new data sources or prediction models
- 📊 **Multi-Source**: Works with multiple Vietnamese financial data providers
- 🧪 **Reproducible**: Consistent results across different runs

---

## Motivation

As a solution architect without finance expertise or Python development background, I want to build an AI model for each company from the VN30 list to predict stock prices in 1, 3, and 6-month horizons with low risk using all available data. This project leverages vnstock for comprehensive Vietnamese market data acquisition and qlib for quantitative finance modeling to create a research-focused prediction system that balances technical sophistication with practical usability.

---

## What You Can Do

### 📈 Predict Stock Movements
- Forecast price changes for 1, 3, or 6 months ahead
- Get confidence scores for each prediction
- Generate buy/hold/sell signals

### 📊 Analyze Performance
- Test predictions against historical data
- Measure accuracy with standard finance metrics
- Simulate portfolio performance with trading costs

### 🔬 Experiment & Research
- Try different machine learning models
- Compare prediction strategies
- Add new data sources or features

---

## Technology Stack

**Core Components:**
- **Python 3.12+** - Modern, reliable programming language
- **vnstock** - Vietnamese market data (prices, financials, news)
- **qlib** - Advanced financial modeling toolkit

**Machine Learning:**
- **PyTorch/TensorFlow** - Deep learning frameworks
- **LightGBM/XGBoost** - Fast, accurate tree-based models
- **scikit-learn** - Traditional ML algorithms

**Data & Visualization:**
- **pandas/numpy** - Data manipulation
- **matplotlib/plotly** - Charts and interactive dashboards

## Project Structure

```
StocketAI/
├── data/
│   ├── symbols/                # Individual stock symbol organization
│   │   └── {symbol}/           # Each symbol as independent data unit
│   │       ├── raw/            # Raw data from vnstock APIs
│   │       ├── processed/      # Cleaned and validated data
│   │       ├── qlib_format/    # Qlib .bin format data
│   │       ├── progress/       # Processing progress and status
│   │       ├── reports/        # Analysis reports and metrics
│   │       └── errors/         # Error logs and debugging info
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

## Installation

### Prerequisites
- Windows 11 with developer tools enabled
- Conda (mandatory - venv/virtualenv not permitted)
- Git for Windows with proper line ending configuration

### Environment Setup

```powershell
# Create conda environment
conda create -n StocketAI python=3.12 -y
conda activate StocketAI

# Install core packages
conda install pip pandas numpy scipy matplotlib seaborn plotly -y
conda install scikit-learn lightgbm xgboost -y
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda install tensorflow -c conda-forge -y

# Install development tools
conda install jupyter jupyterlab pytest flake8 black mypy -y
pip install pre-commit
```

### Source Code Installation

#### vnstock Installation
```powershell
git clone https://github.com/thinh-vu/vnstock.git
cd vnstock
pip install -e .
```

#### qlib Installation
```powershell
git clone https://github.com/microsoft/qlib.git
cd qlib
pip install -e .
```

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd StocketAI
   ```

2. **Set up the environment**
   ```powershell
   conda activate StocketAI
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```powershell
   $env:PYTHONPATH = "$PWD/src;$PWD"
   $env:QLIB_DATA = "$PWD/data/qlib_format"
   ```

4. **Run initial data acquisition**
   ```bash
   jupyter notebook notebooks/vn30/01_load_vn30_constituents.ipynb
   ```

## Usage

The project provides Jupyter notebooks for different use cases:

- `notebooks/vn30/` - VN30 specific workflows
- `notebooks/common/` - Provider-agnostic operations
- `notebooks/[provider_name]/` - Other provider-specific notebooks

Each notebook contains complete, production-ready workflows for data acquisition, processing, model training, and evaluation.

## Development

### Coding Standards
- PEP 8 compliance with 88-character line limit
- Type hints for all functions and methods
- Google-style docstrings for public APIs
- Grouped imports with proper ordering

### Testing
- Unit tests for individual functions with edge cases
- Integration tests for component interactions
- 90%+ code coverage requirement
- Focus on business logic, not external API testing

### Quality Gates
- Code quality: passes flake8, black, mypy
- Functionality: meets all specified requirements
- Testing: comprehensive test suite
- Documentation: complete and accurate

## Contributing

1. Follow the established coding standards and architecture principles
2. Create comprehensive unit tests for new functionality
3. Update documentation for any API changes
4. Ensure all quality gates pass before submitting

## License

GPLv3

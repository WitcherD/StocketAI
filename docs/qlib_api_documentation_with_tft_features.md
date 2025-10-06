# Qlib API Documentation with TFT-Compatible Features

## Overview

This document provides comprehensive API documentation for qlib features, with special emphasis on TFT (Temporal Fusion Transformer) compatible feature categories and technical indicators. It covers the complete workflow from data acquisition to model training and prediction.

## Table of Contents

- [TFT-Compatible Feature Categories](#tft-compatible-feature-categories)
- [Technical Indicators](#technical-indicators)
- [Data Layer APIs](#data-layer-apis)
- [Model Layer APIs](#model-layer-apis)
- [Workflow APIs](#workflow-apis)
- [Practical Examples](#practical-examples)

## TFT-Compatible Feature Categories

### OBSERVED_INPUT (Time-varying Technical Indicators)

These features are time-varying technical indicators that change over time and are observed by the model during training and inference.

**Source Reference**: `qlib/examples/benchmarks/TFT/data_formatters/qlib_Alpha158.py` - Alpha158Formatter class defines the TFT feature categories and data types.

| Feature | Description | Periods | Implementation | Source Reference |
|---------|-------------|---------|----------------|------------------|
| RESI | Residuals (Close - MA) | 5, 10 | `RESI5`, `RESI10` | `qlib/qlib/contrib/data/loader.py` Alpha158DL.get_feature_config() |
| WVMA | Weighted Moving Average | 5, 60 | `WVMA5`, `WVMA60` | `qlib/qlib/contrib/data/loader.py` Alpha158DL.get_feature_config() |
| RSQR | R-squared | 5, 10, 20, 60 | `RSQR5`, `RSQR10`, `RSQR20`, `RSQR60` | `qlib/qlib/contrib/data/loader.py` Alpha158DL.get_feature_config() |
| CORR | Correlation (Close vs Volume) | 5, 10, 20, 60 | `CORR5`, `CORR10`, `CORR20`, `CORR60` | `qlib/qlib/contrib/data/loader.py` Alpha158DL.get_feature_config() |
| CORD | Correlation Difference | 5, 10, 60 | `CORD5`, `CORD10`, `CORD60` | `qlib/qlib/contrib/data/loader.py` Alpha158DL.get_feature_config() |
| ROC | Rate of Change | 60 | `ROC60` | `qlib/qlib/contrib/data/loader.py` Alpha158DL.get_feature_config() |
| VSTD | Volatility (Returns STD) | 5 | `VSTD5` | `qlib/qlib/contrib/data/loader.py` Alpha158DL.get_feature_config() |
| STD | Standard Deviation | 5 | `STD5` | `qlib/qlib/contrib/data/loader.py` Alpha158DL.get_feature_config() |
| KLEN | Trend Length | N/A | `KLEN` | `qlib/qlib/contrib/data/loader.py` Alpha158DL.get_feature_config() |
| KLOW | Lowest Price in Period | 20-day | `KLOW` | `qlib/qlib/contrib/data/loader.py` Alpha158DL.get_feature_config() |

### KNOWN_INPUT (Temporal Features)

These are temporal features that are known in advance and help the model understand time patterns.

| Feature | Description | Values |
|---------|-------------|---------|
| month | Month of year | 1-12 |
| day_of_week | Day of week | 0-6 (Monday-Sunday) |
| year | Year | YYYY format |

### STATIC_INPUT (Static Features)

These are static features that don't change over time for each instrument.

| Feature | Description | Value |
|---------|-------------|-------|
| const | Constant feature | 1.0 |

## Technical Indicators

### Residuals (RESI)

**Formula**: `RESI_period = close - MA(close, period)`

**API Usage**:
```python
from qlib.data.ops import Feature, Mean

# Calculate RESI5
resi5 = Feature('close') - Mean(Feature('close'), 5)
```

### Weighted Moving Average (WVMA)

**Formula**: Weighted moving average with linear decay weights

**Source Reference**: `qlib/qlib/data/ops.py` - WMA class implements weighted moving average calculation

**API Usage**:
```python
from qlib.data.ops import WMA

# Calculate WVMA5
wvma5 = WMA(Feature('close'), 5)
```

### R-squared (RSQR)

**Formula**: R-squared of linear regression over the period

**Source Reference**: `qlib/qlib/data/ops.py` - Rsquare class implements rolling R-squared calculation using optimized Cython functions

**API Usage**:
```python
from qlib.data.ops import Rsquare

# Calculate RSQR10
rsqr10 = Rsquare(Feature('close'), 10)
```

### Correlation (CORR)

**Formula**: Pearson correlation between close price and volume

**Source Reference**: `qlib/qlib/data/ops.py` - Corr class implements rolling correlation calculation with NaN handling for zero variance cases

**API Usage**:
```python
from qlib.data.ops import Corr

# Calculate CORR5
corr5 = Corr(Feature('close'), Feature('volume'), 5)
```

### Correlation Difference (CORD)

**Formula**: `CORD_period = CORR_period - CORR_prev_period`

**Implementation**:
```python
# CORD5 = CORR5 - CORR_prev_available
cord5 = corr5 - corr_prev
```

### Rate of Change (ROC)

**Formula**: `ROC_period = (close / close_period_ago - 1) * 100`

**API Usage**:
```python
# Calculate ROC60
roc60 = (Feature('close') / Ref(Feature('close'), 60) - 1) * 100
```

### Volatility (VSTD)

**Formula**: Standard deviation of returns over the period

**API Usage**:
```python
from qlib.data.ops import Std

# Calculate VSTD5
returns = (Feature('close') - Delay(Feature('close'), 1)) / Delay(Feature('close'), 1)
vstd5 = Std(returns, 5)
```

### Standard Deviation (STD)

**Formula**: Standard deviation of close prices over the period

**Source Reference**: `qlib/qlib/data/ops.py` - Std class implements rolling standard deviation calculation

**API Usage**:
```python
from qlib.data.ops import Std

# Calculate STD5
std5 = Std(Feature('close'), 5)
```

### Momentum Indicators (KLEN, KLOW)

**KLEN**: Length of consecutive up/down trend
**KLOW**: Lowest price in 20-day rolling window

## Data Layer APIs

### Data Preparation

#### Converting CSV to Qlib Format

```bash
python scripts/dump_bin.py dump_all \
    --data_path ~/.qlib/my_data \
    --qlib_dir ~/.qlib/qlib_data/ \
    --include_fields open,close,high,low,volume,factor \
    --file_suffix .csv
```

#### Data Health Check

```bash
python scripts/check_data_health.py check_data \
    --qlib_dir ~/.qlib/qlib_data/cn_data \
    --missing_data_num 30055 \
    --large_step_threshold_volume 94485 \
    --large_step_threshold_price 20
```

### Data Handler API

#### Alpha158 Data Handler

**Source Reference**: `qlib/qlib/contrib/data/handler.py` - Alpha158 class provides 158 technical indicators for quantitative finance modeling.

```python
from qlib.contrib.data.handler import Alpha158

data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": "csi300",
}

handler = Alpha158(**data_handler_config)

# Get all columns
columns = handler.get_cols()

# Fetch features
features = handler.fetch(col_set="feature")

# Fetch labels
labels = handler.fetch(col_set="label")
```

#### Custom Data Handler with TFT Features

```python
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.ops import *

class TFTDataHandler(DataHandlerLP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_feature_config(self):
        return {
            "RESI5": Feature('close') - Mean(Feature('close'), 5),
            "WVMA5": WMA(Feature('close'), 5),
            "RSQR5": Rsquare(Feature('close'), 5),
            "CORR5": Corr(Feature('close'), Feature('volume'), 5),
            "VSTD5": Std((Feature('close') - Delay(Feature('close'), 1)) /
                        Delay(Feature('close'), 1), 5),
            "STD5": Std(Feature('close'), 5),
            "ROC60": (Feature('close') / Ref(Feature('close'), 60) - 1) * 100,
            "month": Feature('date').dt.month,
            "day_of_week": Feature('date').dt.dayofweek,
            "const": 1.0
        }
```

### Dataset API

#### DatasetH Configuration

```python
from qlib.data.dataset import DatasetH

dataset_config = {
    "handler": {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": data_handler_config,
    },
    "segments": {
        "train": ("2008-01-01", "2014-12-31"),
        "valid": ("2015-01-01", "2016-12-31"),
        "test": ("2017-01-01", "2020-08-01"),
    },
}

dataset = DatasetH(**dataset_config)
```

### Expression Engine Operators

#### Basic Operators

```python
from qlib.data.ops import *

# Arithmetic operations
result = Feature('close') + Feature('open')

# Statistical operations
ma5 = Mean(Feature('close'), 5)
std5 = Std(Feature('close'), 5)

# Reference operations
prev_close = Ref(Feature('close'), 1)
future_close = Ref(Feature('close'), -1)

# Rolling operations
rolling_sum = Sum(Feature('volume'), 10)
rolling_corr = Corr(Feature('close'), Feature('volume'), 20)
```

#### Advanced Operators

```python
# Decay linear weighted moving average
decay_ma = DecayLinear(Feature('close'), 5)

# Time series delay
delayed = Delay(Feature('close'), 3)

# Rolling rank
rank = Rank(Feature('close'), 10)

# Cross-sectional operations
cs_mean = CSMean(Feature('close'))
cs_rank = CSRank(Feature('close'))
```

## Model Layer APIs

### Base Model Interface

```python
from qlib.model.base import Model

class CustomModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, dataset):
        """Train the model"""
        pass

    def predict(self, dataset):
        """Make predictions"""
        pass

    def save(self, path):
        """Save model"""
        pass

    def load(self, path):
        """Load model"""
        pass
```

### Tree-based Models

#### LightGBM Model

```python
from qlib.contrib.model.gbdt import LGBModel

lgb_config = {
    "loss": "mse",
    "colsample_bytree": 0.8879,
    "learning_rate": 0.0421,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 20,
}

model = LGBModel(**lgb_config)
model.fit(dataset)
predictions = model.predict(dataset)
```

#### XGBoost Model

```python
from qlib.contrib.model.gbdt import XGBModel

xgb_config = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

model = XGBModel(**xgb_config)
```

### Neural Network Models

#### MLP Model

```python
from qlib.contrib.model.mlp import MLPModel

mlp_config = {
    "input_dim": 158,  # Alpha158 features
    "hidden_dims": [256, 128, 64],
    "output_dim": 1,
    "activation": "relu",
    "dropout": 0.3,
    "lr": 0.001,
    "batch_size": 2048,
    "epochs": 100,
}

model = MLPModel(**mlp_config)
```

#### LSTM Model

```python
from qlib.contrib.model.rnn import LSTMModel

lstm_config = {
    "input_dim": 158,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 0.001,
    "batch_size": 2048,
    "epochs": 100,
}

model = LSTMModel(**lstm_config)
```

#### TFT Model

**Source Reference**: `qlib/examples/benchmarks/TFT/tft.py` - TFTModel class provides TensorFlow-based TFT implementation with Qlib integration.

```python
from qlib.contrib.model.tft import TFTModel

tft_config = {
    "input_dim": 158,
    "hidden_dim": 64,
    "attention_dim": 32,
    "num_heads": 4,
    "dropout": 0.1,
    "lr": 0.001,
    "batch_size": 1024,
    "epochs": 50,
    # TFT-specific parameters
    "num_quantiles": 3,
    "valid_quantiles": [0.1, 0.5, 0.9],
}

model = TFTModel(**tft_config)
```

### Transformer-based Models

#### TRA Model (Transformer with Attention)

```python
from qlib.contrib.model.transformer import TRAModel

tra_config = {
    "input_dim": 158,
    "hidden_dim": 128,
    "num_layers": 3,
    "num_heads": 8,
    "dropout": 0.1,
    "lr": 0.0005,
    "batch_size": 1024,
}

model = TRAModel(**tra_config)
```

#### HIST Model (Hierarchical Inference Strategy Transformer)

```python
from qlib.contrib.model.transformer import HISTModel

hist_config = {
    "input_dim": 158,
    "hidden_dim": 128,
    "num_hierarchies": 3,
    "dropout": 0.1,
    "lr": 0.0005,
}

model = HISTModel(**hist_config)
```

### Specialized Models

#### KRNN Model (Kernel Recurrent Neural Network)

```python
from qlib.contrib.model.rnn import KRNNModel

krnn_config = {
    "input_dim": 158,
    "hidden_dim": 64,
    "kernel_size": 5,
    "dropout": 0.2,
}

model = KRNNModel(**krnn_config)
```

#### DDG-DA Model (Dynamic Data Generation - Domain Adaptation)

```python
from qlib.contrib.model.meta import DDGDA

ddgda_config = {
    "input_dim": 158,
    "hidden_dim": 128,
    "meta_lr": 0.001,
    "adaptation_steps": 5,
}

model = DDGDA(**ddgda_config)
```

## Workflow APIs

### Qrun Configuration

#### Complete Workflow Configuration

```yaml
qlib_init:
    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn

market: &market csi300
benchmark: &benchmark SH000300

data_handler_config: &data_handler_config
    start_time: 2008-01-01
    end_time: 2020-08-01
    fit_start_time: 2008-01-01
    fit_end_time: 2014-12-31
    instruments: *market

task:
    model:
        class: TFTModel
        module_path: qlib.contrib.model.tft
        kwargs:
            input_dim: 158
            hidden_dim: 64
            attention_dim: 32
            num_heads: 4
            dropout: 0.1
            lr: 0.001
            batch_size: 1024
            epochs: 50
            num_quantiles: 3
            valid_quantiles: [0.1, 0.5, 0.9]

    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2008-01-01, 2014-12-31]
                valid: [2015-01-01, 2016-12-31]
                test: [2017-01-01, 2020-08-01]

    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs: {}
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
              config:
                  strategy:
                      class: TopkDropoutStrategy
                      module_path: qlib.contrib.strategy.strategy
                      kwargs:
                          topk: 50
                          n_drop: 5
                          signal: <PRED>
                  backtest:
                      start_time: 2017-01-01
                      end_time: 2020-08-01
                      account: 100000000
                      benchmark: *benchmark
                      exchange_kwargs:
                          limit_threshold: 0.095
                          deal_price: close
                          open_cost: 0.0005
                          close_cost: 0.0015
                          min_cost: 5
```

#### Running the Workflow

```bash
# Run the complete workflow
qrun workflow_config.yaml

# Run in debug mode
python -m pdb qlib/cli/run.py workflow_config.yaml
```

### Programmatic Workflow

```python
from qlib.contrib.model.tft import TFTModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

# Initialize components
data_handler = Alpha158(**data_handler_config)
dataset = DatasetH(handler=data_handler, segments=segments)
model = TFTModel(**tft_config)

# Start experiment
with R.start(experiment_name="tft_experiment"):
    # Train model
    model.fit(dataset)

    # Generate signals
    recorder = R.get_recorder()
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # Portfolio analysis
    par = PortAnaRecord(model, dataset, recorder)
    par.generate()
```

## Practical Examples

### Example 1: TFT Feature Engineering for VN30

```python
from src.feature_engineering.tft_feature_engineer import VN30TFTFeatureEngineer
import pandas as pd

# Create sample data
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
sample_data = {
    'time': dates,
    'open': np.random.uniform(100, 200, len(dates)),
    'high': np.random.uniform(150, 250, len(dates)),
    'low': np.random.uniform(50, 150, len(dates)),
    'close': np.random.uniform(100, 200, len(dates)),
    'volume': np.random.uniform(100000, 1000000, len(dates))
}
df = pd.DataFrame(sample_data)

# Initialize feature engineer
engineer = VN30TFTFeatureEngineer()

# Engineer all TFT-compatible features
engineered_df = engineer.engineer_all_features(df)

# Validate features
validation = engineer.validate_features(engineered_df)
print(f"Validation results: {validation}")

# Get feature summary
summary = engineer.get_feature_summary(engineered_df)
print(summary)
```

### Example 2: Training TFT Model with Custom Features

```python
import qlib
from qlib.contrib.model.tft import TFTModel
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158

# Initialize Qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')

# Data handler configuration
data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": "csi300",
}

# TFT model configuration
tft_config = {
    "input_dim": 158,
    "hidden_dim": 64,
    "attention_dim": 32,
    "num_heads": 4,
    "dropout": 0.1,
    "lr": 0.001,
    "batch_size": 1024,
    "epochs": 50,
    "num_quantiles": 3,
    "valid_quantiles": [0.1, 0.5, 0.9],
}

# Create dataset
handler = Alpha158(**data_handler_config)
dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2008-01-01", "2014-12-31"),
        "valid": ("2015-01-01", "2016-12-31"),
        "test": ("2017-01-01", "2020-08-01"),
    }
)

# Train TFT model
model = TFTModel(**tft_config)
model.fit(dataset)

# Make predictions
predictions = model.predict(dataset)
print(f"Predictions shape: {predictions.shape}")
```

### Example 3: Custom Expression with TFT Features

```python
from qlib.data.ops import *

# Define TFT-compatible expressions
expressions = {
    # Observed inputs
    "RESI5": Feature('close') - Mean(Feature('close'), 5),
    "WVMA5": WMA(Feature('close'), 5),
    "RSQR5": Rsquare(Feature('close'), 5),
    "CORR5": Corr(Feature('close'), Feature('volume'), 5),
    "VSTD5": Std((Feature('close') - Delay(Feature('close'), 1)) /
                 Delay(Feature('close'), 1), 5),
    "STD5": Std(Feature('close'), 5),
    "ROC60": (Feature('close') / Ref(Feature('close'), 60) - 1) * 100,

    # Known inputs (temporal)
    "month": Feature('date').dt.month,
    "day_of_week": Feature('date').dt.dayofweek,
    "year": Feature('date').dt.year,

    # Static inputs
    "const": 1.0,

    # Custom composite features
    "momentum_score": (Feature('close') - Ref(Feature('close'), 20)) /
                     Ref(Feature('close'), 20),
    "volatility_ratio": Std(Feature('close'), 5) / Std(Feature('close'), 20),
}

# Use expressions in data handler
class CustomTFTHandler(DataHandlerLP):
    def get_feature_config(self):
        return expressions
```

### Example 4: Multi-Horizon Forecasting with TFT

```python
# Configuration for multi-horizon forecasting
multi_horizon_config = {
    "model": {
        "class": "TFTModel",
        "module_path": "qlib.contrib.model.tft",
        "kwargs": {
            "input_dim": 158,
            "hidden_dim": 128,
            "attention_dim": 64,
            "num_heads": 8,
            "dropout": 0.1,
            "lr": 0.0005,
            "batch_size": 512,
            "epochs": 100,
            # Multi-horizon settings
            "output_horizons": [1, 3, 6, 12],  # 1, 3, 6, 12 months
            "num_quantiles": 3,
            "valid_quantiles": [0.1, 0.5, 0.9],
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01"),
            },
        },
    },
}

# The model will output predictions for all specified horizons
# predictions shape: (n_samples, n_horizons, n_quantiles)
```

### Example 5: Backtesting TFT Strategy

```python
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest

# Strategy configuration
strategy_config = {
    "class": "TopkDropoutStrategy",
    "module_path": "qlib.contrib.strategy.strategy",
    "kwargs": {
        "topk": 30,  # Select top 30 stocks
        "n_drop": 3,  # Drop 3 worst performers
        "signal": "<PRED>",  # Use model predictions
    },
}

# Backtest configuration
backtest_config = {
    "start_time": "2017-01-01",
    "end_time": "2020-08-01",
    "account": 100000000,  # 100 million initial capital
    "benchmark": "SH000300",
    "exchange_kwargs": {
        "limit_threshold": 0.095,  # 9.5% daily limit for China market
        "deal_price": "close",
        "open_cost": 0.0005,   # 0.05% opening cost
        "close_cost": 0.0015,  # 0.15% closing cost
        "min_cost": 5,         # Minimum transaction cost
    },
}

# Run backtest
strategy = TopkDropoutStrategy(**strategy_config["kwargs"])
executor = backtest(strategy, **backtest_config)

# Analyze results
portfolio_analysis = executor.portfolio_analysis()
print(f"Total return: {portfolio_analysis.total_return}")
print(f"Annual return: {portfolio_analysis.annual_return}")
print(f"Sharpe ratio: {portfolio_analysis.sharpe_ratio}")
print(f"Maximum drawdown: {portfolio_analysis.max_drawdown}")
```

## References and Bibliography

### Core Qlib Documentation
- **Qlib Official Documentation**: `qlib/docs/` - Complete framework documentation
- **Data Layer Documentation**: `qlib/docs/component/data.rst` - Detailed data processing APIs
- **Model Layer Documentation**: `qlib/docs/component/model.rst` - Model training and prediction APIs
- **Workflow Documentation**: `qlib/docs/component/workflow.rst` - Automated workflow management

### TFT Implementation References
- **TFT Original Paper**: Lim et al. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2019)
- **Qlib TFT Implementation**: `qlib/examples/benchmarks/TFT/tft.py` - TensorFlow-based TFT model
- **TFT Data Formatter**: `qlib/examples/benchmarks/TFT/data_formatters/qlib_Alpha158.py` - Feature categorization

### Technical Indicator References
- **Alpha158 Features**: `qlib/qlib/contrib/data/loader.py` - Complete 158 technical indicators
- **Alpha158 Data Handler**: `qlib/qlib/contrib/data/handler.py` - Data processing pipeline
- **Expression Engine**: `qlib/qlib/data/ops.py` - Mathematical operators for feature engineering

### Research Papers
- **Qlib Framework**: "Qlib: An AI-oriented Quantitative Investment Platform" (Microsoft Research)
- **Alpha158**: Guo et al. "Alpha158: A Collection of Technical Indicators" (2021)
- **Temporal Fusion Transformers**: Lim et al. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2019)

### Key Source Files
- `src/feature_engineering/tft_feature_engineer.py` - VN30 TFT feature engineering implementation
- `qlib/examples/benchmarks/TFT/` - Complete TFT benchmark implementation
- `qlib/qlib/contrib/data/` - Data handlers and loaders
- `qlib/qlib/contrib/model/` - Model implementations

This comprehensive documentation covers all major qlib APIs with special focus on TFT-compatible features and practical implementation examples for stock prediction workflows.

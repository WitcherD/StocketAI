# Qlib Research Findings for Task 05: Baseline Model Training

## Overview
This document contains comprehensive research findings from qlib documentation, examples, and API references needed to implement Task 05 baseline model training for VN30 symbols.

## Core Qlib Architecture

### Model Base Classes
- **Base Model**: `qlib.model.base.Model` - All models inherit from this
- **ModelFT**: `qlib.model.base.ModelFT` - Includes finetuning capabilities
- **Key Methods**:
  - `fit(dataset)` - Train the model
  - `predict(dataset)` - Generate predictions
  - `save(path)` - Save model
  - `load(path)` - Load model

### Data Layer Components
- **DataHandlerLP**: Learnable data preprocessing with processors
- **DatasetH**: Dataset wrapper with data handler
- **TSDatasetH**: Time series dataset for sequential models
- **Data Preparation**: CSV/Parquet → Qlib format (.bin files)

## Model Implementations

### TFT Model (Temporal Fusion Transformer)

#### Configuration (from `qlib/examples/benchmarks/TFT/workflow_config_tft_Alpha158.yaml`)
```yaml
task:
    model:
        class: TFTModel
        module_path: tft  # Note: This is a custom module path
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
```

#### TFT Model Implementation (`qlib/examples/benchmarks/TFT/tft.py`)
```python
class TFTModel(ModelFT):
    def __init__(self, **kwargs):
        self.model = None
        self.params = {"DATASET": "Alpha158", "label_shift": 5}
        self.params.update(kwargs)

    def fit(self, dataset: DatasetH, MODEL_FOLDER="qlib_tft_model", USE_GPU_ID=0, **kwargs):
        # Data preparation
        dtrain, dvalid = self._prepare_data(dataset)

        # Process data for TFT format
        train = process_qlib_data(dtrain, DATASET, fillna=True).dropna()
        valid = process_qlib_data(dvalid, DATASET, fillna=True).dropna()

        # Initialize TFT model
        config = ExperimentConfig(DATASET)
        self.data_formatter = config.make_data_formatter()
        self.model_folder = MODEL_FOLDER

        # Training process
        ModelClass = libs.tft_model.TemporalFusionTransformer
        fixed_params = self.data_formatter.get_experiment_params()
        params = self.data_formatter.get_default_model_params()
        params = {**params, **fixed_params}
        params["model_folder"] = self.model_folder

        self.model = ModelClass(params, use_cudnn=use_gpu[0])
        self.model.fit(train_df=train, valid_df=valid)

    def predict(self, dataset):
        # Prediction process
        output_map = self.model.predict(test, return_targets=True)
        p50_forecast = self.data_formatter.format_predictions(output_map["p50"])
        p90_forecast = self.data_formatter.format_predictions(output_map["p90"])
        predict = (predict50 + predict90) / 2
        return predict
```

#### TFT Data Formatter (`qlib/examples/benchmarks/TFT/data_formatters/qlib_Alpha158.py`)
```python
class Alpha158Formatter(GenericDataFormatter):
    _column_definition = [
        ("instrument", DataTypes.CATEGORICAL, InputTypes.ID),
        ("LABEL0", DataTypes.REAL_VALUED, InputTypes.TARGET),
        ("date", DataTypes.DATE, InputTypes.TIME),
        ("month", DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ("day_of_week", DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        # 20 selected TFT features
        ("RESI5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("WVMA5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        # ... more features
        ("const", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    def get_fixed_params(self):
        return {
            "total_time_steps": 6 + 6,
            "num_encoder_steps": 6,
            "num_epochs": 100,
            "early_stopping_patience": 10,
            "multiprocessing_workers": 5,
        }

    def get_default_model_params(self):
        return {
            "dropout_rate": 0.4,
            "hidden_layer_size": 160,
            "learning_rate": 0.0001,
            "minibatch_size": 128,
            "max_gradient_norm": 0.0135,
            "num_heads": 1,
            "stack_size": 1,
        }
```

### LightGBM Model

#### Configuration (from `qlib/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`)
```yaml
task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs:
            loss: mse
            colsample_bytree: 0.8879
            learning_rate: 0.0421
            subsample: 0.8789
            lambda_l1: 205.6999
            lambda_l2: 580.9768
            max_depth: 8
            num_leaves: 210
            num_threads: 20
```

#### LightGBM Model Class (`qlib/qlib/contrib/model/gbdt.py`)
```python
class LGBModel(Model):
    def __init__(self, loss="mse", early_stopping_rounds=50, num_boost_round=1000, **kwargs):
        self.params = {
            "objective": "regression" if loss == "mse" else loss,
            "boosting": "gbdt",
            "early_stopping_rounds": early_stopping_rounds,
            "num_boost_round": num_boost_round,
            "verbose": -1,
        }
        self.params.update(kwargs)

    def fit(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"])
        dtrain = lgb.Dataset(df_train["feature"], label=df_train["label"])
        dvalid = lgb.Dataset(df_valid["feature"], label=df_valid["label"])

        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
        )

    def predict(self, dataset: DatasetH):
        df_test = dataset.prepare("test", col_set=["feature"])
        return pd.Series(self.model.predict(df_test["feature"]), index=df_test.index)
```

### LSTM Model

#### Configuration (from `qlib/examples/benchmarks/LSTM/workflow_config_lstm_Alpha158.yaml`)
```yaml
task:
    model:
        class: LSTM
        module_path: qlib.contrib.model.pytorch_lstm_ts
        kwargs:
            d_feat: 20
            hidden_size: 64
            num_layers: 2
            dropout: 0.0
            n_epochs: 200
            lr: 1e-3
            early_stop: 10
            batch_size: 800
            metric: loss
            loss: mse
            n_jobs: 20
            GPU: 0
    dataset:
        class: TSDatasetH  # Note: Uses TSDatasetH for sequential data
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
            step_len: 20  # Sequence length for LSTM
```

#### LSTM Data Handler Configuration
```yaml
data_handler_config: &data_handler_config
    start_time: 2008-01-01
    end_time: 2020-08-01
    fit_start_time: 2008-01-01
    fit_end_time: 2014-12-31
    instruments: *market
    infer_processors:
        - class: FilterCol
          kwargs:
              fields_group: feature
              col_list: ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10",
                            "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
                            "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"
                        ]
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
        - class: CSRankNorm
          kwargs:
              fields_group: label
    label: ["Ref($close, -2) / Ref($close, -1) - 1"]
```

## Workflow Management

### Qrun Command Interface
```bash
# Run workflow from config file
qrun configuration.yaml

# Run with debug mode
python -m pdb qlib/cli/run.py configuration.yaml
```

### Programmatic Workflow (`qlib/examples/workflow_by_code.py`)
```python
import qlib
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

# Initialize Qlib
qlib.init(provider_uri=provider_uri, region=REG_CN)

# Initialize components
model = init_instance_by_config(task_config["model"])
dataset = init_instance_by_config(task_config["dataset"])

# Start experiment
with R.start(experiment_name="workflow"):
    R.log_params(**flatten_dict(task_config))
    model.fit(dataset)

    # Generate signals
    recorder = R.get_recorder()
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # Portfolio analysis
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()
```

### Record Templates
- **SignalRecord**: Records model predictions and signals
- **SigAnaRecord**: Signal analysis (IC, Rank IC, etc.)
- **PortAnaRecord**: Portfolio analysis and backtesting

## Data Processing Pipeline

### Data Handler Processors
```python
# Available processors in qlib.data.dataset.processor
- DropnaProcessor: Drop N/A features
- DropnaLabel: Drop N/A labels
- TanhProcess: Tanh transformation for noise
- Fillna: Fill N/A values
- MinMaxNorm: Min-max normalization
- ZscoreNorm: Z-score normalization
- RobustZScoreNorm: Robust z-score normalization
- CSZScoreNorm: Cross-sectional z-score
- CSRankNorm: Cross-sectional rank normalization
```

### Dataset Classes
- **DatasetH**: Standard dataset with data handler
- **TSDatasetH**: Time series dataset for sequential models
- **Key Methods**:
  - `prepare(segments, col_set, data_key)`: Prepare data for training/prediction
  - `get_cols()`: Get column information

## Model Training Infrastructure

### Experiment Management
```python
from qlib.workflow import R

# Start experiment
with R.start(experiment_name="experiment_name"):
    # Training code here
    recorder = R.get_recorder()
    # Record results
```

### Configuration Utilities
```python
from qlib.utils import init_instance_by_config, flatten_dict

# Initialize any qlib component from config
model = init_instance_by_config(config_dict)
dataset = init_instance_by_config(dataset_config)

# Flatten nested config for logging
flat_params = flatten_dict(config_dict)
R.log_params(**flat_params)
```

## VN30-Specific Implementation Notes

### Data Source Integration
- Use processed VN30 data from Task 04
- Ensure TFT-compatible features are available
- Map VN30 symbols to qlib instrument format

### Model Storage Structure
Based on user requirements:
```
models/symbols/{symbol}/tft/
models/symbols/{symbol}/lightgbm/
models/symbols/{symbol}/lstm/
```

### Training Loop Pattern
```python
# For each VN30 symbol
for symbol in vn30_symbols:
    # Load symbol-specific data
    dataset = create_symbol_dataset(symbol, processed_data)

    # Train each baseline model
    for model_type in ['tft', 'lightgbm', 'lstm']:
        model = create_model(model_type, symbol_config)
        model.fit(dataset)

        # Save model
        model_path = f"models/symbols/{symbol}/{model_type}/"
        model.save(model_path)

        # Record performance metrics
        predictions = model.predict(dataset)
        metrics = calculate_metrics(predictions, true_labels)
        save_metrics(metrics, symbol, model_type)
```

## Key API Calls and Methods

### Model Training APIs
- `model.fit(dataset)` - Train model on dataset
- `model.predict(dataset)` - Generate predictions
- `model.save(path)` - Save model to disk
- `model.load(path)` - Load model from disk

### Dataset APIs
- `dataset.prepare(segments, col_set, data_key)` - Prepare data
- `dataset.get_cols()` - Get column information
- `DataHandlerLP.fetch(col_set)` - Fetch processed data

### Workflow APIs
- `R.start(experiment_name)` - Start experiment
- `R.get_recorder()` - Get current recorder
- `R.log_params(**params)` - Log parameters
- `R.save_objects(**objects)` - Save objects

### Data Processing APIs
- `qlib.init(provider_uri, region)` - Initialize qlib
- `init_instance_by_config(config)` - Create instance from config
- `flatten_dict(nested_dict)` - Flatten nested dictionaries

## Configuration Patterns

### Common Data Handler Config
```yaml
data_handler_config:
    start_time: 2008-01-01
    end_time: 2020-08-01
    fit_start_time: 2008-01-01
    fit_end_time: 2014-12-31
    instruments: vn30_symbols  # VN30 specific
    freq: day
```

### Common Dataset Config
```yaml
dataset:
    class: DatasetH
    module_path: qlib.data.dataset
    kwargs:
        handler: *data_handler_config
        segments:
            train: [2008-01-01, 2014-12-31]
            valid: [2015-01-01, 2016-12-31]
            test: [2017-01-01, 2020-08-01]
```

### Record Config
```yaml
record:
    - class: SignalRecord
      module_path: qlib.workflow.record_temp
      kwargs:
        model: <MODEL>
        dataset: <DATASET>
    - class: SigAnaRecord
      module_path: qlib.workflow.record_temp
      kwargs:
        ana_long_short: False
        ann_scaler: 252
```

## Implementation Checklist

### Infrastructure Setup
- [ ] Create model training module structure
- [ ] Implement symbol-specific data loading
- [ ] Set up experiment management
- [ ] Create evaluation metrics utilities

### Model Configurations
- [ ] TFT model configuration with VN30 parameters
- [ ] LightGBM model configuration optimized for VN30
- [ ] LSTM model configuration with proper sequence handling

### Training Scripts
- [ ] Main training experiment script (`experiments/02_baseline_model_training.py`)
- [ ] Symbol iteration loop
- [ ] Model training and evaluation per symbol
- [ ] Performance metrics collection and reporting

### Model Storage
- [ ] Implement model serialization utilities
- [ ] Create directory structure: `models/symbols/{symbol}/{model_type}/`
- [ ] Model versioning and metadata storage

### Validation
- [ ] Model loading and prediction validation
- [ ] Performance benchmark generation
- [ ] Cross-model comparison reports

## References
- `qlib/docs/component/model.rst` - Model training documentation
- `qlib/docs/component/data.rst` - Data processing documentation
- `qlib/docs/component/workflow.rst` - Workflow management
- `qlib/examples/benchmarks/` - Complete working examples
- `qlib/examples/workflow_by_code.py` - Programmatic workflow example
- `qlib/examples/run_all_model.py` - Multi-model training utilities

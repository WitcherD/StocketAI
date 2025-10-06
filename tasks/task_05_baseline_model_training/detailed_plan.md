# Task 05: Baseline Model Training - Detailed Implementation Plan

## Overview
This document provides a detailed implementation plan for Task 05: Baseline Model Training for VN30, based on the requirements specified in `task.md`.

## Task Requirements Reference
- **Location**: `tasks/task_05_baseline_model_training/task.md`
- **Goal**: Implement and train baseline models (TFT, LightGBM, LSTM) for each VN30 symbol
- **Dependencies**: Task 04 (Feature Engineering) - requires processed VN30 data with engineered features
- **Success Criteria**: All models trained on VN30 data with established performance benchmarks

## Implementation Phases

### Phase 1: Infrastructure Setup
- [ ] Create model training module structure in `src/model_training/`
- [ ] Set up base configuration classes for common model parameters
- [ ] Implement model factory pattern for different baseline models
- [ ] Create evaluation utilities for consistent metrics across models
- [ ] Set up logging and error handling infrastructure

### Phase 2: Model Configurations
- [ ] **TFT Configuration** (`src/model_training/tft_config.py`)
  - Implement TFT model configuration class
  - Configure VN30-specific parameters (input_dim, hidden_dim, attention_dim, etc.)
  - Set up multi-horizon forecasting capabilities
  - Configure quantile predictions for uncertainty estimation
- [ ] **LightGBM Configuration** (`src/model_training/lightgbm_config.py`)
  - Implement LightGBM baseline using qlib.contrib.model.gbdt.LGBModel
  - Configure hyperparameters for VN30 stock prediction
  - Set up feature importance tracking
- [ ] **LSTM Configuration** (`src/model_training/lstm_config.py`)
  - Implement LSTM baseline using PyTorch/TensorFlow
  - Configure sequence length and hidden dimensions
  - Set up bidirectional LSTM option

### Phase 3: Training Infrastructure
- [ ] Create data loading utilities for VN30 symbols
- [ ] Implement symbol-specific training loops
- [ ] Set up cross-validation framework
- [ ] Create hyperparameter optimization pipeline
- [ ] Implement early stopping and model checkpointing

### Phase 4: Training Experiments
- [ ] **Main Experiment Script** (`experiments/02_baseline_model_training.py`)
  - Load VN30 constituent list
  - Iterate through each symbol
  - Train all three baseline models per symbol
  - Collect and aggregate performance metrics
  - Handle training failures gracefully

### Phase 5: Model Storage and Versioning
- [ ] Create directory structure: `models/symbols/{symbol}/{model_type}/`
- [ ] Implement model serialization utilities
- [ ] Create model metadata storage (training config, performance metrics, timestamps)
- [ ] Set up model versioning system
- [ ] Implement model loading and validation utilities

### Phase 6: Evaluation and Benchmarking
- [ ] Implement comprehensive evaluation metrics:
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - IC (Information Coefficient)
  - Rank IC (Rank Information Coefficient)
- [ ] Create performance comparison utilities
- [ ] Generate benchmark reports for Task 06 comparison
- [ ] Implement model interpretability analysis

### Phase 7: Production Readiness
- [ ] Add comprehensive error handling and logging
- [ ] Implement training progress tracking
- [ ] Create model validation tests
- [ ] Document model configurations and hyperparameters
- [ ] Set up automated training pipelines

## Technical Specifications

### Model Configurations
Based on qlib documentation and TFT requirements:

**TFT Model**:
```python
{
    "input_dim": 158,  # Alpha158 features
    "hidden_dim": 64,
    "attention_dim": 32,
    "num_heads": 4,
    "dropout": 0.1,
    "lr": 0.001,
    "batch_size": 1024,
    "epochs": 50,
    "num_quantiles": 3,
    "valid_quantiles": [0.1, 0.5, 0.9]
}
```

**LightGBM Model**:
```python
{
    "loss": "mse",
    "colsample_bytree": 0.8879,
    "learning_rate": 0.0421,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 20
}
```

**LSTM Model**:
```python
{
    "input_dim": 158,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 0.001,
    "batch_size": 2048,
    "epochs": 100
}
```

### Data Pipeline Integration
- Utilize processed VN30 data from Task 04
- Load TFT-compatible features from `src/feature_engineering/`
- Ensure Qlib format compatibility
- Handle missing data and outliers

### Performance Benchmarks
- Establish baseline metrics for each model type
- Compare performance across VN30 symbols
- Generate comprehensive benchmark reports
- Prepare data for Task 06 enhancement comparison

## Risk Mitigation
- **Training Failures**: Implement robust error handling and recovery
- **Memory Issues**: Optimize batch sizes and data loading
- **Convergence Problems**: Add early stopping and learning rate scheduling
- **Data Quality**: Validate input data integrity before training

## Success Validation
- [ ] All VN30 symbols have trained models for all three baseline types
- [ ] Performance metrics established and documented
- [ ] Model artifacts properly saved and loadable
- [ ] Benchmark reports generated for Task 06
- [ ] Training logs and error handling implemented

## Next Steps Dependencies
This implementation prepares the foundation for:
- **Task 06**: DDG-DA Integration - uses trained baselines for enhancement
- Model comparison and selection
- Advanced feature engineering validation
- Production deployment preparation

## References
- Task Requirements: `tasks/task_05_baseline_model_training/task.md`
- Qlib TFT Documentation: `docs/qlib_api_documentation_with_tft_features.md`
- Feature Engineering: `src/feature_engineering/__init__.py`
- VN30 Data Processing: `docs/vn30_data_processing_pipeline.md`

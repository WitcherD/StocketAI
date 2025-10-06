# Task 05: Baseline Model Training for VN30

## Task Overview
Implement and train baseline models (TFT, LightGBM, LSTM) for each of VN30 symbol stock prediction. Establish performance benchmarks for comparison with enhanced models.

## Task Status
- **Status**: Not Started
- **Priority**: High
- **Estimated Effort**: 3-4 days
- **Dependencies**: Task 04 (Feature Engineering)

## Task Goals
- Set up TFT model configuration for VN30 data
- Implement LightGBM and LSTM baseline models
- Train models on processed VN30 data with engineered features
- Establish baseline performance metrics and benchmarks
- Create model training infrastructure for future enhancements

## Key Deliverables
- `src/model_training/tft_config.py` - TFT model configuration
- `src/model_training/lightgbm_config.py` - LightGBM baseline implementation
- `src/model_training/lstm_config.py` - LSTM baseline implementation
- `experiments/02_baseline_model_training.py` - Training experiments
- `models/baseline_v1.0/` - Trained baseline model artifacts

## Success Criteria
- [ ] TFT model successfully configured and trained on each of VN30 symbol data
- [ ] LightGBM baseline model implemented and optimized
- [ ] LSTM baseline model implemented and trained
- [ ] Baseline performance metrics established and documented
- [ ] Models ready for Task 06 (DDG-DA Integration)

## Next Task Dependencies
**Task 06: DDG-DA Integration** will depend on this task's output:
- Trained baseline models for enhancement
- Performance benchmarks for comparison
- Model training infrastructure and evaluation framework

## Technical Approach
- Use qlib's TFT implementation as primary baseline
- Implement LightGBM using qlib's existing framework
- Create LSTM model using PyTorch/TensorFlow
- Establish consistent evaluation metrics across all baselines
- Set up model versioning and artifact management

## Immediate Next Steps
1. **Set up TFT model configuration**
2. **Implement LightGBM baseline**
3. **Create LSTM baseline model**
4. **Train and evaluate all baseline models**
5. **Establish performance benchmarks**

## Risk Assessment
- **Medium Risk**: Model training optimization and convergence
- **Medium Risk**: VN30-specific hyperparameter tuning
- **Success Probability**: High (established model architectures)

## Resources Required
- **Computational**: GPU recommended for LSTM training
- **Time**: 3-4 days for complete baseline implementation
- **Dependencies**: qlib, PyTorch/TensorFlow, processed VN30 data and features from Task 04

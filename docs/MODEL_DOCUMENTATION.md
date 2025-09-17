# Model Documentation

## 🎯 Model Overview

The Pricing Random Forest model is designed to predict prices based on multiple features including numerical and categorical inputs.

## 📊 Model Architecture

### Algorithm
- **Type**: Random Forest Regressor
- **Estimators**: 466 trees (optimized via Optuna)
- **Max Depth**: 20
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Max Features**: sqrt (square root of total features)

### Feature Engineering Pipeline

The model uses a sophisticated feature engineering pipeline that transforms 4 input features into 39 engineered features:

#### Input Features
1. **feature1**: Numerical feature (float)
2. **feature2**: Numerical feature (float)
3. **feature3**: Numerical feature (float)
4. **category_feature**: Categorical feature (category_a, category_b, category_c)

#### Engineered Features (39 total)
- **Time-based features**: hour, day_of_week, month, quarter
- **Lag features**: price_lag_1, price_lag_2, price_lag_3
- **Rolling window features**: price_rolling_mean_3, price_rolling_std_3, price_rolling_mean_7, price_rolling_std_7
- **Price change features**: price_change_1, price_change_2, price_change_3
- **Categorical encoding**: One-hot encoded category features
- **Interaction features**: Various combinations of numerical features

## 📈 Performance Metrics

### Test Set Performance
- **WAPE (Weighted Absolute Percentage Error)**: 19.77%
- **MAE (Mean Absolute Error)**: 29.52
- **RMSE (Root Mean Square Error)**: 46.28
- **R² Score**: 0.85

### Validation Strategy
- **Method**: Time Series Split (5 folds)
- **Train/Validation/Test**: 60%/20%/20%
- **Time-based splitting**: Ensures no data leakage

## 🔧 Hyperparameter Optimization

### Optuna Configuration
- **Study Name**: pricing-rf-optimization
- **Trials**: 100
- **Direction**: Minimize WAPE
- **Pruning**: MedianPruner

### Optimized Parameters
```python
{
    'n_estimators': 466,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42
}
```

## 📋 Data Requirements

### Input Data Format
```json
{
  "features": {
    "feature1": 1.5,           // Numerical (float)
    "feature2": 2.3,           // Numerical (float)
    "feature3": 0.8,           // Numerical (float)
    "category_feature": "category_a"  // Categorical (string)
  }
}
```

### Data Validation
- **feature1, feature2, feature3**: Must be numeric (int/float)
- **category_feature**: Must be one of ['category_a', 'category_b', 'category_c']
- **Missing values**: Not allowed (will raise validation error)

## 🚀 Model Deployment

### MLflow Integration
- **Model Registry**: pricing-rf
- **Version**: latest
- **Stage**: Production
- **Artifacts**: Model, preprocessor, feature names

### API Endpoints
- **Single Prediction**: POST /predict
- **Batch Prediction**: POST /predict_batch
- **Health Check**: GET /health
- **Model Info**: GET /model_info

## 🔍 Feature Importance

The model provides feature importance scores for interpretability:

1. **price_rolling_mean_7**: 0.15
2. **feature1**: 0.12
3. **price_lag_1**: 0.11
4. **feature2**: 0.10
5. **price_rolling_std_7**: 0.09
6. **category_feature**: 0.08

## 📊 Model Monitoring

### Key Metrics to Monitor
- **Prediction Latency**: < 100ms
- **Error Rate**: < 5%
- **Data Drift**: Monitor feature distributions
- **Model Performance**: Track WAPE over time

### Alerts
- **High Error Rate**: > 10%
- **High Latency**: > 500ms
- **Data Drift**: Significant distribution changes
- **Model Degradation**: WAPE increase > 5%

## 🔄 Model Retraining

### Retraining Triggers
- **Performance Degradation**: WAPE increase > 5%
- **Data Drift**: Significant feature distribution changes
- **Scheduled**: Monthly retraining
- **New Data**: > 1000 new samples

### Retraining Process
1. **Data Collection**: Gather new training data
2. **Feature Engineering**: Apply same pipeline
3. **Hyperparameter Optimization**: Run Optuna study
4. **Model Validation**: Cross-validation with time series split
5. **A/B Testing**: Compare with current model
6. **Deployment**: Update production model if better

## 📚 References

- **Scikit-learn**: Random Forest implementation
- **Optuna**: Hyperparameter optimization
- **MLflow**: Model lifecycle management
- **FastAPI**: API framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

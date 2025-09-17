# Pricing Random Forest API Usage Guide

## 🚀 Quick Start

The API is running at: **http://localhost:8000**

Interactive documentation: **http://localhost:8000/docs**

## 📋 Required Features

The API expects exactly **4 features** in the correct format:

```json
{
  "features": {
    "feature1": 1.5,           // Numerical value
    "feature2": 2.3,           // Numerical value  
    "feature3": 0.8,           // Numerical value
    "category_feature": "category_a"  // String: "category_a", "category_b", or "category_c"
  }
}
```

## ✅ Correct Examples

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "feature1": 1.5,
         "feature2": 2.3,
         "feature3": 0.8,
         "category_feature": "category_a"
       }
     }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict_batch" \
     -H "Content-Type: application/json" \
     -d '[
       {
         "features": {
           "feature1": 1.5,
           "feature2": 2.3,
           "feature3": 0.8,
           "category_feature": "category_a"
         }
       },
       {
         "features": {
           "feature1": 2.0,
           "feature2": 3.0,
           "feature3": 1.0,
           "category_feature": "category_b"
         }
       }
     ]'
```

## ❌ Common Mistakes

### Wrong: Missing category_feature
```json
{
  "features": {
    "feature1": 1.5,
    "feature2": 2.3,
    "feature3": 0.8
    // Missing "category_feature"
  }
}
```
**Error**: `Missing required features: ['category_feature']`

### Wrong: Using category value in wrong field
```json
{
  "features": {
    "feature1": 1.5,
    "feature2": "category_a",  // Wrong! feature2 should be numeric
    "feature3": 0.8,
    "category_feature": "category_a"
  }
}
```
**Error**: `Feature 'feature2' must be numeric, got: category_a`

### Wrong: Invalid category value
```json
{
  "features": {
    "feature1": 1.5,
    "feature2": 2.3,
    "feature3": 0.8,
    "category_feature": "invalid_category"  // Wrong!
  }
}
```
**Error**: `Invalid category_feature: invalid_category. Must be one of: ['category_a', 'category_b', 'category_c']`

## 🎯 Valid Values

- **feature1**: Any numeric value (float/int)
- **feature2**: Any numeric value (float/int)  
- **feature3**: Any numeric value (float/int)
- **category_feature**: Must be exactly one of:
  - `"category_a"`
  - `"category_b"`
  - `"category_c"`

## 📊 Expected Response

```json
{
  "prediction": 115.74,
  "model_version": "latest",
  "confidence": null
}
```

## 🔧 API Endpoints

### Core Prediction Endpoints
- `GET /health` - Health check
- `GET /model_info` - Model information
- `GET /features` - Feature requirements
- `POST /predict` - Single prediction
- `POST /predict_batch` - Multiple predictions

### Monitoring Endpoints
- `GET /monitoring/status` - Monitoring status and configuration
- `GET /monitoring/drift/history` - Drift detection history
- `POST /monitoring/drift/check` - Manual drift detection
- `POST /monitoring/drift/check_batch` - Batch drift detection
- `POST /monitoring/config/update` - Update monitoring configuration
- `POST /monitoring/enable` - Enable monitoring
- `POST /monitoring/disable` - Disable monitoring
- `POST /monitoring/reset` - Reset monitoring state

### Documentation
- `GET /docs` - Interactive API documentation

## 🚨 Troubleshooting

1. **500 Internal Server Error**: Check if the service is running
2. **400 Bad Request**: Check the error message for specific validation issues
3. **Missing features**: Ensure all 4 required features are provided
4. **Invalid types**: Ensure numerical features are numbers, not strings

## 📊 Monitoring and Drift Detection

The API includes automatic data drift monitoring that runs on every prediction request. The system uses Alibi Detect to compare incoming data against the training data distribution.

### Monitoring Features
- **Real-time Drift Detection**: Automatic monitoring on `/predict` and `/predict_batch` endpoints
- **Multiple Detection Methods**: Kolmogorov-Smirnov (KS) and Maximum Mean Discrepancy (MMD) tests
- **Feature-level Monitoring**: Individual feature drift detection
- **Configurable Thresholds**: Adjustable sensitivity for drift detection

### Example: Check Monitoring Status
```bash
curl -X GET "http://localhost:8000/monitoring/status"
```

### Example: Manual Drift Check
```bash
curl -X POST "http://localhost:8000/monitoring/drift/check" \
     -H "Content-Type: application/json" \
     -d '{
       "feature1": 1.5,
       "feature2": 2.3,
       "feature3": 0.8,
       "category_feature": "category_a"
     }'
```

For detailed monitoring configuration and usage, see the [Monitoring Guide](MONITORING_GUIDE.md).

## 📈 Model Performance

- **Test WAPE**: 19.77%
- **Test MAE**: 29.52
- **Test RMSE**: 46.28
- **Model**: Random Forest with 466 estimators
- **Features**: 6 processed features from 4 input features

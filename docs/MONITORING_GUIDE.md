# Monitoring and Drift Detection Guide

## Overview

This guide explains the monitoring and data drift detection system implemented in the pricing algorithm project. The system uses Alibi Detect to monitor incoming data and detect when it drifts from the training data distribution.

## Features

### Data Drift Detection
- **Multiple Detection Methods**: Kolmogorov-Smirnov (KS), Maximum Mean Discrepancy (MMD), and Wasserstein distance
- **Feature-level Monitoring**: Individual feature drift detection
- **Real-time Monitoring**: Automatic drift detection on prediction endpoints
- **Configurable Thresholds**: Customizable drift detection sensitivity

### Monitoring Capabilities
- **Batch and Point Predictions**: Monitor both single and batch prediction requests
- **Drift History**: Track drift detection results over time
- **Alerting**: Configurable alerts when drift is detected
- **Performance Metrics**: Monitor drift detection performance

## Architecture

### Components

1. **DriftDetector**: Core drift detection using Alibi Detect
2. **MonitoringService**: High-level service for integrating with FastAPI
3. **MonitoringEndpoints**: REST API endpoints for monitoring management
4. **MonitoringConfig**: Configuration management for monitoring settings

### Data Flow

```
Incoming Prediction Request
    ↓
MonitoringService.monitor_prediction()
    ↓
DriftDetector.detect_drift()
    ↓
Alibi Detect Methods (KS, MMD, Wasserstein)
    ↓
DriftResult (drift detected/not detected)
    ↓
Logging and Alerting
```

## Configuration

### MonitoringConfig Parameters

```python
class MonitoringConfig:
    drift_threshold: float = 0.05          # P-value threshold for drift detection
    drift_methods: List[str] = ["ks", "mmd", "wasserstein"]  # Detection methods
    feature_drift_enabled: bool = True     # Enable per-feature drift detection
    monitoring_frequency: int = 100        # Check every N predictions
    alert_on_drift: bool = True           # Enable drift alerts
    log_drift_results: bool = True        # Log drift detection results
    save_drift_results: bool = True       # Save results to file
```

### Feature Configuration

```python
# Numerical features for drift detection
numerical_features: List[str] = ["feature1", "feature2", "feature3"]

# Categorical features (one-hot encoded)
categorical_features: List[str] = ["category_feature"]
```

## API Endpoints

### Monitoring Status
```http
GET /monitoring/status
```
Returns current monitoring status, configuration, and drift summary.

### Drift History
```http
GET /monitoring/drift/history?limit=100
```
Returns drift detection history with optional limit.

### Manual Drift Check
```http
POST /monitoring/drift/check
Content-Type: application/json

{
    "feature1": 1.5,
    "feature2": 2.3,
    "category_feature": "A"
}
```

### Batch Drift Check
```http
POST /monitoring/drift/check_batch
Content-Type: application/json

[
    {"feature1": 1.5, "feature2": 2.3, "category_feature": "A"},
    {"feature1": 2.0, "feature2": 1.8, "category_feature": "B"}
]
```

### Configuration Management
```http
POST /monitoring/config/update
Content-Type: application/json

{
    "drift_threshold": 0.01,
    "monitoring_frequency": 50,
    "alert_on_drift": true
}
```

### Monitoring Control
```http
POST /monitoring/enable    # Enable monitoring
POST /monitoring/disable   # Disable monitoring
POST /monitoring/reset     # Reset monitoring state
```

## Usage Examples

### Basic Setup

```python
from monitoring.monitoring_config import MonitoringConfig
from monitoring.monitoring_service import MonitoringService

# Create configuration
config = MonitoringConfig(
    reference_data_path="data/processed.parquet",
    drift_threshold=0.05,
    monitoring_frequency=10,
    alert_on_drift=True
)

# Initialize monitoring service
service = MonitoringService(config)
await service.initialize()
```

### Manual Drift Detection

```python
# Check single prediction
features = {
    "feature1": 1.5,
    "feature2": 2.3,
    "category_feature": "A"
}

result = await service.monitor_prediction(features)
print(f"Drift detected: {result['drift_detected']}")
```

### Batch Monitoring

```python
# Check batch of predictions
features_list = [
    {"feature1": 1.0, "feature2": 2.0, "category_feature": "A"},
    {"feature1": 1.5, "feature2": 2.5, "category_feature": "B"}
]

result = await service.monitor_batch_predictions(features_list)
print(f"Batch drift detected: {result['drift_detected']}")
```

## Drift Detection Methods

### Kolmogorov-Smirnov (KS) Test
- **Use Case**: Detects differences in cumulative distribution functions
- **Strengths**: Good for detecting shifts in distribution shape
- **Parameters**: `alternative="two-sided"` (default)

### Maximum Mean Discrepancy (MMD)
- **Use Case**: Detects differences in feature space using kernel methods
- **Strengths**: Good for high-dimensional data and complex distributions
- **Parameters**: `kernel="rbf"`, `sigma=1.0` (default)

### Wasserstein Distance
- **Use Case**: Measures the minimum cost to transform one distribution to another
- **Strengths**: Robust to outliers, good for detecting distribution shifts
- **Parameters**: None (default)

## Monitoring Best Practices

### 1. Reference Data Selection
- Use representative training data as reference
- Ensure reference data covers the expected input distribution
- Consider temporal aspects for time-series data

### 2. Threshold Configuration
- Start with default threshold (0.05) and adjust based on false positive rate
- Lower thresholds = more sensitive to drift
- Higher thresholds = less sensitive to drift

### 3. Monitoring Frequency
- Balance between detection speed and performance impact
- More frequent checks = faster drift detection but higher overhead
- Consider business requirements for drift detection latency

### 4. Feature Selection
- Monitor all features used in model training
- Consider feature importance when prioritizing monitoring
- Include both numerical and categorical features

### 5. Alert Management
- Set up appropriate alerting channels (email, Slack, etc.)
- Avoid alert fatigue by tuning thresholds
- Implement alert escalation for critical drift

## Troubleshooting

### Common Issues

1. **"No reference data set"**
   - Ensure reference data is loaded during initialization
   - Check file path and data format

2. **"Failed to detect drift"**
   - Check data preprocessing and feature encoding
   - Verify feature names match configuration

3. **High false positive rate**
   - Increase drift threshold
   - Review reference data quality
   - Consider different detection methods

4. **Performance impact**
   - Reduce monitoring frequency
   - Use fewer detection methods
   - Optimize data preprocessing

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger("monitoring").setLevel(logging.DEBUG)
```

## Integration with FastAPI

The monitoring system is automatically integrated with the FastAPI prediction endpoints:

- `/predict` - Monitors single predictions
- `/predict_batch` - Monitors batch predictions
- `/monitoring/*` - Monitoring management endpoints

Drift detection results are logged and can trigger alerts when configured.

## Performance Considerations

### Memory Usage
- Reference data is loaded into memory
- Consider data size when selecting reference dataset
- Monitor memory usage in production

### CPU Usage
- Drift detection adds computational overhead
- Adjust monitoring frequency based on performance requirements
- Consider using fewer detection methods for better performance

### Storage
- Drift results are saved to JSON files by default
- Consider database storage for high-volume deployments
- Implement result retention policies

## Future Enhancements

### Planned Features
- **Model Performance Monitoring**: Track prediction accuracy over time
- **Concept Drift Detection**: Detect changes in input-output relationships
- **Automated Retraining**: Trigger model retraining on drift detection
- **Dashboard Integration**: Web-based monitoring dashboard
- **Advanced Alerting**: Sophisticated alert routing and escalation

### Extensibility
The monitoring system is designed to be extensible:
- Add new drift detection methods
- Implement custom monitoring logic
- Integrate with external monitoring systems
- Add custom alerting mechanisms

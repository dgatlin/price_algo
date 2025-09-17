# Development Guide

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip or conda
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd price_algo
```

2. **Install dependencies**
```bash
# Install in development mode
pip install -e ".[dev]"

# Or install specific dependencies
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "import pricing_rf; print('Installation successful!')"
```

## 🏗️ Project Structure

```
price_algo/
├── docs/                          # Documentation
│   ├── README.md
│   ├── API_USAGE_GUIDE.md
│   ├── MODEL_DOCUMENTATION.md
│   └── DEVELOPMENT_GUIDE.md
├── src/
│   ├── pricing_rf/               # Core package
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration
│   │   ├── data.py               # Data loading
│   │   ├── features.py           # Feature engineering
│   │   ├── metrics.py            # Custom metrics
│   │   ├── model.py              # Model building
│   │   └── utils.py              # Utilities
│   ├── monitoring/               # Monitoring and drift detection
│   │   ├── __init__.py
│   │   ├── monitoring_config.py  # Monitoring configuration
│   │   ├── drift_detector.py     # Core drift detection
│   │   ├── monitoring_service.py # High-level monitoring service
│   │   └── monitoring_endpoints.py # FastAPI monitoring endpoints
│   ├── training/                 # Training scripts
│   │   ├── train.py              # Main training
│   │   └── objective.py          # Optuna objective
│   └── service/                  # API service
│       ├── app.py                # FastAPI app
│       ├── schemas.py            # Pydantic schemas
│       ├── inference.py          # Model inference
│       └── auth.py               # Authentication
├── configs/                      # Configuration files
│   ├── data.yaml
│   ├── train.yaml
│   └── service.yaml
├── tests/                        # Test suite
│   ├── test_features.py
│   ├── test_training.py
│   └── test_service.py
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_prototypes.ipynb
│   └── 03_model_dev_rf.ipynb
├── data/                         # Data directory
│   ├── raw.csv
│   └── processed.parquet
├── mlflow/                       # MLflow configuration
│   └── docker-compose.yml
├── pyproject.toml                # Project configuration
├── Makefile                      # Common commands
└── README.md                     # Project overview
```

## 🔧 Development Workflow

### 1. Data Preparation
```bash
# Generate sample data
python generate_data.py

# Or use existing data
cp data/raw.csv data/processed.parquet
```

### 2. Model Training
```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Train model
python src/training/train.py
```

### 3. API Development
```bash
# Start API service
python run_service.py

# Test API
curl http://localhost:8000/health
```

### 4. Monitoring Development
```bash
# Test monitoring system
python -c "from src.monitoring.monitoring_service import MonitoringService; print('Monitoring system ready')"

# Run monitoring tests
pytest tests/test_monitoring.py -v

# Test drift detection
python -c "
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.monitoring_config import MonitoringConfig
import pandas as pd
import numpy as np

config = MonitoringConfig()
detector = DriftDetector(config)
data = pd.DataFrame({'feature1': np.random.randn(100), 'feature2': np.random.randn(100)})
detector.set_reference_data(data)
print('Drift detection ready')
"
```

### 5. Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_features.py

# Run monitoring tests
pytest tests/test_monitoring.py

# Run with coverage
pytest --cov=pricing_rf
```

## 🧪 Testing

### Test Structure
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **API Tests**: Service endpoint testing
- **Monitoring Tests**: Drift detection and monitoring system testing

### Running Tests
```bash
# All tests
pytest

# Specific test file
pytest tests/test_features.py

# With verbose output
pytest -v

# With coverage
pytest --cov=pricing_rf --cov-report=html
```

### Test Data
- **Synthetic Data**: Generated for testing
- **Mock Data**: For unit tests
- **Real Data**: For integration tests

## 📊 Data Development

### Data Loading
```python
from pricing_rf.data import load_data, clean_data

# Load raw data
df = load_data("data/raw.csv")

# Clean data
df_clean = clean_data(df)
```

### Feature Engineering
```python
from pricing_rf.features import create_all_features

# Create features
df_features = create_all_features(
    df,
    time_column="timestamp",
    target_column="price",
    categorical_features=["category_feature"],
    numerical_features=["feature1", "feature2", "feature3"]
)
```

### Model Training
```python
from pricing_rf.model import build_rf_model
from pricing_rf.metrics import calculate_wape

# Build model
model = build_rf_model()

# Train model
model.fit(X_train, y_train)

# Evaluate
wape = calculate_wape(y_test, y_pred)
```

## 🔄 Continuous Integration

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks
pre-commit run --all-files
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## 🚀 Deployment

### Local Development
```bash
# Start MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Start API
python run_service.py
```

### Docker Deployment
```bash
# Build image
docker build -t pricing-rf-api .

# Run container
docker run -p 8000:8000 pricing-rf-api
```

### Production Deployment
```bash
# Use docker-compose
docker-compose up -d

# Or use Kubernetes
kubectl apply -f k8s/
```

## 🐛 Debugging

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure package is installed
   pip install -e .
   ```

2. **MLflow Connection Issues**
   ```bash
   # Check MLflow server
   curl http://localhost:5000/health
   ```

3. **Model Loading Issues**
   ```bash
   # Check model registry
   mlflow models list
   ```

### Debugging Tools
- **Logging**: Use Python logging module
- **Debugger**: Use pdb or IDE debugger
- **Profiling**: Use cProfile for performance
- **Monitoring**: Use MLflow for model tracking

## 📚 Code Style

### Python Style Guide
- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations
- **Docstrings**: Document all functions and classes
- **Naming**: Use descriptive variable names

### Example Code Style
```python
def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Weighted Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        WAPE score
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
```

## 🔧 Configuration

### Environment Variables
```bash
# MLflow configuration
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="pricing-rf"

# API configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

### Configuration Files
- **data.yaml**: Data configuration
- **train.yaml**: Training configuration
- **service.yaml**: Service configuration

## 📖 Documentation

### Writing Documentation
- **Docstrings**: Use Google style docstrings
- **README**: Keep README up to date
- **API Docs**: Use FastAPI automatic documentation
- **Code Comments**: Explain complex logic

### Documentation Tools
- **Sphinx**: For detailed documentation
- **MkDocs**: For markdown documentation
- **FastAPI**: For API documentation
- **Jupyter**: For notebook documentation

## 🤝 Contributing

### Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch
3. **Make** changes with tests
4. **Run** all tests and checks
5. **Submit** a pull request

### Code Review
- **Review** all code changes
- **Test** new functionality
- **Check** documentation updates
- **Verify** performance impact

## 📞 Support

### Getting Help
- **Issues**: Create GitHub issues
- **Discussions**: Use GitHub discussions
- **Documentation**: Check docs/ folder
- **Examples**: Look at notebooks/

### Common Questions
- **Q**: How do I add new features?
- **A**: Add to features.py and update tests

- **Q**: How do I change the model?
- **A**: Modify model.py and retrain

- **Q**: How do I deploy to production?
- **A**: Use Docker or Kubernetes deployment

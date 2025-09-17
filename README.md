# Pricing Random Forest Model

A machine learning project for price prediction using Random Forest with MLflow integration, FastAPI service, and comprehensive feature engineering.

## Project Structure

```
price_algo/
├─ notebooks/                    # Jupyter notebooks for analysis
│  ├─ 01_eda.ipynb             # Exploratory Data Analysis
│  ├─ 02_feature_prototypes.ipynb  # Feature Engineering
│  └─ 03_model_dev_rf.ipynb    # Model Development
├─ data/                        # Data storage (gitignored)
│  ├─ raw.csv                  # Raw dataset
│  └─ processed.parquet        # Processed dataset
├─ src/                        # Source code
│  ├─ pricing_rf/             # Core ML package
│  │  ├─ __init__.py
│  │  ├─ config.py            # Pydantic settings
│  │  ├─ data.py              # Data loading & cleaning
│  │  ├─ features.py          # Feature engineering
│  │  ├─ metrics.py           # Custom metrics
│  │  ├─ model.py             # Random Forest models
│  │  └─ utils.py             # Utility functions
│  ├─ training/               # Training pipeline
│  │  ├─ train.py             # Main training script
│  │  └─ objective.py         # Optuna objective
│  ├─ monitoring/             # Monitoring and drift detection
│  │  ├─ __init__.py
│  │  ├─ monitoring_config.py # Monitoring configuration
│  │  ├─ drift_detector.py    # Core drift detection
│  │  ├─ monitoring_service.py # High-level monitoring service
│  │  └─ monitoring_endpoints.py # FastAPI monitoring endpoints
│  └─ service/                # FastAPI service
│     ├─ app.py               # FastAPI application
│     ├─ schemas.py           # Pydantic schemas
│     ├─ inference.py         # Model inference
│     └─ auth.py              # Authentication
├─ configs/                   # Configuration files
│  ├─ data.yaml              # Data configuration
│  ├─ train.yaml             # Training configuration
│  └─ service.yaml           # Service configuration
├─ mlflow/                    # MLflow setup
│  └─ docker-compose.yml     # MLflow + Postgres + MinIO
├─ tests/                     # Test suite
│  ├─ test_features.py
│  ├─ test_training.py
│  └─ test_service.py
├─ docs/                      # Documentation
│  ├─ README.md              # Documentation overview
│  ├─ API_USAGE_GUIDE.md     # API usage guide
│  ├─ MODEL_DOCUMENTATION.md # Model technical details
│  ├─ DEVELOPMENT_GUIDE.md   # Development instructions
│  └─ DEPLOYMENT_GUIDE.md    # Production deployment
├─ pyproject.toml            # Dependencies & tooling
├─ Makefile                  # Development commands
└─ README.md
```

## Features

- **Random Forest Model**: Optimized with Optuna hyperparameter tuning
- **Feature Engineering**: Time-based, lag, rolling, and price change features
- **MLflow Integration**: Model versioning, tracking, and deployment
- **FastAPI Service**: RESTful API for model inference
- **Data Drift Detection**: Real-time monitoring with Alibi Detect
- **Custom Metrics**: WAPE, tail MAE, directional accuracy, and more
- **Time Series Validation**: Proper time-based data splitting
- **Authentication**: Optional JWT and IP-based authentication
- **Comprehensive Testing**: Unit tests for all modules
- **Docker Support**: Containerized MLflow with Postgres and MinIO

## Project Optimization

This project has been optimized for distribution and reduced from **~1.03 GB to 7.5 MB** (99.3% size reduction). The following temporary files have been removed:

- `.venv/` directory (virtual environment)
- `__pycache__/` directories (Python bytecode cache)
- `*.pyc` and `*.pyo` files (compiled Python files)
- `.pytest_cache/` directories (pytest cache)
- `.coverage` files (coverage data)
- `htmlcov/` directories (HTML coverage reports)
- `.mypy_cache/` directories (mypy type checker cache)

**For new users**: You'll need to create your own virtual environment and install dependencies as shown in the Quick Start section above.

## What's Included (7.5 MB)

The optimized project includes all essential components for immediate use:

- **Source Code** (256 KB): Complete ML pipeline, FastAPI service, and monitoring system
- **Trained Model** (3.9 MB): MLflow model registry with trained Random Forest model
- **MLflow Database** (220 KB): Model metadata and experiment tracking data
- **Sample Data** (1.8 MB): Raw CSV data and processed Parquet files for drift detection
- **Notebooks** (1.1 MB): Jupyter notebooks for EDA, feature engineering, and model development
- **Documentation** (84 KB): Comprehensive guides and API documentation
- **Tests** (64 KB): Complete test suite with 95%+ coverage
- **Configuration** (12 KB): YAML configs and project settings

**Ready to run**: New users can immediately start the service and make predictions without additional setup.

## Quick Start

> **📋 For detailed setup instructions, see the [Setup Guide](docs/SETUP_GUIDE.md)**

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd price_algo

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install the package and dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

**Note**: This project has been optimized for distribution. The `.venv` directory and Python cache files have been removed to reduce the project size from ~1.03 GB to 7.5 MB (99.3% reduction). New users will need to create their own virtual environment as shown above.

**Alternative Installation Methods:**

```bash
# Method 1: Using requirements.txt
pip install -r requirements.txt

# Method 2: Using conda
conda create -n pricing-rf python=3.11
conda activate pricing-rf
pip install -e ".[dev]"
```

**Note**: If you don't have Python 3.11+, install it from [python.org](https://python.org) or use conda.

### 2. Setup Data

```bash
# Create sample data
make setup-data

# Or place your data in data/raw.csv
```

### 3. Start MLflow

```bash
# Start MLflow with Docker Compose
make run-mlflow

# Or manually:
cd mlflow && docker-compose up -d
```

### 4. Train Model

```bash
# Run training
make run-train

# Or manually:
python -m src.training.train
```

### 5. Start Service

```bash
# Run FastAPI service
make run-service

# Or manually:
python -m src.service.app
```

## Usage

### Training

The training pipeline includes:
- Data loading and cleaning
- Feature engineering
- Time-based data splitting
- Optuna hyperparameter optimization
- MLflow model registration

```python
from src.pricing_rf.config import Config
from src.training.train import train_model

config = Config()
results = train_model(config)
```

### API Usage

Once the service is running, you can make predictions:

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"feature1": 1.5, "feature2": "category_a"}}'

# Batch prediction
curl -X POST "http://localhost:8000/predict_batch" \
     -H "Content-Type: application/json" \
     -d '[{"features": {"feature1": 1.5, "feature2": "category_a"}}]'
```

### Configuration

Configuration is managed through YAML files in `configs/` and environment variables:

```yaml
# configs/data.yaml
raw_data_path: "data/raw.csv"
time_column: "timestamp"
target_column: "price"
categorical_features: ["category_feature"]
numerical_features: ["feature1", "feature2", "feature3"]
```

Environment variables can override YAML settings:
```bash
export DATA_RAW_DATA_PATH="data/my_data.csv"
export TRAIN_N_TRIALS=200
export SERVICE_PORT=8080
```

## Development

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_features.py
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Clean build artifacts
make clean
```

### Jupyter Notebooks

```bash
# Start Jupyter server
make notebooks

# Or manually:
jupyter notebook notebooks/
```

## Model Metrics

The model uses several custom metrics for evaluation:

- **WAPE**: Weighted Absolute Percentage Error
- **Tail MAE**: Mean Absolute Error for high-value predictions
- **Directional Accuracy**: Percentage of correct direction predictions
- **Hit Rate**: Percentage of predictions within threshold

## API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions
- `GET /model_info` - Model information
- `GET /metrics` - Model performance metrics

### Monitoring Endpoints
- `GET /monitoring/status` - Monitoring status and configuration
- `GET /monitoring/drift/history` - Drift detection history
- `POST /monitoring/drift/check` - Manual drift detection
- `POST /monitoring/drift/check_batch` - Batch drift detection
- `POST /monitoring/config/update` - Update monitoring configuration

## Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[Setup Guide](docs/SETUP_GUIDE.md)** - Complete setup instructions for new users
- **[API Usage Guide](docs/API_USAGE_GUIDE.md)** - Complete guide for using the FastAPI service
- **[Model Documentation](docs/MODEL_DOCUMENTATION.md)** - Technical details about the Random Forest model
- **[Monitoring Guide](docs/MONITORING_GUIDE.md)** - Data drift detection and monitoring system
- **[Development Guide](docs/DEVELOPMENT_GUIDE.md)** - Development workflow and best practices
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Production deployment instructions

### Quick Links
- **API Service**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

## Docker

### MLflow Stack

The MLflow stack includes:
- MLflow Tracking Server
- PostgreSQL database
- MinIO object storage

```bash
cd mlflow && docker-compose up -d
```

### Building Service Image

```bash
# Build Docker image
make docker-build

# Run container
make docker-run
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- scikit-learn for machine learning algorithms
- MLflow for model management
- FastAPI for the web service
- Optuna for hyperparameter optimization


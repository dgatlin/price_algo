# Setup Guide for New Users

This guide helps new users set up the Pricing Random Forest project from scratch.

## 🚀 Quick Setup (Recommended)

### Prerequisites
- **Python 3.11+** installed on your system
- **Git** for cloning the repository

### Step 1: Clone and Navigate
```bash
git clone <repository-url>
cd price_algo
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies
```bash
# Method 1: Using pyproject.toml (recommended)
pip install -e ".[dev]"

# Method 2: Using requirements.txt
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import pricing_rf; print('Installation successful!')"
```

## 🔧 Alternative Setup Methods

### Using Conda
```bash
# Create conda environment
conda create -n pricing-rf python=3.11
conda activate pricing-rf

# Install dependencies
pip install -e ".[dev]"
```

### Using Docker
```bash
# Build Docker image
docker build -t pricing-rf-api .

# Run container
docker run -p 8000:8000 pricing-rf-api
```

## 📊 Running the Project

### 1. Start MLflow Server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

### 2. Train the Model
```bash
python src/training/train.py
```

### 3. Start the API Service
```bash
python run_service.py
```

### 4. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"feature1": 1.5, "feature2": 2.3, "feature3": 0.8, "category_feature": "category_a"}}'
```

## 🐛 Troubleshooting

### Common Issues

**1. Python Version Error**
```bash
# Check Python version
python --version

# If not 3.11+, install Python 3.11+
# macOS: brew install python@3.11
# Ubuntu: sudo apt install python3.11
# Windows: Download from python.org
```

**2. Virtual Environment Issues**
```bash
# If .venv doesn't work, try:
python3 -m venv .venv
# or
python3.11 -m venv .venv
```

**3. Permission Errors (macOS/Linux)**
```bash
# Add --user flag
pip install --user -e ".[dev]"
```

**4. Import Errors**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Reinstall package
pip install -e .
```

**5. MLflow Connection Issues**
```bash
# Check if MLflow is running
curl http://localhost:5000/health

# If not running, start it:
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **Python**: 3.11+
- **RAM**: 4GB
- **Storage**: 2GB free space

### Recommended Requirements
- **OS**: macOS 12+, Ubuntu 20.04+, or Windows 11
- **Python**: 3.11+
- **RAM**: 8GB
- **Storage**: 5GB free space

## 🔍 Verification Steps

### 1. Check Python Version
```bash
python --version
# Should output: Python 3.11.x
```

### 2. Check Virtual Environment
```bash
which python
# Should point to .venv/bin/python (macOS/Linux)
# or .venv\Scripts\python.exe (Windows)
```

### 3. Check Package Installation
```bash
pip list | grep pricing-rf
# Should show: pricing-rf 0.1.0
```

### 4. Test Imports
```bash
python -c "
import pricing_rf
import mlflow
import fastapi
import optuna
print('All packages imported successfully!')
"
```

## 📚 Next Steps

After successful setup:

1. **Read the Documentation**: Check `docs/` folder for detailed guides
2. **Explore the Code**: Look at `src/` folder for source code
3. **Run Tests**: Execute `pytest` to run the test suite
4. **Try the API**: Use the interactive docs at http://localhost:8000/docs

## 🆘 Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Read the documentation** in `docs/` folder
3. **Check the logs** for error messages
4. **Create an issue** on GitHub with:
   - Your operating system
   - Python version
   - Error message
   - Steps to reproduce

## 📞 Support

- **Documentation**: `docs/` folder
- **API Docs**: http://localhost:8000/docs (when running)
- **MLflow UI**: http://localhost:5000 (when running)
- **Health Check**: http://localhost:8000/health (when running)

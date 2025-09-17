# Pricing Random Forest Documentation

This directory contains comprehensive documentation for the Pricing Random Forest project.

## 📚 Documentation Structure

- **[Setup Guide](SETUP_GUIDE.md)** - Complete setup instructions for new users
- **[API Usage Guide](API_USAGE_GUIDE.md)** - Complete guide for using the FastAPI service
- **[Model Documentation](MODEL_DOCUMENTATION.md)** - Technical details about the Random Forest model
- **[Monitoring Guide](MONITORING_GUIDE.md)** - Data drift detection and monitoring system
- **[Development Guide](DEVELOPMENT_GUIDE.md)** - Development workflow and best practices
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment instructions

## 🚀 Quick Start

1. **API Service**: http://localhost:8000
2. **Interactive Docs**: http://localhost:8000/docs
3. **MLflow UI**: http://localhost:5000

## 📋 Project Overview

This project implements a Random Forest model for price prediction with:

- **MLflow Integration**: Model tracking, versioning, and registry
- **Optuna Optimization**: Hyperparameter tuning
- **FastAPI Service**: Production-ready inference API
- **Data Drift Detection**: Real-time monitoring with Alibi Detect
- **Feature Engineering**: Time-based, lag, and rolling features
- **Comprehensive Testing**: Unit tests and validation

## 🔧 Key Features

- **Time Series Validation**: Proper train/validation/test splits
- **Feature Engineering**: 39 engineered features from 4 input features
- **Model Performance**: 19.77% WAPE on test set
- **Data Drift Monitoring**: Real-time drift detection with KS and MMD tests
- **Production Ready**: Docker support, health checks, monitoring
- **Comprehensive Validation**: Input validation and error handling

## 📊 Model Performance

- **Test WAPE**: 19.77%
- **Test MAE**: 29.52
- **Test RMSE**: 46.28
- **Model**: Random Forest with 466 estimators
- **Features**: 6 processed features from 4 input features

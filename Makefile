# Makefile for pricing-rf project

.PHONY: help install install-dev test lint format clean run-train run-service run-mlflow

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=src/pricing_rf --cov-report=html --cov-report=term-missing

lint: ## Run linting
	ruff check src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	ruff check --fix src/ tests/

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

run-train: ## Run training script
	python -m src.training.train

run-service: ## Run FastAPI service
	python -m src.service.app

run-mlflow: ## Start MLflow server with Docker Compose
	cd mlflow && docker-compose up -d

stop-mlflow: ## Stop MLflow server
	cd mlflow && docker-compose down

setup-data: ## Create sample data files
	mkdir -p data
	@echo "Creating sample data files..."
	@echo "timestamp,price,feature1,feature2,feature3,category_feature" > data/raw.csv
	@echo "2024-01-01,100.0,1.5,2.3,0.8,category_a" >> data/raw.csv
	@echo "2024-01-02,105.0,1.6,2.4,0.9,category_b" >> data/raw.csv
	@echo "2024-01-03,98.0,1.4,2.2,0.7,category_a" >> data/raw.csv
	@echo "Sample data created in data/raw.csv"

notebooks: ## Start Jupyter notebook server
	jupyter notebook notebooks/

docker-build: ## Build Docker image
	docker build -t pricing-rf .

docker-run: ## Run Docker container
	docker run -p 8000:8000 pricing-rf


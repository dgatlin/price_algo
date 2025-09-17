"""Training script with Optuna optimization and MLflow integration."""

import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any

from pricing_rf.config import Config
from pricing_rf.data import load_data, clean_data, time_based_split
from pricing_rf.features import create_feature_pipeline, create_all_features
from pricing_rf.model import build_rf, evaluate_rf_model
from pricing_rf.metrics import evaluate_model
from .objective import create_objective


def setup_mlflow(config: Config) -> None:
    """Set up MLflow tracking."""
    mlflow.set_tracking_uri(config.service.mlflow_tracking_uri)
    mlflow.set_experiment("pricing-rf-experiment")


def train_model(config: Config) -> Dict[str, Any]:
    """Main training function."""
    logger = logging.getLogger(__name__)
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Load and clean data
    logger.info("Loading data...")
    df = load_data(str(config.data.raw_data_path))
    df_clean = clean_data(df, config.data.time_column, config.data.target_column)
    
    # Create features
    logger.info("Creating features...")
    df_features = create_all_features(
        df_clean,
        config.data.time_column,
        config.data.target_column,
        config.data.categorical_features,
        config.data.numerical_features
    )
    
    # Time-based split
    logger.info("Splitting data...")
    train_df, val_df, test_df = time_based_split(
        df_features,
        config.data.time_column
    )
    
    # Prepare features and target
    feature_columns = [col for col in df_features.columns 
                      if col not in [config.data.time_column, config.data.target_column]]
    
    X_train = train_df[feature_columns]
    y_train = train_df[config.data.target_column]
    X_val = val_df[feature_columns]
    y_val = val_df[config.data.target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[config.data.target_column]
    
    # Create feature pipeline
    preprocessor = create_feature_pipeline(
        config.data.categorical_features,
        config.data.numerical_features
    )
    
    # Fit preprocessor
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Create Optuna objective
    objective = create_objective(
        X_train_processed, y_train,
        X_val_processed, y_val,
        config.training.primary_metric
    )
    
    # Run optimization
    logger.info("Starting Optuna optimization...")
    study = optuna.create_study(
        direction='minimize',
        study_name='pricing-rf-optimization'
    )
    
    study.optimize(objective, n_trials=config.training.n_trials)
    
    # Get best parameters
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")
    
    # Train final model with best parameters
    logger.info("Training final model...")
    final_model = build_rf(**best_params)
    final_model.fit(X_train_processed, y_train)
    
    # Evaluate on test set
    y_pred_test = final_model.predict(X_test_processed)
    test_metrics = evaluate_model(y_test, y_pred_test)
    
    # Log to MLflow
    with mlflow.start_run(run_name="pricing-rf-final"):
        # Log parameters
        mlflow.log_params(best_params)
        mlflow.log_params({
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_test_samples': len(X_test),
            'n_features': X_train_processed.shape[1]
        })
        
        # Log metrics
        mlflow.log_metrics(test_metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            final_model,
            "model",
            registered_model_name=config.service.model_name
        )
        
        # Log preprocessor
        mlflow.sklearn.log_model(
            preprocessor,
            "preprocessor",
            registered_model_name=f"{config.service.model_name}-preprocessor"
        )
    
    # Save feature names for inference
    feature_names = preprocessor.get_feature_names_out()
    feature_info = {
        'feature_names': feature_names.tolist(),
        'categorical_features': config.data.categorical_features,
        'numerical_features': config.data.numerical_features
    }
    
    # Save feature info
    import json
    with open('feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    mlflow.log_artifact('feature_info.json')
    
    logger.info("Training completed successfully!")
    
    return {
        'best_params': best_params,
        'test_metrics': test_metrics,
        'model': final_model,
        'preprocessor': preprocessor,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    config = Config()
    logging.basicConfig(level=logging.INFO)
    
    results = train_model(config)
    print("Training completed!")
    print(f"Best test WAPE: {results['test_metrics']['wape']:.2f}%")


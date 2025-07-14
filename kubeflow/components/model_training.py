"""
Kubeflow component for model training in house price prediction pipeline.
"""

from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple
import yaml
import os


def load_pipeline_config():
    """Load pipeline configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'pipeline_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Load configuration
config = load_pipeline_config()
base_image = config['global']['base_image']


@component(
    base_image=base_image,
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "mlflow==2.7.1",
        "joblib==1.3.2",
        "pyyaml==6.0.1",
        "optuna==3.3.0"
    ]
)
def model_training_component(
    # Input datasets
    train_data: Input[Dataset],
    validation_data: Input[Dataset],
    data_schema: Input[Dataset],
    
    # Training parameters
    algorithm: str = "random_forest",
    hyperparameter_tuning: bool = True,
    cv_folds: int = 5,
    max_trials: int = 50,
    timeout_minutes: int = 120,
    
    # MLflow configuration
    mlflow_tracking_uri: str = "http://mlflow-service.mlops.svc.cluster.local:5000",
    experiment_name: str = "kubeflow-house-price-prediction",
    
    # Outputs
    trained_model: Output[Model],
    training_metrics: Output[Metrics],
    model_artifacts: Output[Dataset]
    
) -> NamedTuple('TrainingOutput', [
    ('model_score', float),
    ('training_time', float),
    ('best_params', str),
    ('model_version', str)
]):
    """
    Advanced model training component with hyperparameter optimization.
    
    Args:
        train_data: Training dataset
        validation_data: Validation dataset  
        data_schema: Data schema definition
        algorithm: ML algorithm to use
        hyperparameter_tuning: Enable hyperparameter optimization
        cv_folds: Cross-validation folds
        max_trials: Maximum optimization trials
        timeout_minutes: Training timeout in minutes
        mlflow_tracking_uri: MLflow server URI
        experiment_name: MLflow experiment name
        
    Returns:
        NamedTuple with training results
    """
    import pandas as pd
    import numpy as np
    import json
    import time
    import logging
    from datetime import datetime
    from typing import NamedTuple
    
    # ML libraries
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import joblib
    
    # MLflow
    import mlflow
    import mlflow.sklearn
    
    # Optuna for advanced hyperparameter optimization
    import optuna
    from optuna.integration import OptunaSearchCV
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    logger.info("🤖 Starting model training component...")
    logger.info(f"Algorithm: {algorithm}, Hyperparameter tuning: {hyperparameter_tuning}")
    
    def setup_mlflow():
        """Setup MLflow tracking"""
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"✅ MLflow configured: {mlflow_tracking_uri}")
        except Exception as e:
            logger.warning(f"⚠️ MLflow setup failed: {e}. Continuing without tracking.")
    
    def load_training_data():
        """Load and prepare training data"""
        logger.info("📊 Loading training data...")
        
        # Load datasets
        train_df = pd.read_csv(train_data.path)
        val_df = pd.read_csv(validation_data.path)
        
        # Load schema
        with open(data_schema.path, 'r') as f:
            schema = json.load(f)
        
        feature_names = schema['feature_names']
        target_name = schema['features']['target']
        
        # Separate features and target
        X_train = train_df[feature_names]
        y_train = train_df[target_name]
        X_val = val_df[feature_names]
        y_val = val_df[target_name]
        
        logger.info(f"📊 Data loaded:")
        logger.info(f"  - Training: {X_train.shape}")
        logger.info(f"  - Validation: {X_val.shape}")
        logger.info(f"  - Features: {len(feature_names)}")
        
        return X_train, X_val, y_train, y_val, feature_names
    
    def get_model_and_params(algorithm: str):
        """Get model class and parameter grid based on algorithm"""
        if algorithm == "random_forest":
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        elif algorithm == "gradient_boosting":
            model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return model, param_grid
    
    def calculate_metrics(model, X, y, dataset_name=""):
        """Calculate comprehensive model metrics"""
        predictions = model.predict(X)
        
        metrics = {
            'r2_score': r2_score(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100
        }
        
        logger.info(f"📊 {dataset_name} Metrics:")
        logger.info(f"  - R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"  - RMSE: {metrics['rmse']:,.2f}")
        logger.info(f"  - MAE: {metrics['mae']:,.2f}")
        logger.info(f"  - MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def hyperparameter_optimization_optuna(model, param_grid, X_train, y_train, cv_folds, max_trials):
        """Advanced hyperparameter optimization using Optuna"""
        logger.info(f"🎯 Starting Optuna hyperparameter optimization...")
        logger.info(f"  - Max trials: {max_trials}")
        logger.info(f"  - CV folds: {cv_folds}")
        
        def objective(trial):
            # Suggest hyperparameters based on algorithm
            if algorithm == "random_forest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                    'max_depth': trial.suggest_categorical('max_depth', [10, 20, None]),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
            else:  # gradient_boosting
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
            
            # Create model with suggested parameters
            model_trial = model.__class__(random_state=42, n_jobs=-1, **params)
            
            # Cross-validation
            scores = cross_val_score(model_trial, X_train, y_train, 
                                   cv=cv_folds, scoring='neg_mean_squared_error')
            
            return -scores.mean()  # Optuna minimizes, so negate MSE
        
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=max_trials, timeout=timeout_minutes * 60)
        
        logger.info(f"🎯 Optimization completed:")
        logger.info(f"  - Best score: {study.best_value:.4f}")
        logger.info(f"  - Best params: {study.best_params}")
        logger.info(f"  - Trials completed: {len(study.trials)}")
        
        # Create final model with best parameters
        best_model = model.__class__(random_state=42, n_jobs=-1, **study.best_params)
        
        return best_model, study.best_params, study.best_value
    
    def hyperparameter_optimization_grid(model, param_grid, X_train, y_train, cv_folds):
        """Traditional grid search optimization"""
        logger.info(f"🔍 Starting Grid Search optimization...")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds, 
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"🔍 Grid Search completed:")
        logger.info(f"  - Best score: {-grid_search.best_score_:.4f}")
        logger.info(f"  - Best params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_, grid_search.best_params_, -grid_search.best_score_
    
    def train_model_with_mlflow(model, X_train, y_train, X_val, y_val, best_params):
        """Train final model with MLflow tracking"""
        logger.info("🚀 Training final model with MLflow tracking...")
        
        try:
            with mlflow.start_run(run_name=f"kubeflow-training-{algorithm}"):
                # Train model
                model.fit(X_train, y_train)
                
                # Calculate metrics
                train_metrics = calculate_metrics(model, X_train, y_train, "Training")
                val_metrics = calculate_metrics(model, X_val, y_val, "Validation")
                
                # Log parameters
                mlflow.log_params(best_params)
                mlflow.log_param("algorithm", algorithm)
                mlflow.log_param("cv_folds", cv_folds)
                mlflow.log_param("training_samples", len(X_train))
                
                # Log metrics
                mlflow.log_metrics({
                    "train_r2": train_metrics['r2_score'],
                    "train_rmse": train_metrics['rmse'],
                    "train_mae": train_metrics['mae'],
                    "val_r2": val_metrics['r2_score'],
                    "val_rmse": val_metrics['rmse'],
                    "val_mae": val_metrics['mae']
                })
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Get run info
                run = mlflow.active_run()
                model_version = run.info.run_id[:8]
                
                logger.info("✅ Model logged to MLflow successfully")
                
                return model, train_metrics, val_metrics, model_version
                
        except Exception as e:
            logger.warning(f"⚠️ MLflow logging failed: {e}")
            # Continue without MLflow
            model.fit(X_train, y_train)
            train_metrics = calculate_metrics(model, X_train, y_train, "Training")
            val_metrics = calculate_metrics(model, X_val, y_val, "Validation")
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            return model, train_metrics, val_metrics, model_version
    
    def save_model_artifacts(model, best_params, train_metrics, val_metrics, feature_names):
        """Save model and related artifacts"""
        logger.info("💾 Saving model artifacts...")
        
        # Save trained model
        joblib.dump(model, trained_model.path)
        
        # Prepare comprehensive metrics
        all_metrics = {
            'algorithm': algorithm,
            'hyperparameter_tuning': hyperparameter_tuning,
            'training_timestamp': datetime.now().isoformat(),
            'training_duration_seconds': time.time() - start_time,
            'best_parameters': best_params,
            'training_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'feature_names': feature_names,
            'model_info': {
                'sklearn_version': '1.3.0',
                'model_class': model.__class__.__name__,
                'n_features': len(feature_names)
            }
        }
        
        # Calculate feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            all_metrics['feature_importance'] = importance_dict
            
            # Log top 5 most important features
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            logger.info("🔝 Top 5 most important features:")
            for i, (feature, importance) in enumerate(sorted_importance[:5], 1):
                logger.info(f"  {i}. {feature}: {importance:.4f}")
        
        # Save training metrics
        with open(training_metrics.path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save additional artifacts
        artifacts = {
            'model_metadata': {
                'algorithm': algorithm,
                'version': model_version,
                'performance': val_metrics,
                'created_at': datetime.now().isoformat()
            },
            'hyperparameters': best_params,
            'training_config': {
                'cv_folds': cv_folds,
                'hyperparameter_tuning': hyperparameter_tuning,
                'max_trials': max_trials if hyperparameter_tuning else None
            }
        }
        
        with open(model_artifacts.path, 'w') as f:
            json.dump(artifacts, f, indent=2)
        
        logger.info("✅ All artifacts saved successfully")
        
        return all_metrics
    
    # ====================================
    # Main Training Logic
    # ====================================
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Load data
        X_train, X_val, y_train, y_val, feature_names = load_training_data()
        
        # Get model and parameter grid
        model, param_grid = get_model_and_params(algorithm)
        
        # Hyperparameter optimization
        if hyperparameter_tuning:
            if max_trials > len(list(param_grid.values())[0]) * 3:  # Use Optuna for large search spaces
                best_model, best_params, best_score = hyperparameter_optimization_optuna(
                    model, param_grid, X_train, y_train, cv_folds, max_trials
                )
            else:
                best_model, best_params, best_score = hyperparameter_optimization_grid(
                    model, param_grid, X_train, y_train, cv_folds
                )
        else:
            # Use default parameters
            best_model = model
            best_params = model.get_params()
            best_score = 0.0
            logger.info("🔧 Using default hyperparameters (no tuning)")
        
        # Train final model with MLflow tracking
        final_model, train_metrics, val_metrics, model_version = train_model_with_mlflow(
            best_model, X_train, y_train, X_val, y_val, best_params
        )
        
        # Save artifacts
        all_metrics = save_model_artifacts(
            final_model, best_params, train_metrics, val_metrics, feature_names
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        
        logger.info(f"✅ Model training completed successfully!")
        logger.info(f"⏱️ Total training time: {training_time:.2f} seconds")
        logger.info(f"🎯 Validation R² Score: {val_metrics['r2_score']:.4f}")
        
        # Return results
        TrainingOutput = NamedTuple('TrainingOutput', [
            ('model_score', float),
            ('training_time', float),
            ('best_params', str),
            ('model_version', str)
        ])
        
        return TrainingOutput(
            model_score=val_metrics['r2_score'],
            training_time=training_time,
            best_params=json.dumps(best_params),
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise


# Additional component for model comparison
@component(
    base_image=base_image,
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3", 
        "scikit-learn==1.3.0",
        "mlflow==2.7.1"
    ]
)
def model_comparison_component(
    model1: Input[Model],
    model2: Input[Model],
    test_data: Input[Dataset],
    comparison_report: Output[Metrics]
) -> NamedTuple('ComparisonOutput', [('better_model', str), ('improvement', float)]):
    """
    Compare two models and select the better one.
    
    Args:
        model1: First model to compare
        model2: Second model to compare
        test_data: Test dataset for comparison
        comparison_report: Output comparison metrics
        
    Returns:
        NamedTuple with comparison results
    """
    import pandas as pd
    import numpy as np
    import joblib
    import json
    from sklearn.metrics import r2_score, mean_squared_error
    from typing import NamedTuple
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("⚖️ Starting model comparison...")
    
    try:
        # Load models
        model_a = joblib.load(model1.path)
        model_b = joblib.load(model2.path)
        
        # Load test data
        test_df = pd.read_csv(test_data.path)
        
        # Assuming last column is target
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]
        
        # Make predictions
        pred_a = model_a.predict(X_test)
        pred_b = model_b.predict(X_test)
        
        # Calculate metrics
        r2_a = r2_score(y_test, pred_a)
        r2_b = r2_score(y_test, pred_b)
        rmse_a = np.sqrt(mean_squared_error(y_test, pred_a))
        rmse_b = np.sqrt(mean_squared_error(y_test, pred_b))
        
        # Determine better model
        if r2_a > r2_b:
            better_model = "model1"
            improvement = (r2_a - r2_b) / r2_b * 100
        else:
            better_model = "model2"
            improvement = (r2_b - r2_a) / r2_a * 100
        
        # Save comparison report
        comparison_results = {
            'model1_metrics': {'r2': r2_a, 'rmse': rmse_a},
            'model2_metrics': {'r2': r2_b, 'rmse': rmse_b},
            'better_model': better_model,
            'improvement_percent': improvement,
            'comparison_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(comparison_report.path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        logger.info(f"✅ Comparison completed. Better model: {better_model}")
        logger.info(f"📊 Improvement: {improvement:.2f}%")
        
        ComparisonOutput = NamedTuple('ComparisonOutput', [('better_model', str), ('improvement', float)])
        return ComparisonOutput(better_model=better_model, improvement=improvement)
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {str(e)}")
        raise
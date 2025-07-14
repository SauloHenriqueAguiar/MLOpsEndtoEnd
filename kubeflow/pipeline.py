"""
Main Kubeflow pipeline for house price prediction MLOps workflow.
"""

import os
import yaml
from typing import Dict, Any

# Kubeflow imports
from kfp import dsl
from kfp.dsl import pipeline, PipelineTask
from kfp import compiler
from kfp.client import Client

# Import components
from components.data_preprocessing import (
    data_preprocessing_component,
    data_validation_component
)
from components.model_training import (
    model_training_component,
    model_comparison_component
)
from components.model_validation import model_validation_component
from components.model_deployment import model_deployment_component


def load_pipeline_config() -> Dict[str, Any]:
    """Load pipeline configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'pipeline_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Load configuration
config = load_pipeline_config()


@pipeline(
    name=config['pipeline']['name'],
    description=config['pipeline']['description'],
    pipeline_root='gs://mlops-artifacts/pipeline-runs'  # Change to your storage
)
def house_price_prediction_pipeline(
    # Data parameters
    n_samples: int = 1000,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    data_quality_threshold: float = 0.95,
    
    # Training parameters
    algorithm: str = "random_forest",
    hyperparameter_tuning: bool = True,
    cv_folds: int = 5,
    max_trials: int = 50,
    timeout_minutes: int = 120,
    
    # Validation parameters
    min_r2_score: float = 0.70,
    max_rmse: float = 100000,
    max_mape: float = 15.0,
    robustness_test: bool = True,
    
    # Deployment parameters
    deployment_name: str = "house-price-api",
    namespace: str = "mlops",
    replicas: int = 3,
    auto_deploy: bool = True,
    
    # MLflow configuration
    mlflow_tracking_uri: str = "http://mlflow-service.mlops.svc.cluster.local:5000",
    experiment_name: str = "kubeflow-house-price-prediction"
):
    """
    Complete MLOps pipeline for house price prediction.
    
    This pipeline includes:
    1. Data preprocessing and validation
    2. Model training with hyperparameter optimization
    3. Comprehensive model validation
    4. Automated deployment (if approved)
    
    Args:
        n_samples: Number of samples for synthetic data
        test_size: Test set proportion
        validation_size: Validation set proportion
        data_quality_threshold: Minimum data quality score
        algorithm: ML algorithm to use
        hyperparameter_tuning: Enable hyperparameter optimization
        cv_folds: Cross-validation folds
        max_trials: Maximum optimization trials
        timeout_minutes: Training timeout
        min_r2_score: Minimum R² score for approval
        max_rmse: Maximum RMSE for approval
        max_mape: Maximum MAPE for approval
        robustness_test: Enable robustness testing
        deployment_name: Kubernetes deployment name
        namespace: Kubernetes namespace
        replicas: Number of deployment replicas
        auto_deploy: Enable automatic deployment
        mlflow_tracking_uri: MLflow server URI
        experiment_name: MLflow experiment name
    """
    
    # ====================================
    # Step 1: Data Preprocessing & Validation
    # ====================================
    
    # Data preprocessing component
    preprocessing_task = data_preprocessing_component(
        n_samples=n_samples,
        test_size=test_size,
        validation_size=validation_size,
        random_state=42,
        data_quality_threshold=data_quality_threshold
    )
    
    # Configure task resources
    preprocessing_task.set_cpu_request(config['components']['data_preprocessing']['resources']['requests']['cpu'])
    preprocessing_task.set_memory_request(config['components']['data_preprocessing']['resources']['requests']['memory'])
    preprocessing_task.set_cpu_limit(config['components']['data_preprocessing']['resources']['limits']['cpu'])
    preprocessing_task.set_memory_limit(config['components']['data_preprocessing']['resources']['limits']['memory'])
    
    # Data validation component
    data_validation_task = data_validation_component(
        dataset=preprocessing_task.outputs['processed_data'],
        schema=preprocessing_task.outputs['data_schema']
    )
    
    # Set task dependencies and conditions
    data_validation_task.after(preprocessing_task)
    
    # ====================================
    # Step 2: Model Training
    # ====================================
    
    # Main model training component
    training_task = model_training_component(
        train_data=preprocessing_task.outputs['train_data'],
        validation_data=preprocessing_task.outputs['validation_data'],
        data_schema=preprocessing_task.outputs['data_schema'],
        algorithm=algorithm,
        hyperparameter_tuning=hyperparameter_tuning,
        cv_folds=cv_folds,
        max_trials=max_trials,
        timeout_minutes=timeout_minutes,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name
    )
    
    # Configure training task resources
    training_task.set_cpu_request(config['components']['model_training']['resources']['requests']['cpu'])
    training_task.set_memory_request(config['components']['model_training']['resources']['requests']['memory'])
    training_task.set_cpu_limit(config['components']['model_training']['resources']['limits']['cpu'])
    training_task.set_memory_limit(config['components']['model_training']['resources']['limits']['memory'])
    
    # Set dependencies
    training_task.after(data_validation_task)
    
    # Conditional training: only proceed if data validation passes
    with dsl.Condition(
        data_validation_task.outputs['is_valid'] == True,
        name="data-validation-passed"
    ):
        training_task_conditional = training_task
    
    # ====================================
    # Step 3: Model Validation
    # ====================================
    
    # Comprehensive model validation
    validation_task = model_validation_component(
        trained_model=training_task.outputs['trained_model'],
        test_data=preprocessing_task.outputs['test_data'],
        validation_data=preprocessing_task.outputs['validation_data'],
        data_schema=preprocessing_task.outputs['data_schema'],
        min_r2_score=min_r2_score,
        max_rmse=max_rmse,
        max_mape=max_mape,
        robustness_test=robustness_test,
        noise_levels=[0.01, 0.05, 0.1],
        confidence_level=0.95
    )
    
    # Configure validation task resources
    validation_task.set_cpu_request(config['components']['model_validation']['resources']['requests']['cpu'])
    validation_task.set_memory_request(config['components']['model_validation']['resources']['requests']['memory'])
    validation_task.set_cpu_limit(config['components']['model_validation']['resources']['limits']['cpu'])
    validation_task.set_memory_limit(config['components']['model_validation']['resources']['limits']['memory'])
    
    # Set dependencies
    validation_task.after(training_task)
    
    # ====================================
    # Step 4: Model Deployment (Conditional)
    # ====================================
    
    # Conditional deployment: only deploy if model is approved and auto_deploy is enabled
    with dsl.Condition(
        (validation_task.outputs['is_approved'] == True) & (auto_deploy == True),
        name="model-approved-for-deployment"
    ):
        
        deployment_task = model_deployment_component(
            trained_model=training_task.outputs['trained_model'],
            model_approval=validation_task.outputs['model_approval'],
            quality_certificate=validation_task.outputs['quality_certificate'],
            deployment_name=deployment_name,
            namespace=namespace,
            replicas=replicas,
            port=8000,
            image_name="house-price-api:latest",
            cpu_request="500m",
            memory_request="1Gi",
            cpu_limit="1",
            memory_limit="2Gi",
            strategy="RollingUpdate",
            max_surge=1,
            max_unavailable=0
        )
        
        # Configure deployment task resources
        deployment_task.set_cpu_request(config['components']['model_deployment']['resources']['requests']['cpu'])
        deployment_task.set_memory_request(config['components']['model_deployment']['resources']['requests']['memory'])
        deployment_task.set_cpu_limit(config['components']['model_deployment']['resources']['limits']['cpu'])
        deployment_task.set_memory_limit(config['components']['model_deployment']['resources']['limits']['memory'])
        
        # Set dependencies
        deployment_task.after(validation_task)
    
    # ====================================
    # Configure Pipeline-level Settings
    # ====================================
    
    # Set global configurations
    dsl.get_pipeline_conf().set_parallelism(config['execution']['parallelism'])
    dsl.get_pipeline_conf().set_timeout(config['execution']['timeout'])
    
    # Set retry policy
    for task in [preprocessing_task, training_task, validation_task]:
        task.set_retry(config['execution']['retry']['max_retries'])
    
    # Set node selector for ML workloads
    if config['execution'].get('node_selector'):
        for task in [preprocessing_task, training_task, validation_task]:
            task.add_node_selector_constraint(
                label_name='workload-type',
                value='ml-training'
            )
    
    # Add tolerations for dedicated ML nodes
    if config['execution'].get('tolerations'):
        for task in [preprocessing_task, training_task, validation_task]:
            task.add_toleration(
                key='ml-workload',
                operator='Equal',
                value='true',
                effect='NoSchedule'
            )


def create_pipeline_variants():
    """Create different pipeline variants for different environments"""
    
    variants = {
        'development': {
            'n_samples': 500,
            'hyperparameter_tuning': False,
            'max_trials': 10,
            'timeout_minutes': 30,
            'replicas': 1,
            'auto_deploy': True
        },
        'staging': {
            'n_samples': 1000,
            'hyperparameter_tuning': True,
            'max_trials': 25,
            'timeout_minutes': 60,
            'replicas': 2,
            'auto_deploy': False  # Manual approval required
        },
        'production': {
            'n_samples': 2000,
            'hyperparameter_tuning': True,
            'max_trials': 50,
            'timeout_minutes': 120,
            'replicas': 3,
            'auto_deploy': False,  # Manual approval required
            'min_r2_score': 0.75,  # Higher standards for production
            'max_rmse': 80000
        }
    }
    
    return variants


@pipeline(
    name="house-price-prediction-comparison-pipeline",
    description="Pipeline for comparing multiple models"
)
def model_comparison_pipeline(
    # Data parameters
    n_samples: int = 1000,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    
    # Training parameters for model A
    algorithm_a: str = "random_forest",
    algorithm_b: str = "gradient_boosting",
    
    # MLflow configuration
    mlflow_tracking_uri: str = "http://mlflow-service.mlops.svc.cluster.local:5000"
):
    """
    Pipeline for comparing two different algorithms.
    
    Args:
        n_samples: Number of samples for synthetic data
        test_size: Test set proportion
        validation_size: Validation set proportion
        algorithm_a: First algorithm to compare
        algorithm_b: Second algorithm to compare
        mlflow_tracking_uri: MLflow server URI
    """
    
    # Data preprocessing (shared)
    preprocessing_task = data_preprocessing_component(
        n_samples=n_samples,
        test_size=test_size,
        validation_size=validation_size,
        random_state=42
    )
    
    # Train model A
    training_task_a = model_training_component(
        train_data=preprocessing_task.outputs['train_data'],
        validation_data=preprocessing_task.outputs['validation_data'],
        data_schema=preprocessing_task.outputs['data_schema'],
        algorithm=algorithm_a,
        hyperparameter_tuning=True,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=f"comparison-{algorithm_a}"
    )
    
    # Train model B
    training_task_b = model_training_component(
        train_data=preprocessing_task.outputs['train_data'],
        validation_data=preprocessing_task.outputs['validation_data'],
        data_schema=preprocessing_task.outputs['data_schema'],
        algorithm=algorithm_b,
        hyperparameter_tuning=True,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=f"comparison-{algorithm_b}"
    )
    
    # Set parallel execution
    training_task_a.after(preprocessing_task)
    training_task_b.after(preprocessing_task)
    
    # Compare models
    comparison_task = model_comparison_component(
        model1=training_task_a.outputs['trained_model'],
        model2=training_task_b.outputs['trained_model'],
        test_data=preprocessing_task.outputs['test_data']
    )
    
    comparison_task.after(training_task_a, training_task_b)


def compile_pipelines():
    """Compile all pipeline variants"""
    
    print("🔨 Compiling Kubeflow pipelines...")
    
    # Compile main pipeline
    compiler.Compiler().compile(
        pipeline_func=house_price_prediction_pipeline,
        package_path='house_price_prediction_pipeline.yaml'
    )
    print("✅ Main pipeline compiled: house_price_prediction_pipeline.yaml")
    
    # Compile comparison pipeline
    compiler.Compiler().compile(
        pipeline_func=model_comparison_pipeline,
        package_path='model_comparison_pipeline.yaml'
    )
    print("✅ Comparison pipeline compiled: model_comparison_pipeline.yaml")
    
    # Create environment-specific pipelines
    variants = create_pipeline_variants()
    
    for env, params in variants.items():
        
        # Create pipeline with environment-specific parameters
        @pipeline(
            name=f"house-price-prediction-{env}",
            description=f"House price prediction pipeline for {env} environment"
        )
        def env_pipeline():
            return house_price_prediction_pipeline(**params)
        
        # Compile environment-specific pipeline
        compiler.Compiler().compile(
            pipeline_func=env_pipeline,
            package_path=f'house_price_prediction_{env}_pipeline.yaml'
        )
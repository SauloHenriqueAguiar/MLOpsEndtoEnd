"""
Kubeflow component for comprehensive model validation.
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
        "scipy==1.11.2",
        "mlflow==2.7.1",
        "joblib==1.3.2",
        "matplotlib==3.7.2",
        "seaborn==0.12.2"
    ]
)
def model_validation_component(
    # Input model and data
    trained_model: Input[Model],
    test_data: Input[Dataset],
    validation_data: Input[Dataset],
    data_schema: Input[Dataset],
    
    # Validation parameters
    min_r2_score: float = 0.70,
    max_rmse: float = 100000,
    max_mape: float = 15.0,
    robustness_test: bool = True,
    noise_levels: list = None,
    confidence_level: float = 0.95,
    
    # Outputs
    validation_report: Output[Metrics],
    model_approval: Output[Dataset],
    quality_certificate: Output[Dataset]
    
) -> NamedTuple('ValidationOutput', [
    ('is_approved', bool),
    ('validation_score', float),
    ('robustness_score', float),
    ('quality_grade', str)
]):
    """
    Comprehensive model validation including performance, robustness, and quality checks.
    
    Args:
        trained_model: Trained model to validate
        test_data: Test dataset
        validation_data: Validation dataset
        data_schema: Data schema definition
        min_r2_score: Minimum R² score required
        max_rmse: Maximum RMSE allowed
        max_mape: Maximum MAPE allowed
        robustness_test: Enable robustness testing
        noise_levels: Noise levels for robustness testing
        confidence_level: Confidence level for intervals
        
    Returns:
        NamedTuple with validation results
    """
    import pandas as pd
    import numpy as np
    import json
    import joblib
    import logging
    from datetime import datetime
    from typing import NamedTuple
    
    # ML and stats libraries
    from sklearn.metrics import (
        mean_squared_error, r2_score, mean_absolute_error,
        mean_absolute_percentage_error
    )
    from scipy import stats
    from scipy.stats import ks_2samp
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set default noise levels
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.1]
    
    logger.info("🧪 Starting comprehensive model validation...")
    logger.info(f"Validation criteria: R²≥{min_r2_score}, RMSE≤{max_rmse}, MAPE≤{max_mape}%")
    
    def load_validation_data():
        """Load model and validation datasets"""
        logger.info("📊 Loading validation data...")
        
        # Load model
        model = joblib.load(trained_model.path)
        
        # Load datasets
        test_df = pd.read_csv(test_data.path)
        val_df = pd.read_csv(validation_data.path)
        
        # Load schema
        with open(data_schema.path, 'r') as f:
            schema = json.load(f)
        
        feature_names = schema['feature_names']
        target_name = schema['features']['target']
        
        # Separate features and target
        X_test = test_df[feature_names]
        y_test = test_df[target_name]
        X_val = val_df[feature_names]
        y_val = val_df[target_name]
        
        logger.info(f"✅ Data loaded:")
        logger.info(f"  - Test samples: {len(X_test)}")
        logger.info(f"  - Validation samples: {len(X_val)}")
        logger.info(f"  - Features: {len(feature_names)}")
        
        return model, X_test, y_test, X_val, y_val, feature_names, schema
    
    def calculate_comprehensive_metrics(model, X, y, dataset_name=""):
        """Calculate comprehensive performance metrics"""
        logger.info(f"📊 Calculating metrics for {dataset_name}...")
        
        predictions = model.predict(X)
        residuals = y - predictions
        
        # Basic metrics
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        
        # Advanced metrics
        mape = np.mean(np.abs(residuals / y)) * 100
        median_ae = np.median(np.abs(residuals))
        max_error = np.max(np.abs(residuals))
        
        # Statistical metrics
        explained_variance = 1 - (np.var(residuals) / np.var(y))
        
        # Residual analysis
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        residual_skewness = stats.skew(residuals)
        residual_kurtosis = stats.kurtosis(residuals)
        
        # Normality test for residuals
        _, normality_p_value = stats.normaltest(residuals)
        
        # Confidence intervals (approximate)
        confidence_interval = 1.96 * residual_std
        
        metrics = {
            'basic_metrics': {
                'r2_score': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'mse': float(mse)
            },
            'advanced_metrics': {
                'median_absolute_error': float(median_ae),
                'max_error': float(max_error),
                'explained_variance': float(explained_variance)
            },
            'residual_analysis': {
                'mean': float(residual_mean),
                'std': float(residual_std),
                'skewness': float(residual_skewness),
                'kurtosis': float(residual_kurtosis),
                'normality_p_value': float(normality_p_value)
            },
            'confidence_metrics': {
                'confidence_interval_95': float(confidence_interval),
                'prediction_std': float(np.std(predictions))
            },
            'sample_statistics': {
                'n_samples': len(y),
                'target_mean': float(y.mean()),
                'target_std': float(y.std()),
                'prediction_mean': float(predictions.mean()),
                'prediction_std': float(predictions.std())
            }
        }
        
        logger.info(f"📊 {dataset_name} Performance:")
        logger.info(f"  - R² Score: {r2:.4f}")
        logger.info(f"  - RMSE: {rmse:,.2f}")
        logger.info(f"  - MAE: {mae:,.2f}")
        logger.info(f"  - MAPE: {mape:.2f}%")
        
        return metrics, predictions, residuals
    
    def robustness_testing(model, X_test, y_test, noise_levels):
        """Test model robustness against input noise"""
        logger.info("🛡️ Starting robustness testing...")
        
        base_predictions = model.predict(X_test)
        base_r2 = r2_score(y_test, base_predictions)
        
        robustness_results = {
            'base_performance': {'r2': float(base_r2)},
            'noise_tests': [],
            'robustness_score': 0.0
        }
        
        performance_drops = []
        
        for noise_level in noise_levels:
            logger.info(f"🔍 Testing with {noise_level*100:.1f}% noise...")
            
            # Add Gaussian noise to numerical features
            X_noisy = X_test.copy()
            
            # Identify numerical columns (assuming they don't contain 'encoded' in name)
            numeric_cols = [col for col in X_test.columns if 'encoded' not in col]
            
            for col in numeric_cols:
                if col in X_noisy.columns:
                    noise = np.random.normal(0, X_noisy[col].std() * noise_level, len(X_noisy))
                    X_noisy[col] = X_noisy[col] + noise
            
            # Make predictions with noisy data
            noisy_predictions = model.predict(X_noisy)
            noisy_r2 = r2_score(y_test, noisy_predictions)
            
            # Calculate performance drop
            performance_drop = (base_r2 - noisy_r2) / base_r2 * 100
            performance_drops.append(performance_drop)
            
            noise_result = {
                'noise_level': float(noise_level),
                'r2_score': float(noisy_r2),
                'performance_drop_percent': float(performance_drop)
            }
            robustness_results['noise_tests'].append(noise_result)
            
            logger.info(f"  - R² with noise: {noisy_r2:.4f}")
            logger.info(f"  - Performance drop: {performance_drop:.2f}%")
        
        # Calculate overall robustness score
        max_drop = max(performance_drops) if performance_drops else 0
        robustness_score = max(0, 1 - max_drop / 20)  # 20% drop = 0 score
        robustness_results['robustness_score'] = float(robustness_score)
        robustness_results['max_performance_drop'] = float(max_drop)
        
        logger.info(f"🛡️ Robustness testing completed:")
        logger.info(f"  - Max performance drop: {max_drop:.2f}%")
        logger.info(f"  - Robustness score: {robustness_score:.3f}")
        
        return robustness_results
    
    def edge_case_testing(model, schema):
        """Test model behavior on edge cases"""
        logger.info("🎯 Testing edge cases...")
        
        edge_cases = [
            {
                'name': 'minimum_values',
                'area': 30, 'quartos': 1, 'banheiros': 1, 'idade': 0,
                'garagem': 0, 'bairro_encoded': 0
            },
            {
                'name': 'maximum_values', 
                'area': 500, 'quartos': 5, 'banheiros': 4, 'idade': 50,
                'garagem': 1, 'bairro_encoded': 4
            },
            {
                'name': 'average_values',
                'area': 120, 'quartos': 3, 'banheiros': 2, 'idade': 10,
                'garagem': 1, 'bairro_encoded': 2
            }
        ]
        
        edge_test_results = []
        
        for case in edge_cases:
            try:
                # Create feature vector with engineered features
                features = {
                    'area': case['area'],
                    'quartos': case['quartos'],
                    'banheiros': case['banheiros'],
                    'idade': case['idade'],
                    'garagem': case['garagem'],
                    'bairro_encoded': case['bairro_encoded'],
                    'area_per_room': case['area'] / (case['quartos'] + 1),
                    'bathroom_ratio': case['banheiros'] / case['quartos'],
                    'age_squared': case['idade'] ** 2,
                    'total_rooms': case['quartos'] + case['banheiros'],
                    'is_new': 1 if case['idade'] < 5 else 0
                }
                
                # Create DataFrame for prediction
                X_case = pd.DataFrame([features])
                prediction = model.predict(X_case)[0]
                
                edge_result = {
                    'case_name': case['name'],
                    'inputs': case,
                    'prediction': float(prediction),
                    'is_reasonable': 50000 <= prediction <= 2000000
                }
                
                edge_test_results.append(edge_result)
                
                logger.info(f"  - {case['name']}: R$ {prediction:,.0f}")
                
            except Exception as e:
                logger.warning(f"  - {case['name']}: Error - {str(e)}")
                edge_test_results.append({
                    'case_name': case['name'],
                    'inputs': case,
                    'prediction': None,
                    'error': str(e),
                    'is_reasonable': False
                })
        
        return edge_test_results
    
    def data_drift_analysis(X_val, X_test):
        """Analyze potential data drift between validation and test sets"""
        logger.info("📊 Analyzing data drift...")
        
        drift_results = {}
        
        for column in X_val.columns:
            # Kolmogorov-Smirnov test for distribution difference
            try:
                ks_stat, p_value = ks_2samp(X_val[column], X_test[column])
                
                # Calculate mean and std differences
                val_mean, val_std = X_val[column].mean(), X_val[column].std()
                test_mean, test_std = X_test[column].mean(), X_test[column].std()
                
                mean_diff = abs(test_mean - val_mean) / val_mean if val_mean != 0 else 0
                std_diff = abs(test_std - val_std) / val_std if val_std != 0 else 0
                
                drift_results[column] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'significant_drift': p_value < 0.05,
                    'mean_difference': float(mean_diff),
                    'std_difference': float(std_diff)
                }
                
            except Exception as e:
                logger.warning(f"Drift analysis failed for {column}: {e}")
                drift_results[column] = {'error': str(e)}
        
        # Count features with significant drift
        drift_count = sum(1 for result in drift_results.values() 
                         if result.get('significant_drift', False))
        
        logger.info(f"📊 Data drift analysis:")
        logger.info(f"  - Features with significant drift: {drift_count}/{len(drift_results)}")
        
        return drift_results
    
    def performance_by_segments(model, X_test, y_test):
        """Analyze performance across different data segments"""
        logger.info("📈 Analyzing performance by segments...")
        
        predictions = model.predict(X_test)
        
        # Performance by price range
        price_quartiles = y_test.quantile([0.25, 0.5, 0.75])
        
        segments = {
            'low_price': y_test <= price_quartiles[0.25],
            'medium_price': (y_test > price_quartiles[0.25]) & (y_test <= price_quartiles[0.75]),
            'high_price': y_test > price_quartiles[0.75]
        }
        
        segment_results = {}
        
        for segment_name, mask in segments.items():
            if mask.sum() > 0:
                segment_r2 = r2_score(y_test[mask], predictions[mask])
                segment_rmse = np.sqrt(mean_squared_error(y_test[mask], predictions[mask]))
                
                segment_results[segment_name] = {
                    'r2_score': float(segment_r2),
                    'rmse': float(segment_rmse),
                    'sample_count': int(mask.sum())
                }
                
                logger.info(f"  - {segment_name}: R²={segment_r2:.3f}, RMSE={segment_rmse:,.0f}, n={mask.sum()}")
        
        return segment_results
    
    def validate_business_rules(predictions, y_test):
        """Validate business logic and constraints"""
        logger.info("💼 Validating business rules...")
        
        business_validations = {
            'prediction_range': {
                'valid_predictions': ((predictions >= 50000) & (predictions <= 2000000)).sum(),
                'total_predictions': len(predictions),
                'pass_rate': ((predictions >= 50000) & (predictions <= 2000000)).mean()
            },
            'extreme_errors': {
                'errors_over_100pct': (np.abs(predictions - y_test) / y_test > 1.0).sum(),
                'total_predictions': len(predictions),
                'extreme_error_rate': (np.abs(predictions - y_test) / y_test > 1.0).mean()
            },
            'prediction_consistency': {
                'coefficient_of_variation': np.std(predictions) / np.mean(predictions),
                'acceptable': np.std(predictions) / np.mean(predictions) < 1.0
            }
        }
        
        logger.info("💼 Business validation results:")
        logger.info(f"  - Valid predictions: {business_validations['prediction_range']['pass_rate']:.1%}")
        logger.info(f"  - Extreme errors: {business_validations['extreme_errors']['extreme_error_rate']:.1%}")
        
        return business_validations
    
    def generate_quality_grade(validation_score, robustness_score, business_score):
        """Generate overall quality grade"""
        overall_score = (validation_score * 0.5 + robustness_score * 0.3 + business_score * 0.2)
        
        if overall_score >= 0.9:
            grade = "A+"
        elif overall_score >= 0.8:
            grade = "A"
        elif overall_score >= 0.7:
            grade = "B+"
        elif overall_score >= 0.6:
            grade = "B"
        elif overall_score >= 0.5:
            grade = "C"
        else:
            grade = "F"
        
        return grade, overall_score
    
    # ====================================
    # Main Validation Logic
    # ====================================
    
    try:
        validation_start_time = datetime.now()
        
        # Load data and model
        model, X_test, y_test, X_val, y_val, feature_names, schema = load_validation_data()
        
        # Step 1: Calculate comprehensive metrics
        logger.info("📊 Step 1: Performance Evaluation")
        test_metrics, test_predictions, test_residuals = calculate_comprehensive_metrics(
            model, X_test, y_test, "Test Set"
        )
        
        val_metrics, val_predictions, val_residuals = calculate_comprehensive_metrics(
            model, X_val, y_val, "Validation Set"
        )
        
        # Step 2: Robustness testing
        logger.info("🛡️ Step 2: Robustness Testing")
        if robustness_test:
            robustness_results = robustness_testing(model, X_test, y_test, noise_levels)
            robustness_score = robustness_results['robustness_score']
        else:
            robustness_results = {'robustness_score': 1.0}
            robustness_score = 1.0
        
        # Step 3: Edge case testing
        logger.info("🎯 Step 3: Edge Case Testing")
        edge_cases = edge_case_testing(model, schema)
        
        # Step 4: Data drift analysis
        logger.info("📊 Step 4: Data Drift Analysis")
        drift_analysis = data_drift_analysis(X_val, X_test)
        
        # Step 5: Performance segmentation
        logger.info("📈 Step 5: Segment Analysis")
        segment_performance = performance_by_segments(model, X_test, y_test)
        
        # Step 6: Business validation
        logger.info("💼 Step 6: Business Validation")
        business_validation = validate_business_rules(test_predictions, y_test)
        
        # Step 7: Overall validation assessment
        logger.info("✅ Step 7: Final Assessment")
        
        # Check validation criteria
        r2_pass = test_metrics['basic_metrics']['r2_score'] >= min_r2_score
        rmse_pass = test_metrics['basic_metrics']['rmse'] <= max_rmse
        mape_pass = test_metrics['basic_metrics']['mape'] <= max_mape
        
        # Calculate validation score
        validation_score = (
            test_metrics['basic_metrics']['r2_score'] * 0.4 +
            (1 - min(test_metrics['basic_metrics']['rmse'] / max_rmse, 1.0)) * 0.3 +
            (1 - min(test_metrics['basic_metrics']['mape'] / max_mape, 1.0)) * 0.3
        )
        
        # Business score
        business_score = (
            business_validation['prediction_range']['pass_rate'] * 0.5 +
            (1 - business_validation['extreme_errors']['extreme_error_rate']) * 0.5
        )
        
        # Generate quality grade
        quality_grade, overall_score = generate_quality_grade(
            validation_score, robustness_score, business_score
        )
        
        # Final approval decision
        is_approved = all([r2_pass, rmse_pass, mape_pass]) and robustness_score >= 0.7
        
        # Compile comprehensive validation report
        validation_report_data = {
            'validation_timestamp': validation_start_time.isoformat(),
            'model_info': {
                'algorithm': model.__class__.__name__,
                'features': feature_names,
                'n_features': len(feature_names)
            },
            'validation_criteria': {
                'min_r2_score': min_r2_score,
                'max_rmse': max_rmse,
                'max_mape': max_mape,
                'robustness_threshold': 0.7
            },
            'performance_metrics': {
                'test_metrics': test_metrics,
                'validation_metrics': val_metrics
            },
            'robustness_analysis': robustness_results,
            'edge_case_testing': edge_cases,
            'data_drift_analysis': drift_analysis,
            'segment_analysis': segment_performance,
            'business_validation': business_validation,
            'validation_results': {
                'r2_pass': r2_pass,
                'rmse_pass': rmse_pass, 
                'mape_pass': mape_pass,
                'validation_score': float(validation_score),
                'robustness_score': float(robustness_score),
                'business_score': float(business_score),
                'overall_score': float(overall_score),
                'quality_grade': quality_grade,
                'is_approved': is_approved
            }
        }
        
        # Save validation report
        with open(validation_report.path, 'w') as f:
            json.dump(validation_report_data, f, indent=2)
        
        # Save model approval decision
        approval_decision = {
            'approved': is_approved,
            'approval_timestamp': datetime.now().isoformat(),
            'approval_criteria': {
                'performance_criteria_met': all([r2_pass, rmse_pass, mape_pass]),
                'robustness_criteria_met': robustness_score >= 0.7,
                'business_criteria_met': business_score >= 0.8
            },
            'model_version': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'next_steps': 'Deploy to production' if is_approved else 'Requires improvement'
        }
        
        with open(model_approval.path, 'w') as f:
            json.dump(approval_decision, f, indent=2)
        
        # Generate quality certificate if approved
        if is_approved:
            certificate = {
                'certificate_id': f"CERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'model_name': 'House Price Prediction Model',
                'certification_date': datetime.now().isoformat(),
                'certification_status': 'APPROVED',
                'quality_grade': quality_grade,
                'performance_metrics': {
                    'r2_score': test_metrics['basic_metrics']['r2_score'],
                    'rmse': test_metrics['basic_metrics']['rmse'],
                    'mape': test_metrics['basic_metrics']['mape']
                },
                'certification_criteria': [
                    f"R² Score: {test_metrics['basic_metrics']['r2_score']:.4f} ≥ {min_r2_score} ✅",
                    f"RMSE: {test_metrics['basic_metrics']['rmse']:,.0f} ≤ {max_rmse:,.0f} ✅",
                    f"MAPE: {test_metrics['basic_metrics']['mape']:.1f}% ≤ {max_mape}% ✅",
                    f"Robustness: {robustness_score:.3f} ≥ 0.7 ✅"
                ],
                'validity_period': '6 months',
                'certified_by': 'Kubeflow Validation Pipeline'
            }
        else:
            certificate = {
                'certificate_id': None,
                'certification_status': 'REJECTED',
                'rejection_reasons': [],
                'recommendations': []
            }
            
            if not r2_pass:
                certificate['rejection_reasons'].append(f"R² score {test_metrics['basic_metrics']['r2_score']:.4f} below minimum {min_r2_score}")
            if not rmse_pass:
                certificate['rejection_reasons'].append(f"RMSE {test_metrics['basic_metrics']['rmse']:,.0f} above maximum {max_rmse:,.0f}")
            if not mape_pass:
                certificate['rejection_reasons'].append(f"MAPE {test_metrics['basic_metrics']['mape']:.1f}% above maximum {max_mape}%")
            if robustness_score < 0.7:
                certificate['rejection_reasons'].append(f"Robustness score {robustness_score:.3f} below threshold 0.7")
        
        with open(quality_certificate.path, 'w') as f:
            json.dump(certificate, f, indent=2)
        
        # Log final results
        logger.info(f"✅ Model validation completed!")
        logger.info(f"📊 Results Summary:")
        logger.info(f"  - Approval: {'✅ APPROVED' if is_approved else '❌ REJECTED'}")
        logger.info(f"  - Quality Grade: {quality_grade}")
        logger.info(f"  - Overall Score: {overall_score:.3f}")
        logger.info(f"  - R² Score: {test_metrics['basic_metrics']['r2_score']:.4f}")
        logger.info(f"  - RMSE: {test_metrics['basic_metrics']['rmse']:,.0f}")
        logger.info(f"  - Robustness: {robustness_score:.3f}")
        
        # Return validation results
        ValidationOutput = NamedTuple('ValidationOutput', [
            ('is_approved', bool),
            ('validation_score', float),
            ('robustness_score', float),
            ('quality_grade', str)
        ])
        
        return ValidationOutput(
            is_approved=is_approved,
            validation_score=validation_score,
            robustness_score=robustness_score,
            quality_grade=quality_grade
        )
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        raise
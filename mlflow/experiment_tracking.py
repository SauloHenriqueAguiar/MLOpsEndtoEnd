import mlflow
from mlflow_setup import setup_mlflow

class ExperimentTracker:
    def __init__(self, experiment_name="ml_experiment"):
        setup_mlflow(experiment_name=experiment_name)
    
    def start_run(self, run_name=None):
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params):
        mlflow.log_params(params)
    
    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, artifact_path="model"):
        mlflow.sklearn.log_model(model, artifact_path)
    
    def end_run(self):
        mlflow.end_run()
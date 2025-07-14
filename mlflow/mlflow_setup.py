import mlflow
import os

def setup_mlflow(tracking_uri="http://localhost:5000", experiment_name="default"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.get_experiment_by_name(experiment_name)

def get_mlflow_client():
    return mlflow.tracking.MlflowClient()
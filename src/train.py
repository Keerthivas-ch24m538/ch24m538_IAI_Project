from utils.mlflow_setup import init_mlflow
import mlflow, mlflow.sklearn
from sklearn.linear_model import LogisticRegression  # Add import for dummy model

cfg = init_mlflow()  # sets up MLflow from yaml config

with mlflow.start_run():
    mlflow.log_param("demo", True)
    mlflow.log_metric("accuracy", 0.5)
    dummy_model = LogisticRegression()  # Create a dummy model instance
    mlflow.sklearn.log_model(dummy_model, artifact_path="model")  # No "model=" keyword!


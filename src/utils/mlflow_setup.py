import yaml, mlflow, os

def init_mlflow(config_path="configs/mlflow.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    if "experiment_name" in cfg:
        mlflow.set_experiment(cfg["experiment_name"])
    return cfg


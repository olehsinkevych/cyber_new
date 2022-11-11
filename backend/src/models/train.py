"""Training and evaluation routine"""

from urllib.parse import urlparse
import mlflow
import numpy as np
from .model import lstm_model


def get_train_val_data(configs: dict) -> tuple:
    path_x_train = configs["x_train"]
    path_y_train = configs["y_train"]
    path_x_val = configs["x_val"]
    path_y_val = configs["y_val"]
    x_train = np.load(path_x_train)
    y_train = np.load(path_y_train)
    x_val = np.load(path_x_val)
    y_val = np.load(path_y_val)
    return x_train, y_train, x_val, y_val


def train_evaluate(configs: dict) -> None:
    """Provides training and accuracy measurement"""

    x_train, y_train, x_val, y_val = get_train_val_data(
        configs["processed_data_config"])
    model = lstm_model(configs["lstm_model"])
    # Mlflow configuration
    mlflow_configs = configs["mlflow"]
    remote_server_uri = mlflow_configs["remote_server_uri0"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_configs["experiment_name"])
    with mlflow.start_run(run_name=mlflow_configs["run_name"]):
        # 1. hyper-parameters optimization
        # model_data = ...
        mlflow.log_param("model_type", "lstm")
        mlflow.log_metric("mean_squared_error", model_data.mse)
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.keras.log_model(
                model, "model",
                registered_model_name=mlflow_configs["registered_model_name"]
            )
        else:
            mlflow.keras.load_model(model, "model")




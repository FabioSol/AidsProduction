import mlflow
import mlflow.sklearn
import os
from AidsModel.pipeline import Pipeline
from AidsModel import model_path
from src import artifact_path
import joblib
from src.controller import Controller
import yaml
from src import hyper_params_path
from AidsModel.data_split import data_split
from AidsModel.evaluate_model import evaluate_model


def train_with_mlflow():
    with open(hyper_params_path, 'r') as file:
        best_params = yaml.safe_load(file)

    with mlflow.start_run():
        # Enable MLflow autologging
        mlflow.autolog()
        controller = Controller()
        data = controller.get_joined_data()
        X_train, X_test, y_train, y_test = data_split(data)
        model = Pipeline(**best_params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metric = evaluate_model(predictions,y_test)


        # Log the accuracy
        mlflow.log_metric("metric", metric)

        # Save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        # Log the model artifact
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        joblib.dump(model, artifact_path)
        mlflow.log_artifact(artifact_path, artifact_path="model")

        print("Model training and logging completed.")
        print(f"Model metric: {metric}")





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
from src.hyperparam_optimization import optimize_hyper_params

mlflow.set_tracking_uri("http://0.0.0.0:5000")


def load_previous_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def load_previous_metrics():
    # Assuming previous metrics are stored in a file
    metrics_path = model_path.replace('model.pkl', 'metrics.yml')
    if (os.path.exists(metrics_path)):
        with open(metrics_path, 'r') as file:
            return yaml.safe_load(file)
    return None


def save_current_metrics(metrics):
    metrics_path = model_path.replace('model.pkl', 'metrics.yml')
    with open(metrics_path, 'w') as file:
        yaml.dump(metrics, file)


def train_with_mlflow():
    with open(hyper_params_path, 'r') as file:
        best_params = yaml.safe_load(file)

    previous_model = load_previous_model()
    previous_metrics = load_previous_metrics()

    def train_and_evaluate(best_params):
        with mlflow.start_run():
            # Enable MLflow autologging
            mlflow.autolog()
            controller = Controller()
            data = controller.get_joined_data()
            X_train, X_test, y_train, y_test = data_split(data)
            model = Pipeline(**best_params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metric = evaluate_model(predictions, y_test)

            # Log the metric
            mlflow.log_metric("metric", metric)

            return model, metric

    if previous_model is None:
        # No previous model, train and save the new model
        model, metric = train_and_evaluate(best_params)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        save_current_metrics({'metric': metric})

        # Log the model artifact
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        joblib.dump(model, artifact_path)
        mlflow.log_artifact(artifact_path, artifact_path="model")

        print("Model training and logging completed.")
        print(f"Model metric: {metric}")

    else:
        # Train and evaluate the new model
        model, metric = train_and_evaluate(best_params)

        if previous_metrics is None or metric > previous_metrics['metric']:
            # Save the new model and metrics
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            save_current_metrics({'metric': metric})

            # Log the model artifact
            os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
            joblib.dump(model, artifact_path)
            mlflow.log_artifact(artifact_path, artifact_path="model")

            print("Model training and logging completed.")
            print(f"Model metric: {metric}")
        else:
            print("Current model is not better than the previous model. Optimizing hyperparameters...")
            best_trial = optimize_hyper_params()
            best_params = best_trial.params
            with open(hyper_params_path, 'w') as file:
                yaml.dump(best_params, file)
            # Retrain with optimized hyperparameters
            train_with_mlflow()


if __name__ == "__main__":
    train_with_mlflow()



import os

src_path = os.path.dirname(__file__)
root_path = os.path.dirname(src_path)
artifact_path = os.path.join(root_path,"mlflow_artifacts")
hyper_params_path = os.path.join(src_path, "hyperparams.yml")
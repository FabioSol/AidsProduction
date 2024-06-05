import os

src_path = os.path.dirname(__file__)
root_path = os.path.dirname(src_path)
model_repo_path = os.path.join(root_path, "AidsModel")
model_path = os.path.join(model_repo_path,"model","model.pkl")
data_path = os.path.join(model_repo_path,"data")
artifact_path = os.path.join(root_path,"mlflow_artifacts")
hyper_params_path = os.path.join(src_path, "hyperparams.yml")
import os.path
import pickle
from app.AidsModel import model_path

def predict(X):
    print(model_path)
    with open(model_path, 'rb') as f:  # Replace 'model.pkl' with your filename
        model = pickle.load(f)


import os.path
import pickle
from AidsModel import model_path

def predict(X):
    with open(model_path, 'rb') as f:  # Replace 'model.pkl' with your filename
        model = pickle.load(f)
    return model.predict(X)



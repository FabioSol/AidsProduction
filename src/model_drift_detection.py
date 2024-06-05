import pandas as pd
from scipy.stats import ks_2samp
from src.controller import Controller


def detect_model_drift(threshold=0.05):
    controller = Controller()
    new_data = controller.get_new_data()['cid']
    train_data = controller.get_train_data()['cid']

    if len(new_data) == 0:
        return False
    stat, p_value = ks_2samp(train_data, new_data)

    return p_value < threshold



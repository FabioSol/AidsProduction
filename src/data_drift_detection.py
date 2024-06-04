import pandas as pd
from scipy.stats import chi2_contingency
from controller import Controller


def detect_data_drift(threshold=0.05):
    controller = Controller()
    new_data = controller.get_join_data_for_data_drift()
    train_data = controller.get_train_data()
    drift_results = {}
    for column in new_data.columns:
        contingency_table = pd.crosstab(train_data[column], new_data[column])
        stat, p_value, _, _ = chi2_contingency(contingency_table)
        drift_results[column] = {
            'test': 'Chi²',
            'statistic': stat,
            'p_value': p_value,
            'drift_detected': p_value < threshold
        }
    return drift_results


def should_retrain():
    drift_results = detect_data_drift()
    return not all([res.get('drift_detected') for res in drift_results])

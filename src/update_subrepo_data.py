import os

import pandas as pd

from AidsProduction.src.controller import Controller
from AidsProduction.src import data_path


def update_subrepo_data():
    controller = Controller()
    df1 = controller.get_train_data()
    df2 = controller.get_new_data()
    merged_data = pd.concat([df1, df2], ignore_index=True)
    output_path = os.path.join(data_path,"data.csv")
    merged_data.to_csv(output_path, index=False)

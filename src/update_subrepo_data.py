import os

import pandas as pd

from app.src.controller import Controller
from app.AidsModel import model_repo_path


def update_subrepo_data():
    controller = Controller()
    df1 = controller.get_train_data()
    df2 = controller.get_new_data()
    merged_data = pd.concat([df1, df2], ignore_index=True)
    output_path = os.path.join(model_repo_path, "data/data.csv")
    merged_data.to_csv(output_path, index=False)

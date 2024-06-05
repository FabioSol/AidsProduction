import pandas as pd
from ucimlrepo import fetch_ucirepo
from src.controller import Controller


def migration():
    aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)
    df:pd.DataFrame = aids_clinical_trials_group_study_175.data.original.drop(columns=['pidnum']).rename(columns={'time':'time_feature'})
    controller = Controller()
    controller.insert_new_data(df)


if __name__ == '__main__':
    migration()
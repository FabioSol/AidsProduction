import yaml
from AidsModel.hyperparam_optimization import optimize_hyperparameters
from AidsProduction.src.controller import Controller
from AidsProduction.src import hyper_params_path

def optimize_hyper_params():
    controller = Controller()
    data = controller.get_joined_data_for_training()
    best_trial = optimize_hyperparameters(data)
    best_params = best_trial.params
    with open(hyper_params_path, 'w') as file:
        yaml.dump(best_params, file)

    print("Best hyperparameters saved to best_hyperparams.yml")
    print(best_params)



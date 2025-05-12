# -*- coding: utf-8 -*-
"""
File to create and store the configuration file for the experiment.

@author: elianemaalouf
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def load_config(path):
    """
    Load the configuration dictionary from a JSON file.

    path:
        Path to the JSON file.
    """
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def save_config(config_dict, path):
    """
    Save the configuration dictionary to a JSON file.

    config_dict:
        Dictionary containing the configuration parameters.
    path:
        Path to save the JSON file.
    """
    exp_dir = Path(path)
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(f"{exp_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=4)

# new experiment to add
xp_name = "det_preds_reg_n500"
xp_config = {
    "xp_name": xp_name,
    "xp_folder": f"{BASE_DIR}/Experiments/deterministic_preds/{xp_name}",
    "train_subset": 500,
    "lambda_x_vec":[1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
    "lambda_y_vec":[0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
    "run_validations": True,
    "validations":{'types':['rmse', 'es', 'vs'],
                   'params':[None, 2, 0.5]},
    "probabilistic": False,
    "prob_sample_size": 1,
    "parameters_file": f"{BASE_DIR}/Data/parameters_matern32_Mu10_Var1p96_CorH30_CorV15_linear_81.txt",
    "test_vecs_ids_to_invert": None,
    "det_pred_refs":{'rmse': {'lower': None, 'center': None, 'upper': None},
                     'es': {'lower': None, 'center': None, 'upper': None},
                     'vs':{'lower': None, 'center': None, 'upper': None}}
}
save_config(xp_config, xp_config["xp_folder"])

# dictionnary of all experiments
all_xp_configs_path = f"{BASE_DIR}/Experiments/all_xp_configs.json"
# verify if the file exists and create it if not
all_xp_configs = {}
if Path(all_xp_configs_path).exists():
    with open(all_xp_configs_path, 'r') as f:
        all_xp_configs = json.load(f)
else:
    all_xp_configs = {}
# add the new experiment to the dictionary
all_xp_configs[xp_name] = xp_config['xp_folder']

# save the updated dictionary
with open(all_xp_configs_path, 'w') as f:
    json.dump(all_xp_configs, f, indent=4)

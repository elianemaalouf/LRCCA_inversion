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
    Loads a configuration file in JSON format from the given file path and returns its contents as a dictionary.

    :param path: str
        The file path to the JSON configuration file.
    :return: dict
        The contents of the JSON file as a dictionary.
    """
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def save_config(config_dict, path):
    """
    Save a configuration dictionary to a JSON file at the specified path. If the
    directory does not exist, it will be created, including any required parent
    directories.

    :param config_dict: Dictionary containing configuration data to be saved in JSON format.
    :param path: Path to the directory where the JSON file will be saved. The filename
                 will always be 'config.json'.
    :return: None
    """
    exp_dir = Path(path)
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(f"{exp_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=4)

# new experiment to add
xp_name = "prob_preds_inv_n500_resims_ly_exp9"
probabilistic = True
run_validations = False # run validations or inversions
train_subset = 500

root_folder = "probabilistic_preds" if probabilistic else "deterministic_preds"
prob_sample_size = 500 if probabilistic else 1
val_subset_size = 50 if probabilistic else 2000
assess_training_metrics = False if probabilistic else True

xp_config = {
    "xp_name": xp_name,
    "xp_folder": f"{BASE_DIR}/Experiments/{root_folder}/{xp_name}",
    "train_subset": train_subset,
    "val_subset_size": val_subset_size,
    "assess_train_metrics":assess_training_metrics,
    "lambda_x_vec":[1e-4],
    "lambda_y_vec":[0.01],
    "run_validations": run_validations,
    "validation_repeats":5,
    "validations":{'types':['rmse','es', 'vs'],
                   'params':[None, 1, 0.5]},
    "probabilistic": probabilistic,
    "prob_sample_size": prob_sample_size,
    "parameters_file": f"{BASE_DIR}/Data/parameters_matern32_Mu10_Var1p96_CorH30_CorV15_linear_81.txt",
    "test_vecs_ids_to_invert": [102, 106, 270, 435, 860, 154, 253, 309, 548, 966, 385, 498, 583, 608, 836, 900, 10, 18,
    19, 20, 1, 3, 45, 96, 140, 157, 179, 191, 204, 223, 262, 269, 283, 304, 305, 347, 363, 379, 506, 517, 521, 546, 573,
    607, 656, 664, 671, 680, 792, 801,],
}

vs_train_refs_filename = "reference_inv_metrics_es1_vs05_rmse.pkl" # "reference_val_metrics_es2_vs05_rmse.pkl" if 'rmse' in xp_config['validations']['types'] else "reference_val_metrics_es1.pkl" #

# make det_preds_refs :
if not probabilistic:
    det_preds_refs = None
else:
    # deterministic prediction references are taken from the deterministic experiments with
    # small noise : lambda_x = 10^-4 and lambda_y = 1
    # large noise : lambda_x = 10^-4 and lambda_y = 10

    det_preds_refs = {}
    det_preds_refs['small_noise'] = {}
    det_preds_refs['large_noise'] = {}

    # check if 'es' is in validations types with param 2
    if 'es' in xp_config['validations']['types']:
        es_param = xp_config['validations']['params'][xp_config['validations']['types'].index('es')]
        if es_param == 1:
            # use the reference values for the probabilistic case
            det_preds_refs['small_noise']['es'] = {'lower': 857.940, 'center': 1037.395, 'upper': 1290.326},
            det_preds_refs['large_noise']['es'] = {'lower': 1021.957, 'center': 1195.132, 'upper': 1435.445}

        if es_param == 2:
            det_preds_refs['small_noise']['es'] = {'lower': 573.469, 'center': 819.821, 'upper': 1224.353}
            det_preds_refs['large_noise']['es'] = {'lower': 829.229, 'center': 1109.691, 'upper': 1541.074}

    # update det_preds_refs with rmse data if rmse is in validations types
    if 'rmse' in xp_config['validations']['types']:
        det_preds_refs['small_noise']['rmse'] = {'lower': 0.535, 'center': 0.640, 'upper': 0.782}
        det_preds_refs['large_noise']['rmse'] = {'lower': 0.644, 'center': 0.745, 'upper': 0.878}

    # update det_preds_refs with vs data if vs is in validations types
    if 'vs' in xp_config['validations']['types']:
        det_preds_refs['small_noise']['vs'] = {'lower': 542623.32, 'center': 697675.493, 'upper': 914168.837}
        det_preds_refs['large_noise']['vs'] = {'lower': 710793.601, 'center': 870780.149, 'upper': 1105397.309}

# update xp_config  with det_preds_refs
xp_config['det_pred_refs'] = det_preds_refs
xp_config['vs_train_refs_filename'] = vs_train_refs_filename

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

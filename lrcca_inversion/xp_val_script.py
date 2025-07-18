"""
Script to run the LRCCA inversion validation experiments

@author: elianemaalouf
"""

import json
from pathlib import Path

from xp_config import load_config

from lrcca_inversion.utils.config import Config
from lrcca_inversion.utils.generic_fn import import_dataset, save_to_disk
from lrcca_inversion.xp_runners import (get_y_obs_from_disk, run_inversion,
                                        run_validation)

BASE_DIR = Path(__file__).resolve().parent.parent

# Load the experiment configuration
all_xp_configs_path = f"{BASE_DIR}/Experiments/all_xp_configs.json"
if Path(all_xp_configs_path).exists():
    with open(all_xp_configs_path, "r") as f:
        all_xp_configs = json.load(f)
else:
    raise ValueError(
        "No experiments found. Please create an experiment configuration file first via xp_config.py"
    )

xp_name = "prob_preds_inv_n500_resims_ly_exp9"
xp_config = load_config(f"{all_xp_configs[xp_name]}/config.json")

# load data configuration and files
config = Config(xp_config["parameters_file"])

datadir = config.datadir
data_folder_location = config.data_folder_location
noises_list = config.noises_list

train_set_size = int(config.set_size * config.train_split)
val_set_size = int(config.set_size * config.val_split - train_set_size)
test_set_size = int(config.set_size - train_set_size - val_set_size)

## load datasets and center
datasets = import_dataset(
    ["train", "val", "test"],
    config_obj=config,
    train_set_size=train_set_size,
    subset_train=train_set_size,
    mean_center=True,
)

x_mean = datasets["means"][0]  # computed on training data
y_mean = datasets["means"][1]  # computed on training data

train_x = datasets["train"][0]
train_y = datasets["train"][1]  # training data is always noiseless

val_x = datasets["val"][0]
val_y = datasets["val"][1]  # noiseless at import time

test_x = datasets["test"][0]
test_y = datasets["test"][1]  # noiseless at import time

del datasets  # free up memory

if xp_config["run_validations"]:
    print("Running the validation...")
    # create a list of all combinations of lambda_x and lambda_y
    lambda_combinations = [
        (lambda_x, lambda_y)
        for lambda_x in xp_config["lambda_x_vec"]
        for lambda_y in xp_config["lambda_y_vec"]
    ]

    # Run the validation with the current combination of lambda_x and lambda_y
    validation_data = run_validation(
        lambda_combinations,
        xp_config["validations"],
        xp_config["probabilistic"],
        xp_config["prob_sample_size"],
        xp_config["train_subset"],
        xp_config["val_subset_size"],
        train_x,
        train_y,
        val_x,
        val_y,
        x_mean,
        noises_list,
        add_val_noise=True,
        assess_train_metrics=xp_config.get("assess_train_metrics", False),
        validation_repeats=xp_config.get("validation_repeats", 1),
    )
    # Save the validation data to disk
    save_to_disk(validation_data, f"{xp_config['xp_folder']}/validation_data.pkl")


else:
    print("Running the inversion...")
    # prepare test data for inversion
    test_y_loaded_from_disk = False
    test_vecs_ids_to_invert = xp_config["test_vecs_ids_to_invert"]

    if test_vecs_ids_to_invert is not None:
        test_y_loaded_from_disk = True
        test_x = test_x[test_vecs_ids_to_invert, :]

        # Load the test_y from disk
        test_y = get_y_obs_from_disk(test_vecs_ids_to_invert, config, y_mean)

    # Run the inversion
    inversion_data, cca_objects = run_inversion(
        xp_config["lambda_x_vec"],
        xp_config["lambda_y_vec"],
        xp_config["probabilistic"],
        xp_config["prob_sample_size"],
        xp_config["train_subset"],
        train_x,
        train_y,
        test_x,
        test_y,
        x_mean,
        noises_list,
        test_y_loaded_from_disk,
    )

    # Save the inversion data and CCA object to disk
    save_to_disk(inversion_data, f"{xp_config['xp_folder']}/inversion_data.pkl")
    save_to_disk(cca_objects, f"{xp_config['xp_folder']}/cca_objects.pkl")

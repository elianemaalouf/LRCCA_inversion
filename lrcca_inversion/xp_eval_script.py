"""
Script to run the LRCCA inversion experiments evaluations

@author: elianemaalouf
"""

import json
from pathlib import Path

from xp_config import load_config

from lrcca_inversion.utils.config import Config
from lrcca_inversion.utils.generic_fn import (import_dataset, load_from_disk,
                                              save_to_disk)
from lrcca_inversion.xp_runners import (run_inv_reference_metrics,
                                        run_inversion_eval,
                                        run_val_reference_metrics,
                                        run_validation_eval)

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

# get vs training references
if not Path(f"{BASE_DIR}/Experiments/{xp_config['vs_train_refs_filename']}").exists():
    print("Running the reference statistics...")
    # Run the reference statistics
    if xp_config["run_validations"]:
        reference_metrics = run_val_reference_metrics(
            n=100,
            m=500,
            train_x=train_x,
            val_x=val_x,
            x_mean=x_mean,
            metric_dict=xp_config["validations"],
        )
    else:
        print("for test...")
        test_vecs_ids_to_invert = (
            xp_config["test_vecs_ids_to_invert"]
            if xp_config["test_vecs_ids_to_invert"] is not None
            else None
        )

        reference_metrics = run_inv_reference_metrics(
            n=None,
            m=500,
            train_x=train_x,
            test_x=test_x,
            test_ids=test_vecs_ids_to_invert,
            x_mean=x_mean,
            metric_dict=xp_config["validations"],
        )

    # Save the reference metrics to disk
    save_to_disk(
        reference_metrics,
        f"{BASE_DIR}/Experiments/{xp_config['vs_train_refs_filename']}",
    )
else:
    # Load the reference metrics from disk
    reference_metrics = load_from_disk(
        f"{BASE_DIR}/Experiments/{xp_config['vs_train_refs_filename']}"
    )

if xp_config["run_validations"]:
    print("Running the validation evaluation...")
    # import validation_data from disk
    validation_data = load_from_disk(f"{xp_config['xp_folder']}/validation_data.pkl")

    # Run the validation evaluation and plots
    reference_metrics_list = [reference_metrics]

    if xp_config["probabilistic"]:
        reference_metrics_list.append(xp_config["det_pred_refs"])

    run_validation_eval(
        validation_data,
        xp_config["xp_folder"],
        reference_metrics_list=reference_metrics_list,
        ref_stat="median",
        reduce_lambda_y_vec=True,
        lambda_y_subset=3,
    )

else:
    print("Running the inversion evaluation...")
    # import inversion_data and pretrained cca object from disk
    inversion_data = load_from_disk(f"{xp_config['xp_folder']}/inversion_data.pkl")
    cca_objects = load_from_disk(f"{xp_config['xp_folder']}/cca_objects.pkl")

    # Run the inversion evaluation and plots
    run_inversion_eval(
        inversion_data,
        cca_objects,
        xp_config,
        config,
        reference_metrics,
        x_mean=x_mean,
        y_mean=y_mean,
        assess_vs_det_pred=False,
        assess_resims=True,
    )

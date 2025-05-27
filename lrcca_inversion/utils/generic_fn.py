# -*- coding: utf-8 -*-
"""
@author: elianemaalouf
"""

import h5py
import numpy as np
import torch


def save_to_disk(data, file_path, json=False):
    """
    Saves the given data object to the disk at the specified file path. The
    data can be saved in either JSON or binary (pickle) format based on the
    value of the `json` parameter. If `json` is True, the data will be saved
    in JSON format; otherwise, it will be saved in binary format using pickle.

    :param data: The data object that needs to be saved to the specified
        file on disk. Its content and type depend on the serialization format.
    :param file_path: The file path where the serialized data will be stored.
    :param json: A boolean flag indicating the serialization format.
        If True, data is saved in JSON format. Defaults to False.
    :return: None
    """
    if json:
        import json

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    else:
        import pickle

        with open(file_path, "wb") as f:
            pickle.dump(data, f)


def load_from_disk(file_path):
    """
    Loads data from a specified file path using pickle.

    This function reads serialized data from the file located at the given
    file path and deserializes it using pickle. It expects the file to
    be in binary mode and contain data compatible with pickle format.

    :param file_path: The path to the file storing the serialized data.
    :return: The deserialized data loaded from the file.
    """
    import pickle

    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def import_dataset(
    datasets_list, config_obj, train_set_size, subset_train=None, mean_center=True
):
    """
    Imports datasets specified in the input list and processes them based on configuration
    parameters, mean-centering, and optional subset specifications for training data. The
    datasets can include training, validation, or testing sets. Each dataset is loaded
    from its respective HDF5 file and may include transformations such as reshaping and
    mean-centering. Returns a dictionary containing processed datasets for further use.

    :param datasets_list: A list containing dataset identifiers to import (e.g., 'train',
        'val', 'test').
    :param config_obj: An object containing configuration details such as data folder
        location and dimensional parameters (nx, ny, rays).
    :param train_set_size: Total number of training examples available in the dataset.
    :param subset_train: Optional; The number of training examples to use in a subset.
        If None, defaults to `train_set_size`.
    :param mean_center: Boolean flag; If True, applies mean-centering to all datasets
        (training, validation, and testing).
    :return: A dictionary containing imported datasets (`train`, `val`, `test`) and their
        calculated mean values (`means`) for input and output features.
    """
    if "train" not in datasets_list:
        raise ValueError("The dataset list should contain 'train' at least")

    result = {}

    data_folder_location = config_obj.data_folder_location
    dim_x = config_obj.nx * config_obj.ny
    dim_y = config_obj.rays

    if "train" in datasets_list:
        if subset_train is None:
            subset_train = train_set_size

        idx_train = np.random.choice(
            np.arange(train_set_size), size=subset_train, replace=False
        )
        train_models_file = h5py.File(f"{data_folder_location}/train_models.h5")
        train_models_all = (
            torch.tensor(train_models_file.get("train_models"), dtype=torch.float64)
            .numpy()
            .reshape(-1, dim_x)
        )
        train_x = train_models_all[idx_train, :]
        train_models_file.close()

        train_truett_file = h5py.File(f"{data_folder_location}/train_truett.h5")
        train_truett_all = (
            torch.tensor(train_truett_file.get("train_truett"), dtype=torch.float64)
            .numpy()
            .reshape(-1, dim_y)
        )
        train_y = train_truett_all[idx_train, :]
        train_truett_file.close()

        x_mean = train_x.mean(axis=0)
        y_mean = train_y.mean(axis=0)

        if mean_center:
            train_x = train_x - x_mean
            train_y = train_y - y_mean

        result["means"] = [x_mean, y_mean]
        result["train"] = [train_x, train_y]

    if "val" in datasets_list:
        # import validation sets
        val_models_file = h5py.File(f"{data_folder_location}/val_models.h5")
        val_models = torch.tensor(
            val_models_file.get("val_models"), dtype=torch.float64
        ).numpy()
        val_x = val_models.reshape(-1, dim_x)
        val_models_file.close()

        val_truett_file = h5py.File(f"{data_folder_location}/val_truett.h5")
        val_truett_noiseless = torch.tensor(
            val_truett_file.get("val_truett"), dtype=torch.float64
        ).numpy()
        val_y = val_truett_noiseless.reshape(-1, dim_y)
        val_truett_file.close()

        if mean_center:
            val_x = val_x - x_mean
            val_y = val_y - y_mean

        result["val"] = [val_x, val_y]
    else:
        result["val"] = [None, None]

    if "test" in datasets_list:
        test_models_file = h5py.File(f"{data_folder_location}/test_models.h5")
        test_models = torch.tensor(
            test_models_file.get("test_models"), dtype=torch.float64
        ).numpy()
        test_x = test_models.reshape(-1, dim_x)
        test_models_file.close()
        test_truett_file = h5py.File(f"{data_folder_location}/test_truett_noNoise.h5")
        test_truett_noiseless = torch.tensor(
            test_truett_file.get("test_truett_noNoise"), dtype=torch.float64
        ).numpy()
        test_y = test_truett_noiseless.reshape(-1, dim_y)
        test_truett_file.close()

        if mean_center:
            test_x = test_x - x_mean
            test_y = test_y - y_mean

        result["test"] = [test_x, test_y]
    else:
        result["test"] = [None, None]

    return result


def get_noise(config_obj, noise_label):
    """
    Retrieve the noise configuration from a list of available configurations
    based on the noise label provided. The function looks up the appropriate
    noise by matching the given label to predefined options, "small_gauss" or
    "large_gauss". If the noise label does not match any recognized options,
    an exception is raised indicating the unrecognized label.

    :param config_obj: Configuration object containing the list of available
        noises in the attribute `noises_list`. Assumes this object has a
        valid property `noises_list`, which is a list where the first element
        corresponds to "small_gauss" noise and the second element corresponds
        to "large_gauss" noise.
    :param noise_label: Label specifying the noise to retrieve. Must be one
        of the following recognized labels: "small_gauss" or "large_gauss".
    :return: The noise configuration corresponding to the given label. Returns
        the first element of `noises_list` for "small_gauss" or the second
        element of `noises_list` for "large_gauss".
    :raises ValueError: If the provided `noise_label` does not match any
        recognized labels ("small_gauss" or "large_gauss").
    """
    if noise_label == "small_gauss":
        return config_obj.noises_list[0]
    elif noise_label == "large_gauss":
        return config_obj.noises_list[1]
    else:
        raise ValueError("Unknown noise label: {}".format(noise_label))

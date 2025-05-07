# -*- coding: utf-8 -*-
"""
@author: elianemaalouf
"""

import h5py
import torch
import numpy as np
import pickle

def save_to_disk(data, file_path):
    """
    Save data to disk using pickle.

    data:
        data to save
    file_path:
        path to save the data
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_from_disk(file_path):
    """
    Load data from disk using pickle.

    file_path:
        path to load the data from
    :return: loaded data
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def import_dataset(datasets_list, config_obj, train_set_size, subset_train = None, mean_center = True):
    """
    Import datasets from the specified list.
    :param datasets_list: list of dataset names, should contain "train" at least
    :param config_obj: config object containing the data folder location
    :param train_set_size: size of the training set
    :param subset_train: number of training samples to select randomly from the training set
    :return : list of imported datasets
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

        result['means'] = [x_mean, y_mean]
        result['train'] = [train_x, train_y]

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

        result['val'] = [val_x, val_y]
    else:
        result['val'] = [None, None]

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
        test_y = test_truett_noiseless.reshape(-1,dim_y)
        test_truett_file.close()

        if mean_center:
            test_x = test_x - x_mean
            test_y = test_y - y_mean

        result['test'] = [test_x, test_y]
    else:
        result['test'] = [None, None]

    return result

def get_noise(config_obj, noise_label):
    """
    Get the noise parameters from the config object
    :param config_obj: config object containing the noise parameters
    :param noise_label: label of the noise
    :return: noise parameters
    """
    if noise_label == "small_gauss":
        return config_obj.noises_list[0]
    elif noise_label == "large_gauss":
        return config_obj.noises_list[1]
    else:
        raise ValueError("Unknown noise label: {}".format(noise_label))




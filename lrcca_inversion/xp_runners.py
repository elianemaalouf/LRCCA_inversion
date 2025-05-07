"""
Define the runners functions for the experiments.

@author: elianemaalouf
"""
import numpy as np

from cca import  CCA
from lrcca_inversion.utils.metrics import rmse, es, vs

AVAILABLE_METRICS = {
    'rmse': rmse,
    'es': es,
    'vs':vs,
}

def sample_noise(noise_dict, sample_size, dim):
    noise_distribution = noise_dict["distribution"]
    noise_loc = noise_dict["location"]
    noise_scale = noise_dict["scale"]

    if noise_distribution.lower() == "Gaussian".lower():
        noise_sample = noise_scale * np.random.randn(sample_size, dim) + noise_loc
        if noise_scale < 2:
            noise_label = "small_noise"
        else:
            noise_label = "large_noise"
    else:
        raise ValueError("noise distribution not supported")

    return noise_sample, noise_label

def run_validation_metrics(predicted, true, validation_type, validation_param):
    """
    Run the validation metrics for the given predicted and true values.
    """
    est_metric = None
    m = predicted.shape[2]
    n = true.shape[0]
    dim = true.shape[1]

    if validation_type not in AVAILABLE_METRICS.keys():
        raise ValueError(f"Validation type {validation_type} not supported. Available types: {list(AVAILABLE_METRICS.keys())}")
    else:
        for i in range(n):
            est_metric = []
            observation = true[i, :].reshape(1, dim)
            samples = predicted[i, :, :].reshape(m, dim)

            if validation_type == "rmse":
                est_metric.extend(rmse(observation, samples))

            elif validation_type == "es":
                est_metric.append(es(observation, samples, validation_param))

            elif validation_type == "vs":
                est_metric.append(vs(observation, samples, validation_param))

    return est_metric

def run_validation(lambda_combinations, validations_dict, probabilistic, prob_sample_size, train_x, train_y,
                   val_x, val_y, x_mean, y_mean, noises_list, add_val_noise=True):
    """
    Run the validation with the current combination of lambda_x and lambda_y.
    """
    if val_x is None:
        raise ValueError("Validation data is None. Please provide valid validation data.")

    validation_types = validations_dict['types']
    validation_params = validations_dict['params']
    out_dim = train_x.shape[1]

    results = {}

    # Loop over noise types
    for noise_i in noises_list:
        noise_val, noise_label = sample_noise(noise_i, val_y.shape[0], val_y.shape[1]) if add_val_noise else None

        results[noise_label] = {}
        results[noise_label]['train'] = {}
        results[noise_label]['val'] = {}

        # loop over combinations of lambda_x and lambda_y
        for comb in lambda_combinations:
            lambda_x, lambda_y = comb

            cca = CCA()
            cca.fit_cca_svd(train_x, train_y, lambda_x=lambda_x, lambda_y=lambda_y)

            # predict on training
            predicted_train_x = CCA.predict(cca.T_x_full_inv_T, cca.T_y_can, train_y, cca.CanCorr, out_dim,
                                            out_mean = x_mean, probabilistic = probabilistic, sample_size = prob_sample_size)

            # add noise to val_y
            val_y = val_y + noise_val if noise_val is not None else val_y
            # predict on validation
            predicted_val_x = CCA.predict(cca.T_x_full_inv_T, cca.T_y_can, val_y, cca.CanCorr, out_dim,
                                            out_mean = x_mean, probabilistic = probabilistic, sample_size = prob_sample_size)

            # re-add mean
            train_x_d = train_x + x_mean
            val_x_d = val_x + x_mean

            # compute validation metrics
            for i, validation_type in enumerate(validation_types):
                results[noise_label]['train'][validation_type] = {}
                results[noise_label]['val'][validation_type] = {}

                validation_param = validation_params[i]
                train_metrics = run_validation_metrics(predicted_train_x, train_x_d, validation_type, validation_param)
                val_metrics = run_validation_metrics(predicted_val_x, val_x_d, validation_type, validation_param)

                results[noise_label]['train'][validation_type][comb] = train_metrics
                results[noise_label]['val'][validation_type][comb] = val_metrics

    return results

def run_inversion():
    pass

def run_validation_eval():
    pass

def run_inversion_eval():
    pass
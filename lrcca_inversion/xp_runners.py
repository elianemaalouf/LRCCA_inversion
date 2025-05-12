"""
Define the runners functions for the experiments.

@author: elianemaalouf
"""
import time

import numpy as np

from cca import  CCA
from lrcca_inversion.utils.metrics import rmse, es, vs

AVAILABLE_METRICS = {
    'rmse': rmse,
    'es': es,
    'vs':vs,
}

def select_random_indices(n, s, with_replacement = False):
    """
    Select s random indices from n, with or without replacement.

    s:
        number of indices to select. int.
    n:
        total number of indices to select from. int.
    with_replacement:
        whether to select with replacement or not (allows repetitions). bool.

    """
    if s > n:
        raise ValueError("s should be less than or equal to n")
    return np.random.choice(n, size=s, replace=with_replacement)

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

def run_metrics(predicted, true, metric, metric_param):
    """
    Run the validation metrics for the given predicted and true values.
    """
    m = predicted.shape[2]
    n = true.shape[0]
    dim = true.shape[1]

    if metric not in AVAILABLE_METRICS.keys():
        raise ValueError(f"Validation type {metric} not supported. Available types: {list(AVAILABLE_METRICS.keys())}")
    else:
        est_metric = []
        for i in range(n):
            observation = true[i, :].reshape(1, dim)
            samples = predicted[i, :, :].reshape(m, dim)

            if metric == "rmse":
                est_metric.extend(rmse(observation, samples))

            elif metric == "es":
                est_metric.append(es(observation, samples, metric_param))

            elif metric == "vs":
                est_metric.append(vs(observation, samples, metric_param))

    return est_metric

def run_validation(lambda_combinations, validations_dict, probabilistic, prob_sample_size, train_subset_size,
                   train_x, train_y, val_x, val_y, x_mean, y_mean, noises_list, add_val_noise=True):
    """
    Run the validation with the current combination of lambda_x and lambda_y.
    """
    if val_x is None:
        raise ValueError("Validation data is None. Please provide valid validation data.")

    validation_types = validations_dict['types']
    validation_params = validations_dict['params']
    out_dim = train_x.shape[1]
    full_train_size = train_x.shape[0]
    val_sample_size = val_x.shape[0] # the maximum we will use during the validation, also from the training set.
                                    # the training of CCA will be done with subset of size train_subset_size
    results = {}

    # Select a random subset of the training data (for validation)
    train_val_subset_indices = select_random_indices(full_train_size, val_sample_size, with_replacement=False)
    train_val_x = train_x[train_val_subset_indices, :]
    train_val_y = train_y[train_val_subset_indices, :]

    # Select a random subset of the training data (for training)
    train_subset_indices = select_random_indices(full_train_size, train_subset_size, with_replacement=False)
    train_x = train_x[train_subset_indices, :]
    train_y = train_y[train_subset_indices, :]

    # Loop over noise types
    for noise_i in noises_list:
        noise_val, noise_label = sample_noise(noise_i, val_y.shape[0], val_y.shape[1]) if add_val_noise else None

        # add noise to val_y if any, else copy original val_y
        val_y_n = val_y.copy() + noise_val if add_val_noise else val_y.copy()

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
              f"Running validation for noise type:{noise_label}")

        results[noise_label] = {}
        results[noise_label]['train'] = {}
        results[noise_label]['val'] = {}

        # make structure for all validation types
        for validation_type in validation_types:
            results[noise_label]['train'][validation_type] = {}
            results[noise_label]['val'][validation_type] = {}

        # loop over combinations of lambda_x and lambda_y
        for comb in lambda_combinations:
            lambda_x, lambda_y = comb
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                  f":Running validation for lambda_x: {lambda_x}, lambda_y: {lambda_y}")

            cca = CCA()
            cca.fit_cca_svd(train_x, train_y, lambda_x=lambda_x, lambda_y=lambda_y)

            # predict on training
            predicted_train_x = CCA.predict(cca.T_x_full_inv_T, cca.T_y_can, train_val_y, cca.CanCorr, out_dim,
                                            out_mean = x_mean, probabilistic = probabilistic, sample_size = prob_sample_size)

            # predict on validation
            predicted_val_x = CCA.predict(cca.T_x_full_inv_T, cca.T_y_can, val_y_n, cca.CanCorr, out_dim,
                                            out_mean = x_mean, probabilistic = probabilistic, sample_size = prob_sample_size)

            # re-add mean
            train_val_x_d = train_val_x.copy() + x_mean
            val_x_d = val_x.copy() + x_mean

            # compute validation metrics
            for i, validation_type in enumerate(validation_types):

                validation_param = validation_params[i]
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                      f"Running validation: {validation_type} with (train) predictions {predicted_train_x.shape}")
                train_metrics = run_metrics(predicted_train_x, train_val_x_d, validation_type, validation_param)

                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                      f"Running validation: {validation_type} with (val) predictions {predicted_val_x.shape}")
                val_metrics = run_metrics(predicted_val_x, val_x_d, validation_type, validation_param)

                results[noise_label]['train'][validation_type][comb] = train_metrics
                results[noise_label]['val'][validation_type][comb] = val_metrics

    return results

def run_inversion():
    pass

def run_validation_eval(validation_data, xp_config_folder, reference_metrics_list=None, **kwargs):
    """
    Run the validation results evaluation and plots.

    validation_data:
        the dictionary of validation data as provided by the run_validation function.
    xp_config_folder:
        the location where the evaluations and plots will be saved to.
    reference_metrics_list:
        reference metrics against which to add to plots. expected as a list.
    kwargs:
        additional arguments to pass to the evals functions such as 'ref_stat' for the reference statistics,
        and plotting configuration in make_val_boxplots such as 'whis_low', 'whis_high', 'lower_lim', 'upper_lim',
        'h_axis_margin', 'v_axis_margin', 'x_ticks_step', 'reduce_lambda_y_vec'
    """
    import lrcca_inversion.utils.evals as evals

    ## get noise labels as first level keys
    noise_labels = list(validation_data.keys())
    ## get the second level keys (train/val)
    train_val_keys = list(validation_data[noise_labels[0]].keys())
    ## get the third level keys (metrics)
    metrics = list(validation_data[noise_labels[0]][train_val_keys[0]].keys())
    ## get the fourth level keys (lambda combinations)
    lambda_combinations = list(validation_data[noise_labels[0]][train_val_keys[0]][metrics[0]].keys())

    # make summary statistics of the validation data for each noise type/train or val/metric type/ combination
    # and find the best combination for each metric for each noise type/train or val/metric type
    # and save them to disk in readable format
    summaries = {}
    best_comb = {}

    # Combined loop for both summaries and best combinations
    for noise_label in noise_labels:
        summaries[noise_label] = {}
        best_comb[noise_label] = {}

        for train_val_key in train_val_keys:
            summaries[noise_label][train_val_key] = {}
            best_comb[noise_label][train_val_key] = {}

            for metric in metrics:
                # Computing summaries
                summaries[noise_label][train_val_key][metric] = {}
                for comb in lambda_combinations:
                    data = validation_data[noise_label][train_val_key][metric][comb]
                    summaries[noise_label][train_val_key][metric][str(comb)] = evals.compute_stats(data)

                # Finding best combination for current metric
                best_comb[noise_label][train_val_key][metric] = evals.get_best_param_comb(
                    validation_data[noise_label][train_val_key], metric, ref_stat=kwargs.get('ref_stat', 'median')
                )

    # Save results
    evals.save_to_disk(summaries, f"{xp_config_folder}/validation_summaries.json")
    evals.save_to_disk(best_comb, f"{xp_config_folder}/best_param_comb.json")

    # make boxplots of the validation data for each noise type/train or val/metric type
    for noise_label in noise_labels:
        for train_val_key in train_val_keys:
            for metric in metrics:
                references_dict = evals.make_ref_dict(reference_metrics_list, metric) if reference_metrics_list else None
                data = validation_data[noise_label][train_val_key][metric]
                evals.make_val_boxplots(data, metric, references_dict=references_dict,
                                        save_path=f"{xp_config_folder}/{noise_label}_{train_val_key}_{metric}.pdf",
                                        reduce_lambda_y_vec=kwargs.get('reduce_lambda_y_vec', False),
                                        lambda_y_subset=kwargs.get('lambda_y_subset', 4))



def run_inversion_eval(inversion_data, xp_config_folder):
    pass

def run_reference_metrics(n, m, train_x, val_x, x_mean, metric_dict):
    """
    Compute reference statistics against training data.
    n:
        number of samples from val_x (considered as observations/ground truths here)
    m:
        number of samples from train_x (considered as predictions here)
    train_x:
        training data (reference data)
    val_x:
        validation data (observations)
    xp_config_folder:
        the location where the evaluations and plots will be saved to.
    :return:
    """
    total_train = train_x.shape[0]
    total_val = val_x.shape[0]
    dim = train_x.shape[1]

    metric_types = metric_dict['types']
    metric_params = metric_dict['params']

    # select the observations/ground truths. shape (n, dim)
    sample_val_indices = select_random_indices(total_val, n, with_replacement=False)
    val_x = val_x[sample_val_indices, :]
    val_x_d = val_x.copy() + x_mean

    # select the predictions and repeat them for each observation. shape (n, dim, m)
    sample_train_indices = select_random_indices(total_train, m, with_replacement=False)
    train_x_d = train_x[sample_train_indices, :].copy() + x_mean
    train_x_d = np.repeat(train_x_d.reshape(1, dim, m), n, axis=0)

    # compute metrics
    metrics = {}
    for i, metric in enumerate(metric_types):
        metrics[metric] = run_metrics(train_x_d, val_x_d, metric, metric_params[i])

    return metrics

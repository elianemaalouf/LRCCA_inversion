"""
Define the runners functions for the experiments.

@author: elianemaalouf
"""

import time
from pathlib import Path

import numpy as np
from xp_config import det_preds_refs

from cca import CCA
from lrcca_inversion.utils.generic_fn import load_from_disk, save_to_disk
from lrcca_inversion.utils.metrics import es, rmse, vs

AVAILABLE_METRICS = {
    "rmse": rmse,
    "es": es,
    "vs": vs,
}


def select_random_indices(n, s, with_replacement=False):
    """
    Selects random indices from a range of size `n`.

    This function selects `s` random indices from a range of numbers `[0, n)`
    (optionally allowing for replacement). If `s` is greater than `n`, a
    `ValueError` is raised. This function uses NumPy's `random.choice`
    method to perform the selection.

    :param n: The total number of elements in the range to choose from.
    :param s: The number of random indices to select.
    :param with_replacement: Boolean indicating if the indices can be
        selected more than once (True for with replacement, False otherwise).
    :return: A NumPy array containing the selected random indices.
    """
    if s > n:
        raise ValueError("s should be less than or equal to n")
    return np.random.choice(n, size=s, replace=with_replacement)


def sample_noise(noise_dict, sample_size, dim):
    """
    Generates a sample of noise from a given distribution and associates a label
    based on the scale of the noise. Currently, this function supports only the
    Gaussian distribution. If the specified scale of noise is smaller than 2,
    the noise is labeled as 'small_noise'. Otherwise, it is labeled as 'large_noise'.

    :param noise_dict: A dictionary containing the noise distribution information:
        - "distribution": The name of the distribution (e.g., 'Gaussian').
        - "location": The mean or center of the distribution.
        - "scale": The standard deviation or spread of the noise.
    :param sample_size: An integer representing the number of samples to generate.
    :param dim: An integer representing the dimensionality of each noise sample.
    :return: A tuple containing:
        - A NumPy array of generated noise samples with shape (sample_size, dim).
        - A string label describing the scale of the noise ('small_noise' or 'large_noise').
    """
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


def get_y_obs_from_disk(ids_list, config, y_mean):
    """
    Extracts processed noisy observation vectors from disk files based on specified configurations and
    returns them categorized by noise levels.

    :param ids_list: A list of identifiers used to locate corresponding noisy test vectors.
    :param config: Configuration object containing data folder paths, noise distribution details,
        and other relevant parameters such as the number of rays.
    :param y_mean: The mean value(s) to subtract from the loaded noisy vectors during preprocessing.
    :return: A dictionary where each key represents a noise category ("small_noise" or "large_noise")
        and the value is a 2D numpy array containing the processed observation vectors for that
        category.
    :rtype: dict
    """
    import pickle

    test_y = {}
    noises_list = config.noises_list
    data_folder_location = config.data_folder_location

    for noise_i in noises_list:
        noise_distribution = noise_i["distribution"]
        noise_loc = noise_i["location"]
        noise_scale = noise_i["scale"]

        if noise_scale < 2:
            noise_label = "small_noise"
        else:
            noise_label = "large_noise"

        test_y[noise_label] = np.zeros((len(ids_list), config.rays))

        noise_test_y_folder = f"{data_folder_location}/noisy_ttvec_{noise_distribution}_loc{noise_loc}_scale{str(noise_scale).replace('.', 'p')}"

        for i, vec_id in enumerate(ids_list):
            # read y_obs
            with open(f"{noise_test_y_folder}/noisy_tt_vec{vec_id}", "rb") as f:
                y_obs = pickle.load(f)

            y_obs = y_obs.numpy().reshape(-1, config.rays)
            y_obs = y_obs - y_mean
            test_y[noise_label][i, :] = y_obs

    return test_y


def run_metrics(predicted, true, metric, metric_param, reduced_sample_size=None):
    """
    Run evaluation metrics on predicted and true data using specified metric and parameters.

    This function takes predicted values, true observations, and a specified metric to
    evaluate the performance of predictions. It supports multiple metrics such as RMSE,
    Energy Score (ES), and Variance Score (VS). For RMSE, the function can optionally
    reduce the number of samples used in the calculation.

    :param predicted: Predicted values with shape (n_samples, n_dimensions, n_predictions).
    :param true: Ground truth data with shape (n_samples, n_dimensions).
    :param metric: The evaluation metric to compute. Must be one of the keys in
        `AVAILABLE_METRICS`.
    :param metric_param: Additional parameter required for specific metrics such as ES
        and VS.
    :param reduced_sample_size: Optional integer specifying the number of reduced
        samples for RMSE computation. If None, no reduction is applied.
    :return: List of metric values calculated for each observation.
    :rtype: list
    """
    m = predicted.shape[2]
    n = true.shape[0]
    dim = true.shape[1]

    if metric not in AVAILABLE_METRICS.keys():
        raise ValueError(
            f"Validation type {metric} not supported. Available types: {list(AVAILABLE_METRICS.keys())}"
        )
    else:
        est_metric = []

        for i in range(n):
            observation = true[i, :].reshape(1, dim)
            samples = predicted[i, :, :].transpose()

            if metric == "rmse":
                if reduced_sample_size is not None:
                    reduced_sample_ids = select_random_indices(
                        m, reduced_sample_size, with_replacement=False
                    )
                    est_metric.extend(rmse(observation, samples[reduced_sample_ids, :]))
                else:
                    est_metric.extend(rmse(observation, samples))

            elif metric == "es":
                est_metric.append(es(observation, samples, metric_param))

            elif metric == "vs":
                est_metric.append(vs(observation, samples, metric_param))

    return est_metric


def run_validation(
    lambda_combinations,
    validations_dict,
    probabilistic,
    prob_sample_size,
    train_subset_size,
    val_subset_size,
    train_x_orig,
    train_y_orig,
    val_x_orig,
    val_y_orig,
    x_mean,
    noises_list,
    add_val_noise=True,
    assess_train_metrics=False,
    validation_repeats=1,
):
    """
    Run validation experiments for given data and parameters, iterating through specified
    lambda combinations, noise configurations, subsets, and validation metrics. The function
    performs validations for a specified number of repeats and optionally computes training
    metrics. Key processes include subset selection, noise sampling, model training with
    Canonical Correlation Analysis (CCA), and evaluation using specified validation methods.

    :param lambda_combinations: List of tuples containing lambda_x and lambda_y values to be
                                used in the CCA training process
    :param validations_dict: Dictionary containing:
                              - 'types': List of validation methods (e.g., metrics types) to apply
                              - 'params': Corresponding parameters for each validation type
    :param probabilistic: Boolean indicating whether to use probabilistic predictions or deterministic ones
    :param prob_sample_size: Integer specifying the sample size for probabilistic predictions
    :param train_subset_size: Integer defining the size of subsets to randomly select from
                              training data for model training
    :param val_subset_size: Integer defining the size of subsets to randomly select from
                            validation data for validation. If None, the full validation set
                            is used
    :param train_x_orig: Numpy array of original training data (input)
    :param train_y_orig: Numpy array of original training data (output)
    :param val_x_orig: Numpy array of original validation data (input)
    :param val_y_orig: Numpy array of original validation data (output)
    :param x_mean: Mean of the training inputs (used for de-centralization)
    :param noises_list: List of noise configurations to be applied to the validation responses
                        during validation
    :param add_val_noise: Boolean indicating whether to add noise to the validation response
    :param assess_train_metrics: Boolean indicating whether to calculate metrics for the training
                                 sets used during validation
    :param validation_repeats: Integer specifying the number of times to run validation
                               experiments with random subsets
    :return: Dictionary containing validation results categorized by noise types, training or
             validation metrics, validation types, and lambda combinations
    """
    if val_x_orig is None:
        raise ValueError(
            "Validation data is None. Please provide valid validation data."
        )

    validation_types = validations_dict["types"]
    validation_params = validations_dict["params"]
    out_dim = train_x_orig.shape[1]
    full_train_size = train_x_orig.shape[0]
    val_subset_size = (
        val_subset_size if val_subset_size is not None else val_x_orig.shape[0]
    )
    # val_subset_size: the maximum we will use during the validation, also from the training set (to limit execution time).
    # the training of CCA will be done with subset of size train_subset_size
    results = {}

    for _ in range(validation_repeats):
        print(f"Validation repeath number {_ + 1} of {validation_repeats}")

        # Select a random subset of the training data (for validation)
        if assess_train_metrics:
            train_val_subset_indices = select_random_indices(
                full_train_size, val_subset_size, with_replacement=False
            )
            train_val_x = train_x_orig[train_val_subset_indices, :]
            train_val_y = train_y_orig[train_val_subset_indices, :]

        # Select a random subset of the training data (for training)
        train_subset_indices = select_random_indices(
            full_train_size, train_subset_size, with_replacement=False
        )
        train_x = train_x_orig[train_subset_indices, :]
        train_y = train_y_orig[train_subset_indices, :]

        if val_subset_size < val_x_orig.shape[0]:
            # select random indices for the validation data
            val_subset_indices = select_random_indices(
                val_x_orig.shape[0], val_subset_size, with_replacement=False
            )
            val_x = val_x_orig[val_subset_indices, :]
            val_y = val_y_orig[val_subset_indices, :]

        # Loop over noise types
        for noise_i in noises_list:
            noise_val, noise_label = (
                sample_noise(noise_i, val_y.shape[0], val_y.shape[1])
                if add_val_noise
                else None
            )

            # add noise to val_y if any, else copy original val_y
            val_y_n = val_y.copy() + noise_val if add_val_noise else val_y.copy()

            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                f"Running validation for noise type:{noise_label}"
            )

            # check if results is empty, if so, initialize it, else, leave it as is
            if noise_label not in results:
                results[noise_label] = {}
                if assess_train_metrics:
                    results[noise_label]["train"] = {}
                results[noise_label]["val"] = {}

            # make structure for all validation types
            for validation_type in validation_types:
                if assess_train_metrics:
                    results[noise_label]["train"][validation_type] = (
                        {}
                        if validation_type not in results[noise_label]["train"]
                        else results[noise_label]["train"][validation_type]
                    )
                results[noise_label]["val"][validation_type] = (
                    {}
                    if validation_type not in results[noise_label]["val"]
                    else results[noise_label]["val"][validation_type]
                )

            # loop over combinations of lambda_x and lambda_y
            for comb in lambda_combinations:
                lambda_x, lambda_y = comb
                print(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                    f":Running validation for lambda_x: {lambda_x}, lambda_y: {lambda_y}"
                )

                cca = CCA()
                cca.fit_cca_svd(train_x, train_y, lambda_x=lambda_x, lambda_y=lambda_y)

                # predict on training
                if assess_train_metrics:
                    predicted_train_x = CCA.predict(
                        cca.T_x_full_inv_T,
                        cca.T_y_can,
                        train_val_y,
                        cca.CanCorr,
                        out_dim,
                        out_mean=x_mean,
                        probabilistic=probabilistic,
                        sample_size=prob_sample_size,
                    )

                # predict on validation
                predicted_val_x = CCA.predict(
                    cca.T_x_full_inv_T,
                    cca.T_y_can,
                    val_y_n,
                    cca.CanCorr,
                    out_dim,
                    out_mean=x_mean,
                    probabilistic=probabilistic,
                    sample_size=prob_sample_size,
                )

                # re-add mean
                val_x_d = val_x.copy() + x_mean

                # compute validation metrics
                for i, validation_type in enumerate(validation_types):

                    validation_param = validation_params[i]

                    if assess_train_metrics:
                        train_val_x_d = train_val_x.copy() + x_mean
                        print(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                            f"Running validation: {validation_type} with (train) predictions {predicted_train_x.shape}"
                        )
                        train_metrics = run_metrics(
                            predicted_train_x,
                            train_val_x_d,
                            validation_type,
                            validation_param,
                            reduced_sample_size=10 if probabilistic else None,
                        )

                        # if results[noise_label]['train'][validation_type][comb] exists, extend it
                        if comb in results[noise_label]["train"][validation_type]:
                            results[noise_label]["train"][validation_type][comb].extend(
                                train_metrics
                            )
                        else:
                            # else create it
                            results[noise_label]["train"][validation_type][
                                comb
                            ] = train_metrics

                    print(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                        f"Running validation: {validation_type} with (val) predictions {predicted_val_x.shape}"
                    )
                    val_metrics = run_metrics(
                        predicted_val_x,
                        val_x_d,
                        validation_type,
                        validation_param,
                        reduced_sample_size=10 if probabilistic else None,
                    )
                    # if results[noise_label]['val'][validation_type][comb] exists, extend it
                    if comb in results[noise_label]["val"][validation_type]:
                        results[noise_label]["val"][validation_type][comb].extend(
                            val_metrics
                        )
                    else:
                        # else create it
                        results[noise_label]["val"][validation_type][comb] = val_metrics

    return results


def run_inversion(
    lambda_x,
    lambda_y,
    probabilistic,
    prob_sample_size,
    train_subset_size,
    train_x,
    train_y,
    test_x,
    test_y,
    x_mean,
    noises_list,
    test_y_loaded_from_disk=False,
):
    """
    Executes the inversion process using the Regularized Canonical Correlation Analysis (RCCA) method
    to handle training and test data with possible noise perturbations. This method calculates
    projections and predictions for different noise levels, as specified by `noises_list`.

    It supports configurations for probabilistic sampling, subsets for training data,
    and adjustments for noise using lambda regularization parameters. Predictions and
    corresponding ground truth data are returned for each noise type.

    :param lambda_x: Regularization parameter(s) for the x dataset.
    :param lambda_y: Regularization parameter(s) for the y dataset.
    :param probabilistic: A flag indicating whether probabilistic modeling is enabled.
    :param prob_sample_size: Number of probabilistic samples to generate.
    :param train_subset_size: Size of the subset of the training dataset to use.
    :param train_x: Input training dataset for the x variables.
    :param train_y: Input training dataset for the y variables.
    :param test_x: Input test dataset for the x variables.
    :param test_y: Input test dataset for the y variables.
    :param x_mean: Mean of the x variables, used to adjust predictions.
    :param noises_list: List of noise configurations to apply during testing.
    :param test_y_loaded_from_disk: Flag indicating whether test_y data is preloaded with noise labels.

    :return: A tuple containing predictions dictionary and a dictionary of fitted RCCA objects, structured by noise labels.
    """

    # test whether lambda_x or lambda_y are vectors of size larger than 1 and assess if it is the same size as noises_list
    if isinstance(lambda_x, (list, np.ndarray)) and len(lambda_x) > 1:
        if len(lambda_x) != len(noises_list):
            raise ValueError(
                "lambda_x should be a vector of size equal to the number of noise types."
            )
    if isinstance(lambda_y, (list, np.ndarray)) and len(lambda_y) > 1:
        if len(lambda_y) != len(noises_list):
            raise ValueError(
                "lambda_y should be a vector of size equal to the number of noise types."
            )

    # if len(lambda_x) == 1 repeat it for all noise types
    if len(lambda_x) == 1:
        lambda_x = np.repeat(lambda_x, len(noises_list))
    if len(lambda_y) == 1:
        lambda_y = np.repeat(lambda_y, len(noises_list))

    out_dim = train_x.shape[1]
    full_train_size = train_x.shape[0]

    train_subset_indices = select_random_indices(
        full_train_size, train_subset_size, with_replacement=False
    )
    train_x = train_x[train_subset_indices, :]
    train_y = train_y[train_subset_indices, :]

    cca_objects = {}
    predictions = {}

    for i, noise_i in enumerate(noises_list):
        noise_test, noise_label = sample_noise(
            noise_i, test_x.shape[0], train_y.shape[1]
        )

        # train RCCA with the training data and the given lambda_x and lambda_y
        cca = CCA()
        cca.fit_cca_svd(train_x, train_y, lambda_x=lambda_x[i], lambda_y=lambda_y[i])

        cca_objects[noise_label] = cca
        predictions[noise_label] = {}

        if test_y_loaded_from_disk:
            test_y_n = test_y[noise_label]
        else:
            test_y_n = test_y.copy() + noise_test

        predicted_test_x = CCA.predict(
            cca.T_x_full_inv_T,
            cca.T_y_can,
            test_y_n,
            cca.CanCorr,
            out_dim,
            out_mean=x_mean,
            probabilistic=probabilistic,
            sample_size=prob_sample_size,
        )

        test_x_d = test_x.copy() + x_mean

        # combine the predictions and the ground truth test_x_d and add them to predictions dictionnary
        predictions[noise_label]["predicted"] = predicted_test_x
        predictions[noise_label]["ground_truth"] = test_x_d

    return predictions, cca_objects


def run_validation_eval(
    validation_data, xp_config_folder, reference_metrics_list=None, **kwargs
):
    """
    Runs the validation evaluation process, computes summary statistics, identifies the best combination of parameters,
    saves results to the disk, and generates boxplots for the validation data. This function is designed to process
    validation data organized hierarchically by noise types, train/validation splits, metric types, and lambda combinations.

    :param validation_data: The hierarchical structure containing validation data. The hierarchy should be:
        - First level: noise labels
        - Second level: train/validation split labels
        - Third level: metric types
        - Fourth level: lambda combinations
    :param xp_config_folder: Folder path where results (summaries, best parameter combinations, and plots) will be saved.
    :param reference_metrics_list: Optional list of reference metrics to include as visual guidance in boxplots.
    :param kwargs: Additional arguments to control the evaluation process:
        - 'ref_stat' (str): Reference statistic used for finding the best parameter combination. Defaults to 'median'.
        - 'reduce_lambda_y_vec' (bool): Whether to reduce lambda-y vectors in plots. Defaults to False.
        - 'lambda_y_subset' (int): Number of lambda-y values to include in subsets for plots. Defaults to 4.

    :return: None
    """
    import lrcca_inversion.utils.evals as evals

    ## get noise labels as first level keys
    noise_labels = list(validation_data.keys())
    ## get the second level keys (train/val)
    train_val_keys = list(validation_data[noise_labels[0]].keys())
    ## get the third level keys (metrics)
    metrics = list(validation_data[noise_labels[0]][train_val_keys[0]].keys())
    ## get the fourth level keys (lambda combinations)
    lambda_combinations = list(
        validation_data[noise_labels[0]][train_val_keys[0]][metrics[0]].keys()
    )

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
                    summaries[noise_label][train_val_key][metric][str(comb)] = (
                        evals.compute_stats(data)
                    )

                # Finding best combination for current metric
                best_comb[noise_label][train_val_key][metric] = (
                    evals.get_best_param_comb(
                        validation_data[noise_label][train_val_key],
                        metric,
                        ref_stat=kwargs.get("ref_stat", "median"),
                    )
                )

    # Save results
    save_to_disk(summaries, f"{xp_config_folder}/validation_summaries.json", json=True)
    save_to_disk(best_comb, f"{xp_config_folder}/best_param_comb.json", json=True)

    # make boxplots of the validation data for each noise type/train or val/metric type
    for noise_label in noise_labels:
        for train_val_key in train_val_keys:
            for metric in metrics:
                references_dict = (
                    evals.make_ref_dict(reference_metrics_list, metric, noise_label)
                    if reference_metrics_list
                    else None
                )
                data = validation_data[noise_label][train_val_key][metric]
                evals.make_val_boxplots(
                    data,
                    metric,
                    references_dict=references_dict,
                    save_path=f"{xp_config_folder}/{noise_label}_{train_val_key}_{metric}.pdf",
                    reduce_lambda_y_vec=kwargs.get("reduce_lambda_y_vec", False),
                    lambda_y_subset=kwargs.get("lambda_y_subset", 4),
                )


def run_inversion_eval(
    inversion_data, cca_obj, xp_config, config, reference_metrics, **kwargs
):
    """
    Runs the inversion evaluation process, including metric calculations, predictions comparison,
    transformation matrix plotting, and boxplot generation. Additionally, saves and loads intermediate
    results to/from disk to enable reproducibility and efficiency across multiple runs. This function
    handles stochastic predictions, deterministic predictions comparison, and resimulations, producing
    various diagnostics like inversion metrics or boxplots for analysis.

    :param inversion_data: Dictionary where each key corresponds to a noise label and maps
        the predictions, ground truths, and possibly deterministic predictions for different
        inversion cases. Predicted data is shaped as (n, dim, m).
    :param cca_obj: Dictionary containing Canonical Correlation Analysis (CCA)
        transformation matrices and relevant elements for each noise label.
    :param xp_config: Dictionary with information about the experiment configuration,
        including validation parameters, folder locations, and experimental IDs for test data.
    :param config: Object storing configuration details regarding shapes, dimensions, and solver matrices.
        It must include attributes `nx`, `ny`, `rays`, and `solver_matrix`.
    :param reference_metrics: Reference metrics to compare with during the evaluation process.
    :param kwargs: Additional optional parameters.
        - x_mean: Mean value used for X normalization or de-centralization (default: None).
        - y_mean: Mean value used for Y normalization or de-centralization (default: None).
        - assess_vs_det_pred: Boolean flag indicating whether to assess results against
          deterministic predictions (default: False).
        - assess_resims: Boolean flag indicating whether to assess resimulateions
          (default: False).

    :return: None
    """
    import lrcca_inversion.utils.evals as evals

    width = config.nx
    height = config.ny

    test_y = None

    x_mean = kwargs.get("x_mean", None)
    y_mean = kwargs.get("y_mean", None)
    assess_vs_det_pred = kwargs.get("assess_vs_det_pred", False)
    assess_resims = kwargs.get("assess_resims", False)

    if assess_vs_det_pred and y_mean is None:
        raise ValueError("y_mean is required to assess vs deterministic predictions.")

    if assess_resims and y_mean is None:
        raise ValueError("y_mean is required to assess resimulations.")

    noise_labels = list(inversion_data.keys())

    metrics_types = xp_config["validations"]["types"]
    metrics_params = xp_config["validations"]["params"]

    inversion_metrics = {}
    loaded_results = False

    ids_examples_to_plot = np.array([37, 12, 10])  # None #
    ids_samples_to_plot = None

    if Path(f"{xp_config['xp_folder']}/inversion_metrics.pkl").exists():
        inversion_metrics = load_from_disk(
            f"{xp_config['xp_folder']}/inversion_metrics.pkl"
        )
        loaded_results = True

    if assess_vs_det_pred:
        det_pred_past_run = (
            "det_prediction" in inversion_data[list(inversion_data.keys())[0]]
        )

        if not det_pred_past_run:
            det_preds_refs = {}
            test_y = get_y_obs_from_disk(
                xp_config["test_vecs_ids_to_invert"], config, y_mean
            )
            update_inversion_data_file = True
        else:
            if Path(f"{xp_config['xp_folder']}/det_preds_refs.pkl").exists():
                det_preds_refs = load_from_disk(
                    f"{xp_config['xp_folder']}/det_preds_refs.pkl"
                )
            update_inversion_data_file = False
    else:
        det_pred_past_run = False
        det_preds_refs = None
        update_inversion_data_file = False

    for noise_label in noise_labels:

        n = inversion_data[noise_label]["predicted"].shape[0]
        m = inversion_data[noise_label]["predicted"].shape[2]
        dim = inversion_data[noise_label]["predicted"].shape[1]

        if assess_vs_det_pred and not det_pred_past_run:
            inversion_data[noise_label]["det_prediction"] = CCA.predict(
                cca_obj[noise_label].T_x_full_inv_T,
                cca_obj[noise_label].T_y_can,
                test_y[noise_label],
                cca_obj[noise_label].CanCorr,
                dim,
                out_mean=x_mean,
                probabilistic=False,
                sample_size=1,
            )
            det_preds_refs[noise_label] = {}
            for i, metric in enumerate(metrics_types):
                det_preds_refs[noise_label][metric] = run_metrics(
                    inversion_data[noise_label]["det_prediction"],
                    inversion_data[noise_label]["ground_truth"],
                    metric,
                    metrics_params[i],
                    reduced_sample_size=None,
                )

        # compute metrics, if they were not already computed
        if not loaded_results:

            inversion_metrics[noise_label] = {}

            for i, metric in enumerate(metrics_types):
                inversion_metrics[noise_label][metric] = run_metrics(
                    inversion_data[noise_label]["predicted"],
                    inversion_data[noise_label]["ground_truth"],
                    metric,
                    metrics_params[i],
                    reduced_sample_size=None,
                )

        # plot predictions examples
        # select random indices
        examples_to_plot = min(n, 3)
        ids_examples_to_plot = (
            select_random_indices(n, examples_to_plot, with_replacement=False)
            if ids_examples_to_plot is None
            else ids_examples_to_plot
        )
        samples_per_example = min(m, 3)
        ids_samples_to_plot = (
            np.array(
                [
                    select_random_indices(
                        m, samples_per_example, with_replacement=False
                    )
                    for _ in range(len(ids_examples_to_plot))
                ]
            )
            if ids_samples_to_plot is None
            else ids_samples_to_plot
        )

        # get ground truth
        ground_truth = inversion_data[noise_label]["ground_truth"][
            ids_examples_to_plot, :
        ].reshape(len(ids_examples_to_plot), dim)
        ground_truth = np.expand_dims(ground_truth, axis=1)

        # get predictions and keep dims

        samples = np.array(
            [
                inversion_data[noise_label]["predicted"][ex_id, :, sample_ids]
                for ex_id, sample_ids in zip(ids_examples_to_plot, ids_samples_to_plot)
            ]
        )

        # concatenate the predictions and ground truth to form shape (examples_to_plot, samples_per_example + 1, dim)
        examples = np.concatenate((ground_truth, samples), axis=1)

        if "rmse" in metrics_types:
            # get rmse for each example
            # deaggregate the rmse values
            rmse_ids_to_read = (
                ((ids_examples_to_plot[:, np.newaxis] * m) + ids_samples_to_plot)
                .flatten()
                .astype(int)
            )
            rmse_values_flat = np.array(inversion_metrics[noise_label]["rmse"])[
                rmse_ids_to_read
            ]
            rmse_values = [
                rmse_values_flat[i : i + samples_per_example]
                for i in range(0, len(rmse_values_flat), samples_per_example)
            ]

        evals.plot_samples(
            examples,
            width,
            height,
            rmse_labels=rmse_values,
            grd_truth=True,
            save_location=f"{xp_config['xp_folder']}/{noise_label}_predictions_{ids_examples_to_plot}.pdf",
        )

        # plot transformation matrices
        # X

        evals.plot_transformations(
            cca_obj[noise_label].T_x_can,
            [config.ny, config.nx],
            groups=3,
            number_of_comp=3,
            save_location=f"{xp_config['xp_folder']}"
            + "/X_CanComp_{}"
            + f"_{noise_label}.pdf",
        )
        # Y
        evals.plot_transformations(
            cca_obj[noise_label].T_y_can,
            [1, config.rays],
            groups=3,
            number_of_comp=3,
            save_location=f"{xp_config['xp_folder']}"
            + "/Y_CanComp_{}"
            + f"_{noise_label}.pdf",
        )

    # save inversion_metrics to disk, to avoid recomputing them next time
    if not loaded_results:
        save_to_disk(
            inversion_metrics, f"{xp_config['xp_folder']}/inversion_metrics.pkl"
        )
    if not det_pred_past_run:
        save_to_disk(det_preds_refs, f"{xp_config['xp_folder']}/det_preds_refs.pkl")
    if update_inversion_data_file:
        save_to_disk(inversion_data, f"{xp_config['xp_folder']}/inversion_data.pkl")

    # plot inversion metrics boxplots

    boxplot_data_dict = {}
    references_dict = {}

    for metric in metrics_types:
        boxplot_data_dict[metric] = {}
        for i, noise_label in enumerate(noise_labels):
            boxplot_data_dict[metric][noise_label] = {}
            boxplot_data_dict[metric][noise_label]["prob"] = inversion_metrics[
                noise_label
            ][metric]
            if assess_vs_det_pred:
                boxplot_data_dict[metric][noise_label]["det"] = det_preds_refs[
                    noise_label
                ][metric]
            references_dict = evals.make_ref_dict(
                [reference_metrics], metric, noise_label
            )
            if i < len(noise_labels) - 1:
                continue
            else:
                evals.make_inv_boxplots(
                    boxplot_data_dict,
                    metric,
                    references_dict=references_dict,
                    save_path=f"{xp_config['xp_folder']}/test_inv_{metric}_dist.pdf",
                )

    # make y_obs vs D_{y_resim} metrics
    solver = config.solver_matrix  # shape (dim_y, dim_x)
    y_resim = {}
    y_resim_metrics = {}
    resim_boxplot_data_dict = {}
    loaded_y_resim = False

    if Path(f"{xp_config['xp_folder']}/y_resim.pkl").exists():
        y_resim = load_from_disk(f"{xp_config['xp_folder']}/y_resim.pkl")
        loaded_y_resim = True

    if Path(f"{xp_config['xp_folder']}/y_resim_metrics.pkl").exists():
        y_resim_metrics = load_from_disk(
            f"{xp_config['xp_folder']}/y_resim_metrics.pkl"
        )
        compute_resim_metrics = False
    else:
        compute_resim_metrics = True

    if compute_resim_metrics:
        if test_y is None:
            test_y = get_y_obs_from_disk(
                xp_config["test_vecs_ids_to_invert"], config, y_mean
            )

        for j, noise_label in enumerate(noise_labels):
            y_resim[noise_label] = np.zeros((n, config.rays, m))
            y_resim_metrics[noise_label] = {}
            test_y_d = test_y[noise_label].copy() + y_mean

            if not loaded_y_resim:
                for i in range(n):
                    predictions_i = inversion_data[noise_label]["predicted"][
                        i, :, :
                    ]  # shape (dim_x, m)
                    y_resim[noise_label][i, :, :] = (
                        solver @ predictions_i
                    )  # shape (dim_y, m)

            # compute metrics for y_obs vs D_{y_resim}
            for i, metric in enumerate(metrics_types):

                if metric not in resim_boxplot_data_dict:
                    resim_boxplot_data_dict[metric] = {}

                resim_boxplot_data_dict[metric][noise_label] = {}

                y_resim_metrics[noise_label][metric] = run_metrics(
                    y_resim[noise_label],
                    test_y_d,
                    metric,
                    metrics_params[i],
                    reduced_sample_size=None,
                )
                resim_boxplot_data_dict[metric][noise_label]["y_resim"] = (
                    y_resim_metrics[noise_label][metric]
                )

                if j < len(noise_labels) - 1:
                    continue
                else:
                    evals.make_inv_boxplots(
                        resim_boxplot_data_dict,
                        metric,
                        references_dict=None,
                        save_path=f"{xp_config['xp_folder']}/test_resims_{metric}_dist.pdf",
                    )

        # save to disk
        if not loaded_y_resim:
            save_to_disk(y_resim, f"{xp_config['xp_folder']}/y_resim.pkl")

        save_to_disk(y_resim_metrics, f"{xp_config['xp_folder']}/y_resim_metrics.pkl")


def run_val_reference_metrics(n, m, train_x, val_x, x_mean, metric_dict):
    """
    Executes validation metrics by comparing randomly selected subsets of training
    and validation data. The function selects `n` observations from the validation
    set and `m` predictions from the training set, computes reference metrics for
    the selected subsets, and combines them with the provided metrics dictionary.

    :param n: Number of random validation observations to select
    :param m: Number of random training predictions to select
    :param train_x: Training data array
    :param val_x: Validation data array
    :param x_mean: Mean or baseline value to add to the data for normalization
    :param metric_dict: Dictionary containing:
        - `types`: List of metric types to evaluate
        - `params`: List of parameters corresponding to each metric
    :return: Dictionary of computed metrics where keys are metric names and values
        are the computed metric results
    """
    total_train = train_x.shape[0]
    total_val = val_x.shape[0]
    dim = train_x.shape[1]

    metric_types = metric_dict["types"]
    metric_params = metric_dict["params"]

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
        metrics[metric] = run_metrics(
            train_x_d, val_x_d, metric, metric_params[i], reduced_sample_size=None
        )

    return metrics

def run_inv_reference_metrics(n, m, train_x, test_x, test_ids, x_mean, metric_dict):
    """
    Executes inverse reference metrics computation for a set of given training and test data.
    This function evaluates various metrics as specified using the metric dictionary, with optional
    customizations such as sample size for test data, features averaged by a mean component, and
    predicted values drawn from training data.

    :param n: Number of test samples to calculate metrics for. Defaults to 50 if not provided. If the
              length of `test_ids` exceeds this value, `n` is set to the maximum of the two.
    :param m: Number of predictions drawn from the training data to evaluate against test data.
    :param train_x: Array containing training data samples.
    :param test_x: Array containing test data samples.
    :param test_ids: Indices of test samples to be used, or None if indices should be randomly generated.
    :param x_mean: Mean values of features, added to both training and test data. Used to shift the data.
    :param metric_dict: Dictionary specifying metrics and their respective parameters. It includes:
                        - 'types': List of metric names as strings.
                        - 'params': List of parameters corresponding to each metric.
    :return: Dictionary of computed metrics, with metric names as keys and their corresponding results.
    """
    total_train = train_x.shape[0]

    n = 50 if n is None else n

    if test_ids is not None:
        if len(test_ids) > 49:  # 50 or more
            n = max(len(test_ids), n)

    dim = train_x.shape[1]

    metric_types = metric_dict["types"]
    metric_params = metric_dict["params"]

    if n > len(test_ids):
        # select the observations/ground truths. shape (n, dim)
        sample_test_indices = select_random_indices(
            test_x.shape[0], n, with_replacement=False
        )
        test_ids = sample_test_indices

    test_x = test_x[test_ids, :]
    test_x_d = test_x.copy() + x_mean

    # select the predictions and repeat them for each observation. shape (n, dim, m)
    sample_train_indices = select_random_indices(total_train, m, with_replacement=False)
    train_x_d = train_x[sample_train_indices, :].copy() + x_mean
    train_x_d = np.repeat(train_x_d.reshape(1, dim, m), n, axis=0)

    # compute metrics
    metrics = {}
    for i, metric in enumerate(metric_types):
        metrics[metric] = run_metrics(
            train_x_d, test_x_d, metric, metric_params[i], reduced_sample_size=None
        )

    return metrics

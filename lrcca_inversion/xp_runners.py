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
            samples = predicted[i, :, :].transpose()

            if metric == "rmse":
                if reduced_sample_size is not None:
                    reduced_sample_ids = select_random_indices(m, reduced_sample_size, with_replacement=False)
                    est_metric.extend(rmse(observation, samples[reduced_sample_ids, :]))
                else:
                    est_metric.extend(rmse(observation, samples))

            elif metric == "es":
                est_metric.append(es(observation, samples, metric_param))

            elif metric == "vs":
                est_metric.append(vs(observation, samples, metric_param))

    return est_metric

def run_validation(lambda_combinations, validations_dict, probabilistic, prob_sample_size, train_subset_size,
                   val_subset_size, train_x_orig, train_y_orig, val_x_orig, val_y_orig, x_mean, noises_list, add_val_noise=True, assess_train_metrics = False,
                   validation_repeats=1):
    """
    Run the validation with the combinations of lambda_x and lambda_y.
    """
    if val_x_orig is None:
        raise ValueError("Validation data is None. Please provide valid validation data.")

    validation_types = validations_dict['types']
    validation_params = validations_dict['params']
    out_dim = train_x_orig.shape[1]
    full_train_size = train_x_orig.shape[0]
    val_subset_size = val_subset_size if val_subset_size is not None else val_x_orig.shape[0]
    # val_subset_size: the maximum we will use during the validation, also from the training set (to limit execution time).
    # the training of CCA will be done with subset of size train_subset_size
    results = {}

    for _ in range(validation_repeats):
        print(f"Validation repeath number {_ + 1} of {validation_repeats}")

        # Select a random subset of the training data (for validation)
        if assess_train_metrics:
            train_val_subset_indices = select_random_indices(full_train_size, val_subset_size, with_replacement=False)
            train_val_x = train_x_orig[train_val_subset_indices, :]
            train_val_y = train_y_orig[train_val_subset_indices, :]

        # Select a random subset of the training data (for training)
        train_subset_indices = select_random_indices(full_train_size, train_subset_size, with_replacement=False)
        train_x = train_x_orig[train_subset_indices, :]
        train_y = train_y_orig[train_subset_indices, :]

        if val_subset_size < val_x_orig.shape[0]:
            # select random indices for the validation data
            val_subset_indices = select_random_indices(val_x_orig.shape[0], val_subset_size, with_replacement=False)
            val_x = val_x_orig[val_subset_indices, :]
            val_y = val_y_orig[val_subset_indices, :]

        # Loop over noise types
        for noise_i in noises_list:
            noise_val, noise_label = sample_noise(noise_i, val_y.shape[0], val_y.shape[1]) if add_val_noise else None

            # add noise to val_y if any, else copy original val_y
            val_y_n = val_y.copy() + noise_val if add_val_noise else val_y.copy()

            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                  f"Running validation for noise type:{noise_label}")

            # check if results is empty, if so, initialize it, else, leave it as is
            if noise_label not in results:
                results[noise_label] = {}
                if assess_train_metrics:
                    results[noise_label]['train'] = {}
                results[noise_label]['val'] = {}

            # make structure for all validation types
            for validation_type in validation_types:
                if assess_train_metrics:
                    results[noise_label]['train'][validation_type] = {} if validation_type not in results[noise_label]['train'] else results[noise_label]['train'][validation_type]
                results[noise_label]['val'][validation_type] = {} if validation_type not in results[noise_label]['val'] else results[noise_label]['val'][validation_type]

            # loop over combinations of lambda_x and lambda_y
            for comb in lambda_combinations:
                lambda_x, lambda_y = comb
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                      f":Running validation for lambda_x: {lambda_x}, lambda_y: {lambda_y}")

                cca = CCA()
                cca.fit_cca_svd(train_x, train_y, lambda_x=lambda_x, lambda_y=lambda_y)

                # predict on training
                if assess_train_metrics:
                    predicted_train_x = CCA.predict(cca.T_x_full_inv_T, cca.T_y_can, train_val_y, cca.CanCorr, out_dim,
                                                out_mean = x_mean, probabilistic = probabilistic, sample_size = prob_sample_size)

                # predict on validation
                predicted_val_x = CCA.predict(cca.T_x_full_inv_T, cca.T_y_can, val_y_n, cca.CanCorr, out_dim,
                                                out_mean = x_mean, probabilistic = probabilistic, sample_size = prob_sample_size)

                # re-add mean
                val_x_d = val_x.copy() + x_mean

                # compute validation metrics
                for i, validation_type in enumerate(validation_types):

                    validation_param = validation_params[i]

                    if assess_train_metrics:
                        train_val_x_d = train_val_x.copy() + x_mean
                        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                              f"Running validation: {validation_type} with (train) predictions {predicted_train_x.shape}")
                        train_metrics = run_metrics(predicted_train_x, train_val_x_d, validation_type, validation_param,
                                                    reduced_sample_size = 10 if probabilistic else None)

                        # if results[noise_label]['train'][validation_type][comb] exists, extend it
                        if comb in results[noise_label]['train'][validation_type]:
                            results[noise_label]['train'][validation_type][comb].extend(train_metrics)
                        else:
                            # else create it
                            results[noise_label]['train'][validation_type][comb] = train_metrics

                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                          f"Running validation: {validation_type} with (val) predictions {predicted_val_x.shape}")
                    val_metrics = run_metrics(predicted_val_x, val_x_d, validation_type, validation_param,
                                              reduced_sample_size = 10 if probabilistic else None)
                    # if results[noise_label]['val'][validation_type][comb] exists, extend it
                    if comb in results[noise_label]['val'][validation_type]:
                        results[noise_label]['val'][validation_type][comb].extend(val_metrics)
                    else:
                        # else create it
                        results[noise_label]['val'][validation_type][comb] = val_metrics

    return results

def run_inversion(lambda_x, lambda_y, probabilistic, prob_sample_size, train_subset_size, train_x, train_y,
                  test_x, test_y, x_mean, noises_list, test_y_loaded_from_disk=False):
    """
    Run the inversion with the given parameters.
    """

    # test whether lambda_x or lambda_y are vectors of size larger than 1 and assess if it is the same size as noises_list
    if isinstance(lambda_x, (list, np.ndarray)) and len(lambda_x) > 1:
        if len(lambda_x) != len(noises_list):
            raise ValueError("lambda_x should be a vector of size equal to the number of noise types.")
    if isinstance(lambda_y, (list, np.ndarray)) and len(lambda_y) > 1:
        if len(lambda_y) != len(noises_list):
            raise ValueError("lambda_y should be a vector of size equal to the number of noise types.")

    # if len(lambda_x) == 1 repeat it for all noise types
    if len(lambda_x) == 1:
        lambda_x = np.repeat(lambda_x, len(noises_list))
    if len(lambda_y) == 1:
        lambda_y = np.repeat(lambda_y, len(noises_list))

    out_dim = train_x.shape[1]
    full_train_size = train_x.shape[0]

    train_subset_indices = select_random_indices(full_train_size, train_subset_size, with_replacement=False)
    train_x = train_x[train_subset_indices, :]
    train_y = train_y[train_subset_indices, :]

    cca_objects = {}
    predictions = {}

    for i, noise_i in enumerate(noises_list):
        noise_test, noise_label = sample_noise(noise_i, test_x.shape[0], train_y.shape[1])

        # train RCCA with the training data and the given lambda_x and lambda_y
        cca = CCA()
        cca.fit_cca_svd(train_x, train_y, lambda_x=lambda_x[i], lambda_y=lambda_y[i])

        cca_objects[noise_label] = cca
        predictions[noise_label] = {}

        if test_y_loaded_from_disk:
            test_y_n = test_y[noise_label]
        else:
            test_y_n = test_y.copy() + noise_test

        predicted_test_x = CCA.predict(cca.T_x_full_inv_T, cca.T_y_can, test_y_n, cca.CanCorr, out_dim,
                                        out_mean=x_mean, probabilistic=probabilistic, sample_size=prob_sample_size)

        test_x_d = test_x.copy() + x_mean

        # combine the predictions and the ground truth test_x_d and add them to predictions dictionnary
        predictions[noise_label]['predicted'] = predicted_test_x
        predictions[noise_label]['ground_truth'] = test_x_d

    return predictions, cca_objects

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
                references_dict = evals.make_ref_dict(reference_metrics_list, metric, noise_label) if reference_metrics_list else None
                data = validation_data[noise_label][train_val_key][metric]
                evals.make_val_boxplots(data, metric, references_dict=references_dict,
                                        save_path=f"{xp_config_folder}/{noise_label}_{train_val_key}_{metric}.pdf",
                                        reduce_lambda_y_vec=kwargs.get('reduce_lambda_y_vec', False),
                                        lambda_y_subset=kwargs.get('lambda_y_subset', 4))

def run_inversion_eval(inversion_data, cca_obj, metrics, xp_config_folder, config, reference_metrics, **kwargs):
    import lrcca_inversion.utils.evals as evals

    width = config.nx
    height = config.ny

    #examples_to_plot = kwargs.get('s_examples_to_plot', 3)

    noise_labels = list(inversion_data.keys())

    metrics_types = metrics['types']
    metrics_params = metrics['params']

    results = {}
    ids_examples_to_plot = None
    ids_samples_to_plot = None

    for noise_label in noise_labels:

        n = inversion_data[noise_label]['predicted'].shape[0]
        m = inversion_data[noise_label]['predicted'].shape[2]
        dim = inversion_data[noise_label]['predicted'].shape[1]

        # compute metrics
        results[noise_label] = {}
        for i, metric in enumerate(metrics_types):
            results[noise_label][metric] = run_metrics(
                inversion_data[noise_label]['predicted'],
                inversion_data[noise_label]['ground_truth'],
                metric,
                metrics_params[i],
                reduced_sample_size=int(m // 2) if m > 1 else None
            )

        # plot predictions examples
        # select random indices
        examples_to_plot = 1 if n==1 else 4
        ids_examples_to_plot = select_random_indices(n, examples_to_plot, with_replacement=False) if ids_examples_to_plot is None else ids_examples_to_plot
        samples_per_example = min(m, 3)
        ids_samples_to_plot = np.array([select_random_indices(m, samples_per_example, with_replacement=False)
                               for _ in range(examples_to_plot)]) if ids_samples_to_plot is None else ids_samples_to_plot

        #ids_samples_to_plot = select_random_indices(m, samples_per_example, with_replacement=False) if m > 1 else 0

        # get ground truth
        ground_truth = inversion_data[noise_label]['ground_truth'][ids_examples_to_plot, :].reshape(examples_to_plot, dim)
        ground_truth = np.expand_dims(ground_truth, axis=1)

        # get predictions and keep dims

        samples = np.array([inversion_data[noise_label]['predicted'][ex_id, :, sample_ids]
                  for ex_id, sample_ids in zip(ids_examples_to_plot, ids_samples_to_plot)])

        # tranpose dim with samples_per_example
        #samples = samples.transpose(0, 2, 1)  # shape (examples_to_plot, samples_per_example, dim)

        # concatenate the predictions and ground truth to form shape (examples_to_plot, samples_per_example + 1, dim)
        examples = np.concatenate((ground_truth, samples), axis=1)

        if 'rmse' in metrics_types:
            # get rmse for each example
            # deaggregate the rmse values
            rmse_ids_to_read = ((ids_examples_to_plot[:, np.newaxis] * m) + ids_samples_to_plot).flatten().astype(int)
            rmse_values_flat = np.array(results[noise_label]['rmse'])[rmse_ids_to_read]
            rmse_values = [rmse_values_flat[i:i+samples_per_example]
                           for i in range(0, len(rmse_values_flat), samples_per_example)]

        evals.plot_samples(examples, width, height, rmse_labels=rmse_values,
                           grd_truth=True, save_location=f"{xp_config_folder}/{noise_label}_predictions_{ids_examples_to_plot}.pdf",)

        # plot transformation matrices
        # X
        evals.plot_transformations(cca_obj[noise_label].T_x_can,
            [config.ny, config.nx],
            groups=3,
            number_of_comp=3,
            save_location=xp_config_folder +"/X_CanComp_{}"+f"_{noise_label}.pdf")
        # Y
        evals.plot_transformations(cca_obj[noise_label].T_y_can,
            [1, config.rays],
            groups=3,
            number_of_comp=3,
            save_location=xp_config_folder + "/Y_CanComp_{}"+f"_{noise_label}.pdf")





    # plot metrics boxplots



    # get vs training reference

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

    pass

def run_val_reference_metrics(n, m, train_x, val_x, x_mean, metric_dict):
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
        metrics[metric] = run_metrics(train_x_d, val_x_d, metric, metric_params[i], reduced_sample_size = None)

    return metrics

def run_inv_reference_metrics(n, m, train_x, test_x, test_ids, x_mean, metric_dict):
    """
    Compute reference statistics against training data.
    n:
        number of samples from test_x (considered as observations/ground truths here). Will be used only if test_ids is None
    m:
        number of samples from train_x (considered as predictions here)
    train_x:
        training data (reference data)
    test_x:
        test data (observations)
    test_ids:
        indices of the test data to use
    x_mean:
        mean to add to the predictions
    metric_dict:
        dictionary of metrics to compute
    :return:
    """
    total_train = train_x.shape[0]

    n = 50 if n is None else n

    if test_ids is not None:
        if len(test_ids) > 49: # 50 or more
            n = max(len(test_ids), n)

    dim = train_x.shape[1]

    metric_types = metric_dict['types']
    metric_params = metric_dict['params']

    if n > len(test_ids):
        # select the observations/ground truths. shape (n, dim)
        sample_test_indices = select_random_indices(test_x.shape[0], n, with_replacement=False)
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
        metrics[metric] = run_metrics(train_x_d, test_x_d, metric, metric_params[i], reduced_sample_size = None)

    return metrics

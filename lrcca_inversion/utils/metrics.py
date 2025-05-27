"""
Metrics functions.

@author: elianemaalouf
"""

import numpy as np
from sklearn.metrics import pairwise_distances


def rmse(observation, samples):
    """
    Calculate the Root Mean Square Error (RMSE) between an observation and multiple samples.

    The RMSE is a standard way to measure the difference between a single
    observation and a set of predicted data points (samples). This function
    expects the observation to be a single sample and computes the RMSE for
    each sample in the given set of samples.

    :param observation: A single sample with dimensions (1, D) representing
        the true values.
    :param samples: A 2D array with dimensions (N, D), where N is the number
        of samples and D is the dimensionality of each sample.

    :return: A 1D array of shape (N,) containing the RMSE values for each
        sample in the samples array compared to the observation.
    """
    assert (
        observation.shape[1] == samples.shape[1]
    ), "Observation and samples must have the same dimension."
    assert observation.shape[0] == 1, "Observation must be a single sample."

    return np.sqrt(np.mean((observation - samples) ** 2, axis=1))


def ks(observation, samples, kernel=None, **kwargs):
    """
    Computes the Kernel Score (KS) as a measure of similarity between an observation and a set
    of samples, using a specified kernel function. The function calculates pairwise kernel values
    and determines the score based on the mean values of these computations.

    :param observation: A single data point of shape (1, d), where d is the dimension of the data.
    :param samples: A set of samples of shape (n, d), where n is the number of samples and d is
                    the dimension of the data.
    :param kernel: Callable or None, a kernel function to compute similarities. If None, the
                   function attempts to use the squared Euclidean distance from sklearn.
    :param kwargs: Additional keyword arguments to pass to the kernel function. Specific arguments
                   such as 'metric' and 'squared' may be included when using the default kernel.

    :return: The kernel score, computed as the difference between the mean kernel similarity of
             the observation and samples, and the pairwise kernel similarity within the samples.
    :rtype: float
    """
    # verify that the observation and the samples have the same dimension
    assert (
        observation.shape[1] == samples.shape[1]
    ), "Observation and samples must have the same dimension."
    assert observation.shape[0] == 1, "Observation must be a single sample."

    if kernel is None:
        # try to import the pairwise distance function from sklearn
        try:
            from sklearn.metrics import pairwise_distances

            kernel = pairwise_distances
            # add to **kwargs the metric to use and the squared flag
            kwargs["metric"] = "euclidean"
            kwargs["squared"] = True  # squared Euclidean distance
            print("Using the Euclidean distance as a kernel.")
        except:
            raise ImportError(
                "The kernel function is None and the import of the Euclidean function from sklearn failed. "
                "Please provide a kernel function."
            )

    n_samples = samples.shape[0]

    # compute the kernel between the observation and the samples
    k_obs = kernel(observation, samples, **kwargs).reshape(-1)
    k_obs = np.mean(k_obs)

    # computes pairwise kernel between samples
    K_pairwise = kernel(samples, samples, **kwargs).reshape(-1)
    K_pairwise = np.sum(K_pairwise) / (2 * n_samples**2)

    # compute the score
    return k_obs - K_pairwise


def es(observation, samples, power=2):
    """
    Compute the kernel score (KS) between an observation and samples using a specified
    distance metric and power. The function leverages the pairwise distances as the
    kernel for comparisons, switching between squared Euclidean distance for power of 2
    and Manhattan (L1) distance for power of 1.

    :param observation: Data point(s) to compare against the samples.

    :param samples: Array of sample data points for comparison with the observation.

    :param power: Power specifying the type of kernel distance to use.
        Defaults to 2. For power value of:
        - 2: squared Euclidean distance
        - 1: Manhattan (L1) distance

    :return: Returns the kernel score (KS) computed between the observation and the samples
        using the specified distance kernel.
    """
    if power == 2:
        kernel_es = pairwise_distances
        kwargs_es = {"metric": "euclidean", "squared": True}
    if power == 1:
        kernel_es = pairwise_distances
        kwargs_es = {"metric": "l1"}  # {"power": 1}

    return ks(observation, samples, kernel=kernel_es, **kwargs_es)


def vs(observation, samples, power=0.5, w=None):
    """
    Compute a score based on the pairwise differences between components of
    a single observation and multiple samples, weighted by a matrix.

    The function takes a single observation and a set of samples, calculates
    pairwise differences for their components raised to the specified power,
    and computes a weighted score based on these differences. A weight matrix
    is used for the computation, ensuring only non-negative values for weights.

    :param observation: A single observation with shape (1, d), where `d` is
        the dimensionality of the data.
    :param samples: A collection of samples with shape (n, d), where `n` is the
        number of samples and `d` is the dimensionality of the data.
    :param power: The power to which pairwise differences are raised. Defaults to 0.5.
    :param w: An optional weight matrix of shape (d, d). Defaults to a matrix with
        all elements set to 1. Must be non-negative if provided.
    :return: A scalar score computed based on the pairwise differences and weight matrix.
    """

    # verify that the observation and the samples have the same dimension
    assert (
        observation.shape[1] == samples.shape[1]
    ), "Observation and samples must have the same dimension."
    assert observation.shape[0] == 1, "Observation must be a single sample."

    n_samples = samples.shape[0]
    dim = samples.shape[1]

    # if w is None, fill it with ones
    if w is None:
        w = np.ones((dim, dim))
    else:
        # verify that none of the elements in w is negative and that w has the correct shape
        assert np.all(w >= 0) and w.shape == (dim, dim)

    # compute the pairwise differences between the observation components
    diff_obs = (
        np.abs((observation.reshape((1, dim)) - observation.reshape((dim, 1)))) ** power
    )

    # compute the pairwise differences between the samples' components
    diff_samples = np.mean(
        np.abs(
            (
                samples.reshape((n_samples, dim, 1))
                - samples.reshape((n_samples, 1, dim))
            )
        )
        ** power,
        axis=0,
    )

    # compute the score
    return np.sum(w * (diff_obs - diff_samples) ** 2)

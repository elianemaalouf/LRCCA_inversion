"""
Metrics functions.

@author: elianemaalouf
"""
import numpy as np
from sklearn.metrics import pairwise_distances

def rmse(observation, samples):
    """
    Compute the root mean square error (RMSE) between an observation and a sample.

    observation:
        observation vector of shape (1,dim), will be automatically broadcast to (n_samples,dim)

    sample:
        sample from the predictive distribution to test of shape (n_samples,dim)

    Returns:
        float: RMSE value.
    """
    assert observation.shape[1] == samples.shape[1], "Observation and samples must have the same dimension."
    assert observation.shape[0] == 1, "Observation must be a single sample."

    return np.sqrt(np.mean((observation - samples) ** 2, axis=1))

def ks(observation, samples, kernel=None, **kwargs):
    """
    Implements sample based estimation of the kernel score between the observation and the samples.

    observation:
        observation vector of shape (1,dim)
    samples:
        sample from the predictive distribution to test of shape (n_samples,dim)
    kernel:
        a function that computes pairwise kernel or distances between two arrays.
        e.g., sklearn.metrics.pairwise.rbf_kernel, sklearn.metrics.pairwise_distances etc.
        If None, the Euclidean distance is used.
    kwargs:
        additional arguments for the kernel function
    :return: kernel score
    """
    # verify that the observation and the samples have the same dimension
    assert observation.shape[1] == samples.shape[1], "Observation and samples must have the same dimension."
    assert observation.shape[0] == 1, "Observation must be a single sample."

    if kernel is None:
        # try to import the pairwise distance function from sklearn
        try:
            from sklearn.metrics import pairwise_distances

            kernel = pairwise_distances
            # add to **kwargs the metric to use and the squared flag
            kwargs["metric"] = "euclidean"
            kwargs["squared"] = True # squared Euclidean distance
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
    Implements the energy score based on the generic kernel score function.
    observation:
        observation vector of shape (1,dim)
    samples:
        sample from the predictive distribution to test of shape (n_samples,dim)
    power:
        power for the distance
    :return: energy score between the observation and the samples
    """
    if power == 2:
        kernel_es = pairwise_distances
        kwargs_es = {"metric": "euclidean", "squared": True}
    if power == 1:
        kernel_es = pairwise_distances
        kwargs_es = {"metric": 'l1'}  # {"power": 1}

    return ks(observation, samples, kernel=kernel_es, **kwargs_es)

def vs(observation, samples, power=0.5, w=None):
    """
    Implements empirical estimation of the variogram score

    observation:
        observation vector of shape (1,dim)
    samples:
        sample from the predictive distribution to test of shape (n_samples,dim)
    power:
        power for the distance
    w:
        weights matrix of shape (dim,dim)
    :return: variogram score between the observation and the samples
    """

    # verify that the observation and the samples have the same dimension
    assert observation.shape[1] == samples.shape[1], "Observation and samples must have the same dimension."
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

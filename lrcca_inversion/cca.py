# -*- coding: utf-8 -*-
"""
Module implementing Linear Canonical Correlation Analysis (CCA) using Singular Value Decomposition (SVD).
It provides methods for fitting the CCA model, transforming data into canonical variates,
reconstructing original data from canonical variates, and predicting one variable from another using
the canonical transformation matrices, both deterministically and probabilistically.
The code also provides a method to validate the regularization of the covariance matrices.

@author: elianemaalouf
"""

import numpy as np
import numpy.linalg as LA
from scipy import linalg as sLA


def _destandardize(_x, dim, _mean=None, _std=None):
    if _mean is None:
        _mean = np.zeros(dim)
    if _std is None:
        _std = np.ones(dim)
    _mean = _mean.reshape(-1, 1)
    _std = _std.reshape(-1, 1)
    return _x * _std + _mean


class CCA:
    def __init__(self):
        """
        Class to hold the attributes necessary for canonical correlation analysis (CCA). The class
        includes transformation matrices, canonical correlation values, data covariance matrices,
        and regularization parameters.

        Attributes
        ----------
        x_dim : int or None
            Dimension of variable x (denoted as `d`).
        y_dim : int or None
            Dimension of variable y (denoted as `p`).
        r : int or None
            Canonical space dimension.
        T_x_full : Any or None
            Full transformation matrix for x (denoted as `A`).
        T_y_full : Any or None
            Full transformation matrix for y (denoted as `B`).
        T_x_full_inv_T : Any or None
            Transposed inverse of the full transformation matrix for x (denoted as `A^{-1}.T`).
        T_y_full_inv_T : Any or None
            Transposed inverse of the full transformation matrix for y (denoted as `B^{-1}.T`).
        T_x_can : Any or None
            Canonical transformation matrix for x only.
        T_y_can : Any or None
            Canonical transformation matrix for y only.
        T_x_can_inv_T : Any or None
            Inverse transformation of x canonical transformations only.
        T_y_can_inv_T : Any or None
            Inverse transformation of y canonical transformations only.
        CanCorr : Any or None
            Canonical correlation values between canonical variables.
        DataCov_noreg : Any or None
            Full original data covariance matrix (unregularized).
        C_x : Any or None
            Covariance matrix of x.
        C_y : Any or None
            Covariance matrix of y.
        C_xy : Any or None
            Cross-covariance matrix between x and y.
        lambda_x : float
            Regularization parameter for x.
        lambda_y : float
            Regularization parameter for y.
        null_dims : Any or None
            Dimensions associated with null space components.
        """
        self.x_dim = None  # dimension of x, i.e., d
        self.y_dim = None  # dimension of y, i.e., p
        self.r = None  # canonical space dimension
        self.T_x_full = None  # full transformation matrix for x, i.e., A
        self.T_y_full = None  # full transformation matrix for y, i.e., B
        self.T_x_full_inv_T = None  # transposed inverse of the full transformation matrix for x, i.e., A^{-1}.T
        self.T_y_full_inv_T = None  # transposed inverse of the full transformation matrix for y, i.e., B^{-1}.T
        self.T_x_can = None  # canonical transformations only
        self.T_y_can = None  # canonical transformation only
        self.T_x_can_inv_T = None  # inverse of x canonical transformations only
        self.T_y_can_inv_T = None  # inverse of y canonical transformations only
        self.CanCorr = None  # canonical correlation values
        self.DataCov_noreg = None  # full original data covariance matrix
        self.C_x = None
        self.C_y = None
        self.C_xy = None
        self.lambda_x = 0
        self.lambda_y = 0
        self.null_dims = None

    def fit_cca_svd(self, X, Y, lambda_x=0, lambda_y=0):
        """
        Performs Canonical Correlation Analysis (CCA) using Singular Value Decomposition (SVD).
        CCA is a statistical technique used to study the relationships between two sets of
        variables, finding linear combinations of variables from each set that are maximally
        correlated with each other. The function computes the canonical correlation values
        along with the transformation matrices for the input datasets `X` and `Y`.

        :param X: The first dataset of shape (n_samples, n_features_X).
        :param Y: The second dataset of shape (n_samples, n_features_Y).
        :param lambda_x: Regularization parameter for the X dataset, default is 0.
        :param lambda_y: Regularization parameter for the Y dataset, default is 0.
        :return: None. All results are stored as attributes of the instance.
        """

        n, d = X.shape
        n, p = Y.shape
        self.r = min(LA.matrix_rank(X), LA.matrix_rank(Y))
        self.x_dim = d
        self.y_dim = p
        # number of dimensions that are not part of the CCA
        self.null_dims = max(self.x_dim - self.r, self.y_dim - self.r)

        self.lambda_x = lambda_x
        self.lambda_y = lambda_y

        C = np.cov(X.T, Y.T)
        C_xx = C[:d, :d] + lambda_x * np.eye(d)
        C_yy = C[d:, d:] + lambda_y * np.eye(p)
        C_xy = C[:d, d:]

        self.DataCov_noreg = C  # full covariance matrix - non-regularized
        self.C_x = C_xx  # covariance matrix of X, regularized if any
        self.C_y = C_yy  # covariance matrix of Y, regularized if any
        self.C_xy = C_xy  # covariance matrix of X and Y
        self.dataset_size = n

        # check first if full-rank:
        if np.linalg.matrix_rank(C_xx) < d or np.linalg.matrix_rank(C_yy) < p:
            raise ValueError(
                "CCA SVD: Covariance matrices are not full-rank. Cannot perform CCA."
            )
        else:
            self.eig_C_x_vals, self.eig_C_x_vecs = np.linalg.eigh(C_xx)
            self.eig_C_y_vals, self.eig_C_y_vecs = np.linalg.eigh(C_yy)

            # squareroot of the inverse of the covariance matrices
            Cxx_sqrt_inv = (
                self.eig_C_x_vecs
                @ np.diag(1 / np.sqrt(self.eig_C_x_vals))
                @ (self.eig_C_x_vecs).T
            )

            Cyy_sqrt_inv = (
                self.eig_C_y_vecs
                @ np.diag(1 / np.sqrt(self.eig_C_y_vals))
                @ (self.eig_C_y_vecs).T
            )

            # check if Cxx_sqrt_inv and Cyy_sqrt_inv contain complex numbers
            # this should not happen with eigh(), we noticed it happened with sLA.sqrtm()
            if np.iscomplexobj(Cxx_sqrt_inv) or np.iscomplexobj(Cyy_sqrt_inv):
                print("CCA SVD: Square-root of covariances matrices are complex.")
                # check if complex part is small enough
                if (
                    np.max(np.abs(np.imag(Cxx_sqrt_inv))) < 1e-10
                    and np.max(np.abs(np.imag(Cyy_sqrt_inv))) < 1e-10
                ):
                    # if so, take only the real part
                    Cxx_sqrt_inv = np.real(Cxx_sqrt_inv)
                    Cyy_sqrt_inv = np.real(Cyy_sqrt_inv)
                    print(
                        "CCA SVD: Taking only the real part of the square-root of the covariance matrices since "
                        "the imaginary part is small (<1e-10)."
                    )
                else:
                    raise ValueError(
                        "CCA SVD: Imaginary part of the square-root of the covariance matrices is too large. "
                        "Cannot perform CCA."
                    )

        M = np.matmul(np.matmul(Cxx_sqrt_inv, C_xy), Cyy_sqrt_inv)

        svd_M = sLA.svd(
            M, full_matrices=True
        )  # returns U (dxr | dx(d-r)), S (rxr), V^T(rxp|(p-r)xp)
        # when p = r and d> p => U (dxr | dx(d-r)), S (rxr), V^T(rxr)

        self.U_full = svd_M[0]
        self.V_full = svd_M[2].T  # transpose to get V(rxr)
        self.T_x_full = Cxx_sqrt_inv @ self.U_full  # Full X transformations
        self.T_y_full = Cyy_sqrt_inv @ self.V_full  # Full Y transformations
        self.CanCorr = svd_M[1]  # Canonical correlation values

        # limit CanCorr to maximum 1 (numerical errors can make it slightly larger than 1)
        self.CanCorr[self.CanCorr > 1] = 1

        # canonical X transformations, first r columns
        self.T_x_can = self.T_x_full[:, : self.r]
        # canonical Y transformations, first r columns
        self.T_y_can = self.T_y_full[:, : self.r]

        # Make transposed inverses of the full transformation matrices
        C_xx_sqrt = (
            self.eig_C_x_vecs
            @ np.diag(np.sqrt(self.eig_C_x_vals))
            @ (self.eig_C_x_vecs).T
        )
        self.T_x_full_inv_T = C_xx_sqrt @ self.U_full

        C_yy_sqrt = (
            self.eig_C_y_vecs
            @ np.diag(np.sqrt(self.eig_C_y_vals))
            @ (self.eig_C_y_vecs).T
        )
        self.T_y_full_inv_T = C_yy_sqrt @ self.V_full

        self.all_can_corr = self.CanCorr

        self.T_x_can_inv_T = self.T_x_full_inv_T[
            :, : self.r
        ]  # keep only the columns that are not part of the null space
        self.T_y_can_inv_T = self.T_y_full_inv_T[:, : self.r]

    @staticmethod
    def reduce(data, T_matrix):
        """
        Reduces the dimensionality of the given data based on the transformation matrix.

        This method applies a matrix multiplication between the transpose of the
        transformation matrix and the transpose of the data. The result is reshaped
        to a specific dimension based on the number of rows in the transformation
        matrix. The primary purpose of this function is to generate a lower-dimensional
        representation of the data while retaining significant variance as defined
        by the transformation matrix.

        :param data: The original data to be transformed. Must be a 2-dimensional
            array where each row represents a data sample and each column represents
            a feature.
        :param T_matrix: Transformation matrix used to reduce the dimensionality of
            the data. The number of columns in this matrix must match the number
            of columns in the data.
        :return: The reduced dimensional representation of the input data, reshaped
            to the number of rows in the transformation matrix and the corresponding columns.
        """
        # can_variates = T_matrix.T * data.T
        can_variates = np.matmul(T_matrix.T, data.T)
        return can_variates.reshape(T_matrix.shape[1], -1)

    @staticmethod
    def reconstruct(
        T_matrix_inv_T, can_variates, data_dim, data_mean=None, data_std=None
    ):
        """
        Reconstructs the original data based on the provided inverse transformation
        matrix, canonical variates, and optional de-standardization parameters.

        This method computes the original data by applying the inverse transformation
        to the canonical variates, optionally de-standardizing the data using provided
        mean and standard deviation, and then reshaping it to the specified dimensions.

        :param T_matrix_inv_T: The inverse of the transformation matrix transposed,
            used to revert the canonical variates to the original data space.

        :param can_variates: The transformed data in the canonical variates space
            which needs to be reverted.

        :param data_dim: The dimensions to which the reconstructed data should be
            reshaped after reversion, specified as a tuple.

        :param data_mean: The mean values used for de-standardization of the data.
            This is optional and used if the data was standardized earlier. Defaults to None.

        :param data_std: The standard deviation values used for de-standardization
            of the data. This is optional and used if the data was standardized earlier. Defaults to None.

        :return: The reconstructed data reshaped to the specified dimensions.
        """
        # can_variates = T_matrix.T * data.T
        # data = T_matrix_inv_T * can_variates

        data = np.matmul(T_matrix_inv_T, can_variates)

        data = _destandardize(data, dim=data_dim, _mean=data_mean, _std=data_std)

        return data.reshape(data_dim, -1)

    @staticmethod
    def make_z_posterior(z_obs, canCorr, null_dims, square_root_cov=True):
        """
        Creates the posterior distribution for canonical variates using the observed data,
        canonical correlations, and specified dimensions of null space.

        :param z_obs: Observed canonical variates. Shape (r,) or (r, 1).
        :param canCorr: Canonical correlations of shape (r,).
        :param null_dims: Number of null dimensions to append to the model.
        :param square_root_cov: Indicates whether to return the square root of the covariance
            matrix (True) or the full covariance matrix (False). Default is True.
        :return: Tuple containing:
            - z_mean: Mean vector of the canonical variates. Shape (output_dim, 1).
            - z_cov: Either the square root of the covariance matrix (if square_root_cov=True)
              or the full covariance matrix (if square_root_cov=False). Shape (output_dim, output_dim).
        """
        r = len(canCorr)
        z_obs = z_obs.reshape(r, 1)  # shape (r x 1)
        # make canonical variates mean vector
        z_mean = np.matmul(np.diag(canCorr), z_obs)  # shape (r x 1)
        z_mean = np.append(z_mean, np.zeros((null_dims, 1)), axis=0).reshape(
            -1, 1
        )  # shape (out_dim x 1)

        # make canonical variates square root of the covariance matrix
        z_sq_cov = np.sqrt(np.eye(r) - np.diag(canCorr**2))
        z_sq_cov = np.append(z_sq_cov, np.zeros((r, null_dims)), axis=1)
        z_sq_cov = np.append(
            z_sq_cov,
            np.append(np.zeros((null_dims, r)), np.eye(null_dims), axis=1),
            axis=0,
        )
        if square_root_cov:
            # return the square root of the covariance matrix
            return z_mean, z_sq_cov
        else:
            return z_mean, z_sq_cov @ z_sq_cov.T

    @staticmethod
    def predict(
        T_matrix_out_inv_T,
        T_matrix_in,
        new_in_data,
        canCorr,
        out_dim,
        out_mean=None,
        out_std=None,
        probabilistic=False,
        **kwargs
    ):
        """
        Predict the outputs using input data and precomputed projection matrices, canonical correlations,
        and other configurations. This function allows for deterministic or probabilistic prediction generation
        depending on the value of `probabilistic`. For probabilistic predictions, sampling from the posterior
        distribution is performed based on provided or default configurations.

        :param T_matrix_out_inv_T: Precomputed transformation matrix for output space.
        :param T_matrix_in: Precomputed transformation matrix for input space.
        :param new_in_data: The new input data samples for which predictions are made.
        :param canCorr: Array of canonical correlation coefficients computed during the training phase.
        :param out_dim: Dimensionality of the output space or the number of reconstructed output features.
        :param out_mean: Mean vector for output features used for reconstruction. Defaults to None.
        :param out_std: Standard deviation vector for output features used for reconstruction. Defaults to None.
        :param probabilistic: Whether to perform probabilistic prediction. Defaults to False.
        :param kwargs: Additional parameters such as `sample_size` for probabilistic sampling.
        :return: A numpy array containing the predictions. The shape depends on the probabilistic flag.
                 For deterministic prediction: (num_samples, out_dim, 1).
                 For probabilistic prediction: (num_samples, out_dim, sample_size).
        """
        in_samples = new_in_data.shape[0]
        r = len(canCorr)
        null_dims = out_dim - r
        new_in_can_variates = CCA.reduce(new_in_data, T_matrix_in)

        if not probabilistic:
            sample_size = 1
            # assess if T_matrix_in column numbers are equal to the number of canonical correlations
            if T_matrix_in.shape[1] != r:
                raise ValueError(
                    "T_matrix_in should have the same number of columns as the number of canonical correlations."
                )

            # complement with zeros the canonical variates (if mistmatched dimensions)
            new_out_can_variates = np.matmul(np.diag(canCorr), new_in_can_variates)
            new_out_can_variates = np.append(
                new_out_can_variates, np.zeros((null_dims, in_samples)), axis=0
            )

            predictions = CCA.reconstruct(
                T_matrix_out_inv_T, new_out_can_variates, out_dim, out_mean, out_std
            )

            return predictions.T.reshape(in_samples, out_dim, sample_size)
        else:
            # read sample_size from kwargs if any, if not set to 100
            sample_size = kwargs.get("sample_size", 100)

            def sample_z_posterior(new_in_can_var_vec):
                z_mean, z_sq_cov = CCA.make_z_posterior(
                    new_in_can_var_vec, canCorr, null_dims, square_root_cov=True
                )

                # sample canonical variates from the posterior distribution by sampling from standard normal and transforming
                z_can = z_mean + z_sq_cov @ np.random.randn(
                    out_dim, sample_size
                )  # shape (out_dim x sample_size)

                return z_can

            predictions = np.zeros((in_samples, out_dim, sample_size))
            for i in range(in_samples):
                new_in_can_var_vec = new_in_can_variates[:, i]
                z_can = sample_z_posterior(new_in_can_var_vec)
                predictions[i, :, :] = CCA.reconstruct(
                    T_matrix_out_inv_T, z_can, out_dim, out_mean, out_std
                )

            return predictions


def test_probcca_vs_analytical(cca_obj, new_y_c):
    """
    Function to verify that the probabilistic CCA mean and covariance are equal to the analytical solution ones.

    cca_obj
        A CCA object fitted with the training set
    new_y_c
        A new input data to be inverted, e,g. y_obs. It should be mean-centered.
    """

    # make analytical solution mean and covariance
    C_y_inv = (
        cca_obj.eig_C_y_vecs
        @ np.diag(1 / cca_obj.eig_C_y_vals)
        @ (cca_obj.eig_C_y_vecs).T
    )

    analytical_mean = np.matmul(np.matmul(cca_obj.C_xy, C_y_inv), new_y_c.T).reshape(-1)
    analytical_cov = cca_obj.C_x - cca_obj.C_xy @ C_y_inv @ (cca_obj.C_xy).T

    analytical_sample = np.random.multivariate_normal(
        analytical_mean, analytical_cov, size=5000
    )
    est_analytical_mean = np.mean(analytical_sample, axis=0)
    est_analytical_cov = np.cov(analytical_sample.T)

    # make probabilistic solution mean and covariance
    z_obs = CCA.reduce(new_y_c, cca_obj.T_y_can)
    prob_cca_mean_z, prob_cca_cov_z = CCA.make_z_posterior(
        z_obs, cca_obj.CanCorr, cca_obj.null_dims, square_root_cov=False
    )

    prob_cca_mean = np.matmul(cca_obj.T_x_full_inv_T, prob_cca_mean_z).reshape(-1)
    prob_cca_cov = cca_obj.T_x_full_inv_T @ prob_cca_cov_z @ (cca_obj.T_x_full_inv_T).T

    prob_cca_sample = np.random.multivariate_normal(
        prob_cca_mean, prob_cca_cov, size=5000
    )
    est_prob_cca_mean = np.mean(prob_cca_sample, axis=0)
    est_prob_cca_cov = np.cov(prob_cca_sample.T)

    # check if the means are equal
    assert np.allclose(
        analytical_mean, prob_cca_mean, atol=1e-10
    ), "Means are not equal"
    # check if the covariances are equal
    assert np.allclose(
        analytical_cov, prob_cca_cov, atol=1e-10
    ), "Covariances are not equal"

    # estimations based on sampled data do not have to be equal, uncomment the next lines to check

    # assert np.allclose(est_analytical_mean, analytical_mean, atol=1e-10), "Analytical mean and estimated means are not equal"
    # assert np.allclose(est_analytical_cov, analytical_cov, atol=1e-10), "Analytical covariance and estimated covariances are not equal"

    # assert np.allclose(est_prob_cca_mean, prob_cca_mean, atol=1e-10), "Probabilistic mean and estimated means are not equal"
    # assert np.allclose(est_prob_cca_cov, prob_cca_cov, atol=1e-10), "Probabilistic covariance and estimated covariances are not equal"

    # assert np.allclose(est_analytical_mean, est_prob_cca_mean, atol=1e-10), "Estimated means are not equal"
    # assert np.allclose(est_analytical_cov, est_prob_cca_cov, atol=1e-10), "Estimated covariances are not equal"


# test
if __name__ == "__main__":
    # test CCA
    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 5)
    x_mean = np.mean(X, axis=0)
    y_mean = np.mean(Y, axis=0)
    # mean center the data
    X = X - x_mean
    Y = Y - y_mean

    cca = CCA()
    cca.fit_cca_svd(X, Y, lambda_x=0.1, lambda_y=0.1)

    print("Canonical correlation values: ", cca.CanCorr)
    print("Canonical variates of X: ", cca.T_x_can)
    print("Canonical variates of Y: ", cca.T_y_can)

    # test reduce
    Z_x = CCA.reduce(X, cca.T_x_can)
    Z_y = CCA.reduce(Y, cca.T_y_can)

    print("Canonical variates of X: ", Z_x)
    print("Canonical variates of Y: ", Z_y)
    # test reconstruct
    X_reconstructed = CCA.reconstruct(cca.T_x_can_inv_T, Z_x, cca.x_dim)
    Y_reconstructed = CCA.reconstruct(cca.T_y_can_inv_T, Z_y, cca.y_dim)

    print("Reconstructed X: ", X_reconstructed)
    print("Reconstructed Y: ", Y_reconstructed)

    # test predict
    new_in_data = np.random.randn(1, 5)
    new_in_data = new_in_data - y_mean

    predictions = CCA.predict(
        cca.T_x_full_inv_T, cca.T_y_can, new_in_data, cca.CanCorr, cca.x_dim
    )
    print("Predictions: ", predictions)
    # test predict with probabilistic
    predictions_prob = CCA.predict(
        cca.T_x_full_inv_T,
        cca.T_y_can,
        new_in_data,
        cca.CanCorr,
        cca.x_dim,
        probabilistic=True,
        sample_size=20,
    )
    print("Probabilistic predictions: ", predictions_prob)

    test_probcca_vs_analytical(cca, new_in_data)

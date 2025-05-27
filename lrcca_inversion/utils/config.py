# -*- coding: utf-8 -*-
"""
@author: elianemaalouf
"""

import ast
import random

import numpy as np
import torch


def get_prior(h5_file, mu):
    """
    Get prior information including covariance matrix, mean vector, and
    the square root of the covariance matrix for a Gaussian field prior.

    This function reads an HDF5 file containing a Gaussian covariance matrix,
    applies Cholesky decomposition to compute its square root, and constructs
    a mean vector filled with a specified constant value.

    :param h5_file: Path to the HDF5 file containing the Gaussian covariance matrix.
    :param mu: The value to populate the mean vector with.
    :return: A dictionary containing:
        - **ntot** (*int*): Dimensionality of the covariance matrix.
        - **covariance_matrix** (*np.ndarray*): The covariance matrix extracted from the HDF5 file.
        - **covM_squareRoot** (*np.ndarray*): Cholesky decomposition (square root) of the covariance matrix.
        - **prior_mean** (*np.ndarray*): A mean vector for the prior distribution.
    """
    import h5py

    cov_matrix_file = h5py.File(h5_file)
    cov_matrix = cov_matrix_file.get("GaussianCovarianceMatrix")
    cov_matrix = np.array(cov_matrix)
    cov_matrix_file.close()

    dim = cov_matrix.shape[0]
    # create mean vector
    m_prior = np.repeat(mu, dim)  # Gaussian filed prior mean

    # cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)  # square root of prior covariance matrix

    return {
        "ntot": dim,
        "covariance_matrix": cov_matrix,
        "covM_squareRoot": L,
        "prior_mean": m_prior,
    }


def get_linear_solver(h5_file):
    """
    Retrieve a linear solver matrix from an HDF5 file and return it along with
    the number of data points (rows) in the matrix.

    The function reads an HDF5 file specified by the provided file path to
    extract a dataset named "LinearSolverMatrix." The dataset is converted
    into a NumPy array and returned alongside its dimensions.

    :param h5_file: The file path of the HDF5 file containing the linear solver matrix.
    :return: A dictionary containing the solver matrix under the key "solver_matrix"
        and the number of rows of the matrix under the key "ndata".
    """
    import h5py

    F_matrix_file = h5py.File(h5_file)
    F_matrix = F_matrix_file.get("LinearSolverMatrix")
    F_matrix = np.array(F_matrix)
    F_matrix_file.close()

    return {"solver_matrix": F_matrix, "ndata": F_matrix.shape[0]}


class Config:
    """Class to read environment configuration parameters for Goephysics applications."""

    def __init__(self, parameters_file, setup_solver=True, setup_prior=True):
        """
        Initializes the class instance by loading configuration parameters from a file,
        fixing random seeds, and optionally setting up the prior mean and covariance,
        along with a forward solver if required. The configuration parameters are read
        from the provided file and used to configure various attributes related to
        Gaussian fields, solver setup, data directories, and model characteristics.

        :param parameters_file: Path to the file containing model configuration parameters.
        :param setup_solver: A flag indicating whether to set up the forward solver. Defaults to True.
        :param setup_prior: A flag indicating whether to set up the prior mean and covariance matrix. Defaults to True.

        :raises ValueError: If the solver type specified in the configuration file is not supported.
        """
        with open(parameters_file) as f:
            self.parameters = f.read()

        self.parameters = ast.literal_eval(self.parameters)

        # Read paramters:
        self.cov_kernel = self.parameters["Gaussian covariance kernel"]
        self.nc = self.parameters["Number of channels in subsurface images"]
        self.nx = self.parameters["Number of cells on the horizontal axis (pixels)"]
        self.ny = self.parameters["Number of cells on the vertical axis (pixels)"]
        self.spacing = self.parameters["Width of a grid cell in meters"]
        self.lxtrue = self.parameters["Horizontal correlation length (pixels)"]
        self.lytrue = self.parameters["Vertical correlation length (pixels)"]
        self.sigma2 = self.parameters["Gaussian field prior variance"]
        self.mu = self.parameters["Gaussian field prior mean"]
        self.jitter = self.parameters["Covariance matrix jitter"]
        self.solver_type = self.parameters["solver_type"]
        self.sources_x = self.parameters["Sources x position"]
        self.rays = self.parameters["rays"]
        self.set_size = self.parameters["Dataset size"]
        self.train_split = self.parameters["train_split"]
        self.val_split = self.parameters["val_split"]
        self.noises_list = self.parameters["All noises configurations"]
        self.rootdir = self.parameters["rootdir"]
        self.ndata = self.rays
        self.ntot = self.nx * self.ny

        # fix seed for all random number generators:
        np.random.seed(self.parameters["FixedSeed"])
        os.environ["PYTHONHASHSEED"] = str(self.parameters["FixedSeed"])
        random.seed(self.parameters["FixedSeed"])
        torch.manual_seed(self.parameters["FixedSeed"])
        torch.cuda.manual_seed_all(self.parameters["FixedSeed"])
        np.random.seed(self.parameters["FixedSeed"])
        print("Random seed fixed to:{}".format(self.parameters["FixedSeed"]))

        ## setup directories and import necessary files
        self.datadir = self.rootdir + "/Data"
        self.data_folder_location = (
            self.datadir
            + "/{}_Mu{}_Var{}_CorH{}_CorV{}_{}_{}".format(
                self.cov_kernel,
                self.mu,
                str(round(self.sigma2, 2)).replace(".", "p"),
                self.lxtrue,
                self.lytrue,
                self.solver_type,
                self.ndata,
            )
        )

        if setup_prior:
            # set up prior mean and covariance
            prior = get_prior(f"{self.data_folder_location}/GaussCovMatrix.h5", self.mu)
            self.CM = prior[
                "covariance_matrix"
            ]  # Gaussian field prior covariance matrix
            self.L = prior["covM_squareRoot"]  # square root of prior covariance matrix
            self.m_prior = prior["prior_mean"]  # Gaussian filed prior mean
            self.ntot = prior["ntot"]

        if setup_solver:
            # setup forward solver _ only linear solver here
            if self.solver_type == "linear":
                self.solver_setup_dict = get_linear_solver(
                    f"{self.data_folder_location}/linearForwardMatrix.h5"
                )
                self.solver_matrix = self.solver_setup_dict["solver_matrix"]
                self.ndata = self.solver_setup_dict["ndata"]
            else:
                raise ValueError("Solver type not supported")


if __name__ == "__main__":
    import os

    os.chdir("/home/dl-rookie/PycharmProjects/LRCCA_inversion")
    parameters_file = (
        "./Data/parameters_matern32_Mu10_Var1p96_CorH30_CorV15_linear_81.txt"
    )
    params = Config(parameters_file)

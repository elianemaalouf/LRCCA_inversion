# -*- coding: utf-8 -*-
"""
@author: elianemaalouf
"""

import ast
import os
import random

import numpy as np
import torch

from tomokernel_straight import tomokernel_straight_2D

def matern32(xx, yy, lxtrue, lytrue, sigma2, jitter):
    """matern32
    Matern 3/2 covariance kernel

    :param xx: horizontal axis positions differences (columns of the grid, not the rows) -> x_i - x_j
    :param yy: vertical axis positions differences (rows of the grid, not the columns) -> y_i - y_j
    :param lxtrue: correlation length in horizontal direction
    :param lytrue: correlation length in vertical direction
    :param sigma2: variance at the origin
    """
    ntot = xx.shape[0]
    Hm = np.sqrt(np.power(xx / lxtrue, 2) + np.power(yy / lytrue, 2))
    CM = (
        sigma2 * (1 + np.sqrt(3) * Hm) * np.exp(-np.sqrt(3) * Hm)
        + np.identity(ntot) * jitter
    )  # covariance matrix
    return CM

def prior_setup(nx, ny, lxtrue, lytrue, sigma2, cov_kernel, jitter, mu):
    ## grid
    ntot = nx * ny
    xx, yy = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    # print('xx: ', xx.shape)
    # print('yy:', yy.shape)

    xf = xx.flatten(
        order="C"
    )  # flatten 2D to 1D, reading from left to right, columns first then rows
    yf = yy.flatten(order="C")

    ## coordinates differences
    xx = np.subtract.outer(
        xf, xf
    )  # nx by nx matrix containing all 2 by 2 differences between horizontal coordinates
    yy = np.subtract.outer(
        yf, yf
    )  # ny by ny matrix containing all 2 by 2 differences between vertical coordinates

    ## create covariance matrix:
    if cov_kernel=='matern32':
        CM = matern32(
            xx, yy, lxtrue, lytrue, sigma2, jitter=jitter
        )  # Gaussian field prior covariance matrix
    else:
        raise ValueError("Covariance kernel not supported")

    L = np.linalg.cholesky(CM)  # square root of prior covariance matrix

    ## mean vector
    m_prior = np.repeat(mu, ntot)  # Gaussian filed prior mean

    return {
        "ntot": ntot,
        "covariance_matrix": CM,
        "covM_squareRoot": L,
        "prior_mean": m_prior,
    }

def linearSolver_matrix(nx, ny, spacing, sources_x, start_y, step_y):
    """linearSolver_matrix
    Function to generate the linear solver matrix given the subsurface domain grid

    :param nx: number of cells on the horizontal axis (number of columns and not the number of rows) (corresponds to pixels)
    :param ny: number of cells on the vertical axis (number of rows and not the number of columns) (corresponds to pixels)
    :param spacing: width of a cell in meters
    :param sources_x: position of the sources on the horizontal axis
    :param start_y: locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
    :param step_y: distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)
    :return: dictionnary containing the following keys {'solver_matrix' (the solver matrix) ,
    'ndata' (length of data/measurement vector)}
    """
    x = np.arange(0, (nx * spacing) + spacing, spacing)  # in meters
    y = np.arange(0, (ny * spacing) + spacing, spacing)

    sourcex = sources_x
    sourcey = (
        np.arange(start_y, ny, step_y) * spacing
    )  # sets the sources in the middle of each cell, in meters
    receiverx = nx * spacing
    receivery = (
        np.arange(start_y, ny, step_y) * spacing
    )  # sets the receivers in the middle of each cell, in meters
    nsource = len(sourcey)
    nreceiver = len(receivery)

    ndata = nsource * nreceiver  # number of rays

    data = np.zeros((ndata, 4))
    # Calculate acquisition geometry (multiple-offset gather)
    for jj in range(0, nsource):
        for ii in range(0, nreceiver):
            data[jj * nreceiver + ii, :] = np.array(
                [sourcex, sourcey[jj], receiverx, receivery[ii]]
            )
        # Calculate forward modeling kernel (from Matlab code by Dr. James Irving, UNIL)
    A = tomokernel_straight_2D(
        data, x, y
    )  # Distance of ray-segment in each cell for each ray
    A = np.array(A.todense())
    del data

    return {"solver_matrix": A, "ndata": ndata}

class Config:
    """Class to read environment configuration parameters for Goephysics applications."""

    def __init__(self, parameters_file, setup_solver=True, setup_prior=True):
        """
        Initializes Config instance attributes from values stored in a configuration file on disk.
        :param parameters_file: configuration file
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

        # fix seed for all random number generators:
        np.random.seed(self.parameters["FixedSeed"])
        os.environ["PYTHONHASHSEED"] = str(self.parameters["FixedSeed"])
        random.seed(self.parameters["FixedSeed"])
        torch.manual_seed(self.parameters["FixedSeed"])
        torch.cuda.manual_seed_all(self.parameters["FixedSeed"])
        np.random.seed(self.parameters["FixedSeed"])
        print("Random seed fixed to:{}".format(self.parameters["FixedSeed"]))

        if setup_prior:
            # set up prior mean and covariance
            prior = prior_setup(
                self.nx,
                self.ny,
                self.lytrue,
                self.lytrue,
                self.sigma2,
                self.cov_kernel,
                self.jitter,
                self.mu,
            )
            self.CM = prior[
                "covariance_matrix"
            ]  # Gaussian field prior covariance matrix
            self.L = prior["covM_squareRoot"]  # square root of prior covariance matrix
            self.m_prior = prior["prior_mean"]  # Gaussian filed prior mean
            self.ntot = prior["ntot"]
        else:
            self.ntot = self.nx * self.ny

        if setup_solver:
            # setup forward solver _ only linear solver
            if self.solver_type == "linear":
                if self.rays == 81:
                    start_y = 5.5  # locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
                    # (.5 to place in middle of the cell, relevant for linear solver)
                    step_y = 5  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)
                if self.rays == 576:
                    start_y = 2.5  # locate first source/receiver in the vertical direction (i.e. in the rows of the domain grid)
                    # (.5 to place in middle of the cell, relevant for linear solver)
                    step_y = 2  # distance between source/receiver in the vertical direction (i.e. in the rows of the domain grid)

                self.solver_setup_dict = linearSolver_matrix(
                    self.nx, self.ny, self.spacing, self.sources_x, start_y, step_y
                )
                self.solver_matrix = self.solver_setup_dict["solver_matrix"]
                self.ndata = self.solver_setup_dict["ndata"]
            else:
                raise ValueError("Solver type not supported")
        else:
            self.ndata = self.rays

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

if __name__ == "__main__":
    parameters_file = "../../Data/parameters_matern32_Mu10_Var1p96_CorH30_CorV15_linear_81.txt"
    params = Config(parameters_file)

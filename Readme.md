# Regularized Canonical Correlation Analysis (RCCA) for solving linear inverse problems

Data and Python code repository to reproduce the results in chapter 3 of the thesis 
"Contributions to data-driven Bayesian solutions to inverse problems, with classical multivariate statistics and modern
generative neural networks" by Eliane Maalouf (University of Neuchâtel, Switzerland).

# Contents
- `Data/`: Contains the data used in the experiments.
- `lrcca_inversion/`: Python package containing the code to perform the RCCA inversion.
Main files:
  - `cca.py`: main class for performing Regularized Canonical Correlation Analysis (RCCA). 
    It contains methods for fitting CCA with SVD and regularization, for dimensionality reduction,
    and for generating deterministic and probabilistic predictions.
  - `xp_val_script.py`: script to run the experiments and generate the results.
  - `xp_eval_script.py`: script to evaluate the results of the experiments.
  - `xp_runners.py`: main functions called by the scripts to run the experiments and evaluations.
- `lrcca_inversion/utils/`: Contains utility functions for configuration, plotting and metrics computation.
- `Experiments/`: Contains the generated data from all the experiments mentioned in the chapter. 
- `xp_config.py`: script to configure the experiments, including the data paths and parameters for the RCCA inversion.
For every new experiment, this script creates a new directory and generates a `config.json` file with the experiment 
parameters. The script also updates the `Experiments/all_xp_configs.json` file with the new experiment name and folder 
location.

# Experiments naming convention and reproducibility
All deterministic predictions experiments are stored in the `Experiments/deterministic_preds/` folder. 
All probabilistic predictions experiments are stored in the `Experiments/probabilistic_preds/` folder.
Regularization validation experiments are named as `det_preds_reg_XYZ` (for deterministic predictions) or `prob_preds_reg_XYZ` 
(for probabilistic predictions). Inversion experiments are named as `det_preds_inv_XYZ` (for deterministic predictions) 
or `prob_preds_inv_XYZ` (for probabilistic predictions). Each experiment folder name corresponds to the experiment
name contained in its `config.json` file and listed in the `Experiments/all_xp_configs.json` file.
## Usage
1. To run a new experiment, edit the `xp_config.py` file to set the parameters and data paths. Then run 
   `xp_val_script.py` changing the `xp_name` variable to the name of the new experiment name 
    (as chosen in the `xp_config.py` file). Proceed in the same way to run the evaluation script by `xp_eval_script.py`.
2. To rerun an existing experiment, simply run the `xp_val_script.py` or `xp_eval_script.py` script with the 
   `xp_name` variable set to the name of the experiment you want to rerun. Make sure the experiment name and folder
    are correctly set in the `Experiments/all_xp_configs.json` file. 

For each of the validation experiments, we provide the full data that resulted from the validation runs in 
`validation_data.pkl` file. For each inversion experiment, we provide the full data that resulted from the 
inversion runs in `inversion_data.pkl` file and the trained CCA objects in `cca_objects.pkl`.

All experiments ran with a fixed seed of 42, fixed in the data parameters file 
`Data/parameters_matern32_Mu10_Var1p96_CorH30_CorV15_linear_81.txt` and enforced in `lrcca_inversion/utils/config.py` 
for numpy, torch, and random libraries.  

For convenience, we ran the scripts from the console of Pycharm IDE, but they can be easily adapted to run 
from the command line. 

# Requirements
- Python 3.8 or higher, we used 3.10 
- numpy 
- scipy 
- scikit-learn
- matplotlib
- mpl_toolkits
- pandas
- seaborn
- h5py, to load the data
- torch, data is stored as PyTorch tensors
- pickle

# Disclaimer 
This software is provided 'as is' without any warranty, express or implied. 
Please see the `LICENSE` file for full details.





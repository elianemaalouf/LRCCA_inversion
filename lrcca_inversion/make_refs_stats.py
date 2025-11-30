"""
Generate summary statistics for the reference metrics.
"""
"""

from pathlib import Path
from lrcca_inversion.utils.generic_fn import load_from_disk, save_to_disk
from lrcca_inversion.utils.evals import compute_stats

BASE_DIR = Path(__file__).resolve().parent.parent

file_name = "reference_inv_metrics_es1_vs05_rmse.pkl"#"reference_val_metrics_es2_vs05_rmse.pkl" # "reference_val_metrics_es1.pkl" #
output_file_name = "reference_inv_metrics_es1_vs05_rmse.json"#"reference_val_metrics_es2_vs05_rmse_summaries.json" # "reference_val_metrics_es1_summaries.json" #

reference_metrics = load_from_disk(f"{BASE_DIR}/Experiments/{file_name}")

summaries = {}
for metric in reference_metrics.keys():
    summaries[metric] = compute_stats(reference_metrics[metric])

save_to_disk(summaries, f"{BASE_DIR}/Experiments/{output_file_name}", json=True)

"""
# %%

from pathlib import Path

from lrcca_inversion.utils.evals import compute_stats
from lrcca_inversion.utils.generic_fn import load_from_disk, save_to_disk

BASE_DIR = Path(__file__).resolve().parent.parent

xp_name = "non_linear_prob_preds_inv_n500"
file_name = "y_resim_metrics.pkl"
output_file_name = "resim_metrics_summaries"

reference_metrics = load_from_disk(
    f"{BASE_DIR}/Experiments/probabilistic_preds/{xp_name}/{file_name}"
)

summaries = {}
for noise_label in reference_metrics.keys():
    summaries[noise_label] = {}
    for metric in reference_metrics[noise_label].keys():
        summaries[noise_label][metric] = compute_stats(
            reference_metrics[noise_label][metric]
        )

save_to_disk(
    summaries,
    f"{BASE_DIR}/Experiments/probabilistic_preds/{xp_name}/{output_file_name}",
    json=True,
)

# %%
"""
## Median resim RMSE plots
import matplotlib.pyplot as plt
import numpy as np

import lrcca_inversion.utils.evals as evals

# data from resim experiments 
lambda_y = [0.0001, 0.001, 0.01, 0.1, 1, 10 , 100]

# from resim_metrics_summaries.json of each experiment
# small noise (SN) stats
sn_median_rmse = [1.563, 1.362, 0.929, 0.746, 0.853, 1.249, 2.238]
sn_lower = [1.315, 1.158,  0.805, 0.662, 0.768, 1.100, 1.852] # lower - IQR
sn_upper = [1.965, 1.689, 1.089, 0.860, 0.952, 1.417, 2.682] # upper - IQR

# large noise (LN) stats
ln_median_rmse = [12.181, 10.234, 5.199, 2.563, 2.434, 2.667, 3.301]
ln_lower = [8.114, 6.954, 3.978, 2.359, 2.288, 2.522, 3.032] # lower - IQR
ln_upper = [15.763, 13.896, 6.821, 2.807, 2.559, 2.808, 3.624] # upper - IQR

sn_std = 0.5
ln_std = 2.5

# plot the median RMSE for SN and LN and fill the area between the lower and upper bounds
mpl, plt, make_axes_locatable, tick = evals.plots_imports()
evals.base_config(mpl)

plt.plot(np.log10(lambda_y), sn_median_rmse, label="SN", marker="o")
plt.fill_between(np.log10(lambda_y), sn_lower, sn_upper, color="blue", alpha=0.1)
plt.plot(np.log10(lambda_y), ln_median_rmse, label="LN", marker="o")
plt.fill_between(np.log10(lambda_y), ln_lower, ln_upper, color="orange", alpha=0.1)
plt.plot(
    np.log10(lambda_y),
    np.sqrt(lambda_y),
    label="y = sqrt(lambda_y)",
    linestyle="--",
    color="gray",
)

# plot points
# plt.scatter(np.log10([sn_std, ln_std]), [sn_std, ln_std], marker='x', color = 'red', label="Std Dev Points", zorder=5)

# add horizontal lines for the standard deviations and make them transparent
plt.axhline(sn_std, color="blue", linestyle=":", label="SN Std Dev", alpha=0.5)
plt.axhline(ln_std, color="orange", linestyle=":", label="LN Std Dev", alpha=0.5)

# remove upper and right spines
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)

plt.xlabel(r"$\log_{10}(\lambda_Y)$")
plt.ylabel("Median resimulations RMSE")
plt.legend()
plt.savefig("resim_median_rmse_non_linear.pdf", dpi=600, bbox_inches="tight")
"""

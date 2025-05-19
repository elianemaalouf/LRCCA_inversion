"""
Evaluation and plotting functions.

@author: elianemaalouf
"""
import numpy as np

def save_to_disk(data, filepath):
    """
    Save data to disk in JSON format.

    Parameters
    ----------
    data : dict
        The data to save, typically a nested dictionary structure
    filepath : str
        The full path where to save the JSON file

    Returns
    -------
    None
    """
    import json

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def compute_stats(np_array):
    """
    Compute statistics for an array.
    Statistics computed are mean, median, 25th and 75th percentiles, 2.5th and 97.5th percentiles.

    np_array:
        1D numpy array

    idx_lowest_stat:
        return the index of the lowest statistic selected, default is 'median'.

    :returns: dictionary of statistics
    """
    if np_array is not None:
        stats = {
            'mean': np.mean(np_array),
            'std': np.std(np_array),
            'median': np.median(np_array)
        }
        # Calculate all quantiles at once
        quantiles = np.quantile(np_array,
                                q=[0.025, 0.25, 0.75, 0.975],
                                interpolation="nearest")
        stats.update({
            'q025': quantiles[0],
            'q25': quantiles[1],
            'q75': quantiles[2],
            'q975': quantiles[3]
        })
        return stats
    return {k: None for k in ['mean', 'std', 'median', 'q25', 'q75', 'q025', 'q975']}

def get_best_param_comb(data_dict, metric, ref_stat = 'median'):
    """
    Get the best parameter combination for a given metric.

    data_dict:
        Dictionary containing the data to evaluate. The combination of the parameters is a key in the dictionary at a
        level lower than metric. e.g. {'rmse':{(lambda_x, lambda_y): np.array([1,2,3])}}.
    metric:
        Metric to evaluate. e.g. 'rmse', 'es', 'vs'. Could be a list of more than one metric.
    ref_stat:
        Reference statistic to use for evaluation. Default is 'median'. Could be 'mean'.

    :returns: dictionary of best parameter combination
    """
    if isinstance(metric, str):
        metric = [metric]

    best_comb = {}
    for m in metric:
        if m not in data_dict.keys():
            raise ValueError(f"Metric {m} not found in data dictionary. Available metrics: {list(data_dict.keys())}")

        #best_comb[m] = {}
        best_comb = {}
        for comb, values in data_dict[m].items():
            stats = compute_stats(values)
            best_comb[comb] = stats[ref_stat]

        # get the best combination
        best_comb = min(best_comb, key=best_comb.get)

    return best_comb

def make_ref_dict(ref_data_dict_list, metric, noise_label=None):
    """
    Prepares a dictionary of reference values for the given metric to be used in the boxplots.

    ref_data_dict_list:
        list of dictionnaries containing the reference data. When doing deterministic inversion, the reference data
        is the one against training.
        When doing probabilistic inversion, the reference data is the one against training and
        the data from the deterministic inversion.

    metric:
        metric to filter for

    :return: a dictionary containing the reference values for the metric. It will be structured as follows:
        {'ref1': {'lower': value, 'center': value, 'upper': value},
        'ref2': {'lower': value, 'center': value, 'upper': value}}.
        For relevance of interpretation, the center value represents a median and lower-upper should represent
        IQR bounds.
    """
    ref_dict = {}
    for i, ref_data_dict in enumerate(ref_data_dict_list):
        if noise_label is not None and noise_label in ref_data_dict.keys():
            ref_data_dict = ref_data_dict[noise_label]

        if metric not in ref_data_dict.keys():
            raise ValueError(f"Metric {metric} not found in reference data dictionary. Available metrics: {list(ref_data_dict.keys())}")

        #if ref_data_dict[metric] is already a dict with keys 'lower', 'center', 'upper', skip the computation
        if isinstance(ref_data_dict[metric], dict):
            ref_dict[f'ref{i+1}'] = ref_data_dict[metric]
            continue
        # get the reference values
        stats = compute_stats(ref_data_dict[metric])
        ref_dict[f'ref{i+1}'] = {
            'lower': stats['q25'],
            'center': stats['median'],
            'upper': stats['q75']
        }

    return ref_dict

# plotting functions
def plots_imports():
    import matplotlib as mpl

    mpl.use("module://backend_interagg")
    # mpl.use('pdf')  # choose pdf renderer for vector graphic # default was: 'module://backend_interagg'
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tick
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    return mpl, plt, make_axes_locatable, tick

def base_config(
    mpl,
    figsize=(8, 6),
    family="serif",
    fonttype="Liberation Serif",
    texfontType="dejavuserif",
    dpi=600,
    fontsize=14,
):
    """base_config
    Function to setup common figures configuration
    figsize:
        specify figure size (width, height) in inches. Default: (8in , 6in)
    family:
        font family for text. Default: serif
    fonttype:
        specific font type in family. Default: Liberation Serif
    texfontType:
        Tex (for math text) font type. Default: dejavuserif
    dpi:
        specify image resolution in dots per inch. Default: 600
    fontsize:
        font size in points. Default: 14 pt
    """
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["figure.dpi"] = dpi
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["font.family"] = family
    mpl.rcParams["font.{}".format(family)] = fonttype
    mpl.rcParams["mathtext.fontset"] = texfontType
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["lines.linewidth"] = 1

def make_val_boxplots(data_dict, metric, references_dict= None, labels_dict = None, save_path=None, show = False, **kwargs):
    """
    Create the boxplots for the chosen validation metrics for all the parameter combinations.
    data_dict:
        Dictionary containing the data to evaluate. The combination of the parameters is a key in the dictionary at a
        level lower than metric. e.g. {'rmse':{(lambda_x, lambda_y): np.array([1,2,3])}}.
    metric:
        the metric to plot. e.g. 'rmse', 'es', 'vs'. One metric at a time.
    labels_dict:
        a dictionary containing {'plot_title':'', 'x_label':'', 'y_label':''}.
    references_dict:
        a dictionary containing the reference values for the metrics. It should be structured as follows:
        {'ref1': {'lower': value, 'center': value, 'upper': value},
        'ref2': {'lower': value, 'center': value, 'upper': value}}.
        For relevance of interpretation, the center value should represent a median and lower-upper should represent
        IQR bounds.
    save_path:
        location where the plot will be stored as pdf.
    kwargs:
        configure plotting parameters. For example, 'whis_low', 'whis_high', 'lower_lim', 'upper_lim', 'h_axis_margin',
        'v_axis_margin', 'x_ticks_step', 'reduce_lambda_y_vec', 'lambda_y_subset'.
    """

    import pandas as pd
    import seaborn as sns

    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl)

    # extract the lists parameter combinations and their corresponding values
    combinations = list(data_dict.keys())
    lambda_x_vec = sorted(list(set([comb[0] for comb in combinations])))
    lambda_y_vec = sorted(list(set([comb[1] for comb in combinations])))
    reduce_lambda_y_vec = kwargs.get('reduce_lambda_y_vec', False)
    lambda_y_subset = kwargs.get('lambda_y_subset', 4)
    if reduce_lambda_y_vec:
        # take only the last 4 values of lambda_y_vec
        lambda_y_vec = lambda_y_vec[-lambda_y_subset:]


    # make dataframe
    all_values_df = pd.DataFrame(columns=["lambda_x", "lambda_y", "value"])
    for comb in combinations:
        if comb[1] not in lambda_y_vec:
            continue
        else:
            new_data = pd.DataFrame({
                "lambda_x": comb[0],
                "lambda_y": comb[1],
                "value": data_dict[comb]
            })
        all_values_df = pd.concat([all_values_df, new_data], ignore_index=True)


    # get kwargs if any or set default values
    whis_low = kwargs.get('whis_low', 2.5)
    whis_high = kwargs.get('whis_high', 97.5)
    lower_lim = kwargs.get('lower_lim', None)
    upper_lim = kwargs.get('upper_lim', None)
    h_axis_margin = kwargs.get('h_axis_margin', 0.7)
    v_axis_margin = kwargs.get('v_axis_margin', 0.1)
    x_ticks_step = kwargs.get('x_ticks_step', 1)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    sns.boxplot(
        x="lambda_x",
        y="value",
        data=all_values_df,
        hue="lambda_y",
        ax=ax,
        fliersize=2,
        palette="vlag",
        whis=[whis_low, whis_high],
    )

    # set lower and upper limits based on data.
    if lower_lim is None:
        # take mininmum from data and references if any
        if references_dict is not None:
            lower_lim = min(all_values_df["value"].min(), min([ref['lower'] for ref in references_dict.values()])) - v_axis_margin
        else:
            lower_lim = all_values_df["value"].min() - v_axis_margin
    if upper_lim is None:
        # take maximum from data and references if any
        if references_dict is not None:
            upper_lim = max(all_values_df["value"].max(), max([ref['upper'] for ref in references_dict.values()])) + v_axis_margin
        else:
            upper_lim = all_values_df["value"].max() + v_axis_margin

    # draw reference lines if any, changing the color for each reference
    if references_dict is not None:
        for i, (ref_name, ref_values) in enumerate(references_dict.items()):
            ax.axhline(ref_values['lower'], color=f"C{i}", linestyle=":", linewidth = 1, label=f"{ref_name} 25th percentile")
            ax.axhline(ref_values['center'], color=f"C{i}", linestyle="--", linewidth = 1, label=f"{ref_name} median")
            ax.axhline(ref_values['upper'], color=f"C{i}", linestyle="-.", linewidth = 1, label=f"{ref_name} 75th percentile")

    # set axis labels and title
    if labels_dict is not None:
        ax.set_title(labels_dict.get('plot_title', ''))
        ax.set_xlabel(labels_dict.get('x_label', ''))
        ax.set_ylabel(labels_dict.get('y_label', ''))
    else:
        ax.set_title(f"{metric} boxplots")
        ax.set_xlabel(r"$\log_{10}(\lambda_X)$")
        ax.set_ylabel(metric)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(lower_lim, upper_lim)

    # set scientific notation for y axis
    if lower_lim > 1e2:
        sci_bound = 6 if upper_lim > 1e6 else 3
    else:
        sci_bound = 0
    ax.ticklabel_format(
        axis="y", style="sci", scilimits=(sci_bound, sci_bound), useMathText=True
    )

    # set x axis ticks and labels using log10
    ax.set_xticks(list(range(len(lambda_x_vec[::x_ticks_step]))))
    ax.set_xticklabels(
        [str(np.round(np.log10(l), 2)) for l in lambda_x_vec[::x_ticks_step]]
    )

    plt.gca().spines["left"].set_position(("data", -h_axis_margin))
    plt.gca().spines["bottom"].set_position(("data", lower_lim - v_axis_margin))

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.close()

def plot_transformations(
        T_matrix, dims, groups=3, number_of_comp=3, save_location=None, dpi=600
    ):
        """
        Function to plot the first "number_of_comp" canonical transformations
        T_matrix
            the matrix of canonical transformations.
        dims
            provides the shape of the components to plot.
            Should be either (1, dim) for a 1D vector or (height, width) for a 2D matrix
        groups
            number of grouped components to plot
        number_of_comp
            the number of components to plot per group. Total number of components is groups*number_of_comp
        save_location
            folder location where to save the resulting plot
        dpi
            the image resolution in dots per inch. Default: 600
        """

        mpl, plt, make_axes_locatable, tick = plots_imports()
        base_config(mpl)

        T_matrix = T_matrix.T
        height, width = dims

        for g in range(groups):
            fig, axes = plt.subplots(nrows=1, ncols=number_of_comp)

            if hasattr(axes, "__len__"):
                axes = axes
            else:
                axes = [axes]

            for i in range(number_of_comp):
                if height == 1:
                    im = axes[i].plot(
                        T_matrix[g * number_of_comp + i, :].reshape(width)
                    )
                    axes[i].spines["right"].set_visible(False)
                    axes[i].spines["top"].set_visible(False)
                    axes[i].set_xlim([0, width - 1])
                    axes[i].set_xticks(list(range(0, width, 20)))
                    axes[i].set_xticklabels([str(x) for x in range(0, width, 20)])

                else:
                    im = axes[i].imshow(
                        T_matrix[g * number_of_comp + i, :].reshape(height, width)
                    )
                    axes[i].set_xticks([])
                    axes[i].set_yticks([])
                    ax_divider = make_axes_locatable(axes[i])
                    cax = ax_divider.append_axes(
                        "right", size="5%", pad="2%", frameon=False
                    )
                    cax.set_xticks([])
                    cax.set_yticks([])

                    fig.colorbar(
                        im, cax=cax, format=tick.FormatStrFormatter("%.2f")
                    )  # , extend = 'both')

                axes[i].title.set_text(
                    "Transform. #{}".format(g * number_of_comp + i + 1)
                )
            plt.tight_layout()
            plt.savefig(save_location.format(g), dpi=dpi, bbox_inches="tight")

            plt.close()

def plot_samples(
    examples,
    width,
    height,
    rmse_labels=None,
    ssim_labels = None,
    grd_truth=True,
    save_location=None,
    dpi=600,
    show = False,
):
    """
    Function to plot given set of examples
    examples:
        a set of examples to plot. Expects format as (number of examples, samples per example, height*width)
    width:
        the width of the image in pixels
    height:
        the height of the image in pixels
    rmse_labels:
        a list of RMSE values to add on top of the sample
    ssim_labels:
        a list of SSIM values to add on top of the sample
    grd_truth:
        if True, the first example provided is the ground truth
    save_location:
        location where to save the generated plot
    dpi:
        resolution of the image in dots per inch (dpi)
    """

    import matplotlib.gridspec as gridspec

    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl)

    fig = plt.figure()

    rows = examples.shape[0]
    cols = examples.shape[1]

    gspec = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)

    vmin = np.min(examples)
    vmax = np.max(examples)

    for i in range(rows):

        for j in range(cols):
            ax = fig.add_subplot(gspec[i, j])

            im = ax.imshow(examples[i, j, :].reshape(height, width))
            ax.set_xticks([])
            ax.set_yticks([])
            im.set_clim(vmin, vmax)
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="5%", pad="2%", frameon=False)
            cax.set_xticks([])
            cax.set_yticks([])

            if j == cols - 1:
                fig.colorbar(im, cax=cax, format=tick.FormatStrFormatter("%.2f"))

            if grd_truth:
                if j == 0:
                    ax.title.set_text(r"Ground truth")
                else:
                    label = f"RMSE = {rmse_labels[i][j-1]:.2f} ns/m" if rmse_labels is not None else f"Example #{j}"
                    label = f"{label}; SSIM = {ssim_labels[i][j-1]:.2f}" if ssim_labels is not None else label
                    ax.title.set_text(label)
            else:
                label = f"RMSE = {rmse_labels[i][j]:.2f} ns/m" if rmse_labels is not None else f"Example #{j}"
                label = f"{label}; SSIM = {ssim_labels[i][j]:.2f}" if ssim_labels is not None else label
                ax.title.set_text(label)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()

def plot_boxplots(
    values_all,
    labels,
    axes_plot_titles=None,
    lower_lim=0.2,
    upper_lim=None,
    whis_low=2.5,
    whis_high=97.5,
    y_scale = 'linear',
    save_location=None,
    dpi=600,
    show=False,
):
    """
    Function to plot boxplots of given values. It detects if one or more boxplots need to be drawn and assigns given labels
    to them.

    values_all:
        array containing the values to boxplot. Format as [number of subplots, number of boxplots, values per boxplot]
    to have multiple sublopts in the same figure and multiple boxplots in each subplot.

    labels :
        labels to give to each boxplot, should be a list with size "number of plots"

    axes_plot_titles :
        provide title to use for axes and plot. Format as [list of subplot_titles, horizontal_axis_title, vertical_axis_title]

    lower_lim:
        lower limit of the y axis

    upper_lim:
        upper limit of the y axis

    whis_low:
        lower limit of the whiskers

    whis_high:
        upper limit of the whiskers

    y_scale:
        scale of the y axis. Default: 'linear'

    save_location:
        location where to save the generated plot

    dpi:
        resolution of the image in dots per inch (dpi)

    show:
        if True, the plot is shown but not saved. if False, it is only saved
    """
    import pandas as pd
    import seaborn as sns

    mpl, plt, make_axes_locatable, tick = plots_imports()
    base_config(mpl)

    number_of_subplots = values_all.shape[0]
    number_of_box = values_all.shape[1]

    fig, ax = plt.subplots(nrows=1, ncols=number_of_subplots, sharey=True)
    ax.set_yscale(y_scale)
    if hasattr(ax, "__len__"):
        axes = ax
    else:
        axes = [ax]

    whis_low = whis_low
    whis_high = whis_high

    upper_lim = upper_lim
    lower_lim = lower_lim

    if len(labels) != values_all.shape[1]:
        print("Number of labels does not match number of boxplots to produce!")
        return None
    else:
        for i in range(number_of_subplots):
            values = pd.DataFrame(values_all[i, :, :].reshape(number_of_box, -1)).T
            values.columns = labels

            sns.boxplot(
                data=values,
                whis=[whis_low, whis_high],
                fliersize=2,
                palette="vlag",
                ax=axes[i],
            )
            axes[i].spines["top"].set_visible(False)
            axes[i].spines["right"].set_visible(False)
            axes[i].set_ylim(top=upper_lim, bottom=lower_lim)
            axes[i].yaxis.set_major_formatter(tick.FormatStrFormatter("%.2f"))

            if axes_plot_titles:
                axes[i].title.set_text(axes_plot_titles[0][i])
                if i == 0:
                    axes[0].set_ylabel(axes_plot_titles[2])

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(save_location, dpi=dpi, bbox_inches="tight")

    plt.close()




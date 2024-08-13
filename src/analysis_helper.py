import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_pickle_data(p):
    try:
        with open(p, "rb") as f:
            return [pickle.load(f) for _ in range(14)]
    except EOFError:
        print(f'EOF Error at {p}')
        return [None] * 14


def update_logs(pickle_data):
    bias, ind, preds, p_fit, n, fit_track, average_change, average_retention, p_size, histograms, mut_effect, xov_effect, sharp_list, sharp_std, density = pickle_data

    mut_list_effect = mut_effect[0]
    mut_cumul_effect = mut_effect[1]
    xov_list_effect = mut_effect[0]
    xov_cumul_effect = mut_effect[1]
    sharp_in_list = sharp_list[0]
    sharp_out_list = sharp_list[1]
    sharp_in_std = sharp_std[0]
    sharp_out_std = sharp_std[1]

    if np.isnan(p_fit):
        p_fit = np.PINF

    return {
        'logs': (bias, ind, preds, p_fit),
        'p_log': p_fit,
        'track_log': fit_track,
        'node_log': n,
        'prop_log': n / len(ind) if ind else np.nan,
        'change_log': average_change,
        'retent_log': average_retention,
        'p_size_log': p_size,
        'histog_log': histograms,
        'mut_cumul_log': mut_cumul_effect,
        'mut_list_log': mut_list_effect,
        'xov_cumul_log': xov_cumul_effect,
        'xov_list_log': xov_list_effect,
        'sharp_in_list_log': sharp_in_list,
        'sharp_in_std_log': sharp_in_std,
        'sharp_out_list_log': sharp_out_list,
        'sharp_out_std_log': sharp_out_std,
        'den_log': density
    }


def get_logs_cgp(base_path, max_e, f_name):
    full_logs, full_fits, fit_tracks, active_nodes = [], [], [], []
    node_prop, avg_chg, avg_ret, p_sz_li, hist_li = [], [], [], [], []
    mut_cumul, mut_list, xov_cumul, xov_list = [], [], [], []
    sharp_in_mean, sharp_out_mean, sharp_in_std, sharp_out_std, den = [], [], [], [], []

    for name in f_name:
        print(f"Loading {base_path}{name}")

        logs, p_log, track_log, node_log, prop_log = [], [], [], [], []
        change_log, retent_log, p_size_log, histog_log = [], [], [], []
        mut_cumul_log, mut_list_log, xov_cumul_log, xov_list_log = [], [], [], []
        sharp_in_list_log, sharp_out_list_log = [], []
        sharp_in_std_log, sharp_out_std_log, den_log = [], [], []

        for e in range(1, max_e + 1):
            p = f'{base_path}{name}/log/output_{e}.pkl'
            pickle_data = load_pickle_data(p)

            if pickle_data[0] is not None:
                updated_logs = update_logs(pickle_data)

                logs.append(updated_logs['logs'])
                p_log.append(updated_logs['p_log'])
                track_log.append(updated_logs['track_log'])
                node_log.append(updated_logs['node_log'])
                prop_log.append(updated_logs['prop_log'])
                change_log.append(updated_logs['change_log'])
                retent_log.append(updated_logs['retent_log'])
                p_size_log.append(updated_logs['p_size_log'])
                mut_cumul_log.append(updated_logs['mut_cumul_log'])
                mut_list_log.append(updated_logs['mut_list_log'])
                xov_cumul_log.append(updated_logs['xov_cumul_log'])
                xov_list_log.append(updated_logs['xov_list_log'])
                sharp_in_list_log.append(updated_logs['sharp_in_list_log'])
                sharp_out_list_log.append(updated_logs['sharp_out_list_log'])
                sharp_in_std_log.append(updated_logs['sharp_in_std_log'])
                sharp_out_std_log.append(updated_logs['sharp_out_std_log'])
                den_log.append(updated_logs['den_log'])
                try:
                    histog_log.append(updated_logs['histog_log'])
                except:
                    print(updated_logs['histog_log'])

        full_logs.append(logs)
        full_fits.append(p_log)
        fit_tracks.append(track_log)
        node_prop.append(prop_log)
        avg_chg.append(change_log)
        avg_ret.append(retent_log)
        p_sz_li.append(p_size_log)
        hist_li.append(histog_log)
        active_nodes.append(node_log)
        mut_cumul.append(mut_cumul_log)
        mut_list.append(mut_list_log)
        xov_cumul.append(xov_cumul_log)
        xov_list.append(xov_list_log)
        sharp_in_mean.append(sharp_in_list_log)
        sharp_out_mean.append(sharp_out_list_log)
        sharp_in_std.append(sharp_in_std_log)
        sharp_out_std.append(sharp_out_std_log)
        den.append(den_log)

    return [full_logs, np.array(full_fits), np.array(fit_tracks), np.array(active_nodes), np.array(node_prop),
            np.array(avg_chg), np.array(avg_ret), np.array(p_sz_li), hist_li, mut_list, mut_cumul, xov_list, xov_cumul,
            sharp_in_mean, sharp_in_std, sharp_out_mean, sharp_out_std, den]


def data_dict(data):
    return {
        'logs': data[0],
        'p_fits': data[1],
        'fit_track': data[2],
        'nodes': data[3],
        'prop': data[4],
        'average_change': data[5],
        'average_retention': data[6],
        'p_size': data[7],
        'histograms': data[8],
        'mut_cumul_drift': data[9],
        'mut_list_drift': data[10],
        'xov_cumul_drift': data[11],
        'xov_list_drift': data[12],
        'sharp_in_mean': data[13],
        'sharp_in_std': data[14],
        'sharp_out_mean': data[15],
        'sharp_out_std': data[16],
        'density_distro': data[17]
    }


def load_all_data(base_paths, max_e, f_names):
    return {key: data_dict(get_logs_cgp(base_paths[key], max_e, f_names[key])) for key in base_paths}


# Refactored by chatGPT
def calculate_avg_and_std(data, axis=None):
    """
    Calculate the average and standard deviation across the axis=0
    for each element in the input list.
    """
    ax = 0 if axis is None else axis
    avgs = np.array([np.average(d, axis=ax) for d in data])
    std_devs = np.array([np.std(d, axis=ax) for d in data])
    return avgs, std_devs


def get_avg_gens(data, axis=None):
    """
    Calculate the averages and standard deviations,
    then return the averages along with their corresponding error ribbons.
    """
    avgs, std_devs = calculate_avg_and_std(data, axis)
    return avgs, avgs - std_devs, avgs + std_devs


def get_err_ribbon(avgs, std_devs):
    """
    Return the upper and lower bounds of the error ribbon.
    """
    return avgs + std_devs, avgs - std_devs


def create_boxplot(ax, data, colors, method_names):
    """
    Create a boxplot for the provided data on the given axis.

    Parameters:
    - ax: The axis to draw the boxplot on.
    - data: A list of data arrays to plot.
    - colors: A list of colors for each box in the boxplot.
    - method_names: A list of method names for labeling the x-axis.
    """
    boxes = ax.boxplot(data, showfliers=False, patch_artist=True)
    for box, color in zip(boxes['boxes'], colors):
        box.set_facecolor(color)
    ax.set_yscale('log')
    ax.set_xticks(list(range(1, len(method_names) + 1)), method_names)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)
    return boxes['boxes']


def configure_axes(ax, title, ylabel, method_names, y_bottom=1e-6):
    """
    Configure the appearance of the axes.

    Parameters:
    - ax: The axis to configure.
    - title: The title for the axis.
    - ylabel: The label for the y-axis.
    - method_names: A list of method names for labeling the x-axis.
    """
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(bottom=y_bottom)


def plot_box_plots(f_names, data_dicts, method_names, method_names_long, color_order, key, name, title, y_label):
    """
    Create a figure with subplots for the fitness evaluation on SR problems.

    Parameters:
    - f_names: List of problem names (used for titles).
    - data_dicts: Dictionary of data (one entry per method).
    - method_names: List of method names for labeling the x-axis.
    - method_names_long: List of full method names for the legend.
    - color_order: List of colors corresponding to the methods.
    """
    fig, axs = plt.subplots(len(f_names), 1, figsize=(10.45, 12.1))
    fig.subplots_adjust(hspace=0)

    legend_objects = None

    for n, name in enumerate(f_names):
        data = [
            data_dicts['cgp_base'][key][n],
            data_dicts['cgp_40'][key][n],
            data_dicts['cgp_1x'][key][n],
            data_dicts['cgp_vlen'][key][n],
            data_dicts['lgp_1x'][key][n],
            data_dicts['cgp_2x'][key][n],
            data_dicts['lgp_2x'][key][n],
            data_dicts['cgp_sgx'][key][n],
            data_dicts['lgp_base'][key][n],
        ]

        boxes = create_boxplot(axs[n], data, color_order, method_names)
        configure_axes(axs[n], name, y_label, method_names)

        if n == 0:  # Capture legend objects from the first plot
            legend_objects = boxes

    fig.suptitle(title, fontsize=16)
    fig.legend(legend_objects, method_names_long, fontsize=10, ncol=2, bbox_to_anchor=(0.5, 0.965), loc='upper center')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
    fig.savefig(f"../output/{name}.png")


def plot_with_error(ax, x_data, y_data, lower_bound, upper_bound, label, color, alpha=0.10):
    """
    Plot a line with an error ribbon on a given axis.

    Parameters:
    - ax: The axis to plot on.
    - x_data: The x-axis data (usually the range of generations).
    - y_data: The y-axis data (averages).
    - lower_bound: The lower bound of the error ribbon.
    - upper_bound: The upper bound of the error ribbon.
    - label: The label for the line.
    - color: The color of the line and the ribbon.
    - alpha: The transparency of the ribbon.
    """
    ax.plot(y_data, label=label, c=color)
    ax.fill_between(x_data, lower_bound, upper_bound, color=color, alpha=alpha)


def configure_subplot(ax, title, ylabel, xlabel=None):
    """
    Configure the appearance of a subplot axis.

    Parameters:
    - ax: The axis to configure.
    - title: The title for the axis.
    - ylabel: The label for the y-axis.
    - xlabel: The label for the x-axis (optional).
    """
    ax.set_title(title, fontsize=12)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)


def plot_over_generations(f_names, avgs, method_names_long, color_order, name, title, y_label):
    """
    Create a figure with subplots for the similarity evaluation across methods.

    Parameters:
    - f_names: List of problem names (used for titles).
    - avgs: Dictionary of average retention data (per method).
    - stds: Dictionary of standard deviation data (per method).
    - method_names_long: List of full method names for the legend.
    - color_order: List of colors corresponding to the methods.
    - x_range: The range of x values (usually the range of generations).
    """
    fig, axs = plt.subplots(4, 2, figsize=(9.75, 6.75))
    fig.subplots_adjust(hspace=0.3)
    legend_objects = []
    x_range = range(avgs[0][0].shape[0])  # Assuming all methods have the same number of generations

    for n, ax in enumerate(axs.flat[:len(f_names)]):
        for method, color in zip(avgs.keys(), color_order):
            plot_with_error(
                ax, x_range, avgs[method][n][0], avgs[method][n][1], avgs[method][n][2], label=method, color=color
            )
            if n == 0:  # Capture legend objects from the first subplot
                legend_objects.append(ax.plot([], [], label=method, color=color)[0])

        configure_subplot(ax, f_names[n], y_label, xlabel="Generations" if n > 5 else None)

    axs.flat[-1].set_visible(False)
    fig.suptitle(title, fontsize=16)
    fig.legend(legend_objects, method_names_long, fontsize=10, loc='lower right', bbox_to_anchor=(1, 0.05), ncol=2)
    plt.show()
    fig.tight_layout()
    fig.savefig(f"../output/{name}.png")


def plot_individuals(avgs, f_names, method_names_long, color_order, name, title, y_label, x_range, n_methods=9,
                     n_problems=11):
    fig, axs = plt.subplots(n_problems, n_methods)
    colors = ['red', 'black', 'green']
    for n, ax in enumerate(axs.flat):
        for color in color_order:
            plot_with_error(
                ax, x_range, avgs[n][0], avgs[n][1], avgs[n][2], y_label
            )


def prepare_avgs(data, metric):
    """
    Prepares averages and standard deviation values

    Parameters:
    - data: Dictionary of loaded data
    - metric: String indicating the metric we want to measure
    """
    return {key: get_avg_gens(data[key][metric]) for key in data}



def prepare_avgs_multi(data, metric):
    return {key: get_avg_gens(data[key][metric], axis=1) for key in data}


def plot_multiple_series(n_problems, n_methods, data, f_names, method_names_long, color_order, name, title, y_label):
    """
    Create a figure with subplots for multiple series across problems and methods.

    Parameters:
    - n_problems: Number of rows (problems) in the subplot grid.
    - n_methods: Number of columns (methods) in the subplot grid.
    - data: List of data where each element is a tuple of three lists: (average, lower_bound, upper_bound).
    - f_names: List of problem names (used for titles).
    - method_names_long: List of full method names for the legend.
    - color_order: List of colors corresponding to the methods.
    - name: Filename to save the plot.
    - title: The title of the entire figure.
    - y_label: Label for the y-axis.
    """
    fig, axs = plt.subplots(n_problems, n_methods, figsize=(9.75, 6.75))
    fig.subplots_adjust(hspace=0.3)
    legend_objects = []

    for i in range(n_problems):
        for j in range(n_methods):
            ax = axs[i, j]
            method_index = i * n_methods + j
            if method_index >= len(data):
                ax.set_visible(False)
                continue

            avg, lower_bound, upper_bound = data[method_index]
            x_range = range(len(avg))

            ax.plot(x_range, avg, color=color_order[method_index], label=f_names[i])
            ax.fill_between(x_range, lower_bound, upper_bound, color=color_order[method_index], alpha=0.3)

            if i == 0:  # Capture legend objects from the first row
                legend_objects.append(ax.plot([], [], color=color_order[method_index])[0])

            ax.set_title(f_names[i], fontsize=12)
            if j == 0:
                ax.set_ylabel(y_label, fontsize=10)
            if i == n_problems - 1:
                ax.set_xlabel("Generations", fontsize=10)
            ax.set_ylim(bottom=0)

    fig.suptitle(title, fontsize=16)
    fig.legend(legend_objects, method_names_long, fontsize=10, loc='lower right', bbox_to_anchor=(1, 0.05), ncol=2)
    plt.show()
    fig.tight_layout()
    fig.savefig(f"../output/{name}.png")

import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_pickle_data(p):
    try:
        with open(p, "rb") as f:
            return [pickle.load(f) for _ in range(16)]
    except EOFError:
        print(f'EOF Error at {p}')
        return [None] * 16


def update_logs(pickle_data):
    bias, ind, preds, p_fit, n, fit_track, average_change, average_retention, p_size, histograms, mut_effect, xov_effect, sharp_list, sharp_std, density, density_list = pickle_data
    mut_list_effect = mut_effect[0]
    mut_cumul_effect = mut_effect[1]
    xov_list_effect = xov_effect[0]
    xov_cumul_effect = xov_effect[1]
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
        'den_log': density,
        'den_list': density_list
    }


def get_logs_cgp(base_path, max_e, problem_names):
    full_logs, full_fits, fit_tracks, active_nodes = [], [], [], []
    node_prop, avg_chg, avg_ret, p_sz_li, hist_li = [], [], [], [], []
    mut_cumul, mut_list, xov_cumul, xov_list = [], [], [], []
    sharp_in_mean, sharp_out_mean, sharp_in_std, sharp_out_std, den, den_list = [], [], [], [], [], []

    for name in problem_names:
        print(f"Loading {base_path}{name}")

        logs, p_log, track_log, node_log, prop_log = [], [], [], [], []
        change_log, retent_log, p_size_log, histog_log = [], [], [], []
        mut_cumul_log, mut_list_log, xov_cumul_log, xov_list_log = [], [], [], []
        sharp_in_list_log, sharp_out_list_log = [], []
        sharp_in_std_log, sharp_out_std_log, den_log, den_list_log = [], [], [], []

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
                den_list_log.append(updated_logs['den_list'])
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
        den_list.append(den_list_log)

    return [full_logs, np.array(full_fits), np.array(fit_tracks), np.array(active_nodes), np.array(node_prop),
            np.array(avg_chg), np.array(avg_ret), np.array(p_sz_li), hist_li, mut_list, mut_cumul, xov_list, xov_cumul,
            sharp_in_mean, sharp_in_std, sharp_out_mean, sharp_out_std, den, den_list]


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
        'density_distro': data[17],
        'density_list': data[18]
    }


def load_all_data(base_paths, max_e, f_names, problem_names):
    return {key: data_dict(get_logs_cgp(base_paths[key], max_e, problem_names)) for key in base_paths}


# Refactored by chatGPT
def calculate_avg_and_std(data, axis=1):
    data = np.array(data)
    if data.ndim == 1:
        axis = 0  # Adjust to axis 0 if data is 1D
    print(type(data))
    avgs = np.array([np.nanmean(d, axis=axis) for d in data])
    std_devs = np.array([np.nanstd(d, axis=axis) for d in data])
    return avgs, std_devs


def get_avg_gens(data, axis=1):
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


def plot_box_plots(f_names, problem_names, data_dicts, method_names, method_names_long, color_order, key, plot_name,
                   title, y_label):
    """
    Create a figure with subplots for the fitness evaluation on SR problems.

    Parameters:
    - f_names: List of problem names (used for titles).
    - data_dicts: Dictionary of data (one entry per method).
    - method_names: List of method names for labeling the x-axis.
    - method_names_long: List of full method names for the legend.
    - color_order: List of colors corresponding to the methods.
    """
    fig, axs = plt.subplots(len(problem_names), 1, figsize=(10.45, 12.1))
    fig.subplots_adjust(hspace=0)

    legend_objects = None

    for n, name in enumerate(problem_names):
        data = [
            data_dicts['cgp_base'][key][n],
            # data_dicts['cgp_40'][key][n],
            data_dicts['cgp_1x'][key][n],
            # data_dicts['cgp_vlen'][key][n],
            data_dicts['cgp_diversity'][key][n]
            # data_dicts['lgp_1x'][key][n],
            # data_dicts['cgp_2x'][key][n],
            # data_dicts['lgp_2x'][key][n],
            # data_dicts['cgp_sgx'][key][n],
            # data_dicts['lgp_base'][key][n],
        ]

        boxes = create_boxplot(axs[n], data, color_order, method_names)
        configure_axes(axs[n], name, y_label, method_names)

        if n == 0:  # Capture legend objects from the first plot
            legend_objects = boxes

    fig.suptitle(title, fontsize=16)
    fig.legend(legend_objects, method_names_long, fontsize=10, ncol=2, bbox_to_anchor=(0.5, 0.965), loc='upper center')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
    fig.savefig(f"../output/{plot_name}.png", format='png')


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

    # This is chatgpt's innovation
    def flatten_if_needed(arr):
        return arr.flatten() if arr.ndim > 1 else arr

    y_data, lower_bound, upper_bound = map(flatten_if_needed, [y_data, lower_bound, upper_bound])

    ax.fill_between(x_data, lower_bound, upper_bound, color=color, alpha=alpha)
    ax.plot(y_data, label=label, c=color)


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


import matplotlib.pyplot as plt


def plot_over_generations(f_names, problem_names, avgs, method_names_long, color_order, plot_name, title, y_label):
    """
    Create a figure with subplots for the similarity evaluation across methods.

    Parameters:
    - f_names: List of problem names (used for titles).
    - avgs: Dictionary of average retention data (per method).
    - method_names_long: List of full method names for the legend.
    - color_order: List of colors corresponding to the methods.
    - plot_name: Name of the file to save the plot.
    - title: Title of the plot.
    - y_label: Label for the y-axis.
    """
    n_problems = len(problem_names)
    fig, axs = plt.subplots(n_problems, 2, figsize=(9.75, 6.75))
    fig.subplots_adjust(hspace=0.3)
    legend_objects = []

    # Get the number of generations from the first method
    first_method_key = next(iter(avgs))  # Get the first key from the dictionary
    x_range = range(avgs[first_method_key][0][0].shape[0])  # Access the first array in the tuple of the first method

    first_method_key = next(iter(avgs))  # Get the first key from the dictionary
    sample_data = np.array(avgs[first_method_key])
    ndim = sample_data.ndim
    x_range = range(sample_data.shape[-1])  # Assuming x-axis is the last dimension

    for n, ax in enumerate(axs.flat[:len(problem_names)]):
        for method, color in zip(avgs.keys(), color_order):
            avg_data = np.array(avgs[method])

            # Create a tuple of slices based on the number of dimensions
            slices = tuple([slice(None) if i in [0, *range(2, ndim)] else n for i in range(ndim)])
            avg_data_to_plot = avg_data[slices]
            plot_with_error(
                ax, x_range, avg_data_to_plot[0], avg_data_to_plot[1], avg_data_to_plot[2],
                label=method, color=color
            )
            if n == 0:  # Capture legend objects from the first subplot
                legend_objects.append(ax.plot([], [], label=method, color=color)[0])
        configure_subplot(ax, problem_names[n], y_label, xlabel="Generations" if n > int(n_problems / 2) else None)

    axs.flat[-1].set_visible(False)
    fig.suptitle(title, fontsize=16)
    fig.legend(legend_objects, method_names_long, fontsize=10, loc='lower right', bbox_to_anchor=(1, 0.05), ncol=2)
    plt.show()
    fig.tight_layout()
    fig.savefig(f"../output/{plot_name}.png", format='png')


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
    return {key: get_avg_gens(data[key][metric], axis=0) for key in data}


# chatgpt whipped this up, it's some kind of magic
def prepare_avgs_cumul(data, metric, n_problems):
    """
    Prepares averages and standard deviation values for cumulative data.

    Parameters:
    - data: Dictionary of loaded data.
    - metric: String indicating the metric we want to measure.
    - n_problems: Number of problems.
    """

    def slice_array(arr, n, m):
        """
        Helper function to dynamically slice arrays based on their number of dimensions.
        """
        ndim = arr.ndim
        slices = [slice(None)] * (ndim - 2)  # Middle dimensions: keep all elements
        return arr[(n,) + tuple(slices) + (m,)]

    return {
        key: {
            n: {
                stat: np.nanmean(slice_array(np.array(data[key][metric]), n, m), axis=0)
                if stat.endswith('_avg') else np.nanstd(slice_array(np.array(data[key][metric]), n, m), axis=0)
                for stat, m in zip(['d_avg', 'n_avg', 'b_avg', 'd_std', 'n_std', 'b_std'], [0, 1, 2, 0, 1, 2])
            } for n in range(n_problems)
        } for key in data
    }


def prepare_avgs_density_distro(data, n_problems):
    """
    Prepares averages and standard deviation values for density_distro data.

    Parameters:
    - data: Dictionary of loaded data.
    - metric: String indicating the metric we want to measure.
    - n_problems: Number of problems.
    """

    def slice_array(arr, n, m):
        """
        Helper function to dynamically slice arrays based on their number of dimensions.
        """
        result = []
        for a in arr:
            result.append(a[n][m])
        return np.array(result)

    metric = 'density_distro'
    return {
        key: {
            n: {
                stat: np.nanmean(slice_array(np.array(data[key][metric]), n, m), axis=0)
                if stat.endswith('_avg') else np.nanstd(slice_array(np.array(data[key][metric]), n, m), axis=0)
                for stat, m in
                zip(['d_avg', 'n_avg', 'b_avg', 'd_std', 'n_std', 'b_std'], ['d', 'n', 'b', 'd', 'n', 'b'])
            } for n in range(n_problems)
        } for key in data
    }


def prepare_avgs_density_list(data):
    """
    Prepares average density lists for each problem in the data.

    Parameters:
    - data: Dictionary of loaded data

    Returns:
    - Averages of density lists organized by key, problem, and replicate.
    """
    metric = 'density_list'

    return {
        key: [
            [
                {cat: np.array(r[cat]) for cat in ['d', 'n', 'b']}
                for r in p
            ]
            for p in problem_list[metric]
        ]
        for key, problem_list in data.items()
    }


def prepare_avgs_multi(data, metric):
    def average_over_time(arrays):
        # Stack arrays and compute the mean over the first axis
        stacked = np.stack(arrays, axis=0)
        return np.mean(stacked, axis=0)

    if metric != 'density_distro':
        return {key: get_avg_gens(data[key][metric], axis=1) for key in data}
    else:
        result = {}
        for key in data:
            metric_data = data[key][metric]
            if isinstance(metric_data, list) and all(isinstance(x, list) for x in metric_data):
                # Initialize dictionaries to store lists of arrays
                avg_dicts = {'b': [], 'n': [], 'd': []}

                # Iterate over each list in the outer list
                for sublist in metric_data:
                    # Iterate over each dictionary in the sublist
                    for item in sublist:
                        if isinstance(item, dict):
                            for sub_key in ['b', 'n', 'd']:
                                if sub_key in item:
                                    avg_dicts[sub_key].append(item[sub_key])

                # Compute averages for each key
                result[key] = {sub_key: average_over_time(avg_dicts[sub_key]) for sub_key in ['b', 'n', 'd']}
            else:
                raise TypeError(f"Unexpected data type for metric '{metric}' in key '{key}': {type(metric_data)}")
        return result


def plot_multiple_series(f_names, problem_names, drift_colors, drift_names, drift_categories, data, name, title,
                         y_label, x_label=None, histogram=False):
    """
    Create a figure with subplots for multiple series across problems and methods.

    Parameters:
    - f_names: List of problem names (used for titles).
    - problem_names: List of problems (used for rows).
    - data: Dictionary of data where each element is a tuple of three lists: (average, lower_bound, upper_bound).
    - method_names_long: List of full method names for the legend.
    - color_order: List of colors corresponding to the methods.
    - name: Filename to save the plot.
    - title: The title of the entire figure.
    - y_label: Label for the y-axis.
    """
    n_problems = len(problem_names)
    n_methods = len(f_names)
    fig, axs = plt.subplots(n_problems, n_methods, figsize=(9.75, 6.75))
    fig.subplots_adjust(hspace=0.3)
    legend_objects = []

    for i, p_name in enumerate(problem_names):
        for j, f_name in enumerate(f_names):
            ax = axs[i, j]

            for k, series in enumerate(drift_categories):
                avg = data[f_name][i][f'{series}_avg']
                std = data[f_name][i][f'{series}_std']
                upper_bound = avg + std
                lower_bound = avg - std
                x_range = range(len(avg))
                if not histogram:
                    ax.plot(avg, color=drift_colors[k], label=drift_names[k])
                    ax.fill_between(x_range, lower_bound, upper_bound, color=drift_colors[k], alpha=0.3)
                else:
                    marker = ['*', 'o', 'v', '^', '<', '>']
                    try:
                        # ax.scatter(x_range, avg, color=drift_colors[k], label=drift_names[i], alpha=0.85, marker=marker[k])
                        ax.errorbar(x_range, avg, yerr=[lower_bound, upper_bound], fmt=marker[k], color=drift_colors[k],
                                    label=drift_names[k], capsize=3)
                    except ValueError:
                        print(f'ValueError in d_distro {f_name} {p_name}')

            if i == 0 and j == 0:  # Capture legend objects from the first subplot
                legend_objects.extend([ax.plot([], [], color=color)[0] for color in drift_colors])
            if i == 0:
                print(f_name)
                ax.set_title(f'{f_names[f_name]}\n{p_name}')
            else:
                ax.set_title(f'{p_name}')
            ax.set_title(p_name, fontsize=12)
            if j == 0:
                ax.set_ylabel(f'{y_label} {p_name}', fontsize=10)
            if i == n_problems - 1:
                xlab = "Generations" if x_label is None else x_label
                ax.set_xlabel(xlab, fontsize=10)
            ax.set_ylim(bottom=0)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.suptitle(title, fontsize=16)
    fig.legend(legend_objects, drift_names, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    plt.show()
    fig.savefig(f"../output/{name}.png", format='png')


import numpy as np
import matplotlib.pyplot as plt

def plot_density_heatmap_split(f_name_list, problem_names, replicates, data, name, title, y_label, drift_names, x_label=None):
    """
    Create split heatmaps showing the distribution of crossover points across generations
    for each method ('d', 'n', 'b') with a color legend.

    Parameters:
    - f_names: List of method names (used for titles).
    - problem_names: List of problems (used for rows).
    - replicates: Number of replicates in the data.
    - data: Dictionary of data structured as data[f_name][problem][replicate]['d'|'n'|'b'][generation][xover point].
    - name: Filename to save the plot.
    - title: The title of the entire figure.
    - y_label: Label for the y-axis (crossover points).
    - x_label: Label for the x-axis (generations) (optional).
    """
    f_names = f_name_list
    del f_names['cgp_base']
    n_problems = len(problem_names)
    n_methods = len(f_names)
    fig, axs = plt.subplots(n_problems, n_methods * 3, figsize=(18, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    method_names = ['d', 'n', 'b']
    method_titles = {'d': 'Method D', 'n': 'Method N', 'b': 'Method B'}

    for i, p_name in enumerate(problem_names):
        for j, f_name in enumerate(f_names):

            gens = len(data[f_name][j][0]['d'])  # Number of generations
            zeros = len(data[f_name][j][0]['d'][0])  # Number of crossover points (bins for y-axis)

            # Initialize 2D histogram arrays for each method
            value_data = {method: np.zeros((gens, zeros)) for method in method_names}

            for g in range(gens):
                for method in method_names:
                    for r in range(replicates):
                        # Get crossover points for current generation and method
                        xover_points = data[f_name][i][r][method][g]
                        for point in range(len(xover_points)):
                            value_data[method][g, point] = xover_points[point]

            # Plot heatmaps for each method in separate subplots
            #print(value_data)
            for k, method in enumerate(method_names):
                ax = axs[i, j * 3 + k]
                cax = ax.imshow(value_data[method].T, aspect='auto', origin='lower',
                                extent=[0, gens, 0, zeros], cmap='viridis')
                ax.set_title(f"{f_names[f_name]}, {problem_names[i]} - {drift_names[k]}")
                ax.set_xlabel(x_label or 'Generation')
                if j == 0:
                    ax.set_ylabel(y_label)
                # Add a color bar (legend) next to the heatmap
                fig.colorbar(cax, ax=ax, orientation='vertical', label='Count')

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.suptitle(title, fontsize=16)
    # fig.legend(legend_objects, drift_names, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    plt.show()
    fig.savefig(f"../output/{name}.png", format='png')

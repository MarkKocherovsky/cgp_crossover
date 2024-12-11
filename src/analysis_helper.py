import pickle
import psutil
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import pandas as pd
from numpy import log
from pandas import cut
from copy import copy, deepcopy
from sharpness import *
from helper import processSharpness
from scikit_posthocs import posthoc_mannwhitney
from scipy.signal import savgol_filter
intermediate_results = "../output/intermediate_results/"
desired_generations = 10000

def save_logs_to_disk(problem_name, results, folder="../output/intermediate_results/"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = os.path.join(folder, f"{problem_name}_results.pkl")
    print(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)


def load_pickle_data(p):
    data = []
    try:
        with open(p, "rb") as f:
            for _ in range(20):
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break  # Stop loading if EOF is reached
        return data
    except EOFError as e:
        print(e)
        print('analysis_helper.py::load_pickle_data()')
        return None


def update_logs(pickle_data):
    if pickle_data is None:
        return None

    try:
        (bias, ind, preds, p_fit, n, fit_track, average_change, average_retention, 
         p_size, histograms, mut_effect, xov_effect, sharp_list, sharp_std, 
         density, density_list, mut_den, mut_den_list, div_list, fit_mean) = pickle_data

        l_ind = n / len(ind) if ind else np.nan
        
        mut_list_effect, mut_cumul_effect = mut_effect
        xov_list_effect, xov_cumul_effect = xov_effect
        sharp_in_list, sharp_out_list = sharp_list
        sharp_in_std, sharp_out_std = sharp_std
        
        if np.isnan(p_fit):
            p_fit = np.inf
        return {
            'ind_log': ind,
            'p_log': p_fit,
            'track_log': fit_track,
            'node_log': n,
            'prop_log': l_ind,
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
            'den_list': density_list,
            'mut_den_log': mut_den,
            'mut_den_list_log': mut_den_list,
            'div_list_log': div_list,
            'fit_mean_log': fit_mean
        }
    except (ValueError, IndexError) as e:
        print(f"Error in update_logs: {e}")
        return None


def get_logs_cgp(base_path, f_name, max_e, problem_names):
    """
    full_fits, fit_tracks, active_nodes = [], [], []
    node_prop, avg_chg, avg_ret, p_sz_li, hist_li = [], [], [], [], []
    mut_cumul, mut_list, xov_cumul, xov_list = [], [], [], []
    sharp_in_mean, sharp_out_mean, sharp_in_std, sharp_out_std, den, den_list = [], [], [], [], [], []
    mut_den, mut_den_list = [], []
    div_list, fit_mean = [], []
    """
    for name in problem_names:
        print(f"Loading {base_path}{name}")
        print('RAM Used (GB):', psutil.virtual_memory()[3] / 1_000_000_000)

        # Initialize logs for the current problem
        p_log, track_log, node_log, prop_log = [], [], [], []
        change_log, retent_log, p_size_log, histog_log = [], [], [], []
        mut_cumul_log, mut_list_log, xov_cumul_log, xov_list_log = [], [], [], []
        sharp_in_list_log, sharp_out_list_log = [], []
        sharp_in_std_log, sharp_out_std_log, den_log, den_list_log = [], [], [], []
        mut_den_log, mut_den_list_log = [], []
        div_list_log, ind_log, fit_mean_log = [], [], []

        # Process logs one by one to avoid loading everything into memory at once
        for e in range(1, max_e + 1):
            p = f'{base_path}{name}/log/output_{e}.pkl'
            pickle_data = load_pickle_data(p)
            
            if pickle_data and pickle_data[0] is not None:
                updated_logs = update_logs(pickle_data)
                if updated_logs:  # Ensure the logs are not empty
                    ind_log.append(updated_logs['ind_log'])
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
                    mut_den_log.append(updated_logs['mut_den_log'])
                    mut_den_list_log.append(updated_logs['mut_den_list_log'])
                    div_list_log.append(updated_logs['div_list_log'])
                    fit_mean_log.append(updated_logs['fit_mean_log'])
                    try:
                        histog_log.append(updated_logs['histog_log'])
                    except KeyError:
                        print(updated_logs['histog_log'])
                
            # Free memory after each file is processed
            del pickle_data, updated_logs
        results = {
            'ind': ind_log,
            'p_fits': np.array(p_log),
            'fit_track': np.array(track_log),
            'nodes': np.array(node_log),
            'prop': np.array(prop_log),
            'average_change': np.array(change_log),
            'average_retention': np.array(retent_log),
            'p_size': np.array(p_size_log),
            #'histograms': hist_li,
            'mut_cumul_drift': np.array(mut_cumul_log),
            'mut_list_drift': np.array(mut_list_log),
            'xov_cumul_drift': np.array(xov_cumul_log),
            'xov_list_drift': np.array(xov_list_log),
            'sharp_in_mean': np.array(sharp_in_list_log),
            'sharp_in_std': np.array(sharp_in_std_log),
            'sharp_out_mean': np.array(sharp_out_list_log),
            'sharp_out_std': np.array(sharp_out_std_log),
            'density_distro': np.array(den_log),
            'density_list': np.array(den_list_log),
            'mut_density_distro': np.array(mut_den_log),
            'mut_density_list': np.array(mut_den_list_log),
            'div_list': np.array(div_list_log),
            'fit_mean': np.array(fit_mean_log)
        }

        save_logs_to_disk(f'{f_name}_{name}', results)
        # Free memory used by problem-level logs
        del (ind_log, p_log, track_log, node_log, prop_log, change_log, retent_log, p_size_log, histog_log,
             mut_cumul_log, mut_list_log, xov_cumul_log, xov_list_log, sharp_in_list_log, sharp_out_list_log,
             sharp_in_std_log, sharp_out_std_log, den_log, den_list_log, mut_den_log, mut_den_list_log,
             div_list_log, fit_mean_log, results)
        # Set logs to None to completely dereference them
        p_log = track_log = node_log = prop_log = change_log = retent_log = p_size_log = histog_log = None
        mut_cumul_log = mut_list_log = xov_cumul_log = xov_list_log = None
        sharp_in_list_log = sharp_out_list_log = sharp_in_std_log = sharp_out_std_log = None
        den_log = den_list_log = mut_den_log = mut_den_list_log = None
        div_list_log = ind_log = fit_mean_log = results = None

        gc.collect()  # Force garbage collection 
        print(f"Finished processing {name}. RAM Used (GB):", psutil.virtual_memory()[3] / 1_000_000_000)


def load_all_data(base_paths, max_e, f_names, problem_names):
    for key in base_paths:
        get_logs_cgp(base_paths[key], key, max_e, problem_names)
        gc.collect()


# Refactored by chatGPT
def calculate_avg_and_std(data, axis=1):
    try:
        data = np.array(data)
    except Exception as e:
        raise ValueError('analysis_helper.py::calculate_avg_and_std: unable to make np array from data') from e    
    if data.ndim == 1:
        axis = 0  # Adjust to axis 0 if data is 1D
    print(data.shape)
    avgs = np.nanmean(data, axis=axis)
    std_devs = np.nanstd(data, axis=axis)
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


def create_boxplot(ax, data, colors, method_names, log):
    print(f"Creating boxplot with {len(data)} data arrays and {len(method_names)} method names.")
    boxes = ax.boxplot(data, showfliers=False, patch_artist=True)
    medians = np.nanmedian(data, axis=1)
    for box, color in zip(boxes['boxes'], colors):
        box.set_facecolor(color)
    ax.set_xticks(list(range(1, len(method_names) + 1)), method_names)
    ax.axhline(y=np.min(medians), color='black', linestyle='--', linewidth=1.5)
    ax.axhline(y=np.max(medians), color='darkgray', linestyle='dotted', linewidth=1.25)
    if log:
       ax.set_yscale('log')
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
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=12)
    #ax.set_ylim(bottom=y_bottom)


def plot_box_plots(f_names, problem_names, method_names, method_names_long, color_order, metric, plot_name,
                   title, y_label, log=False):
    """
    Create a figure with subplots for the fitness evaluation on SR problems.

    Parameters:
    - f_names: List of problem names (used for titles).
    - data_dicts: Dictionary of data (one entry per method).
    - method_names: List of method names for labeling the x-axis.
    - method_names_long: List of full method names for the legend.
    - color_order: List of colors corresponding to the methods.
    """
    fig, axs = plt.subplots(len(problem_names), 1, figsize=(7.5, 9.5))#figsize=(10.45, 12.1))
    fig.subplots_adjust(hspace=0)
    axs = np.atleast_1d(axs)
    legend_objects = None
    for n, p_name in enumerate(problem_names):
        data = []
        for f, f_name in enumerate(f_names):
            with open(f'{intermediate_results}{f_name}_{p_name}_results.pkl', 'rb') as f:
                print(f'opening {intermediate_results}{f_name}_{p_name}_results.pkl')
                data.append(pickle.load(f)[metric])
        boxes = create_boxplot(axs[n], data, color_order, method_names, log)
        data = None
        configure_axes(axs[n], p_name, y_label, method_names)
        if n == 0:  # Capture legend objects from the first plot
            legend_objects = boxes
        boxes = None
        gc.collect()
    fig.suptitle(title, fontsize=24)
    fig.legend(legend_objects, method_names_long, fontsize=14, ncol=5, bbox_to_anchor=(0.5, 0.95), loc='upper center')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
    with open(f'../output/graphs_raw/{plot_name}.pkl', "wb") as f:
        pickle.dump(fig, f)
    gc.collect() 
    print('BoxPlots: RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    #fig.savefig(f"../output/{plot_name}.png", format='png')


def plot_with_error(ax, x_data, y_data, lower_bound, upper_bound, bottom, top, label, color, alpha=0.025, log = False, logx = False, smoothing=False, marker_space=500):
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
    global desired_generations
    # This is chatgpt's innovation
    def flatten_if_needed(arr):
        return arr.flatten() if arr.ndim > 1 else arr

    y_data, lower_bound, upper_bound = map(flatten_if_needed, [y_data, lower_bound, upper_bound])
    print(len(x_data))
    if len(x_data) != y_data.shape[0]:
        x_data = list(range(y_data.shape[0]))
    try:
        x_data = x_data[-desired_generations:]
        y_data = y_data[-desired_generations:]
        lower_bound = lower_bound[-desired_generations:]
        upper_bound = upper_bound[-desired_generations:]
        if smoothing:
            y_data = savgol_filter(y_data, 100, 2)
            lower_bound = savgol_filter(lower_bound, 100, 2)
            upper_bound = savgol_filter(upper_bound, 100, 2)
    except IndexError as e:
        print("analysis_helper.py::plot_with_error")
        print(e)
        print(len(x_data))
    marker_x = np.arange(marker_space, len(x_data), marker_space).astype(np.int32)
    marker_y = y_data[marker_x]
    #x_data = savgol_filter(x_data, 100, 3)
    #y_data = savgol_filter(y_data, 100, 3)
    #lower_bound = savgol_filter(lower_bound, 100, 3)
    #upper_bound = savgol_filter(upper_bound, 100, 3)
    try:
        if log:
            ax.set_yscale('log')
        if logx:
            ax.set_xscale('log')
            x_data[0] = 1e-12
        ax.fill_between(x_data, lower_bound, upper_bound, color=color, alpha=alpha)
        ax.plot(y_data, label=label, c=color)
        print(marker_x, marker_y)
        ax.scatter(marker_x, marker_y, c=color, s=40)
        ax.set_ylim(bottom, top)
    except ValueError as e:
        print(e)
        print(f'x_data.shape: {len(x_data)}')
        print(f'y_data.shape: {y_data.shape}')
        print(f'lower_bound.shape: {lower_bound.shape}')
        print(f'upper_bound.shape: {upper_bound.shape}')
        exit(1)


def configure_subplot(ax, title, ylabel, xlabel=None, log = False):
    """
    Configure the appearance of a subplot axis.

    Parameters:
    - ax: The axis to configure.
    - title: The title for the axis.
    - ylabel: The label for the y-axis.
    - xlabel: The label for the x-axis (optional).
    """
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)


import matplotlib.pyplot as plt


def plot_over_generations(f_names, problem_names, metric, bottom, top, method_names_long, color_order, plot_name, title, y_label, log = False, logx = False, marker_space=500, smoothing=False, legend=False):
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
    print(f'Starting {plot_name}')
    n_problems = len(problem_names)
    n_functions = len(f_names)
    print(f'{n_problems} problems\t{n_functions} xover methods')
    if n_problems > 1:
        fig, axs = plt.subplots(int(np.round(n_problems/2)), 2, figsize=(12, 12))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(12,12))
    fig.subplots_adjust(hspace=0.3)
    legend_objects = []
    x_range = None
    ndim = 3
    if isinstance(axs, np.ndarray):
        axes = axs.flat[:len(problem_names)]
    else:
        axes = [axs]  # Wrap it in a list for consistent iteration
    for n, ax in enumerate(axes):
        p_name = problem_names[n]
        for j, f_name in enumerate(f_names):
            with open(f'{intermediate_results}{f_name}_{p_name}_results.pkl', 'rb') as f:
                print(f'opening {intermediate_results}{f_name}_{p_name}_results.pkl')
                data = pickle.load(f)[metric]
            color = color_order[j]
            if metric == 'p_size' and 'dnc' in f_name: #this is the most braindead way of doing it but stfu
                data *= (64 + 1 + 11)
                data = data.astype(np.int32)
            #avgs = np.nanmean(data, axis=0)
            stats = np.percentile(data, [0, 25, 50, 75, 100], axis = 0)
            meds = stats[2, :]
            #stds = np.nanstd(data, axis = 0)
            first_q = stats[1, :]
            third_q = stats[3, :]
            del(data)
            # Create a tuple of slices based on the number of dimensions
            if x_range is None:
                x_range = list(range(len(meds)))
            plot_with_error(
                ax, x_range, meds, first_q, third_q, bottom, top,
                label=f_names[f_name], color=color, log = log, logx = logx, marker_space = marker_space, smoothing=smoothing
            )
            if n == 0:  # Capture legend objects from the first subplot
                legend_objects.append(ax.plot([], [], label=f_names[f_name], color=color)[0])
        configure_subplot(ax, problem_names[n], y_label, xlabel="Generations" if n > int(n_problems / 2) else None, log=log)
        avgs = None
    avg_data = None
    if n_problems % 2 != 0 and n_problems > 1:
        axs.flat[-1].set_visible(False)
    fig.suptitle(title, fontsize=24, y = 0.55)
    n_cols = 2 if n_problems <= 1 else 4
    y_anchor = 0.95 if n_problems <= 1 else 0.95
    if legend:
        fig.legend(legend_objects, method_names_long, fontsize=18, ncol=3, bbox_to_anchor=(0.5, y_anchor), loc='upper center')
        fig.tight_layout()
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    #pickle.dump(f'../output/graphs_raw/{plot_name}.pkl')
    with open(f'../output/graphs_raw/{plot_name}.pkl', "wb") as f:
        pickle.dump(fig, f)
    gc.collect()
    #fig.savefig(f"../output/{plot_name}.png", format='png')


def plot_individuals(avgs, f_names, method_names_long, color_order, name, title, y_label, x_range, n_methods=9,
                     n_problems=11):
    fig, axs = plt.subplots(n_problems, n_methods)
    colors = ['red', 'black', 'green']
    for n, ax in enumerate(axs.flat):
        for color in color_order:
            plot_with_error(
                ax, x_range, avgs[n][0], avgs[n][1], avgs[n][2], y_label
            )


def prepare_avgs(data):
    """
    Prepares averages and standard deviation values

    Parameters:
    - data: Dictionary of loaded data
    - metric: String indicating the metric we want to measure
    """
    print([data[key].shape for key in data])
    return {key: get_avg_gens(data[key], axis=0) for key in data}


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


def prepare_avgs_density_distro(data, metric='density_distro'):
    """
    Prepares averages and standard deviation values for density_distro data.

    Parameters:
    - data: Dictionary of loaded data.
    - metric: String indicating the metric we want to measure.
    - n_problems: Number of problems.
    """

    def slice_array(arr, m):
        """
        Helper function to dynamically slice arrays based on their number of dimensions.
        """
        result = []
        for a in arr:
            result.append(a[m])
        return np.array(result)

    return { 
             stat: np.nanmean(slice_array(np.array(data), m), axis=0)
             if stat.endswith('_avg') else np.nanstd(slice_array(np.array(data), m))
             for stat, m in zip(['d_avg', 'n_avg', 'b_avg', 'd_std', 'n_std', 'b_std'], ['d', 'n', 'b', 'd', 'n', 'b'])
           }
    """
        key: {
            n: {
                stat: np.nanmean(slice_array(np.array(data[key][metric]), n, m), axis=0)
                if stat.endswith('_avg') else np.nanstd(slice_array(np.array(data[key][metric]), n, m), axis=0)
                for stat, m in
                zip(['d_avg', 'n_avg', 'b_avg', 'd_std', 'n_std', 'b_std'], ['d', 'n', 'b', 'd', 'n', 'b'])
            } for n in range(n_problems)
        } for key in data
    }
   """


def prepare_avgs_density_list(data, metric = 'density_list'):
    """
    Prepares average density lists for each problem in the data.

    Parameters:
    - data: Dictionary of loaded data

    Returns:
    - Averages of density lists organized by key, problem, and replicate.
    """

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


def plot_multiple_series(f_names, problem_names, drift_colors, drift_names, drift_categories, metric, name, title,
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
            if f_name == 'cgp_base':
                continue
            ax = axs[i, j]
            with open(f'{intermediate_results}{f_name}_{p_name}_results.pkl', 'rb') as f:
                print(f'opening {intermediate_results}{f_name}_{p_name}_results.pkl')
                data = pickle.load(f)[metric]
            for k, series in enumerate(drift_categories):
                series_data = [data[r][series] for r in range(len(data))]
                stats = np.percentile(data, [0, 25, 50, 75, 100], axis = 0)
                meds = stats[2, :]
                #stds = np.nanstd(data, axis = 0)
                first_q = stats[1, :]
                third_q = stats[3, :]
                del(data)
                """
                avg = np.nanmean(series_data, axis=0)
                std = np.nanstd(series_data, axis=0)
                upper_bound = avg + std
                lower_bound = avg - std
                """
                x_range = range(len(avg))
                if not histogram:
                    ax.plot(meds, color=drift_colors[k], label=drift_names[k])
                    ax.fill_between(x_range, first_q, third_q, color=drift_colors[k], alpha=0.025)
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
    
    #pickle.dump(f'../output/graphs_raw/{plot_name}.pkl')
    with open(f'../output/graphs_raw/{name}.pkl', "wb") as f:
        pickle.dump(fig, f)
    gc.collect()
    #fig.savefig(f"../output/{name}.png", format='png')


import numpy as np
import matplotlib.pyplot as plt

#thanks chat
def save_density_csv(data, file_name, f_names, p_names):
    csv_data = []
    f_name = 'cgp_1x'
    p = 0
    p_name = p_names[p]
    k = 0
    replicate = data[f_name][p][k]
    for drift_type in ['d', 'n', 'b']:  # Assuming drift is one of these keys
        print(f'{file_name} {f_name} {p_name} {k} {drift_type}')
        for g, generation in enumerate(replicate[drift_type]):
            for index, value in enumerate(replicate[drift_type][generation]):
                # Create a row with the relevant data
                row = [f_name, p_name, k, drift_type, g, index, value]
                csv_data.append(row)
    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(csv_data, columns=['Algorithm', 'Problem', 'Replicate', 'Drift', 'Generation', 'Index', 'Value'])

    # Save DataFrame as a CSV file
    df.to_csv('data_by_problem.csv', index=False)

    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(csv_data, columns=['Algorithm', 'Problem', 'Replicate', 'Drift', 'Generation', 'Index', 'Value'])
    # Save DataFrame as a CSV file
    df.to_csv(f'../output/density_logs/{file_name}.csv', index=False)
    print(f'Saved to ../output/density_logs/{file_name}.csv')
def calculate_bins(max_bin):
    if np.isnan(max_bin) or max_bin <= 1:
        max_bin = 1
        bins = [0, 1]
    elif max_bin < 15:
        bins = np.concatenate((np.array([0, 1]), np.arange(2, max_bin + 2, 1)))
    elif max_bin < 50:
        bins = np.concatenate((np.array([0, 1]), np.arange(5, max_bin + 5, 5)))
    elif max_bin < 100:
        bins = np.concatenate((np.array([0, 1]), np.arange(10, max_bin + 10, 10)))
    elif max_bin < 400:
        bins = np.concatenate((np.array([0, 1]), np.arange(25, max_bin + 25, 25)))
    elif max_bin < 500:
        bins = np.concatenate((np.array([0, 1]), np.arange(50, max_bin + 50, 50)))
    elif max_bin < 1000:
        bins = np.concatenate((np.array([0, 1]), np.arange(100, max_bin + 100, 100)))
    elif max_bin < 5000:
        bins = np.concatenate((np.array([0, 1]), np.arange(500, max_bin + 500, 500)))
    elif max_bin < 10000:
        bins = np.concatenate((np.array([0, 1]), np.arange(500, max_bin + 500, 500)))
    else:
        bins = np.concatenate((np.array([0, 1]), np.arange(1000, max_bin + 1000, 1000)))

    return bins


def plot_density_heatmap_split(f_name_list, short_name_list, problem_names, replicates, metric, name, title, y_label, drift_names, p_count=3, x_label=None, g_count=100, max_gen=None):
    """
    Create split heatmaps showing the distribution of crossover points across generations
    for each method ('d', 'n', 'b') with a color legend.
    """
    f_names = copy(f_name_list)
    if 'mut' not in metric:
        for key in ['cgp_base', 'cgp_ep', 'cgp_rl_small', 'cgp_rl']:
            f_names.pop(key, None)

    n_problems = len(problem_names)
    n_algorithms = len(f_names)
    n_methods = 3  # Fixed number of methods: d, n, b

    if n_problems == 1:
        n_rows = n_algorithms
        n_cols = n_methods
    else:
        n_rows = n_problems
        n_cols = n_algorithms * n_methods

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(80, 15))
    fig.subplots_adjust(hspace=0.1, wspace=0.25)

    method_names = ['d', 'n', 'b']
    method_binning = {'d': [], 'n': [], 'b': []}
    group_count = g_count

    for i, p_name in enumerate(problem_names):
        for j, f_name in enumerate(f_names):
            print(f'{intermediate_results}{f_name}_{p_name}_results.pkl')
            with open(f'{intermediate_results}{f_name}_{p_name}_results.pkl', 'rb') as f:
                data = pickle.load(f)[metric]
            point_count = 1 if 'sgx' in f_name or 'vlen' in f_name else p_count
            gens = len(data[0]['d']) if max_gen is None else max_gen
            zeros = len(data[0]['d'][0]) // point_count
            if ('dnc' in f_name or 'unx' in f_name or '2x' in f_name) and point_count > 1:
                zeros += 1
            if 'vlen' in f_name and 'mut' not in metric:
                zeros = 64

            gen_groups = gens // group_count
            value_data = {method: np.zeros((gen_groups, zeros)) for method in method_names}

            for method in method_names:
                for r in range(replicates):
                    for g_group in range(gen_groups):
                        gen_range = slice(g_group * group_count, (g_group + 1) * group_count)
                        for g in range(gen_range.start, gen_range.stop):
                            xover_points = data[r][method][g][:64] if 'vlen' in f_name and 'mut' not in metric else data[r][method][g]
                            xover_points_grouped = np.add.reduceat(xover_points, np.arange(0, len(xover_points), point_count))
                            value_data[method][g_group] += np.where(xover_points_grouped > 0.0, xover_points_grouped, 0)

                max_bin = np.nanmax([g for g in value_data[method]])
                bins = calculate_bins(max_bin)
                method_binning[method] = deepcopy(bins)

            method_binning = {method: np.array(method_binning[method]) for method in method_binning}
            axs = np.atleast_2d(axs)

            for k, method in enumerate(method_names):
                ax = axs[j if n_problems == 1 else i, j * n_methods + k] if n_problems > 1 else axs[j, k]
                colors = ['black', 'hotpink', 'red', 'orange', 'yellow', 'white', 'blue', 'cyan']
                color_list = np.linspace(0, 1, len(method_binning[method]))
                cmap = mpl.colors.ListedColormap(plt.get_cmap(mpl.colors.LinearSegmentedColormap.from_list("custom_cmap", colors))(color_list))
                norm = mpl.colors.BoundaryNorm(method_binning[method], cmap.N, clip=True)
                cax = ax.imshow((value_data[method]).T, aspect='auto', origin='lower', extent=[0, gens, 0, zeros], cmap=cmap, norm=norm)
                ax.set_title(f"{short_name_list[f_name]}, {problem_names[i]} - {drift_names[k]}", size=18)
                ax.set_xlabel(x_label or 'Generation (Grouped by every 100)', size=14)
                if j == 0 and n_problems > 1:
                    ax.set_ylabel(y_label, size = 10)
                elif k == 0 and n_problems == 1:
                    ax.set_ylabel(y_label, size = 14)
                cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Operations')
                cbar.set_ticks(method_binning[method])
                cbar.set_ticklabels([str(int(b)) for b in method_binning[method]], size=14)
                cbar.set_label('Operations', size=14)


    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.suptitle(title, fontsize=24)
    with open(f'../output/graphs_raw/{name}.pkl', "wb") as f:
        pickle.dump(fig, f)


def get_medians(f_name_list, problem_names, metric, name='medians'):
    medians = pd.DataFrame(index=f_name_list, columns=problem_names)
    print('collecting medians')
    global intermediate_results
    for f_name in f_name_list:
        for j, p_name in enumerate(problem_names):
            with open(f'{intermediate_results}{f_name}_{p_name}_results.pkl', 'rb') as f:
                print(f'opening {intermediate_results}{f_name}_{p_name}_results.pkl')
                data = pickle.load(f)[metric]
            medians[p_name][f_name] = np.nanmedian(data)
    data = None
    gc.collect()
    medians.to_csv(f'../output/{name}.csv')
    return medians


def get_significance(f_name_list, problem_names, metric, name='significance'):
    # Creating a list of method names (from f_name_list keys if it's a dict)
    name_list = [f_name_list[key] for key in f_name_list] if isinstance(f_name_list, dict) else f_name_list

    for i, p_name in enumerate(problem_names):
        # Collect the metric data
        data = []
        for j, f_name in enumerate(f_name_list):
            with open(f'{intermediate_results}{f_name}_{p_name}_results.pkl', 'rb') as f:
                print(f'opening {intermediate_results}{f_name}_{p_name}_results.pkl')
                data.append(pickle.load(f)[metric])

        try:
            metric_data = np.array(data)
        except (TypeError, ValueError) as e:
            print(f'{p_name}')
            print(f'src/analysis_helper.py::get_significance: {e}')
            print([d.shape for d in data])
            print(data)
            exit(1)

        # Reshape the data into long format for the posthoc test
        df = pd.DataFrame(metric_data.T, columns=name_list)
        df_long = df.melt(var_name='algorithm', value_name='value')

        # Perform the posthoc test
        sig_df = posthoc_mannwhitney(df_long, val_col='value', group_col='algorithm')
        print(f"Significance results for {p_name}:")
        print(sig_df)
        sig_df.to_csv(f'../output/significance/{name}_{p_name}.csv')
        data = df = df_long = sig_df = metric_data = None
        gc.collect()

from functions import *  # Assuming necessary imports are here
from cgp_fitness import *  # Assuming necessary imports are here

def plot_sharpness(f_names, p_names, problems, colors, method_names_long, n_points=100, n_replicates=3):
    bias = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    def createInputVector(x_in, c, num_inputs=1):
        vec = np.zeros((x_in.shape[0], c.shape[0] + 1))
        x_in = x_in.reshape(-1, num_inputs)
        vec[:, :num_inputs] = x_in
        vec[:, num_inputs:] = c
        return vec

    models = {}
    n_problems = len(p_names)
    n_fs = len(f_names)
    
    # Create figure and subplots
    fig, axs = plt.subplots(n_replicates, n_problems * n_fs, figsize=(6 * n_problems * n_fs, 6 * n_replicates))
    fig.subplots_adjust(hspace=0.3)
    legend_objects = []
    labels = []
    
    for p, p_name in enumerate(p_names):
        problem = problems[p]
        low_bound, up_bound = problem.x_rng
        x = np.linspace(low_bound, up_bound, n_points)
        y = np.fromiter(map(problem, list(x)), dtype=np.float32)
        for r in range(n_replicates):
            for f, f_name in enumerate(f_names):
                # Accessing the correct subplot
                ax_idx = p * n_fs + f  # Correct index for the subplot
                axs[r, ax_idx].plot(x, y, color='black', label='Ground Truth', linewidth=1) 
        for j, f_name in enumerate(f_names):
            # Load the data for the current problem and method
            with open(f'{intermediate_results}{f_name}_{p_name}_results.pkl', 'rb') as f:
                print(f'opening {intermediate_results}{f_name}_{p_name}_results.pkl')
                data = pickle.load(f)
            
            # Extract models and sort based on p_fit
            for r in range(len(data['ind'])):
                models[f'model_{r}'] = [data['ind'][r], data['p_fits'][r]]
            
            # Sort models based on p_fit and take the top n_replicates
            sorted_models = sorted(models.items(), key=lambda item: item[1][1])
            top_n_models = dict(sorted_models[:n_replicates])
            
            for m, model in enumerate(top_n_models):
                train_x = createInputVector(x, bias)
                fitness = Fitness()
                sharp_in_manager = SAM_IN(train_x)
                sharp_out_manager = SAM_OUT()
                
                sharp_in_list, sharp_out_list, _ = [], [], []
                sharp_in_list, _, sharp_out_list, _ = processSharpness(train_x, 1, 0, 1, problem, 
                                                                       sharp_in_manager, [fitness], 
                                                                       [top_n_models[model][1]], 
                                                                       [top_n_models[model][0]], 
                                                                       y, sharp_in_list, _, 
                                                                       sharp_out_manager, sharp_out_list, _,
                                                                       n_neighbors = 1000)
                
                # Generate predictions using the model
                preds = np.array(fitness(train_x, y, top_n_models[model][0], opt=1)[0])
                
                # Access the correct subplot
                ax_idx = p * n_fs + j  # Correct subplot index for problem p and method j
                axs[m, ax_idx].plot(x, preds, color=colors[j],
                                    label=f'{f_names[f_name]}_elite{m} | fitness = {np.round(top_n_models[model][1], 5)} | SAM-in = {np.round(sharp_in_list[0], 3)} | SAM-Out = {np.round(sharp_out_list[0], 3)}')
                axs[m, ax_idx].set_title(f'{f_names[f_name]}_elite{m}\nfitness = {top_n_models[model][1]:.5e}\nSAM-in = {sharp_in_list[0]:.5e}\nSAM-Out = {sharp_out_list[0]:.5e}')
                #axs[m, ax_idx].legend()
    
    # Set a super title for the entire figure
    fig.suptitle('Elite Comparison on Selected Problems', fontsize=16)
    fig.tight_layout()
    
    # Save the figure
    plot_name = 'sharpness_elites'
    with open(f'../output/graphs_raw/{plot_name}.pkl', "wb") as f:
        pickle.dump(fig, f)
    
    # Clean up memory
    gc.collect()
         
def get_elite_sharpness(f_names, p_names, problems, color_order, method_names, method_names_long, y_label = 'r$\mathrm{Average}$(SAM-IN)', title='sharpness', log=True, n_points=50):
    bias = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    def createInputVector(x_in, c, num_inputs=1):
        vec = np.zeros((x_in.shape[0], c.shape[0] + 1))
        x_in = x_in.reshape(-1, num_inputs)
        vec[:, :num_inputs] = x_in
        vec[:, num_inputs:] = c
        return vec

    models = {}
    sharpness = {key: {fkey: [] for fkey in f_names} for key in p_names}
    n_problems = len(p_names)
    n_fs = len(f_names)
    
    for p, p_name in enumerate(p_names):
        problem = problems[p]
        low_bound, up_bound = problem.x_rng
        x = np.linspace(low_bound, up_bound, n_points)
        y = np.fromiter(map(problem, list(x)), dtype=np.float32)
        for j, f_name in enumerate(f_names):
            # Load the data for the current problem and method
            with open(f'{intermediate_results}{f_name}_{p_name}_results.pkl', 'rb') as f:
                print(f'opening {intermediate_results}{f_name}_{p_name}_results.pkl')
                data = pickle.load(f)
            
            # Extract models and sort based on p_fit
            for r in range(len(data['ind'])):
                models[f'model_{r}'] = [data['ind'][r], data['p_fits'][r]]
            
            # Sort models based on p_fit and take the top n_replicates
            sorted_models = sorted(models.items(), key=lambda item: item[1][1])
            
            train_x = createInputVector(x, bias)
            fitness = Fitness()
            sharp_in_manager = SAM_IN(train_x)
            sharp_out_manager = SAM_OUT()
            
            for model in sorted_models:
                sharp_in_list, sharp_out_list, _ = [], [], []
                sharp_in_list, _, sharp_out_list, _ = processSharpness(train_x, 1, 0, 1, problem, 
                                                                   sharp_in_manager, [fitness], 
                                                                   [model[1][1]], 
                                                                   [model[1][0]], 
                                                                   y, sharp_in_list, _, 
                                                                   sharp_out_manager, sharp_out_list, _, 2,
                                                                   n_neighbors = 1)
               
                
                sharpness[p_name][f_name].append(sharp_in_list[0])
        sharpness[p_name][f_name] = np.array(sharpness[p_name][f_name]) 
    # Clean up memory
    fig, axs = plt.subplots(len(p_names), 1, figsize=(7.5, 9.5))#figsize=(10.45, 12.1))
    fig.subplots_adjust(hspace=0)

    legend_objects = None

    for n, p_name in enumerate(p_names):
        data = [sharpness[p_name][f] for f in f_names]
        boxes = create_boxplot(axs[n], data, color_order, method_names, log)
        data = None
        configure_axes(axs[n], p_name, y_label, method_names)
        if n == 0:  # Capture legend objects from the first plot
            legend_objects = boxes
        boxes = None
        gc.collect()
    fig.suptitle(title, fontsize=24)
    fig.legend(legend_objects, method_names_long, fontsize=14, ncol=5, bbox_to_anchor=(0.5, 0.965), loc='upper center')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
    with open(f'../output/graphs_raw/{title}.pkl', "wb") as f:
        pickle.dump(fig, f)
    gc.collect() 
    print('BoxPlots: RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    return sharpness
    gc.collect()

from scipy.stats import spearmanr

np.random.seed(42)
def sharpnessCorrelation(f_names, p_names, problems, colors, method_names_long):
    bias = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    def createInputVector(x_in, c, num_inputs=1):
        vec = np.zeros((x_in.shape[0], c.shape[0] + 1))
        x_in = x_in.reshape(-1, num_inputs)
        vec[:, :num_inputs] = x_in
        vec[:, num_inputs:] = c
        return vec
    sharp_data = pd.DataFrame(columns=['Problem', 'Xover', 'Replicate', 'Fitness', 'SAM-In', 'SAM-Out'])
    corr_in_data = pd.DataFrame(index=p_names, columns=f_names)
    corr_out_data = pd.DataFrame(index=p_names, columns=f_names)
    models = {}
    legend_objects = []
    n_problems = len(p_names)
    n_fs = len(f_names)
    if n_problems > 1:
        fig, axs = plt.subplots(int(np.round(n_problems/2)), 2, figsize=(12, 12))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(12,12))
    if isinstance(axs, np.ndarray):
        axes = axs.flat[:len(p_names)]

    for p, p_name in enumerate(p_names):
        problem = problems[p]
        low_bound, up_bound = problem.x_rng
        x = np.linspace(low_bound, up_bound, 30)
        y = np.fromiter(map(problem, list(x)), dtype=np.float32)
        for j, f_name in enumerate(f_names):
            sharpness_data = []
            # Load the data for the current problem and method
            with open(f'{intermediate_results}{f_name}_{p_name}_results.pkl', 'rb') as f:
                print(f'opening {intermediate_results}{f_name}_{p_name}_results.pkl')
                data = pickle.load(f)
            
            # Extract models and sort based on p_fit
            for r in range(len(data['ind'])):
                model = data['ind'][r]
                replicate_fitness = data['p_fits'][r]
            
                train_x = createInputVector(x, bias)
                fitness = Fitness()
                sharp_in_manager = SAM_IN(train_x)
                sharp_out_manager = SAM_OUT()
            
                sharp_in_list, sharp_out_list, _ = [], [], []

                sharp_in_list, _, sharp_out_list, _ = processSharpness(train_x, 1, 0, 1, problem, 
                                                                   sharp_in_manager, [fitness], 
                                                                   [replicate_fitness], [model], 
                                                                   y, sharp_in_list, _, 
                                                                   sharp_out_manager, sharp_out_list, _, 2,
                                                                   n_neighbors = 25)
                
                sharpness_data.append([p_name, f_name, r, replicate_fitness, sharp_in_list[0], sharp_out_list[0]])
            sharp_data = pd.concat([sharp_data, pd.DataFrame(sharpness_data, columns=sharp_data.columns)], ignore_index=True) 
            corr_data = sharp_data.query(f'Problem == "{p_name}" and Xover == "{f_name}"')
            corr_in_data.at[p_name, f_name] = spearmanr(corr_data['Fitness'], corr_data['SAM-In']).statistic
            corr_out_data.at[p_name, f_name] = spearmanr(corr_data['SAM-Out'], corr_data['Fitness']).statistic
            axes[p].scatter(corr_data['Fitness'], corr_data['SAM-Out'], color = colors[j], s=5)
            #axes[p].set_yscale('log')
            #axes[p].set_xscale('log')
            axes[p].set_title(f'{p_name}')
            axes[p].set_ylabel('SAM-Out')
            axes[p].set_xlabel('Fitness')
            axes[p].text(0.87, 0.07, 'Less Fit')
            axes[p].text(0.07, 0.92, 'Sharper')
            axes[p].arrow(0.87, 0.05, 0.05, 0, head_width=0.02)
            axes[p].arrow(0.05, 0.87, 0, 0.05, head_width=0.02)
            if p == 0:  # Capture legend objects from the first subplot
                legend_objects.append(axes[p].plot([], [], label=f_names[f_name], color=colors[j])[0])

        
    corr_in_data.to_csv('../output/SAM-In.csv')
    corr_out_data.to_csv('../output/SAM-Out.csv')
    sharp_data.to_csv('../output/sharpness_data.csv')
    fig.suptitle('SAM-Out vs. Fitness', fontsize=24)
    print(legend_objects, method_names_long)
    fig.legend(legend_objects, method_names_long, fontsize=14, ncol=5, bbox_to_anchor=(0.5, 0.95), loc='upper center')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
    with open(f'../output/graphs_raw/fit_sharp_out.pkl', "wb") as f:
        pickle.dump(fig, f)
    gc.collect() 
    


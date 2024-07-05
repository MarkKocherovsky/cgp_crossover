import numpy as np
import warnings
from numpy import random
from pathlib import Path
from functions import *
from effProg import *
from similarity import *
from cgp_selection import *
from cgp_mutation import *
from cgp_xover import *
from cgp_fitness import *
from cgp_operators import *
from cgp_parents import *
from cgp_impact import *
from cgp_plots import *
from sharpness import *
from selection_impact import *
from sys import argv
from copy import deepcopy
warnings.filterwarnings('ignore')

def initialize_globals(t, max_g, max_n, max_c):
    print("started")
    print(f'trial {t}')
    print(f'generations {max_g}')
    print(f'max body nodes {max_n}')
    print(f'children {max_c}')
    biases = np.arange(0, 10, 1).astype(np.int32)
    print(f'biases {biases}')
    random.seed(t + 50)
    return biases

def setup_functions(func_index, fit_index):
    func_bank = Collection()
    func = func_bank.func_list[func_index]
    func_name = func_bank.name_list[func_index]
    fits = FitCollection()
    fit = fits.fit_list[fit_index]
    fit_name = fits.name_list[fit_index]
    print(fit)
    print(fit_name)
    return func, func_name, fit, fit_name

def create_input_vector(x, c, inputs=1):
    vec = np.zeros((x.shape[0], c.shape[0] + 1))
    x = x.reshape(-1, inputs)
    vec[:, :inputs] = x
    vec[:, inputs:] = c
    return vec

def get_noise(shape, inputs, func, sharp_in_manager):
    noisy_x = np.zeros(shape)
    noisy_x[:, :inputs] = sharp_in_manager.perturb_data()[:, :inputs]
    noisy_x[:, inputs:] = sharp_in_manager.perturb_constants()[:, inputs:]
    noisy_y = np.fromiter(map(func.func, list(noisy_x[:, :inputs].flatten())), dtype=np.float32)
    return noisy_x, noisy_y

def get_neighbor_map(preds, sharp_out_manager, fitness, train_y):
    neighborhood = sharp_out_manager.perturb(preds)
    return [fitness.fit(neighbor, train_y) for neighbor in neighborhood]

def measure_sharpness(train_x_bias, train_y, parent, func, fitness, sharp_in_manager, sharp_out_manager):
    p_fit, p_A, p_B = fitness(train_x_bias, train_y, parent)

    noisy_x, noisy_y = get_noise(train_x_bias.shape, 1, func, sharp_in_manager)
    p_sharp, _, _ = fitness(noisy_x, noisy_y, parent)
    sharp_list = [np.abs(p_fit - p_sharp)]
    sharp_std = [0]

    preds, _, _ = fitness(train_x_bias, train_y, parent, opt=1)
    neighbor_map = get_neighbor_map(preds, sharp_out_manager, fitness, train_y)
    sharp_out_list = [np.std(neighbor_map) ** 2]  # variance
    sharp_out_std = [0]

    return sharp_list, sharp_std, sharp_out_list, sharp_out_std, p_fit, p_A, p_B

def measure_changes(children, fitness_objects, train_x_bias, train_y, parent, p_fit, func, sharp_in_manager, sharp_out_manager):
    c_fit = np.array([fitness_objects[x](train_x_bias, train_y, child) for child, x in zip(children, range(len(children)))])
    best_child_index = np.argmin(c_fit[:, 0])
    best_c_fit = c_fit[best_child_index, 0]
    best_child = children[best_child_index]

    change_list = np.array([percent_change(c, p_fit) for c in c_fit[:, 0]])
    change_list = change_list[np.isfinite(change_list)]
    cl_std = np.nanstd(change_list)
    avg_hist_list = [(np.histogram(change_list, bins=5, range=(cl_std * -2, cl_std * 2)))]

    ret_avg_list = [(find_similarity(best_child[0], parent[0], best_child[1], parent[1], mode='cgp', method='distance'))]
    ret_std_list = [0.0]
    std_change_list = [0]
    p_size = [cgp_active_nodes(parent[0], parent[1], opt=2)]

    drift_cum = np.array([0, 0, 0])
    for c in c_fit[:, 0]:
        if change(c, p_fit) > 0.1:
            drift_cum[0] += 1
        elif change(c, p_fit) < -0.1:
            drift_cum[2] += 1
        else:
            drift_cum[1] += 1

    noisy_x, noisy_y = get_noise(train_x_bias.shape, 1, func, sharp_in_manager)
    preds, _, _ = fitness(train_x_bias, train_y, parent, opt=1)
    neighbor_map = get_neighbor_map(preds, sharp_out_manager, fitness, train_y)
    c_sharp = [np.abs(p_fit - fitness(noisy_x, noisy_y, parent)[0])]
    o_sharp = [np.std(neighbor_map) ** 2]

    for i in range(len(children)):
        c = children[i]
        noisy_x[:, 1:] = sharp_in_manager.perturb_constants()[:, 1:]
        noisy_y = np.fromiter(map(func.func, list(noisy_x[:, :1].flatten())), dtype=np.float32)
        c_sharp.append(np.abs(c_fit[i] - fitness(noisy_x, noisy_y, c)[0]))
        preds, _, _ = fitness(train_x_bias, train_y, c, opt=1)
        neighbor_map = get_neighbor_map(preds, sharp_out_manager, fitness, train_y)
        o_sharp.append(np.std(neighbor_map) ** 2)

    sharp_list = np.mean(c_sharp)
    sharp_std = np.std(c_sharp)
    sharp_out_list = np.mean(o_sharp)
    sharp_out_std = np.std(o_sharp)

    return best_child, best_c_fit, avg_hist_list, ret_avg_list, ret_std_list, std_change_list, p_size, drift_cum, sharp_list, sharp_std, sharp_out_list, sharp_out_std

def plot_fitness(t, max_g, max_n, max_c, g, fit_track, run_name):
    import matplotlib.pyplot as plt
    plt.plot(fit_track)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title(f'Fitness over Generations (Trial {t})')
    plt.savefig(f'./cgp/{run_name}/fitness_{t}_{max_g}_{max_n}_{max_c}_{g}.png')
    plt.close()

def plot_change(t, max_g, max_n, max_c, g, avg_change_list, std_change_list, run_name):
    import matplotlib.pyplot as plt
    plt.plot(avg_change_list)
    plt.fill_between(range(len(avg_change_list)), avg_change_list - std_change_list, avg_change_list + std_change_list, alpha=0.3)
    plt.xlabel('Generations')
    plt.ylabel('Average Percent Change')
    plt.title(f'Average Percent Change over Generations (Trial {t})')
    plt.savefig(f'./cgp/{run_name}/change_{t}_{max_g}_{max_n}_{max_c}_{g}.png')
    plt.close()

def plot_sharpness(t, max_g, max_n, max_c, g, sharp_list, sharp_std, sharp_out_list, sharp_out_std, run_name):
    import matplotlib.pyplot as plt
    plt.plot(sharp_list, label='Sharpness')
    plt.fill_between(range(len(sharp_list)), np.array(sharp_list) - np.array(sharp_std), np.array(sharp_list) + np.array(sharp_std), alpha=0.3)
    plt.plot(sharp_out_list, label='Sharpness Out')
    plt.fill_between(range(len(sharp_out_list)), np.array(sharp_out_list) - np.array(sharp_out_std), np.array(sharp_out_list) + np.array(sharp_out_std), alpha=0.3)
    plt.xlabel('Generations')
    plt.ylabel('Sharpness')
    plt.legend()
    plt.title(f'Sharpness over Generations (Trial {t})')
    plt.savefig(f'./cgp/{run_name}/sharpness_{t}_{max_g}_{max_n}_{max_c}_{g}.png')
    plt.close()

def plot_population_size(t, max_g, max_n, max_c, g, p_size, run_name):
    import matplotlib.pyplot as plt
    plt.plot(p_size)
    plt.xlabel('Generations')
    plt.ylabel('Population Size')
    plt.title(f'Population Size over Generations (Trial {t})')
    plt.savefig(f'./cgp/{run_name}/pop_size_{t}_{max_g}_{max_n}_{max_c}_{g}.png')
    plt.close()

def plot_retention(t, max_g, max_n, max_c, g, ret_avg_list, ret_std_list, run_name):
    import matplotlib.pyplot as plt
    plt.plot(ret_avg_list, label='Retention Avg')


# cgp_utils.py
import numpy as np
import matplotlib.pyplot as plt
import warnings
from numpy import random, sin, cos, tan, sqrt, exp, log, abs, floor, ceil
from math import log, pi
from sys import path
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
from copy import deepcopy

# Custom imports (assuming these are in the same directory or available in the PYTHONPATH)
from functions import *
from effProg import *
from similarity import *
from cgp_selection import *
from cgp_mutation import *
from cgp_xover import *
from cgp_fitness import *
from cgp_operators import *
from cgp_parents import *
from sharpness import *
from cgp_plots import *

# Suppress warnings
warnings.filterwarnings('ignore')

# Utility functions
def initialize_population(max_p, max_n, bank, inputs=1, outputs=1, arity=2, first_body_node=11):
    return generate_parents(max_p, max_n, bank, first_body_node=first_body_node, outputs=outputs, arity=arity)

def calculate_fitness(fitness_objects, train_x_bias, train_y, individuals, max_p):
    fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(range(max_p), individuals)])
    fitnesses = fit_temp[:, 0].copy().flatten()
    alignment_a = fit_temp[:, 1].copy()
    alignment_b = fit_temp[:, 2].copy()
    return fitnesses, alignment_a, alignment_b

def get_noise(shape, pop_size, inputs, func, sharp_in_manager, opt=0):
    x, y = [], []
    if opt == 1:
        fixed_inputs = sharp_in_manager.perturb_data()[:, :inputs]
    for p in range(pop_size):
        noisy_x = np.zeros((shape))
        if opt == 1:
            noisy_x[:, :inputs] = deepcopy(fixed_inputs)
        else:
            noisy_x[:, :inputs] = sharp_in_manager.perturb_data()[:, :inputs]
        noisy_x[:, inputs:] = sharp_in_manager.perturb_constants()[:, inputs:]
        noisy_y = np.fromiter(map(func.func, list(noisy_x[:, :inputs].flatten())), dtype=np.float32)
        x.append(noisy_x)
        y.append(noisy_y)
    return np.array(x), np.array(y)

def calculate_sharpness(fitness_objects, noisy_x, noisy_y, individuals, max_p):
    return np.array([fitness_objects[i](noisy_x[i], noisy_y[i], ind)[0] for i, ind in zip(range(max_p), individuals)])

def get_neighbor_map(preds, sharp_out_manager, fitness, train_y):
    neighborhood = sharp_out_manager.perturb(preds)
    return [fitness.fit(neighbor, train_y) for neighbor in neighborhood]

def log_results(log_dir, func_name, t, data):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    import pickle
    with open(f"{log_dir}/output_{t}.pkl", "wb") as f:
        for item in data:
            pickle.dump(item, f)

def process_retention(retention, pop, fitnesses, max_p):
    change_list = []
    ret_list = []
    full_change_list = []

    for p in retention:
        ps = [pop[p], pop[p + 1]]
        p_fits = np.array([fitnesses[p], fitnesses[p + 1]])
        cs = [pop[p + max_p], pop[p + max_p + 1]]
        c_fits = np.array([fitnesses[p + max_p], fitnesses[p + max_p + 1]])
        best_p, best_c = np.argmin(p_fits), np.argmin(c_fits)

        change_list.append(percent_change(c_fits[best_c], p_fits[best_p]))
        ret_list.append(find_similarity(cs[best_c][0], ps[best_p][0], cs[best_c][1], ps[best_p][1], 'cgp'))
        full_change_list.extend([percent_change(c, p_fits[best_p]) for c in c_fits])

    return change_list, ret_list, full_change_list


def initialize_parameters(argv):
    t = int(argv[1])
    max_g = int(argv[2])
    max_n = int(argv[3])
    max_p = int(argv[4])
    max_c = int(argv[5])
    outputs = 1
    inputs = 1
    biases = np.arange(0, 10, 1).astype(np.int32)
    bias = biases.shape[0]
    arity = 2
    p_mut = float(argv[8])
    p_xov = float(argv[9])
    random.seed(t + 250)
    
    return t, max_g, max_n, max_p, max_c, outputs, inputs, biases, bias, arity, p_mut, p_xov

import numpy as np
import matplotlib.pyplot as plt
import warnings
from numpy import random, sqrt
from math import isnan
from sys import argv
from pathlib import Path
from copy import deepcopy

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

warnings.filterwarnings('ignore')

# Helper functions
def setup_environment():
    """Setup environment and return configuration parameters."""
    trial = int(argv[1])
    max_g = int(argv[2])
    max_n = int(argv[3])
    max_c = int(argv[4])
    func_index = int(argv[5])
    fit_index = int(argv[6])
    
    biases = np.arange(0, 10, 1).astype(np.int32)
    inputs = 1
    arity = 2
    
    print(f"Trial {trial}")
    print(f"Generations {max_g}")
    print(f"Max Body Nodes {max_n}")
    print(f"Children {max_c}")
    print(f"Biases {biases}")
    print(f"Seed = {sqrt(trial)}")
    
    random.seed(trial + 50)
    
    func_bank = Collection()
    func = func_bank.func_list[func_index]
    func_name = func_bank.name_list[func_index]
    
    fits = FitCollection()
    fit = fits.fit_list[fit_index]
    fit_name = fits.name_list[fit_index]
    
    return trial, max_g, max_n, max_c, biases, inputs, arity, func, func_name, fit, fit_name

def create_input_vector(x, c, inputs):
    """Create input vector with biases."""
    vec = np.zeros((x.shape[0], c.shape[0] + 1))
    vec[:, :inputs] = x.reshape(-1, inputs)
    vec[:, inputs:] = c
    return vec

def initialize_population(max_n, bank, arity, first_body_node=11):
    """Generate initial parent population."""
    return generate_parents(1, max_n, bank, first_body_node=first_body_node, outputs=1, arity=arity)

def get_noise(shape, inputs, func, sharp_in_manager):
    """Generate noisy input and output data."""
    noisy_x = np.zeros(shape)
    noisy_x[:, :inputs] = sharp_in_manager.perturb_data()[:, :inputs]
    noisy_x[:, inputs:] = sharp_in_manager.perturb_constants()[:, inputs:]
    noisy_y = np.fromiter(map(func.func, noisy_x[:, :inputs].flatten()), dtype=np.float32)
    return noisy_x, noisy_y

def evaluate_neighbors(preds, sharp_out_manager, fitness, train_y):
    """Evaluate fitness of neighbors."""
    neighborhood = sharp_out_manager.perturb(preds)
    return [fitness.fit(neighbor, train_y) for neighbor in neighborhood]

def process_generation(g, max_g, parent, train_x_bias, train_y, fit, func, func_name, biases, arity, max_n, max_c, bank, fitness_objects):
    """Process a single generation of the evolutionary algorithm."""
    mutate = mutate_1_plus_4
    
    # Generate offspring
    children = [mutate(deepcopy(parent)) for _ in range(max_c)]
    c_fit = np.array([fitness_objects[i](train_x_bias, train_y, child) for i, child in enumerate(children)])
    
    # Select best child
    best_child_index = np.argmin(c_fit[:, 0])
    best_c_fit = c_fit[best_child_index, 0]
    best_child = children[best_child_index]
    
    # Track metrics
    avg_change_list = [percent_change(best_c_fit, p_fit)]
    change_list = np.array([percent_change(c, p_fit) for c in c_fit[:, 0]])
    avg_hist_list = []
    
    if not np.all(change_list == 0.0):
        cl_std = np.nanstd(change_list)
        avg_hist_list.append((g, np.histogram(change_list, bins=5, range=(cl_std * -2, cl_std * 2))))
    
    # Calculate similarity
    ret_avg_list = [find_similarity(best_child[0], parent[0], best_child[1], parent[1], mode='cgp', method='distance')]
    ret_std_list = [0.0]
    std_change_list = [0]
    
    # Drift and sharpness calculations
    drift_cum, sharp_list, sharp_std, sharp_out_list = np.array([0, 0, 0]), [], [], []
    
    for i in range(max_c):
        c = children[i]
        noisy_x, noisy_y = get_noise(train_x_bias.shape, inputs, func, sharp_in_manager)
        c_sharp = [np.abs(p_fit - fitness(noisy_x, noisy_y, parent)[0])]
        preds, _, _ = fitness(train_x_bias, train_y, c, opt=1)
        neighbor_map = evaluate_neighbors(preds, sharp_out_manager, fitness, train_y)
        o_sharp = [np.std(neighbor_map) ** 2]
        sharp_list.append(np.mean(c_sharp))
        sharp_std.append(np.std(c_sharp))
        sharp_out_list.append(np.mean(o_sharp))
    
    if any(c_fit <= p_fit) and random.rand() > 1 / max_c:
        best = np.argmin(c_fit)
        parent = deepcopy(children[best])
        p_fit = np.min(c_fit)
        p_A = c_fit[best, 1]
        p_B = c_fit[best, 2]
    
    if g % 100 == 0:
        print(f"Gen {g} Best Fitness: {p_fit}\tMean SAM-In: {np.round(np.mean(c_sharp), 5)}\tMean SAM-Out: {np.round(np.mean(o_sharp), 5)}")
    
    return parent, p_fit, avg_change_list, avg_hist_list, ret_avg_list, ret_std_list, std_change_list, drift_cum, sharp_list, sharp_std, sharp_out_list

def main():
    trial, max_g, max_n, max_c, biases, inputs, arity, func, func_name, fit, fit_name = setup_environment()
    
    bank = (add, sub, mul, div)
    bank_string = ("+", "-", "*", "/")
    
    train_x_bias = create_input_vector(func.x_dom, biases, inputs)
    
    parent = initialize_population(max_n, bank, arity)
    
    fitness = Fitness()
    sharp_in_manager = SAM_IN(train_x_bias)
    sharp_out_manager = SAM_OUT()
    
    p_fit, p_A, p_B = fitness(train_x_bias, func.y_test, parent)
    
    fit_track, p_size = [], []
    avg_change_list, std_change_list = [], []
    avg_hist_list, ret_avg_list, ret_std_list = [], [], []
    drift_list, drift_cum = [], np.array([0, 0, 0])
    
    fitness_objects = [Fitness() for _ in range(max_c)]
    
    for g in range(1, max_g + 1):
        results = process_generation(g, max_g, parent, train_x_bias, func.y_test, fit, func, func_name, biases, arity, max_n, max_c, bank, fitness_objects)
        
        parent, p_fit, avg_change_list, avg_hist_list, ret_avg_list, ret_std_list, std_change_list, drift_cum, sharp_list, sharp_std, sharp_out_list = results
        
        fit_track.append(p_fit)
        p_size.append(cgp_active_nodes(parent[0], parent[1], opt=2))
    
    avg_change_list, std_change_list, p_size = np.array(avg_change_list), np.array(std_change_list), np.array(p_size)
    
    print(cgp_active_nodes(parent[0], parent[1], opt=1))
    print(cgp_active_nodes(parent[0], parent[1]))
    print(f"Trial {trial}: Best Fitness = {p_fit}")
    print(f"Mutations:\tDeleterious\tNeutral\tBeneficial")
    print(f"\t{drift_cum[0]}\t{drift_cum[1]}\t{drift_cum[2]}")
    print(f'Sharpness: {sharp_list[-1]}')
    print('Biases:', biases)
    print('Best Individual:', parent)
    
    Path(f"../output/cgp/{func_name}/log/").mkdir(parents=True, exist_ok=True)
    
    with open(f"../output/cgp/{func_name}/log/output_{trial}.pkl", "wb") as f:
        pickle.dump({
            'biases': biases,
            'parent': parent,
            'preds': preds,
            'p_fit': p_fit,
            'n': cgp_active_nodes(parent[0], parent[1]),
            'fit_track': fit_track,
            'avg_change_list': avg_change_list,
            'ret_avg_list': ret_avg_list,
            'p_size': p_size,
            'bin_centers': bin_centers,
            'hist_gens': hist_gens,
            'avg_hist_list': avg_hist_list,
            'drift_list': drift_list,
            'drift_cum': drift_cum,
            'sharp_list': sharp_list,
            'sharp_std': sharp_std
        }, f)
    
if __name__ == "__main__":
    main()


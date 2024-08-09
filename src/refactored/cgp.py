import numpy as np
import matplotlib.pyplot as plt
import warnings
from numpy import random
from sys import argv
from copy import deepcopy
from collections import Counter
from pathlib import Path
import pickle

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

from cgp_utils import create_input_vector, get_noise, get_neighbor_map, calculate_drift, update_parent, evaluate_sharpness

warnings.filterwarnings('ignore')

print("started")
t = int(argv[1])
print(f'trial {t}')
max_g = int(argv[2])
print(f'generations {max_g}')
max_n = int(argv[3])
print(f'max body nodes {max_n}')
max_c = int(argv[4])
print(f'children {max_c}')
outputs, inputs = 1, 1
biases = np.arange(0, 10, 1).astype(np.int32)
bias = biases.shape[0]
print(f'biases {biases}')
arity = 2
random.seed(t + 50)
print(f'Seed = {np.sqrt(t)}')

bank = (add, sub, mul, div)
bank_string = ("+", "-", "*", "/")
func_bank = Collection()
func = func_bank.func_list[int(argv[5])]
func_name = func_bank.name_list[int(argv[5])]
train_x = func.x_dom
train_y = func.y_test

f = int(argv[6])
fits = FitCollection()
fit = fits.fit_list[f]
fit_name = fits.name_list[f]
print(fit)
print(fit_name)

mutate = mutate_1_plus_4
final_fit = []
fit_track = []
ind_base = np.zeros(((arity + 1) * max_n,), np.int32).reshape(-1, arity + 1)
train_x_bias = create_input_vector(train_x, biases)
print("instantiating parent")
parent = generate_parents(1, max_n, bank, first_body_node=11, outputs=1, arity=2)

fitness = Fitness()
sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()
p_fit, p_A, p_B = fitness(train_x_bias, train_y, parent)

noisy_x, noisy_y = get_noise(train_x_bias.shape, sharp_in_manager, inputs, func)
p_sharp, _, _ = fitness(noisy_x, noisy_y, parent)
sharp_list = [np.abs(p_fit - p_sharp)]
sharp_std = [0]

preds, _, _ = fitness(train_x_bias, train_y, parent, opt=1)
neighbor_map = get_neighbor_map(preds, sharp_out_manager, fitness, train_y)
sharp_out_list = [np.std(neighbor_map) ** 2]
sharp_out_std = [0]

f_change = np.zeros((max_c,))
avg_change_list, avg_hist_list, std_change_list = [], [], []
p_size = [cgp_active_nodes(parent[0], parent[1], opt=2)]
ret_avg_list, ret_std_list = [], []
fitness_objects = [Fitness() for _ in range(max_c)]

drift_list = []
drift_cum = np.array([0, 0, 0])
run_name = 'cgp'

for g in range(1, max_g + 1):
    children = [mutate(deepcopy(parent)) for _ in range(max_c)]
    c_fit = np.array([fitness_objects[x](train_x_bias, train_y, child) for child, x in zip(children, range(max_c))])
    
    best_child_index = np.argmin(c_fit[:, 0])
    best_c_fit = c_fit[best_child_index, 0]
    best_child = children[best_child_index]
    
    avg_change_list.append(percent_change(best_c_fit, p_fit))
    change_list = np.array([percent_change(c, p_fit) for c in c_fit[:, 0]])
    change_list = change_list[np.isfinite(change_list)]
    cl_std = np.nanstd(change_list)
    if not all(cl == 0.0 for cl in change_list):
        avg_hist_list.append((g, np.histogram(change_list, bins=5, range=(cl_std * -2, cl_std * 2))))
    
    ret_avg_list.append(find_similarity(best_child[0], parent[0], best_child[1], parent[1], mode='cgp', method='distance'))
    ret_std_list.append(0.0)
    std_change_list.append(0)
    
    a, b = c_fit[:, 1].copy().flatten(), c_fit[:, 2].copy().flatten()
    c_fit = c_fit[:, 0].flatten()
    if any(np.isnan(c_fit)):
        c_fit[np.isnan(c_fit)] = np.PINF
    
    drift = calculate_drift(c_fit, p_fit)
    drift_cum += drift
    drift_list.append(drift_cum.copy())

    noisy_x, noisy_y = get_noise(train_x_bias.shape, sharp_in_manager, inputs, func)
    preds, _, _ = fitness(train_x_bias, train_y, parent, opt=1)
    neighbor_map = get_neighbor_map(preds, sharp_out_manager, fitness, train_y)
    
    c_sharp_mean, c_sharp_std, o_sharp_mean, o_sharp_std = evaluate_sharpness(max_c, train_x_bias, train_y, children, fitness, sharp_in_manager, sharp_out_manager, inputs, func, p_fit)
    sharp_list.append(c_sharp_mean)
    sharp_std.append(c_sharp_std)
    sharp_out_list.append(o_sharp_mean)
    sharp_out_std.append(o_sharp_std)
    
    parent, p_fit, p_A, p_B = update_parent(children, c_fit, parent, p_fit, p_A, p_B)

    if g % 100 == 0:
        print(f"Gen {g} Best Fitness: {p_fit}\tMean SAM-In: {np.round(c_sharp_mean, 5)}\tMean SAM-Out: {np.round(o_sharp_mean, 5)}")
    
    fit_track.append(p_fit)
    p_size.append(cgp_active_nodes(parent[0], parent[1], opt=2))

avg_change_list, std_change_list, p_size = np.array(avg_change_list), np.array(std_change_list), np.array(p_size)
print(cgp_active_nodes(parent[0], parent[1], opt=1))
print(cgp_active_nodes(parent[0], parent[1]))
print(f"Trial {t}: Best Fitness = {p_fit}")
print(f"Mutations:\tDeleterious\tNeutral\tBeneficial")
print(f"\t{drift_cum[0]}\t{drift_cum[1]}\t{drift_cum[2]}")
print(f'Sharpness: {sharp_list[-1]}')
print('biases')
print(biases)
print('best individual')
print(parent)
print('preds')
preds, p_A, p_b = fitness(train_x_bias, train_y, parent, opt=1)
print(preds)
print(fitness(train_x_bias, train_y, parent, opt=0))

output_dir = Path(f"../output/cgp/{func_name}/log/")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / f"output_{t}.pkl", "wb") as f:
    pickle.dump(biases, f)
    pickle.dump(parent[0], f)
    pickle.dump(parent[1], f)
    pickle.dump(preds, f)
    pickle.dump(p_fit, f)
    pickle.dump(p_size, f)
    pickle.dump([avg_change_list], f)
    pickle.dump([ret_avg_list], f)
    pickle.dump([bin_centers, hist_gens, avg_hist_list], f)
    pickle.dump(drift_list, f)
    pickle.dump(drift_cum, f)
    pickle.dump(sharp_list, f)
    pickle.dump(sharp_std, f)


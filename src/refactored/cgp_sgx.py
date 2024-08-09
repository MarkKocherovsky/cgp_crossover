import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
from numpy import random
from pathlib import Path
from copy import deepcopy
from scipy.stats import skew, kurtosis
from sys import argv

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

from cgp_utils import *

warnings.filterwarnings('ignore')
print("started")

# Initialize parameters
t, max_g, max_n, max_p, max_c, outputs, inputs, biases, bias, arity, p_mut, p_xov, run_name = initialize_parameters(argv)

bank = (add, sub, mul, div)
bank_string = ("+", "-", "*", "/")

func_bank = Collection()
func = func_bank.func_list[int(argv[6])]
func_name = func_bank.name_list[int(argv[6])]
train_x = func.x_dom
train_y = func.y_test

f = int(argv[7])
fits = FitCollection()
fit = fits.fit_list[f]
fit_name = fits.name_list[f]

alignment = np.zeros((max_p + max_c, 2))
alignment[:, 0] = 1.0

train_x_bias = np.zeros((train_x.shape[0], biases.shape[0] + 1))
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = biases

mutate = basic_mutation
select = tournament_elitism
parents = generate_parents(max_p, max_n, bank, first_body_node=11, outputs=1, arity=2)
density_distro = np.zeros(max_n * (outputs + arity), dtype=np.int32)

sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()

fitness_objects = [Fitness() for _ in range(max_p + max_c)]
fitnesses, alignment = evaluate_fitness(train_x_bias, train_y, parents, fitness_objects)

noisy_x, noisy_y = generate_noise_data(sharp_in_manager, train_x_bias.shape, max_p + max_c, inputs, func)
sharpness = update_sharpness(noisy_x, noisy_y, parents, fitness_objects, max_p, max_c)

sharp_in_list = [np.mean(sharpness)]
sharp_in_std = [np.std(sharpness)]

preds = [fitness_objects[i](train_x_bias, train_y, parent, opt=1)[0] for i, parent in zip(range(max_p), parents)]
neighbor_map = np.array([get_neighbor_map(pred, sharp_out_manager, fitness_objects[i], train_y) for i, pred in zip(range(max_p), preds)])
sharp_out_list = [np.mean(compute_sharpness(neighbor_map))]
sharp_out_std = [np.std(compute_sharpness(neighbor_map))]

fit_track, ret_avg_list, ret_std_list = [], [], []
avg_change_list, avg_hist_list, std_change_list = [], [], []
best_i = np.argmin(fitnesses[:max_p])
p_size = [cgp_active_nodes(parents[best_i][0], parents[best_i][1], opt=2)]

mut_impact = MutationImpact(neutral_limit=0.1)
num_elites = 7

for g in range(1, max_g + 1):
    children, retention, density_distro = xover(deepcopy(parents), density_distro, method='Subgraph')
    children = mutate(deepcopy(children))
    pop = parents + children
    fitnesses, alignment = evaluate_fitness(train_x_bias, train_y, pop, fitness_objects)

    if any(np.isnan(fitnesses)):
        fitnesses[np.isnan(fitnesses)] = np.PINF
    mut_impact(fitnesses, max_p)

    change_list, full_change_list, ret_list = evaluate_retention(pop, fitnesses, retention, max_p, percent_change, find_similarity)

    full_change_list = np.array(full_change_list).flatten()
    full_change_list = full_change_list[np.isfinite(full_change_list)]
    cl_std = np.nanstd(full_change_list)
    if not all(cl == 0.0 for cl in full_change_list):
        avg_hist_list.append((g, np.histogram(full_change_list, bins=10, range=(cl_std * -2, cl_std * 2))))
    avg_change_list.append(np.nanmean(change_list))
    std_change_list.append(np.nanstd(change_list))
    ret_avg_list.append(np.nanmean(ret_list))
    ret_std_list.append(np.nanstd(ret_list))

    noisy_x, noisy_y = generate_noise_data(sharp_in_manager, train_x_bias.shape, max_p + max_c, inputs, func)
    sharpness = update_sharpness(noisy_x, noisy_y, parents, fitness_objects, max_p, max_c)
    sharp_in_list.append(np.mean(sharpness))
    sharp_in_std.append(np.std(sharpness))

    preds = [fitness_objects[i](train_x_bias, train_y, parent, opt=1)[0] for i, parent in zip(range(max_p), parents)]
    neighbor_map = np.array([get_neighbor_map(pred, sharp_out_manager, fitness_objects[i], train_y) for i, pred in zip(range(max_p), preds)])
    sharp_out_list.append(np.mean(compute_sharpness(neighbor_map)))
    sharp_out_std.append(np.std(compute_sharpness(neighbor_map)))

    best_i = np.argmin(fitnesses[:max_p])
    parents = select(fitnesses, alignment, parents, max_p, num_elites)
    fit_track.append(np.min(fitnesses[:max_p]))
    p_size.append(cgp_active_nodes(parents[best_i][0], parents[best_i][1], opt=2))

    print(f"Generation {g} completed with best fitness {np.min(fitnesses[:max_p])}")

# Save results
output_path = f"{run_name}_results.pkl"
save_results(output_path, biases, parents[best_i], preds, np.min(fitnesses[:max_p]), max_g, fit_track, avg_change_list, ret_avg_list, p_size, None, None, avg_hist_list, None, None, sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std, density_distro)
print("Results saved to", output_path)


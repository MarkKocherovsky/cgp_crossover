import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from copy import deepcopy
from scipy.signal import savgol_filter
from sys import argv

from cgp_utils import (
    initialize_parameters, initialize_func_bank, initialize_fitness, getNoise, 
    get_neighbor_map, log_and_save_results
)
from sharpness import SAM_IN, SAM_OUT
from cgp_operators import cgp_active_nodes
from cgp_selection import tournament_elitism
from cgp_mutation import basic_mutation
from cgp_xover import xover
from cgp_fitness import Fitness
from cgp_parents import generate_parents
from similarity import find_similarity
from effProg import percent_change

warnings.filterwarnings('ignore')
print("started")

# Initialize parameters
t, max_g, max_n, max_p, max_c, p_mut, p_xov = initialize_parameters(argv)
print(f'trial {t}')
print(f'generations {max_g}')
print(f'max body nodes {max_n}')
print(f'Parents {max_p}')
print(f'children {max_c}')

outputs, inputs = 1, 1
biases = np.arange(0, 10, 1).astype(np.int32)
bias = biases.shape[0]
print(f'biases {biases}')
arity = 2
random.seed(t + 200)
bank = (add, sub, mul, div)
bank_string = ("+", "-", "*", "/")
run_name = 'cgp_2x'

# Initialize function bank and fitness
func, func_name, train_x, train_y = initialize_func_bank(argv)
fit, fit_name = initialize_fitness(argv)

print(train_x)
print(f'Fitness Function')
print(fit)
print(fit_name)

alignment = np.zeros((max_p + max_c, 2))
alignment[:, 0] = 1.0

train_x_bias = np.zeros((train_x.shape[0], biases.shape[0] + 1))
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = biases
print(train_x_bias)

mutate = basic_mutation
select = tournament_elitism
parents = generate_parents(max_p, max_n, bank, first_body_node=11, outputs=1, arity=2)
density_distro = np.zeros(max_n * (outputs + arity), dtype=np.int32)

fitness_objects = [Fitness() for _ in range(max_p + max_c)]
fitnesses = np.zeros((max_p + max_c,))
fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, parent) for i, parent in zip(range(max_p), parents)])
fitnesses[:max_p] = fit_temp[:, 0].copy().flatten()
alignment[:max_p, 0] = fit_temp[:, 1].copy()
alignment[:max_p, 1] = fit_temp[:, 2].copy()
print(np.round(fitnesses, 4))

sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()

# SAM-IN
noisy_x, noisy_y = getNoise(train_x_bias.shape, max_p, max_c, inputs, func, sharp_in_manager)
sharpness = np.array([fitness_objects[i](noisy_x[i], noisy_y[i], parent)[0] for i, parent in zip(range(max_p), parents)])
sharp_in_list, sharp_in_std = [np.mean(sharpness)], [np.std(sharpness)]

# SAM-OUT
preds = [fitness_objects[i](train_x_bias, train_y, parent, opt=1)[0] for i, parent in zip(range(max_p), parents)]
neighbor_map = np.array([get_neighbor_map(pred, sharp_out_manager, fitness_objects[i], train_y) for i, pred in zip(range(max_p), preds)])
print(neighbor_map.shape)
sharp_out_list = [np.mean(np.std(neighbor_map, axis=1) ** 2)]
sharp_out_std = [np.std(np.std(neighbor_map, axis=1))]

print(np.round(sharpness, 4))
print(np.round(np.std(neighbor_map, axis=1) ** 2, 4))

fit_track, ret_avg_list, ret_std_list = [], [], []
avg_change_list, avg_hist_list, std_change_list = [], [], []

best_i = np.argmin(fitnesses[:max_p])
p_size = [cgp_active_nodes(parents[best_i][0], parents[best_i][1], opt=2)]

for g in range(1, max_g + 1):
    children, retention, density_distro = xover(deepcopy(parents), density_distro, method='TwoPoint')
    children = mutate(deepcopy(children))
    pop = parents + children
    fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(range(max_p + max_c), pop)])

    fitnesses = fit_temp[:, 0].copy().flatten()
    alignment = fit_temp[:, 1].copy()
    alignment = fit_temp[:, 2].copy()
    if any(np.isnan(fitnesses)):
        nans = np.isnan(fitnesses)
        fitnesses[nans] = np.PINF

    change_list, full_change_list, ret_list = [], [], []
    for p in retention:
        ps = [pop[p], pop[p + 1]]
        p_fits = np.array([fitnesses[p], fitnesses[p + 1]])
        cs = [pop[p + max_p], pop[p + max_p + 1]]
        c_fits = np.array([fitnesses[p + max_p], fitnesses[p + max_p + 1]])
        best_p = np.argmin(p_fits)
        best_c = np.argmin(c_fits)

        change_list.append(percent_change(c_fits[best_c], p_fits[best_p]))
        ret_list.append(find_similarity(cs[best_c][0], ps
        best_p][0], cs[best_c][1], ps[best_p][1], 'cgp'))

        full_change_list.append([percent_change(c, best_p) for c in c_fits])

    full_change_list = np.array(full_change_list).flatten()
    full_change_list = full_change_list[np.isfinite(full_change_list)]
    cl_std = np.nanstd(full_change_list)
    if not all(cl == 0.0 for cl in full_change_list):
        avg_hist_list.append((g, np.histogram(full_change_list, bins=10, range=(cl_std * -2, cl_std * 2))))
    avg_change_list.append(np.nanmean(change_list))
    std_change_list.append(np.nanstd(change_list))
    ret_avg_list.append(np.nanmean(ret_list))
    ret_std_list.append(np.nanstd(ret_list))

    # SAM-IN
    noisy_x, noisy_y = getNoise(train_x_bias.shape, max_p, max_c, inputs, func, sharp_in_manager, opt=1)
    sharpness = np.array([fitness_objects[i](noisy_x[i], noisy_y[i], individual)[0] for i, individual in zip(range(max_p + max_c), pop)])
    sharp_in_list.append(np.mean(sharpness))
    sharp_in_std.append(np.std(sharpness))

    # SAM-OUT
    preds = [fitness_objects[i](train_x_bias, train_y, individual, opt=1)[0] for i, individual in zip(range(max_p + max_c), pop)]
    neighbor_map = np.array([get_neighbor_map(pred, sharp_out_manager, fitness_objects[i], train_y) for i, pred in zip(range(max_p + max_c), preds)])
    out_sharpness = np.std(neighbor_map, axis=1) ** 2
    sharp_out_list.append(np.mean(out_sharpness))  # variance
    sharp_out_std.append(np.std(out_sharpness))

    best_i = np.argmin(fitnesses)
    best_fit = fitnesses[best_i]
    if g % 100 == 0:
        print(f"Gen {g} Best Fitness: {best_fit}\tMean SAM-In: {sharp_in_list[-1]}\tMean SAM-Out: {sharp_out_list[-1]}")
        # sort sharpness
        indices = np.argsort(fitnesses)[:num_elites]
        elites = [pop[i] for i in indices]
        sharpness = np.array(sharpness)
        out_sharpness = np.array(out_sharpness)
        in_sharp = sharpness[indices]
        out_sharp = out_sharpness[indices]

    fit_track.append(best_fit)
    p_size.append(cgp_active_nodes(pop[best_i][0], pop[best_i][1], opt=2))
    parents = select(pop, fitnesses, max_p)

# Final evaluation
pop = parents + children
fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(range(max_p + max_c), pop)])
fitnesses = fit_temp[:, 0].copy().flatten()
best_i = np.argmin(fitnesses)
best_fit = fitnesses[best_i]
best_pop = pop[best_i]
print(f"Trial {t}: Best Fitness = {best_fit}")
print('best individual')
print(pop[best_i])

# Update and print density distribution
print('# of mutations at each point')
print(density_distro)
density_distro = density_distro / np.sum(density_distro)
print('Probability Density Function')
print(density_distro)

# Save results
output_dir = f"../output/cgp_2x/{func_name}/log/"
Path(output_dir).mkdir(parents=True, exist_ok=True)
log_and_save_results(output_dir, t, (
    fit_track, ret_avg_list, ret_std_list, avg_change_list, avg_hist_list, std_change_list, p_size
))

print(f"Trial {t} results saved.")


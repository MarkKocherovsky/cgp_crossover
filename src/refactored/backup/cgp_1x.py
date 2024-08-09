import numpy as np
import warnings
import pickle
from sys import argv
from pathlib import Path
from copy import deepcopy
from scipy.stats import skew, kurtosis

from functions import Collection, FitCollection
from effProg import *
from similarity import find_similarity, percent_change
from cgp_selection import tournament_elitism
from cgp_plots import scatter, fit_plot, proportion_plot, change_avg_plot, retention_plot, drift_plot, sharp_bar_plot, sharp_plot, cgp_graph
from cgp_mutation import basic_mutation
from cgp_xover import xover
from cgp_fitness import Fitness
from cgp_operators import cgp_active_nodes
from cgp_parents import generate_parents
from cgp_impact import MutationImpact
from sharpness import SAM_IN, SAM_OUT
from cgp_utils import get_noise, get_neighbor_map, process_generation_data, save_results

warnings.filterwarnings('ignore')

def main():
    # Command-line arguments
    trial = int(argv[1])
    max_g = int(argv[2])
    max_n = int(argv[3])
    max_p = int(argv[4])
    max_c = int(argv[5])
    func_index = int(argv[6])
    fit_index = int(argv[7])
    p_mut = float(argv[8])
    p_xov = float(argv[9])
    
    # Configuration and setup
    np.random.seed(trial + 100)
    print(f"Trial {trial}, Generations {max_g}, Max Nodes {max_n}, Parents {max_p}, Children {max_c}")
    
    func_bank = Collection()
    func = func_bank.func_list[func_index]
    func_name = func_bank.name_list[func_index]
    
    fits = FitCollection()
    fit = fits.fit_list[fit_index]
    fit_name = fits.name_list[fit_index]
    
    train_x = func.x_dom
    train_y = func.y_test
    biases = np.arange(0, 10, 1).astype(np.int32)
    
    train_x_bias = np.hstack((train_x, np.tile(biases, (train_x.shape[0], 1))))
    
    parents = generate_parents(max_p, max_n, (add, sub, mul, div), first_body_node=11, outputs=1, arity=2)
    fitness_objects = [Fitness() for _ in range(max_p + max_c)]
    
    sharp_in_manager = SAM_IN(train_x_bias)
    sharp_out_manager = SAM_OUT()

    fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, parent) for i, parent in enumerate(parents)])
    fitnesses = fit_temp[:, 0].copy()
    alignment = fit_temp[:, 1:3].copy()
    
    # Initial sharpness calculations
    noisy_x, noisy_y = get_noise(train_x_bias.shape, max_p + max_c, func, opt=0)
    sharpness = np.array([fitness_objects[i](noisy_x[i], noisy_y[i], parent)[0] for i, parent in enumerate(parents)])
    sharp_in_list, sharp_in_std = [np.mean(sharpness)], [np.std(sharpness)]
    
    # Initial SAM-OUT calculations
    preds = [fitness_objects[i](train_x_bias, train_y, parent, opt=1)[0] for i, parent in enumerate(parents)]
    neighbor_map = np.array([get_neighbor_map(pred, sharp_out_manager, fitness_objects[i]) for i, pred in enumerate(preds)])
    out_sharpness = np.std(neighbor_map, axis=1) ** 2
    sharp_out_list, sharp_out_std = [np.mean(out_sharpness)], [np.std(out_sharpness)]

    fit_track, ret_avg_list, ret_std_list, avg_change_list = [], [], [], []
    std_change_list, avg_hist_list, drift_list = [], [], []
    
    mut_impact = MutationImpact(neutral_limit=0.1)
    
    for g in range(1, max_g + 1):
        # Generate children, apply mutation and evaluate
        children, _, density_distro = xover(deepcopy(parents), np.zeros(max_n * (1 + 2)), method='OnePoint', max_n=max_n)
        children = basic_mutation(deepcopy(children))
        pop = parents + children
        
        fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in enumerate(pop)])
        fitnesses = fit_temp[:, 0].copy()
        alignment = fit_temp[:, 1:3].copy()
        
        if np.any(np.isnan(fitnesses)):
            fitnesses[np.isnan(fitnesses)] = np.PINF
        
        mut_impact(fitnesses, max_p)
        change_list, full_change_list, ret_list = process_generation_data(pop, fitnesses, max_p)
        
        noisy_x, noisy_y = get_noise(train_x_bias.shape, max_p + max_c, func, opt=1)
        sharpness = np.array([fitness_objects[i](noisy_x[i], noisy_y[i], individual)[0] for i, individual in enumerate(pop)])
        sharp_in_list.append(np.mean(sharpness))
        sharp_in_std.append(np.std(sharpness))
        
        preds = [fitness_objects[i](train_x_bias, train_y, individual, opt=1)[0] for i, individual in enumerate(pop)]
        neighbor_map = np.array([get_neighbor_map(pred, sharp_out_manager, fitness_objects[i]) for i, pred in enumerate(preds)])
        out_sharpness = np.std(neighbor_map, axis=1) ** 2
        sharp_out_list.append(np.mean(out_sharpness))
        sharp_out_std.append(np.std(out_sharpness))
        
        best_i = np.argmin(fitnesses)
        best_fit = fitnesses[best_i]
        
        if g % 100 == 0:
            print(f"Gen {g} Best Fitness: {best_fit}\tMean SAM-In: {sharp_in_list[-1]}\tMean SAM-Out: {sharp_out_list[-1]}")
            # Additional plotting code omitted for brevity
        
        fit_track.append(best_fit)
        p_size = [cgp_active_nodes(pop[best_i][0], pop[best_i][1], opt=2)]
        parents = tournament_elitism(pop, fitnesses, max_p)
    
    # Final output and saving results
    best_i = np.argmin(fitnesses)
    best_pop = pop[best_i]
    best_fit = fitnesses[best_i]
    
    print(f"Trial {trial}: Best Fitness = {best_fit}")
    drift_cum, drift_list = mut_impact.return_lists(option=1)
    
    save_results(func_name, run_name='cgp_1x', trial=trial, best_pop=best_pop, preds=preds, best_fit=best_fit,
                 p_size=p_size, fit_track=fit_track, avg_change_list=avg_change_list, ret_avg_list=ret_avg_list,
                 bin_centers=None, hist_gens=None, avg_hist_list=None, drift_list=drift_list, drift_cum=drift_cum,
                 sharp_in_list=sharp_in_list, sharp_out_list=sharp_out_list, sharp_in_std=sharp_in_std,
                 sharp_out_std=sharp_out_std, density_distro=density_distro)

if __name__ == "__main__":
    main()


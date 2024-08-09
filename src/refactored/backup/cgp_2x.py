import numpy as np
import warnings
from pathlib import Path
from sys import argv
from functions import Collection, FitCollection
from effProg import *
from similarity import find_similarity, percent_change
from cgp_selection import tournament_elitism
from cgp_mutation import basic_mutation
from cgp_xover import xover
from cgp_fitness import Fitness
from cgp_operators import cgp_active_nodes
from cgp_parents import generate_parents
from sharpness import SAM_IN, SAM_OUT
from cgp_utils import get_noise, get_neighbor_map, process_generation_data, save_results, plot_active_nodes, change_histogram_plot

warnings.filterwarnings('ignore')
print("Started")

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
    np.random.seed(trial + 200)
    print(f"Trial {trial}, Generations {max_g}, Max Nodes {max_n}, Parents {max_p}, Children {max_c}")

    func_bank = Collection()
    func = func_bank.func_list[func_index]
    func_name = func_bank.name_list[func_index]
    train_x = func.x_dom
    train_y = func.y_test

    fits = FitCollection()
    fit = fits.fit_list[fit_index]
    fit_name = fits.name_list[fit_index]

    biases = np.arange(0, 10, 1).astype(np.int32)
    train_x_bias = np.hstack((train_x, biases.reshape(-1, 1)))

    parents = generate_parents(max_p, max_n, (add, sub, mul, div), first_body_node=11, outputs=1, arity=2)
    fitness_objects = [Fitness() for _ in range(max_p + max_c)]
    
    # Initialize managers
    sharp_in_manager = SAM_IN(train_x_bias)
    sharp_out_manager = SAM_OUT()

    # Evaluate initial population
    fitnesses, alignment = evaluate_population(train_x_bias, train_y, parents, fitness_objects)
    sharp_in_list, sharp_in_std = evaluate_sharpness(train_x_bias, fitness_objects, parents, sharp_in_manager, opt=0)
    neighbor_map, sharp_out_list, sharp_out_std = evaluate_sharpness(train_x_bias, fitness_objects, parents, sharp_out_manager, opt=1)

    fit_track, avg_change_list, ret_avg_list, std_change_list = [], [], [], []
    avg_hist_list, drift_list = [], []
    
    mut_impact = MutationImpact(neutral_limit=0.1)
    num_elites = 7
    
    for g in range(1, max_g + 1):
        children, retention, density_distro = xover(deepcopy(parents), np.zeros(max_n * (1 + 2)), method='TwoPoint')
        children = basic_mutation(deepcopy(children))
        pop = parents + children
        fitnesses, alignment = evaluate_population(train_x_bias, train_y, pop, fitness_objects)
        
        if np.any(np.isnan(fitnesses)):
            fitnesses[np.isnan(fitnesses)] = np.PINF
        
        mut_impact(fitnesses, max_p)
        change_list, full_change_list, ret_list = process_generation_data(pop, fitnesses, max_p)
        
        avg_hist, avg_change, std_change, ret_avg, ret_std = process_generation_statistics(full_change_list, change_list, ret_list, g)
        avg_hist_list.append(avg_hist)
        avg_change_list.append(avg_change)
        std_change_list.append(std_change)
        ret_avg_list.append(ret_avg)
        ret_std_list.append(ret_std)
        
        sharp_in_list, sharp_in_std = update_sharpness(train_x_bias, fitness_objects, pop, sharp_in_manager, sharp_in_list, sharp_in_std)
        neighbor_map, sharp_out_list, sharp_out_std = update_sharpness(train_x_bias, fitness_objects, pop, sharp_out_manager, sharp_out_list, sharp_out_std, opt=1)

        best_i = np.argmin(fitnesses)
        best_fit = fitnesses[best_i]
        
        if g % 100 == 0:
            print(f"Gen {g} Best Fitness: {best_fit}\tMean SAM-In: {sharp_in_list[-1]}\tMean SAM-Out: {sharp_out_list[-1]}")
            # Add any specific plotting or logging here if needed

        fit_track.append(best_fit)
        parents = tournament_elitism(pop, fitnesses, max_p)
    
    # Final evaluation
    best_i = np.argmin(fitnesses)
    best_pop = pop[best_i]
    best_fit = fitnesses[best_i]
    
    print(f"Trial {trial}: Best Fitness = {best_fit}")
    drift_cum, drift_list = mut_impact.return_lists(option=1)
    
    # Save results
    save_results(func_name, run_name='cgp_2x', trial=trial, best_pop=best_pop, preds=None, best_fit=best_fit,
                 p_size=[cgp_active_nodes(best_pop[0], best_pop[1], opt=2)],
                 fit_track=fit_track, avg_change_list=avg_change_list, ret_avg_list=ret_avg_list,
                 bin_centers=None, hist_gens=None, avg_hist_list=avg_hist_list, drift_list=drift_list,
                 drift_cum=drift_cum, sharp_in_list=sharp_in_list, sharp_out_list=sharp_out_list,
                 sharp_in_std=sharp_in_std, sharp_out_std=sharp_out_std, density_distro=density_distro)

    # Final plotting
    plot_active_nodes(best_pop[0], best_pop[1], inputs + biases.shape[0], (add, sub, mul, div), biases, inputs, *Fitness()(train_x_bias, train_y, best_pop, opt=1))
    change_histogram_plot(avg_hist_list, func_name, 'cgp_2x', trial, max_g, opt=1)

if __name__ == "__main__":
    main()


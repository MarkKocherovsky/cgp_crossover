import numpy as np
import warnings
from pathlib import Path
import pickle
from sys import argv

from functions import *
from effProg import *
from similarity import *
from cgp_selection import *
from cgp_mutation import *
from cgp_xover import *
from cgp_fitness import *
from cgp_operators import *
from cgp_plots import *
from sharpness import *
from cgp_parents import *
from helpers import create_input_vector, initialize_population, get_noise_data, evaluate_neighbors, process_generation

warnings.filterwarnings('ignore')

def setup_environment():
    """Setup environment and return configuration parameters."""
    trial = int(argv[1])
    max_g = int(argv[2])
    max_n = int(argv[3])
    max_p = int(argv[4])
    max_c = int(argv[5])
    func_index = int(argv[6])
    fit_index = int(argv[7])
    
    biases = np.arange(0, 10, 1).astype(np.int32)
    inputs = 1
    arity = 2
    func_bank = Collection()
    func = func_bank.func_list[func_index]
    func_name = func_bank.name_list[func_index]
    
    fits = FitCollection()
    fit = fits.fit_list[fit_index]
    fit_name = fits.name_list[fit_index]
    
    return trial, max_g, max_n, max_p, max_c, biases, inputs, arity, func, func_name, fit, fit_name

def main():
    trial, max_g, max_n, max_p, max_c, biases, inputs, arity, func, func_name, fit, fit_name = setup_environment()
    
    bank = (add, sub, mul, div)
    parents = initialize_population(max_p, max_n, bank, arity)
    
    train_x_bias = create_input_vector(func.x_dom, biases, inputs)
    fitness_objects = [Fitness() for _ in range(max_p + max_c * max_p)]
    
    sharp_in_manager = SAM_IN(train_x_bias)
    sharp_out_manager = SAM_OUT()
    
    fit_track, p_size = [], []
    avg_change_list, std_change_list = [], []
    avg_hist_list, ret_avg_list, ret_std_list = [], [], []
    drift_list, drift_cum = [], np.array([0, 0, 0])
    
    for g in range(1, max_g + 1):
        results = process_generation(g, max_g, parents, train_x_bias, func.y_test, fitness_objects, sharp_in_manager, sharp_out_manager, max_c, func, arity)
        
        pop, parents, fit_temp, fitnesses, best_i, best_fit, sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std, avg_change_list, std_change_list, ret_avg_list, avg_hist_list, p_size = results
        
        fit_track.append(best_fit)
        p_size.append(cgp_active_nodes(pop[best_i][0], pop[best_i][1], opt=2))
        
        if g % 100 == 0:
            print(f"Gen {g} Best Fitness: {best_fit}\tMean SAM-In: {sharp_in_list[-1]}\tMean SAM-Out: {sharp_out_list[-1]}")
        
        parents = tournament_elitism(pop, fitnesses, max_p)
    
    best_fit = fit_temp[:, 0].min()
    best_individual = pop[np.argmin(fit_temp[:, 0])]
    
    print(f"Trial {trial}: Best Fitness = {best_fit}")
    print(f"Operators:\tDeleterious\tNeutral\tBeneficial")
    print(f"\t{drift_cum[0]}\t{drift_cum[1]}\t{drift_cum[2]}")
    
    Path(f"../output/{func_name}/log").mkdir(parents=True, exist_ok=True)
    pickle.dump(fit_track, open(f"../output/{func_name}/log/fit_track_{trial}.pkl", "wb"))
    
if __name__ == "__main__":
    main()


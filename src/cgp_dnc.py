#CGP 1 Point Crossover
import os
import sys
from functions import *
from helper import saveResults
from similarity import *
from cgp_plots import *
from cgp_fitness import *
from cgp_operators import *
from sharpness import *
from sys import argv

import torch
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dnc'))
if module_path not in sys.path:
    sys.path.append(module_path)

from cgp_ga import *
from multiparent_wrapper import NeuralCrossoverWrapper

print("started")
t = int(argv[1])  #trial
print(f'trial {t}')
max_g = int(argv[2])  #max generations
print(f'generations {max_g}')
max_n = int(argv[3])  #max body nodes
print(f'max body nodes {max_n}')
max_p = int(argv[4])  #max parents
print(f'Parents {max_p}')
outputs = 1
inputs = 1
biases = np.arange(0, 10, 1).astype(np.int32)
bias = biases.shape[0]  #number of biases
print(f'biases {biases}')
arity = 2
p_mut = float(argv[6])
p_xov = float(argv[7])
random.seed(t + 420)
print(f'Seed = {t + 420}')

run_name = 'cgp_dnc'

bank = (add, sub, mul, div)
bank_string = ("+", "-", "*", "/")

func_bank = Collection()
print(func_bank.func_list)
func = func_bank.func_list[int(argv[5])]
func_name = func_bank.name_list[int(argv[5])]
train_x = func.x_dom
train_y = func.y_test
train_x_bias = np.zeros((train_x.shape[0], biases.shape[0] + 1))
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = biases

print(train_x)
n_items = len(train_x_bias)
params_dict = {
    'n_generations': max_g,
    'population_size': max_p,
    'crossover_prob': p_xov,
    'mutation_prob': p_mut,
    'ind_length': max_n,
    'save_every_n_generations': 1000,
    'min_selection_val': 0,
    'max_selection_val': n_items - 1,
    'flip_mutation_prob': 0.1,
    'tournament_size': 4,
    'save_population_info': True,
    'save_fitness_info': True,
    'elitism': False,
    'n_parents': 2,
    'target_function': func
}

alignment = np.zeros((max_p, 2))
alignment[:, 0] = 1.0

torch.manual_seed(t + 420)
fitness = Fitness()
ncs = NeuralCrossoverWrapper(embedding_dim=64, sequence_length=params_dict['ind_length'] * 3 + 1,
                             num_embeddings=180 + 1,
                             running_mean_decay=0.95,
                             get_fitness_function=lambda ind: fitness(train_x_bias, train_y, ind),
                             batch_size=4, freeze_weights=True,
                             load_weights_path=None, learning_rate=1e-4,
                             epsilon_greedy=0.3, use_scheduler=False, use_device='cpu', n_parents=2)
ga_class = SelectionGA(**params_dict, random_state=t + 42)
PATH_TO_EXP = 'dnc/'
data = ga_class.fit(PATH_TO_EXP, train_x_bias, train_y, fitness, crossover_func=ncs.cross_pairs)
best_pop = data[0]
fit_track = data[1]
best_fit = fit_track[-1]
avg_change_list = data[2]
ret_avg_list = data[3]
p_size = data[4]
avg_hist_list = data[5]
mut_impact = data[6]
sharp_in_list = data[7]
sharp_out_list = data[8]
sharp_in_std = data[9]
sharp_out_std = data[10]
density_distro = data[11]
preds, p_A, p_B = fitness(train_x_bias, train_y, best_pop, opt=1)

mut_cum, mut_list, xov_cum, xov_list = mut_impact.returnLists(option=0)

Path(f"../output/{run_name}/{func_name}/log/").mkdir(parents=True, exist_ok=True)

first_body_node = inputs + bias
bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g)
n = plot_active_nodes(best_pop[0], best_pop[1], first_body_node, bank_string, biases, inputs, p_A, p_B, func_name,
                      run_name, t, opt=1)
saveResults(run_name, func_name, t, biases, best_pop, preds, best_fit, n, fit_track, avg_change_list, ret_avg_list,
            p_size, bin_centers, hist_gens, avg_hist_list, mut_list, mut_cum,
            xov_list, xov_cum, sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std, density_distro)


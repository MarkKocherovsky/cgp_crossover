#CGP 1 Point Crossover
import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from functions import *
from helper import saveResults
from similarity import *
from cgp_plots import *
from cgp_fitness import *
from cgp_operators import *
from sharpness import *
from sys import argv

import torch
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dnc'))
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
p_mut = float(argv[5])
print(f'p_mut: {p_mut}')
p_xov = float(argv[6])
print(f'p_xov: {p_xov}')
random.seed(t + 420)
print(f'Seed = {t + 420}')
xover_type = int(argv[7])
xover_list = ["Uniform", "OnePoint", "TwoPoint"]
try:
    learning_rate = float(argv[8])
except:
    learning_rate = 1e-4
try:
    xover = xover_list[xover_type]
    print(xover)
    run_name = f'cgp_dnc_{xover}'
except IndexError:
    print('cgp_dnc.py:\t unknown Crossover Type')
    xover = 'OnePoint'
    run_name = 'cgp_dnc_OnePoint'
run_name = f'{run_name}_real_world_lr_{learning_rate:.1e}'
print(run_name)

func_name = 'Diabetes'
bank, bank_string = loadBank()


func = Diabetes()
print(len(func()))
train_x, test_x, train_y, test_y = func()
dims = train_x.shape[1]
inputs = dims
first_body_node = inputs+bias

load_parents = False
if load_parents:
    run_name += '_parents'
    print('opening data/cgp_parents.pkl')
    with open('data/cgp_parents.pkl', 'rb') as f:
        parents = pickle.load(f)
else:
    parents = None
print(train_x)
train_x_bias = prepareConstants(train_x, biases)
test_x_bias = prepareConstants(test_x, biases)
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
                             num_embeddings=75,
                             running_mean_decay=0.95,
                             get_fitness_function=lambda ind: fitness(train_x_bias, train_y, ind, arity),
                             batch_size=820, freeze_weights=True,
                             load_weights_path=None, learning_rate=learning_rate,
                             epsilon_greedy=0.2, use_scheduler=False, use_device='cpu', n_parents=2, xover=xover)
ga_class = SelectionGA(**params_dict, random_state=t + 42)
PATH_TO_EXP = 'dnc/'
data = ga_class.fit(PATH_TO_EXP, train_x_bias, train_y, fitness, crossover_func=ncs.cross_pairs, parents=parents, first_body_node = first_body_node)
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
density_distro_list = data[12]
mut_density_distro = data[13]
mut_density_distro_list = data[14]
div_list = data[15]
fit_mean = data[16]
preds, p_A, p_B = fitness(train_x_bias, train_y, best_pop, arity, opt=1)

mut_cum, mut_list, xov_cum, xov_list = mut_impact.returnLists(option=0)

Path(f"../output/{run_name}/{func_name}/log/").mkdir(parents=True, exist_ok=True)

first_body_node = inputs + bias
bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g)
n = plot_active_nodes(best_pop[0], best_pop[1], first_body_node, bank_string, biases, inputs, p_A, p_B, func_name,
                      run_name, t, arity, opt=1)
saveResults(run_name, func_name, t, biases, best_pop, preds, best_fit, n, fit_track, avg_change_list, ret_avg_list,
            p_size, bin_centers, hist_gens, avg_hist_list, mut_list, mut_cum,
            xov_list, xov_cum, sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std, density_distro, density_distro_list, mut_density_distro, mut_density_distro_list, div_list, fit_mean)
print('mutation')
print(mut_density_distro)
print('xover')
print(density_distro)
xovers = density_distro['d']+density_distro['n'] + density_distro['b']
xovers = sum(xovers)
print(f'# xovers = {xovers}, expected amount = {p_xov*max_p/2*max_g}')

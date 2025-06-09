from cgp_ga import SelectionGA
from multiparent_wrapper import NeuralCrossoverWrapper
from cgp_fitness import *
import numpy as np
import torch
import json
import os

PERMUTATION = False
datasets_json = json.load(open('./datasets/koza1.json', 'r'))
PATH_TO_EXP = f'./experiments/cgp/DNC/'
dataset_name = 'koza1'
train_x = np.array(datasets_json[dataset_name]['x'])
constants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
train_x = np.array([[x, *constants] for x in train_x])
train_y = np.array(datasets_json[dataset_name]['y'])
n_items = len(train_x)
n_parents = 2
print(dataset_name, n_items)
try:
    os.makedirs(os.path.join(PATH_TO_EXP, dataset_name))
except FileExistsError:
    pass

params_dict = {
    'n_generations': 50,
    'population_size': 40,
    'crossover_prob': 0.5,
    'mutation_prob': 0.025,
    'ind_length': 64,
    'save_every_n_generations': 5,
    'min_selection_val': 0,
    'max_selection_val': n_items - 1,
    'flip_mutation_prob': 0.1,
    'tournament_size': 5,
    'save_population_info': False,
    'save_fitness_info': False,
    'elitism': False,
    'n_parents': n_parents
}

#CGP
fitness = Fitness()

torch.manual_seed(4242)
ncs = NeuralCrossoverWrapper(embedding_dim=64, sequence_length=params_dict['ind_length']*3+1, num_embeddings=180 + 1,
                             running_mean_decay=0.95,
                             get_fitness_function=lambda ind: fitness(train_x, train_y, ind),
                             batch_size=2048, freeze_weights=True,
                             load_weights_path=None, learning_rate=1e-4,
                             epsilon_greedy=0.3, use_scheduler=False, use_device='cpu', n_parents=n_parents)
ga_class = SelectionGA(**params_dict, random_state=42)
ga_class.fit(PATH_TO_EXP, train_x, train_y, fitness, crossover_func=ncs.cross_pairs)

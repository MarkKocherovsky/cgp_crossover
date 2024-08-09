# selection_utils.py
import numpy as np
from numpy import random

def get_best_individual(pop, f_list):
    best_f_i = np.argmin(f_list)
    return pop[best_f_i], best_f_i

def tournament_selection(pop, f_list, n_con):
    idx = np.arange(len(pop))
    c_id = random.choice(idx, n_con, replace=False)  # Get contestants' ids
    f_c = f_list[c_id]
    winner = np.argmin(f_c)
    w_id = c_id[winner]
    return pop[w_id], w_id


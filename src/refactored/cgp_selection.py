# main.py
import numpy as np
from numpy import random
from selection_utils import get_best_individual, tournament_selection

def tournament_elitism(pop, f_list, max_p, n_con=4):
    new_p = []
    p_distro = np.zeros(len(pop))
    
    # Keep best individual
    best_ind, best_f_i = get_best_individual(pop, f_list)
    p_distro[best_f_i] += 1
    new_p.append(best_ind)
    
    # Perform tournament selection
    while len(new_p) < max_p:
        winner, w_id = tournament_selection(pop, f_list, n_con)
        p_distro[w_id] += 1
        new_p.append(winner)
    
    return new_p, p_distro  # Uncomment if p_distro is needed


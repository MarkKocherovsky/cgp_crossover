from copy import deepcopy

import numpy as np
from numpy import random


#Selection
#pop: population
#f_list: fitnesses
#n_con: number of contestants
def tournament_elitism(pop, f_list, max_p, n_con=4):
    new_p = []
    p_distro = np.zeros(len(pop))
    idx = np.array(range(0, len(pop)))
    #keep best ind
    best_f_i = np.argmin(f_list)
    p_distro[best_f_i] += 1
    new_p.append(pop[best_f_i])
    while len(new_p) < max_p:
        c_id = random.choice(idx, (n_con,), replace=False)  #get contestants id
        f_c = f_list[c_id]
        winner = np.argmin(f_c)
        w_id = c_id[winner]
        p_distro[w_id] += 1
        new_p.append(pop[w_id])
    return new_p  #, p_distro


def selectElite(current_parent: tuple, current_children: list, parent_fitness: float, child_fitness: [float],
                parent_a, parent_b, child_a, child_b) -> object:
    max_children = len(current_children)
    if any(child_fitness <= parent_fitness) and random.rand() > 1 / max_children:
        best = np.argmin(child_fitness)
        current_parent = deepcopy(current_children[best])
        p_fi = np.argmin(child_fitness)
        # parent_distro[0] += 1
        parent_fitness = np.min(child_fitness)
        parent_a = child_a[p_fi]
        parent_b = child_b[p_fi]
    return current_parent, parent_fitness, parent_a, parent_b

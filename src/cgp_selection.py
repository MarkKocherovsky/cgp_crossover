from copy import deepcopy

import numpy as np
from numpy import random
from cgp_parents import *
from copy import copy, deepcopy


#Selection
#pop: population
#f_list: fitnesses
#n_con: number of contestants

import numpy as np
from copy import deepcopy

def individuals_are_unique(new_p, candidate):
    # Check if the candidate individual is already in new_p
    try:
        body_tuple = tuple(candidate[0].flatten())
        output_tuple = tuple(candidate[1])
        combined = (body_tuple, output_tuple)
    except IndexError as e:
        print(f'cgp_selection::individuals_are_unique {e}')
        print(f'candidate\n{candidate}')
        exit(1) 
    for ind in new_p:
        ind_body_tuple = tuple(ind[0].flatten())
        ind_output_tuple = tuple(ind[1])
        if (ind_body_tuple, ind_output_tuple) == combined:
            return False  # Found a duplicate
    return True  # Unique individual

def tournament_elitism(population, f_list, max_p, n_con=4, replace=True, fixed_length=True):
    new_p = []
    pop = deepcopy(population)  # Ensure a deep copy of the population
    available_idx = list(range(len(pop)))

    # Keep best individual
    best_f_i = np.argmin(f_list)
    best_individual = (np.copy(pop[best_f_i][0]), np.copy(pop[best_f_i][1]))
    new_p.append(best_individual)

    if not replace:
        available_idx.remove(best_f_i)

    # Selection loop
    while len(new_p) < max_p:
        remaining_unique = len(available_idx) - (max_p - len(new_p))
        if remaining_unique <= 0 or len(available_idx) <= 0:
            break  # Early exit if not enough unique candidates left
        c_id = np.random.choice(available_idx, (min(n_con, len(available_idx)),), replace=False)
        f_c = f_list[c_id]

        # Find minimum fitness values and their indices
        min_fitness = np.min(f_c)
        min_indices = np.where(f_c == min_fitness)[0]  # Get indices of individuals with min fitness
        winner = np.random.choice(min_indices) if len(min_indices) > 1 else min_indices[0]
        w_id = c_id[winner]

        if not replace:
            available_idx.remove(w_id)
        selected_individual = (np.copy(pop[w_id][0]), np.copy(pop[w_id][1]))
        # Check for duplicates before adding
        if individuals_are_unique(new_p, selected_individual):
            new_p.append(selected_individual)
    # Handle single parent generation outside the loop if only one is needed
    if max_p - len(new_p) == 1:
        new_individual = generate_parents(1, fixed_length)
        if individuals_are_unique(new_p, new_individual):
            new_p.append(copy(new_individual))

    # Generate multiple individuals if more than one is needed
    while len(new_p) < max_p:
        new_individuals = generate_parents(max_p - len(new_p), fixed_length)
        for new_ind in new_individuals:
            if individuals_are_unique(new_p, new_ind):
                new_p.append(copy(new_ind))
                if len(new_p) >= max_p:
                    break  # Exit if we have reached the desired population size

    return new_p

"""
def tournament_elitism(population, f_list, max_p, n_con=4, replace=True):
    new_p = []
    pop = deepcopy(population)
    p_distro = np.zeros(len(pop))
    available_idx = list(range(len(pop)))

    # Keep best individual
    best_f_i = np.argmin(f_list)
    p_distro[best_f_i] += 1
    new_p.append(pop[best_f_i])
    if not replace:
        available_idx.remove(best_f_i)

    while len(new_p) < max_p:
        c_id = np.random.choice(available_idx, (n_con,), replace=False)  # Get contestants' ids
        f_c = f_list[c_id]
        winner = np.argmin(f_c)
        w_id = c_id[winner]
        p_distro[w_id] += 1
        new_p.append(pop[w_id])
        if not replace:
            available_idx.remove(w_id)

    return new_p  # , p_distro
"""

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

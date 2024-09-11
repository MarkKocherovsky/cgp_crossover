import numpy as np
from numpy import random
from copy import deepcopy
from cgp_parents import generate_single_instruction


def mutate_node(ind, out, i, arity, in_size, bank_len, density):
    if i >= ind.shape[0]:  # output node
        density[i] += 1
        i -= ind.shape[0]
        out[i] = random.randint(0, ind.shape[0] + in_size)

    else:  # body node
        j = int(ind.shape[1] * random.random_sample())
        if j < arity:
            ind[i, j] = random.randint(0, i + in_size)
        else:
            ind[i, j] = random.randint(0, bank_len)
        density[i * ind.shape[1] + j] += 1
    return (ind, out), density


def mutate_1_plus_4(individual, len_bank=4, arity=2, in_size=11):
    ind, out = individual
    density = np.zeros(ind.shape[0] * ind.shape[1] + out.shape[0])
    i = int((ind.shape[0] + out.shape[0]) * random.random_sample())
    new_ind, density = mutate_node(ind, out, i, arity, in_size, len_bank, density)
    return new_ind, density, [i]


def basic_mutation(subjects, arity=2, in_size=11, p_mut=0.025, bank_len=4):
    mutated_individuals = []
    density = np.zeros((len(subjects), subjects[0][0].shape[0] * subjects[0][0].shape[1] + 1))
    for m in range(len(subjects)):
        if random.random() < p_mut:
            mutant = deepcopy(subjects[m])
            ind, out = mutant
            i = int((ind.shape[0] + out.shape[0]) * random.random_sample())
            subjects[m], density[m] = mutate_node(ind, out, i, arity, in_size, bank_len, density[m])
            mutated_individuals.append(m)
    return subjects, mutated_individuals, density


def macromicro_mutation(subjects, arity=2, in_size=11, p_mut=0.025, bank_len=4, n_max=64):
    mutated_individuals = []
    for m in range(len(subjects)):
        if random.random() < p_mut:
            mutated_individuals.append(m)
            mutant = deepcopy(subjects[m])
            ind, out = np.array(mutant[0].copy()), np.array(mutant[1].copy())
            i = int((ind.shape[0] + out.shape[0]) * random.random_sample())

            if i >= ind.shape[0]:  # output node
                out[i - ind.shape[0]] = random.randint(0, ind.shape[0] + in_size)
            else:  # body node
                mutation = determine_mutation(len(ind), n_max)
                if mutation == 0:  # deletion
                    ind, out = delete_instruction(ind, out)
                elif mutation == 1:  # point mutation
                    ind, out = mutate_node(ind, out, i, arity, in_size, bank_len)
                elif mutation == 2:  # insertion
                    ind = insert_instruction(ind, generate_single_instruction(i))
                else:
                    raise ValueError(f'src::cgp_mutation::macromicro_mutation: asked for mutation == {mutation}')

            subjects[m] = (ind, out)
    return subjects, mutated_individuals


def determine_mutation(ind_len, n_max):
    if ind_len < 2:
        return random.randint(1, 3)  # skip deletion if too small
    elif ind_len >= n_max:
        return random.randint(0, 2)  # skip insertion if too large
    else:
        return random.randint(0, 3)  # any mutation


def delete_instruction(ind, out):
    j = random.randint(0, len(ind))
    ind = np.delete(ind, j, axis=0)
    whr = np.where(ind >= len(ind))[0]
    for instance in whr:
        ind[instance] = random.randint(0, len(ind))
    out = [random.randint(0, len(ind)) if x >= len(ind) else x for x in out]
    return ind, out


def insert_instruction(ind, new_instruction):
    j = random.randint(0, len(ind))
    return np.insert(ind, j, new_instruction, axis=0)

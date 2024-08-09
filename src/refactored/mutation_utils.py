# mutation_utils.py
import numpy as np
from numpy import random
from copy import deepcopy

def mutate_node(ind, out, arity, in_size, len_bank):
    i = int((ind.shape[0] + out.shape[0]) * random.random_sample())
    if i >= ind.shape[0]:  # output node
        i = i - ind.shape[0]
        out[i] = random.randint(0, ind.shape[0] + in_size)
    else:  # body node
        j = int(ind.shape[1] * random.random_sample())
        if j < arity:
            ind[i, j] = random.randint(0, i + in_size)
        else:
            ind[i, j] = random.randint(0, len_bank)
    return ind, out

def adjust_individual(ind, out, arity, in_size, len_bank, n_max, generate_single_instruction):
    i = int((ind.shape[0] + out.shape[0]) * random.random_sample())
    if i >= ind.shape[0]:  # output node
        i = i - ind.shape[0]
        out[i] = random.randint(0, ind.shape[0] + in_size)
    else:  # body node
        if len(ind) < 2:  # too small
            mutation = random.randint(1, 3)
        elif len(ind) >= n_max:  # too big
            mutation = random.randint(0, 2)
        else:  # any mutation
            mutation = random.randint(0, 3)
        if mutation == 0:  # remove instruction
            j = random.randint(0, len(ind))
            ind = np.delete(ind, j, axis=0)
            ind = _fix_indices(ind, out)
        elif mutation == 1:  # point mutation
            ind, out = mutate_node(ind, out, arity, in_size, len_bank)
        elif mutation == 2:  # add instruction
            j = random.randint(0, len(ind))
            new_inst = generate_single_instruction(j)
            ind = np.insert(ind, j, new_inst, axis=0)
        else:
            raise ValueError(f'src::cgp_mutation::macromicro_mutation: asked for mutation == {mutation}')
    ind = _prevent_recursion(ind, in_size)
    return ind, out

def _fix_indices(ind, out):
    whr = np.array(np.where(ind >= ind.shape[0]))
    if len(whr) > 0:
        for instance in range(len(whr)):
            ind[whr[instance]] = random.randint(0, ind.shape[0])
    for o in range(len(out)):
        if out[o] >= ind.shape[0]:
            out[o] = random.randint(0, ind.shape[0])
    return ind

def _prevent_recursion(ind, in_size):
    for k in range(len(ind)):  # prevent recursion
        for l in range(ind[k, :-1].shape[0]):
            if ind[k, l] == k + in_size:
                ind[k, l] = random.randint(0, k + in_size)
    return ind


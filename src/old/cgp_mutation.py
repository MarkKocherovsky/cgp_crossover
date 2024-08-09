import numpy as np
from numpy import random
from copy import deepcopy
from cgp_parents import generate_single_instruction


#Subjects: Individuals to go through the loop

def mutate_1_plus_4(individual, len_bank=4, arity=2, in_size=11):  #temporary to make cgp(1+4) work
    ind = individual[0]
    out = individual[1]
    i = int((ind.shape[0] + out.shape[0]) * random.random_sample())
    if i >= ind.shape[0]:  #output node
        i = i - ind.shape[0]
        out[i] = random.randint(0, ind.shape[0] + in_size)
    else:  #body node
        j = int(ind.shape[1] * random.random_sample())
        #print(i,j)
        if j < arity:
            ind[i, j] = random.randint(0, i + in_size)
        else:
            ind[i, j] = random.randint(0, len_bank)
    return ind, out


def basic_mutation(subjects, arity=2, in_size=11, p_mut=0.025, bank_len=4):
    #mutants = parents[mut_id]
    for m in range(len(subjects)):
        if random.random() >= p_mut:
            continue
        mutant = deepcopy(subjects[m])
        ind = mutant[0].copy()
        out = mutant[1].copy()
        i = int((ind.shape[0] + out.shape[0]) * random.random_sample())
        if i >= ind.shape[0]:  #output node
            i = i - ind.shape[0]
            out[i] = random.randint(0, ind.shape[0] + in_size)
        else:  #body node
            j = int(ind.shape[1] * random.random_sample())
            #print(i,j)
            if j < arity:
                ind[i, j] = random.randint(0, i + in_size)
            else:
                ind[i, j] = random.randint(0, bank_len)
        subjects[m] = (ind, out)
    return subjects


def macromicro_mutation(subjects, arity=2, in_size=11, p_mut=0.025, bank_len=4, n_max=64):
    for m in range(len(subjects)):
        if random.random() >= p_mut:
            continue
        mutant = deepcopy(subjects[m])
        ind = np.array(mutant[0].copy())
        out = np.array(mutant[1].copy())
        i = int((ind.shape[0] + out.shape[0]) * random.random_sample())
        if i >= ind.shape[0]:  #output node
            i = i - ind.shape[0]
            out[i] = random.randint(0, ind.shape[0] + in_size)
        else:  #body node
            if len(ind) < 2:  #too small
                mutation = random.randint(1, 3)
            elif len(ind) >= n_max:  #too big
                mutation = random.randint(0, 2)
            else:  #any mutation
                mutation = random.randint(0, 3)
            if mutation == 0:  #refactored with chatgpt
                j = random.randint(0, len(ind) - 1)  # Ensure valid index
                ind = np.delete(ind, j, axis=0)
                whr = np.where(ind >= len(ind))[0]  # Ensure whr is a 1D array
                for instance in whr:
                    ind[instance] = random.randint(0, len(ind) - 1)
                out = [random.randint(0, len(ind) - 1) if x >= len(ind) else x for x in out]
            elif mutation == 1:  #point mutation
                j = int(ind.shape[1] * random.random_sample())
                if j < arity:
                    ind[i, j] = random.randint(0, i + in_size)
                else:
                    ind[i, j] = random.randint(0, bank_len)
            elif mutation == 2:  #Add instruciton
                j = random.randint(0, len(ind))
                np.insert(ind, j, generate_single_instruction(j), axis=0)
            else:
                raise ValueError(f'src::cgp_mutation::macromicro_mutation: asked for mutation == {mutation}')
            for k in range(len(ind)):  #refactored with chatgpt, prevent recursion
                ind[k, :-1] = np.where(ind[k, :-1] == k + in_size, random.randint(0, k + in_size), ind[k, :-1])

        subjects[m] = (ind, out)
    return subjects

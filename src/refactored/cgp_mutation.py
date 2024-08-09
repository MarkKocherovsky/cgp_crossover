# main.py
import numpy as np
from numpy import random
from copy import deepcopy
from cgp_parents import generate_single_instruction
from mutation_utils import mutate_node, adjust_individual

def mutate_1_plus_4(individual, len_bank=4, arity=2, in_size=11):
    ind, out = individual
    ind, out = mutate_node(ind, out, arity, in_size, len_bank)
    return ind, out

def basic_mutation(subjects, arity=2, in_size=11, p_mut=0.025, bank_len=4):
    for m in range(len(subjects)):
        if random.random() < p_mut:
            subjects[m] = mutate_subject(subjects[m], arity, in_size, bank_len)
    return subjects

def macromicro_mutation(subjects, arity=2, in_size=11, p_mut=0.025, bank_len=4, n_max=64):
    for m in range(len(subjects)):
        if random.random() < p_mut:
            subjects[m] = mutate_macromicro_subject(subjects[m], arity, in_size, bank_len, n_max)
    return subjects

def mutate_subject(subject, arity, in_size, bank_len):
    mutant = deepcopy(subject)
    ind, out = mutant
    ind, out = mutate_node(ind, out, arity, in_size, bank_len)
    return ind, out

def mutate_macromicro_subject(subject, arity, in_size, bank_len, n_max):
    mutant = deepcopy(subject)
    ind, out = mutant
    ind, out = adjust_individual(ind, out, arity, in_size, bank_len, n_max, generate_single_instruction)
    return ind, out


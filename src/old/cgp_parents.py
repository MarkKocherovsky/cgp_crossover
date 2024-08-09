import numpy as np
from numpy import random


def generate_parents(max_p, max_n, bank, first_body_node=11, outputs=1, arity=2, fixed_length=True):
    parents = []
    for p in range(0, max_p):
        if fixed_length:
            n = max_n
        else:
            n = random.randint(1, max_n + 1)
        ind_base = np.zeros((n, arity + 1), np.int32)
        for i in range(0, n):
            #print(i < inputs+bias)
            for j in range(0, arity):  #input
                #if i < (first_body_node):
                #	ind_base[i,j] = random.randint(0, first_body_node)
                #else:
                ind_base[i, j] = random.randint(0, i + first_body_node)
            ind_base[i, -1] = random.randint(0, len(bank))
            output_nodes = random.randint(0, n + first_body_node, (outputs,), np.int32)
        if max_p == 1:
            return ind_base.copy(), output_nodes.copy()
        else:
            parents.append((ind_base.copy(), output_nodes.copy()))
    return parents


def generate_single_instruction(idx, bank_len=4, first_body_node=11, arity=2):
    instruction = np.zeros((1 + arity,))
    for i in range(0, arity):
        instruction[i] = random.randint(0, idx + first_body_node)
    instruction[-1] = random.randint(0, bank_len)
    return instruction

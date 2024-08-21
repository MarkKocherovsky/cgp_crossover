import numpy as np
from numpy import random


def generate_parents(max_p, max_n, bank, inputs=1, n_constants=10, outputs=1, arity=2, fixed_length=True):
    parents = []
    first_body_node = inputs+n_constants
    for _ in range(max_p):
        n = max_n if fixed_length else random.randint(1, max_n + 1)
        ind_base = generate_individual_base(n, arity, first_body_node, len(bank))
        output_nodes = random.randint(0, n + first_body_node, outputs, np.int32)
        parent = (ind_base.copy(), output_nodes.copy())
        parents.append(parent)
    return parents


def generate_individual_base(n, arity, first_body_node, bank_len):
    ind_base = np.zeros((n, arity + 1), np.int32)
    for i in range(n):
        ind_base[i, :-1] = [random.randint(0, i + first_body_node) for _ in range(arity)]
        ind_base[i, -1] = random.randint(0, bank_len)
    return ind_base


def generate_single_instruction(idx, first_body_node, bank_len=4, arity=2):
    instruction = np.zeros((arity + 1,))
    for i in range(arity):
        instruction[i] = random.randint(0, idx + first_body_node)
    instruction[-1] = random.randint(0, bank_len)
    return instruction

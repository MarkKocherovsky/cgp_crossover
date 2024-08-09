# main.py
from utils import np, randint, generate_individual, generate_output_nodes

def generate_parents(max_p, max_n, bank, first_body_node=11, outputs=1, arity=2, fixed_length=True):
    parents = []
    bank_len = len(bank)
    for _ in range(max_p):
        n = max_n if fixed_length else randint(1, max_n + 1)
        ind_base = generate_individual(n, arity, first_body_node, bank_len)
        output_nodes = generate_output_nodes(n, first_body_node, outputs)
        parent = (ind_base.copy(), output_nodes.copy())
        if max_p == 1:
            return parent
        else:
            parents.append(parent)
    return parents

def generate_single_instruction(idx, bank_len=4, first_body_node=11, arity=2):
    instruction = np.zeros((1 + arity,))
    for i in range(arity):
        instruction[i] = randint(0, idx + first_body_node)
    instruction[-1] = randint(0, bank_len)
    return instruction


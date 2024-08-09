import numpy as np
import random
from subgraph import SubgraphCrossover


def crossover_segments(ind1, ind2, i, j, fixed_length):
    front_1, back_1 = ind1[:i], ind1[i:]
    front_2, back_2 = ind2[:j], ind2[j:]
    return np.concatenate((front_1, back_2)), np.concatenate((front_2, back_1))


def crossover_segments_2x(ind1, ind2, i, j):
    front_1, mid_1, back_1 = ind1[:i], ind1[i:j], ind1[j:]
    front_2, mid_2, back_2 = ind2[:i], ind2[i:j], ind2[j:]
    return np.concatenate((front_1, mid_2, back_1)), np.concatenate((front_2, mid_1, back_2))


def adjust_outputs(out1, out2, i, j):
    if len(out1.shape) <= 1 and out1.shape[0] == 1:
        return out2.copy(), out1.copy()
    else:
        out1 = np.concatenate((out1[:i], out2[j:]))
        out2 = np.concatenate((out2[:j], out1[i:]))
        return out1, out2


def limit_values(out, ind_shape, first_body_node):
    whr_o = np.where(out >= ind_shape[0]+first_body_node)
    out[whr_o] = ind_shape[0] + first_body_node - 1
    return out


def trim_individual(ind, max_n):
    if ind.shape[0] > max_n:
        idxs = np.arange(0, max_n)
        to_del = np.random.choice(idxs, ind.shape[0] - max_n, replace=False)
        ind = np.delete(ind, to_del, axis=0)
    return ind


def adjust_indices(ind, bank_len, first_body_node):
    whr = np.array(np.where(ind[:, :-1] >= ind.shape[0]+first_body_node)).T
    for w in whr:
        ind[w[0], w[1]] = np.random.randint(0, ind.shape[0]+first_body_node) if ind.shape[0] > 1 else 0

    ar = np.array(list(zip(*np.where(ind[:, -1] >= bank_len))))
    for a in ar:
        ind[a[0], -1] = np.random.randint(0, bank_len)
    return ind

def prevent_recursion(ind, first_body_node):
    for i in range(ind.shape[0]):
        # Identify positions where the value could cause recursion
        positions = np.where(ind[i, :-1] >= i + first_body_node)
        for w in zip(*positions):
            # Set a new random value at the identified position to prevent recursion
            ind[i, w[0]] = np.random.randint(0, i + first_body_node) if ind.shape[0] > 1 else 0
    
    return ind

def xover_1x(p1, p2, max_n, first_body_node=11, fixed_length=True, bank_len=4):
    ind1, ind2 = np.array(p1[0]), np.array(p2[0])
    out1, out2 = np.array(p1[1]), np.array(p2[1])
    d_distro = np.zeros(max_n * 3)
    s1 = ind1.shape[1]

    if fixed_length:
        s2 = s1
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
    else:
        s2 = ind2.shape[1]

    i = np.random.randint(1, ind1.shape[0] - 1)
    j = i if fixed_length else np.random.randint(1, ind2.shape[0] - 1)
    d_distro[i] += 1
    if not fixed_length:
        d_distro[j] += 1

    ind1, ind2 = crossover_segments(ind1, ind2, i, j, fixed_length)

    i_out = np.random.randint(0, out1.shape[0] - 1) if out1.shape[0] > 1 else 0
    if fixed_length or out2.shape[0] <= 1:
        j_out = i_out
    else:
        j_out = np.random.randint(0, out2.shape[0] - 1)

    out1, out2 = adjust_outputs(out1, out2, i_out, j_out)

    ind1, ind2 = ind1.reshape(-1, s1), ind2.reshape(-1, s2)
    ind1, ind2 = trim_individual(ind1, max_n), trim_individual(ind2, max_n)
    ind1, ind2 = adjust_indices(ind1, bank_len, first_body_node), adjust_indices(ind2, bank_len, first_body_node)
    ind1, ind2 = prevent_recursion(ind1, first_body_node), prevent_recursion(ind2, first_body_node)
    out1, out2 = limit_values(out1, ind1.shape, first_body_node), limit_values(out2, ind2.shape, first_body_node)
    return [(ind1, out1), (ind2, out2), d_distro]


def xover_2x(p1, p2, max_n, first_body_node=11, fixed_length=False, bank_len=4):
    ind1, ind2 = p1[0].flatten(), p2[0].flatten()
    out1, out2 = p1[1], p2[1]
    d_distro = np.zeros(max_n * 3)
    s = p1[0].shape

    i = random.randint(1, ind1.shape[0] - 1)
    j = random.randint(i, ind1.shape[0] - 1)
    d_distro[i] += 1
    d_distro[j] += 1

    ind1, ind2 = crossover_segments_2x(ind1, ind2, i, j)

    i_out = random.randint(0, out2.shape[0])
    out1, out2 = adjust_outputs(out1, out2, i_out, i_out)

    return [(ind1.reshape(s), out1), (ind2.reshape(s), out2), d_distro]


def xover_sgx(p1, p2, max_n, inputs=1, first_body_node=11, fixed_length=False, bank_len=4):
    c1, c1_distro = SubgraphCrossover(p1, p2, max_n, inputs)
    c2, c2_distro = SubgraphCrossover(p2, p1, max_n, inputs)
    d_distro = c1_distro + c2_distro
    return c1, c2, d_distro


def xover(parents, max_n, method='None', p_xov=0.5, first_body_node=11, fixed_length=True, bank_len=4):
    children = []
    d_distro = np.zeros((len(parents), max_n * 3))
    methods = {
        'None': 'None',
        'OnePoint': xover_1x,
        'TwoPoint': xover_2x,
        'Subgraph': xover_sgx,
    }

    xover_method = methods.get(method, 'None')
    retention = []

    for i in range(0, len(parents), 2):
        if method == 'None' or random.random() < p_xov:
            children.append(parents[i])
            children.append(parents[i + 1])
        elif method in methods:
            c1, c2, d_distro[i:i + 1, :] = xover_method(parents[i], parents[i + 1], max_n=max_n,
                                                        fixed_length=fixed_length,
                                                        first_body_node=first_body_node, bank_len=bank_len)
            children.append(c1)
            children.append(c2)
            retention.append(i)
    return children, np.array(retention, dtype=np.int32), d_distro

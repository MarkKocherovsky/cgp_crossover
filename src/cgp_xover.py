import numpy as np
import random
from subgraph import SubgraphCrossover


def crossover_segments(ind1, ind2, i, j, fixed_length):
    front_1, back_1 = ind1[:i], ind1[i:]
    front_2, back_2 = ind2[:j], ind2[j:]
    return np.concatenate((front_1, back_2)), np.concatenate((front_2, back_1))


def crossover_segments_2x(ind1, ind2, i1, i2, j1, j2):
    front_1, mid_1, back_1 = ind1[:i1], ind1[i1:i2], ind1[i2:]
    front_2, mid_2, back_2 = ind2[:j1], ind2[j1:j2], ind2[j2:]
    return np.concatenate((front_1, mid_2, back_1)), np.concatenate((front_2, mid_1, back_2))


def adjust_outputs(out1, out2, i, j):
    if len(out1.shape) <= 1 and out1.shape[0] == 1:
        return out2.copy(), out1.copy()
    else:
        out1 = np.concatenate((out1[:i], out2[j:]))
        out2 = np.concatenate((out2[:j], out1[i:]))
        return out1, out2


def limit_values(out, ind_shape, first_body_node):
    out = np.atleast_1d(out)
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

def xover_1x(p1, p2, max_n, first_body_node, fixed_length=True, bank_len=4):
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
    try:
        i = np.random.randint(1, ind1.shape[0] - 1)
    except ValueError:
        print(f'ind1 shape = {ind1.shape[0]}, cannot perform randomness, setting i to 0.')
        i = 0
    try:
        j = i if fixed_length else np.random.randint(1, ind2.shape[0] - 1)
    except ValueError as e:
        print('cgp_xover.py::xover_1x')
        print(e)
        print(ind1.shape)
        print(ind2.shape)
        print(i)
        if not fixed_length:
            j = 0
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

def xover_2x(p1, p2, max_n, first_body_node, fixed_length=True, bank_len=4, outputs=1):
    ind1, ind2 = p1[0], p2[0]
    out1, out2 = p1[1], p2[1]
    d_distro = np.zeros(max_n * 3 + outputs)
    s1 = p1[0].shape[1]
    if fixed_length:
        s2 = s1
        l1 = len(ind1.flatten())
        ind1 = np.concatenate((ind1.flatten(), out1))
        ind2 = np.concatenate((ind2.flatten(), out2))
    else:
        s2 = ind2.shape[1]
    try:
        i1, i2 = sorted(random.sample(range(1, ind1.shape[0]), 2))
    except ValueError:
        i1, i2 = 0, 1
    try:
        j1, j2 = (i1, i2) if fixed_length else sorted(random.sample(range(1, ind2.shape[0]), 2))
    except ValueError as e:
        print('cgp_xover.py::xover_2x')
        print(e)
        print(ind1.shape)
        print(ind2.shape)
        print(i1, i2)
        if not fixed_length:
            j1, j2 = 0, 1

    indices = [i1, i2] + ([j1, j2] if not fixed_length else [])
    for idx in indices:
        d_distro[idx] += 1


    ind1, ind2 = crossover_segments_2x(ind1, ind2, i1, i2, j1, j2)

    i_out = np.random.randint(0, out1.shape[0] - 1) if out1.shape[0] > 1 else 0
    if fixed_length or out2.shape[0] <= 1:
        j_out = i_out
    else:
        j_out = np.random.randint(0, out2.shape[0] - 1)

    if fixed_length:
        out1, out2= ind1[l1:].copy(), ind2[l1:].copy()
        ind1, ind2 = ind1[:l1], ind2[:l1]
    out1, out2 = adjust_outputs(out1, out2, i_out, j_out)

    ind1, ind2 = ind1.reshape(-1, s1), ind2.reshape(-1, s2)
    ind1, ind2 = trim_individual(ind1, max_n), trim_individual(ind2, max_n)
    ind1, ind2 = adjust_indices(ind1, bank_len, first_body_node), adjust_indices(ind2, bank_len, first_body_node)
    ind1, ind2 = prevent_recursion(ind1, first_body_node), prevent_recursion(ind2, first_body_node)
    out1, out2 = limit_values(out1, ind1.shape, first_body_node), limit_values(out2, ind2.shape, first_body_node)
    return [(ind1, out1), (ind2, out2), d_distro]


def normalize_indices(length1, length2):
    """Normalize indices for variable-length parents."""
    samples = [np.arange(length1+1), np.arange(length2+1)]
    samples_norm = [samples[0] / length1, samples[1] / length2]
    return samples, samples_norm

def concatenate_and_sort(p1, samples, samples_norm, d_distro, parent_length):
    """Concatenate nodes and sort based on normalized indices."""
    fu_list = np.concatenate(samples)
    no_list = np.concatenate(samples_norm)

    # Sort by normalized order
    sorted_indices = no_list.argsort()
    split_idx = len(samples[0])
    out_idx = len(sorted_indices) - 2  # Index for the last two outputs

    # Prepare a list to hold the output node indices
    sorted_out = []

    # Update d_distro based on sorted indices for nodes
    for i in sorted_indices[:parent_length]:
        if i < split_idx:
            # For body1
            sample_idx = i
            d_distro[0, samples[0][sample_idx]] += 1
        elif split_idx <= i < out_idx:
            # For body2
            sample_idx = i - split_idx
            d_distro[0, samples[1][sample_idx]] += 1
        else:
            # Handle the last two indices (output nodes)
            sorted_out.append(p1[i])  # Append the output node to the list

    # Only update d_distro for output nodes if they are exchanged
    if len(sorted_out) == 2:
        d_distro[0, -1] += 1  # Increment for the output node count if they are part of the crossover

    # Create a trimmed version of p1, excluding the last two elements
    trimmed_p1 = p1[:-2]  # Keep all but the last two elements

    # Return trimmed_p1 and handle the outputs based on whether they were exchanged
    if len(sorted_out) == 2:
        return np.array(trimmed_p1), np.atleast_1d(sorted_out[0]), np.atleast_1d(sorted_out[1]), d_distro
    else:
        # If outputs are not exchanged, return them as they were
        return np.array(trimmed_p1), p1[-2], p1[-1], d_distro



def split_nodes(combined_sorted, out1, out2, len1, len2, shape1, shape2):
    # Assuming combined_sorted is structured to include both body and output nodes
    # Split the body nodes back into two parts
    new_p1_body = combined_sorted[:len1]
    new_p2_body = combined_sorted[len1:len1 + len2]
    return (new_p1_body, out1), (new_p2_body, out2)


def generate_mask(genome_length):
    half_length = genome_length // 2
    if genome_length % 2 == 0:
        extra = 0
    else:
        extra = 1
    mask = np.array([True] * half_length + [False] * half_length + [False] * extra)
    np.random.shuffle(mask)  # Shuffle the array to randomize the order
    return mask


def xover_uniform(p1, p2, max_n, first_body_node, fixed_length=True, bank_len=4):
    """Main crossover function handling both fixed-length and variable-length parents."""
    # Flatten and concatenate genomes for fixed-length, otherwise exchange nodes
    out1, out2 = p1[1], p2[1] 
    if fixed_length:
        d_distro = np.zeros((1, max_n * 3 + 1))
        p1_flat = np.concatenate((p1[0].flatten(), out1))
        p2_flat = np.concatenate((p2[0].flatten(), out2))
        genome_length = len(p1_flat)
        
        # Generate mask and apply crossover for flattened genomes
        mask = generate_mask(genome_length)
        d_distro[0, mask] += 1
        p1_flat[mask], p2_flat[mask] = p2_flat[mask].copy(), p1_flat[mask].copy()
        body_len = len(p1_flat)-len(out1)
        new_p1 = (p1_flat[:body_len].reshape(p1[0].shape), np.atleast_1d(p1_flat[body_len:].flatten()))
        new_p2 = (p2_flat[:body_len].reshape(p2[0].shape), np.atleast_1d(p2_flat[body_len:].flatten()))
  
    else:
        # Handle variable-length by node-level crossover (axis 1)
        d_distro = np.zeros((1, max_n + 1))
        len1, len2 = p1[0].shape[0], p2[0].shape[0]
    
        # Include output nodes in length calculations
        output1_len = p1[1].shape[0]
        output2_len = p2[1].shape[0]
    
        # Normalize indices for body nodes
        samples, samples_norm = normalize_indices(len1, len2)
        print(p1[1], p2[1])
        # Concatenate body nodes and output nodes
        combined_nodes = list(p1[0]) + list(p2[0]) + list(p1[1]) + list(p2[1])
        combined_sorted, out1, out2, d_distro = concatenate_and_sort(combined_nodes, samples, samples_norm, d_distro, len1)
        print(out1, out2)
        print('---')
        # Perform crossover on outputs
        stupid_mask = np.full(d_distro.shape, False).flatten()
        # Split back into separate genomes based on node count (including outputs)
        new_p1, new_p2 = split_nodes(combined_sorted, out1, out2, len1, len2, p1[0].shape, p2[0].shape)
        new_p1 = list(new_p1)
        new_p2 = list(new_p2)
        new_p1[0], new_p2[0] = trim_individual(new_p1[0], max_n), trim_individual(new_p2[0], max_n)
        new_p1[0], new_p2[0] = adjust_indices(new_p1[0], bank_len, first_body_node), adjust_indices(new_p2[0], bank_len, first_body_node)
        new_p1[0], new_p2[0] = prevent_recursion(new_p1[0], first_body_node), prevent_recursion(new_p2[0], first_body_node)
        new_p1[1], new_p2[1] = limit_values(new_p1[1], new_p1[0].shape, first_body_node), limit_values(new_p2[1], new_p2[0].shape, first_body_node)
     
    return [tuple(new_p1), tuple(new_p2), d_distro]


def xover_sgx(p1, p2, max_n, inputs=11, first_body_node=11, fixed_length=True, bank_len=4):
    c1, c1_distro = SubgraphCrossover(p1, p2, max_n, inputs)
    c2, c2_distro = SubgraphCrossover(p2, p1, max_n, inputs)
    d_distro = c1_distro + c2_distro
    return c1, c2, d_distro

def xover_real(p1, p2, func_total, first_body_node, outputs, arity):
    """
    p1: First Parent
    p2: Second Parent
    func_total: number of functions in the bank
    inputs: number of inputs
    bias: number of bias terminals
    outputs: number of output nodes
    arity: arity of the function
    """
    n_c = 2
    s = p1[0].shape
    out1, out2 = p1[1], p2[1]
    p1, p2 = np.concatenate((p1[0].flatten(), out1)), np.concatenate((p2[0].flatten(), out2))
    assert len(p1) == len(p2)

    def decode(genome, func_total, first_body_node, outputs=1, arity=2):
        node_length = 1 + arity  # Each node has 1 operator and arity inputs
        n_nodes = (len(genome) // node_length) + outputs + first_body_node - 1
        decoded_genome = np.zeros(len(genome))

        # Decode each node
        for i in range(0, len(genome) - outputs, node_length):
            node_term = (i // node_length) + outputs + first_body_node - 1
            decoded_genome[i:i + arity] = np.floor(genome[i:i + arity] * node_term)
            decoded_genome[i + arity] = np.floor(genome[i + arity] * func_total)

        # Decode outputs
        for i in range(outputs):
            decoded_genome[-(i + 1)] = np.floor(genome[-(i + 1)] * (n_nodes - i))

        return decoded_genome.astype(np.int32)

    def encode(decoded_genome, func_total, first_body_node, outputs=1, arity=2):
        node_length = 1 + arity  # Each node has 1 operator and arity inputs
        n_nodes = (len(decoded_genome) // node_length) + outputs + first_body_node - 1
        genome = np.zeros(len(decoded_genome))

        # Encode each node
        for i in range(0, len(decoded_genome) - outputs, node_length):
            node_term = (i // node_length) + outputs + first_body_node - 1
            genome[i:i + arity] = decoded_genome[i:i + arity] / node_term  # Reverse the floor(genei * node_term)
            genome[i + arity] = decoded_genome[i + arity] / func_total  # Reverse the floor(genei * func_total)

        # Encode outputs
        for i in range(outputs):
            genome[-(i + 1)] = decoded_genome[-(i + 1)] / (n_nodes - i)

        return np.round(genome, 2)

    children = []
    p1 = encode(p1, func_total, first_body_node, outputs, arity)
    p2 = encode(p2, func_total, first_body_node, outputs, arity)
    for i in range(n_c):
        r_i = random.uniform(0, 1)
        o_i = (1 - r_i) * p1 + r_i * p2
        children.append(o_i)
    children = [decode(child, func_total, first_body_node, outputs, arity) for child in children]
    ind1 = children[0]
    ind2 = children[1]
    out1 = ind1[-outputs:]
    out2 = ind2[-outputs:]
    return [(ind1[:-out1.shape[0]].reshape(s), out1), (ind2[:-out2.shape[0]].reshape(s), out2)]

def xover(parents, max_n, first_body_node, method='None', p_xov=0.5, fixed_length=True, bank_len=4,
          shape_override=None, n_inputs=1):
    children = []
    d_distro = np.zeros((len(parents), max_n * 3)) if shape_override is None else np.zeros(shape_override)
    methods = {
        'None': 'None',
        'OnePoint': xover_1x,
        'TwoPoint': xover_2x,
        'Uniform': xover_uniform,
        'Subgraph': xover_sgx,
        'Real': xover_real
    }

    xover_method = methods.get(method, 'None')
    retention = []

    for i in range(0, len(parents), 2):
        if method == 'None' or p_xov == 0 or random.random() > p_xov or len(parents[i][0]) < 2 or len(parents[i+1][0]) < 2:
            children.append(parents[i])
            children.append(parents[i + 1])
        elif method == 'Real':
            c1, c2 = xover_real(parents[i], parents[i+1], bank_len, first_body_node, 1, 2)
            children.append(c1)
            children.append(c2)
            retention.append(i)
        elif method in methods:
            if method == 'Uniform':
                c1, c2, d_distro[i, :] = xover_method(parents[i], parents[i + 1], max_n, first_body_node,
                                                        fixed_length=fixed_length,
                                                        bank_len=bank_len)
            elif method == 'Subgraph':
                c1, c2, d_distro[i:i + 1, :] = xover_method(parents[i], parents[i + 1], max_n, first_body_node=first_body_node,
                                                        fixed_length=fixed_length,
                                                        bank_len=bank_len, inputs=first_body_node)
            else:
                c1, c2, d_distro[i:i + 1, :] = xover_method(parents[i], parents[i + 1], max_n, first_body_node,
                                                        fixed_length=fixed_length,
                                                        bank_len=bank_len)
            children.append(c1)
            children.append(c2)
            retention.append(i)
    return children, np.array(retention, dtype=np.int32), d_distro


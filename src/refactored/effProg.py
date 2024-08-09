import numpy as np
from pathlib import Path
import graphviz as gv

def effProg(n_calc, prog_long, fb_node=12):
    """Calculate effective programs."""
    n_calc += fb_node
    prog = prog_long.copy().astype(np.int32)
    last = len(prog)
    eff_i = np.zeros(last, dtype=np.int32)
    eff_prog = []
    dep_set = [0]  # assume only register 0 is output

    for i in range(last - 1, -1, -1):
        if eff_i[i] == 1:
            continue
        if prog[i][0] in dep_set:
            eff_i[i] = 1
            dep_set.remove(prog[i][0])
            for k in [2, 3]:
                if prog[i][k] < n_calc + 1 and prog[i][k] not in dep_set:
                    dep_set.append(prog[i][k])
            dep_set = list(set(dep_set))  # remove duplicates
    eff_prog = [prog[i] for i in range(last) if eff_i[i] == 1]
    return np.array(eff_prog, dtype=np.int32)

def save_to_file(file_path, content, mode='a'):
    """Helper function to write content to a file."""
    with open(file_path, mode) as f:
        f.write(content)

def lgp_print_individual(ind, a, b, run_name, func_name, bank_string, t, bias, n_inp=1, fb_node=12):
    """Print and save the individual program representation."""
    Path(f"../output/{run_name}/{func_name}/best_program/").mkdir(parents=True, exist_ok=True)
    
    def put_r(reg):
        if reg == 0:
            return 'R_0'
        if 0 < reg <= n_inp:
            return f'R_{reg}'
        return f'R_{reg - fb_node + 2}'

    ind = np.array(ind)
    registers = list(map(put_r, ind[:, 0].astype(np.int32)))
    operators = [bank_string[i] for i in ind[:, 1].astype(np.int32)]
    operands = ind[:, 2:].astype(np.int32)
    
    content = 'R\tOp\tI\n'
    for reg, op, ops in zip(registers, operators, operands):
        nums = [f'R_{n}' if 0 < n <= n_inp else f'R_{n - fb_node + 2}' for n in ops]
        content += f"{reg}\t{op}\t{nums}\n"
    
    content += f"Scaling\nR_0 = {a}*R_0+{b}\n\n{ind}"
    
    save_to_file(f"../output/{run_name}/{func_name}/best_program/best_{t}.txt", content)
    print(content)

def draw_graph(ind, i_a, i_b, fb_node=12, n_inp=1, max_d=4, n_bias=10, bias=np.arange(0, 10, 1).astype(np.int32), bank_string=("+", "-", "*", "/")):
    """Draw and return a graph of the program."""
    dot = gv.Digraph()
    ind = ind.astype(np.int32)
    
    # Add nodes
    for i in range(1):  # outputs
        dot.node(f'O_{i}', f'R_{i}', shape='square', rank='same', fillcolor='lightblue', style='filled')
    for i in range(1, n_inp + 1):  # inputs
        dot.node(f'I_{i}', f'R_{i}', shape='square', rank='same', fillcolor='orange', style='filled')
    for b in range(n_bias):  # biases
        dot.node(f'B_{b + n_inp + 1}', f"{bias[b]}", shape='square', rank='same', fillcolor='yellow', style='filled')
    dot.attr(rank='same')
    for i in range(1 + n_inp, max_d + 1 + n_inp):  # calculation registers
        dot.node(f'D_{i}', f'R_{i}', shape='square', rank='same', fillcolor='darkseagreen1', style='filled')
    for o, op in enumerate(bank_string):
        dot.node(f'Op_{o}', op)
    
    # Add edges
    for i, (reg, op, ops) in enumerate(ind):
        op_node = f'Op_{int(op)}'
        target = f'O_0' if reg == 0 else f'D_{reg - fb_node + 2}'
        dot.edge(op_node, target, label=f'i.{i+1}')
        for n in ops:
            source = f'O_0' if n == 0 else f'I_{n}' if 0 < n <= n_inp else f'B_{n}' if n < fb_node else f'D_{n - fb_node + 2}'
            dot.edge(source, op_node, label=f'(i.{i+1}|a.{list(ops).index(n)+1})')
    
    # Add scaling nodes
    dot.attr(rank='max')
    dot.node(f'A', f'*{np.round(i_a, 5)}', shape='diamond', fillcolor='green', style='filled')
    dot.node(f'B', f'+({np.round(i_b, 5)})', shape='diamond', fillcolor='green', style='filled')
    dot.edge(f'A', f'B')
    dot.edge(f'O_0', f'A')
    
    return dot

def get_thickness(ind, fb_node=12, max_d=4, operators=4):
    """Compute thickness matrices."""
    length = fb_node + max_d
    to_operator = np.zeros((operators, length), dtype=int)
    from_operator = np.zeros((operators, length), dtype=int)
    
    for reg, op, *operands in ind:
        from_operator[int(op), reg] += 1
        for op in operands:
            to_operator[int(op), reg] += 1
    
    return from_operator, to_operator

def draw_graph_thickness(ind, i_a, i_b, fb_node=12, n_inp=1, max_d=4, n_bias=10, bias=np.arange(0, 10, 1).astype(np.int32), bank_string=("+", "-", "*", "/")):
    """Draw and return a graph with edge thickness representing connections."""
    f_mat, t_mat = get_thickness(ind)
    dot = gv.Digraph()
    ind = ind.astype(np.int32)
    
    # Add nodes
    for i in range(1):  # outputs
        dot.node(f'N_{i}', f'R_{i}', shape='square', rank='same', fillcolor='lightblue', style='filled')
    for i in range(1, n_inp + 1):  # inputs
        dot.node(f'N_{i}', f'R_{i}', shape='square', rank='same', fillcolor='orange', style='filled')
    for b in range(n_bias):  # biases
        dot.node(f'N_{b + n_inp + 1}', f"{bias[b]}", shape='square', rank='same', fillcolor='yellow', style='filled')
    dot.attr(rank='same')
    for i in range(max_d):  # calculation registers
        dot.node(f'N_{fb_node + i}', f'R_{i + n_inp + 1}', shape='square', rank='same', fillcolor='darkseagreen1', style='filled')
    for o, op in enumerate(bank_string):
        dot.node(f'Op_{o}', op)
    
    # Add edges with thickness
    for i, op_node in enumerate(f_mat):
        for j, thickness in enumerate(op_node):
            if thickness > 0:
                dot.edge(f'Op_{i}', f'N_{j}', penwidth=str(thickness))
    for i, op_node in enumerate(t_mat):
        for j, thickness in enumerate(op_node):
            if thickness > 0:
                dot.edge(f'N_{j}', f'Op_{i}', penwidth=str(thickness))
    
    # Add scaling nodes
    dot.attr(rank='max')
    dot.node(f'A', f'*{np.round(i_a, 5)}', shape='diamond', fillcolor='green', style='filled')
    dot.node(f'B', f'+({np.round(i_b, 5)})', shape='diamond', fillcolor='green', style='filled')
    dot.edge(f'A', f'B')
    dot.edge(f'N_0', f'A')
    
    return dot

def percent_change(new, old):
    """Calculate percent change between new and old values."""
    return new - old if np.isfinite(new) and np.isfinite(old) else np.nan

def cgp_active_nodes(ind_base, output_nodes, outputs=1, first_body_node=11, opt=0):
    """Determine active nodes in a CGP program."""
    active_nodes = set()

    def plot_body_node(n_node):
        node = ind_base[n_node - first_body_node]
        for prev_node in node[:-1]:
            if prev_node not in active_nodes:
                active_nodes.add(prev_node)
                if prev_node >= 0:
                    plot_body_node(prev_node)
                    
    if opt == 0:
        for n in output_nodes:
            if n >= 0:
                plot_body_node(n)
    return list(active_nodes)


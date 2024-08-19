from pathlib import Path

import numpy as np


def effProg(n_calc, prog_long, fb_node):  #Wolfgang gave this to me
    n_calc = n_calc + fb_node
    prog = prog_long.copy().astype(np.int32)
    prog = np.atleast_2d(prog)
    last = len(prog)
    eff_i = np.zeros((last,)).astype(np.int32)
    eff_prog = []
    dep_set = [0]  #assume only register 0 is output
    for i in range(last - 1, -1, -1):
        ind = i  #+1
        #print(f'ind {ind}')
        for j in range(0, len(dep_set)):
            #print(f'eff_i {eff_i}')
            if eff_i[ind] == 1:
                break
            #print(f'prog[ind][0] {prog[ind][0]} | dep_set[j] {dep_set[j]}')
            try:
                if prog[ind][0] == dep_set[j]:
                    eff_i[ind] = 1  #Mark instruction as effective
                    #x = 0
                    a = dep_set.copy()
                    #print(a)
                    #print(dep_set[j])
                    #print(list(filter(lambda x: x != dep_set[j], a)))
                    a = [x for x in a if x != dep_set[j]]
                    dep_set = a.copy()
                    for k in [2, 3]:
                        #print(f'prog[ind][{k}] {prog[ind][k]}')
                        if prog[ind][k] < n_calc + 1:  #add 1st source register if calculation
                            #print(f'{prog[ind][k]} {prog[ind][k] not in dep_set}')
                            if prog[ind][k] not in dep_set:
                                dep_set.append(prog[ind][k])
                    new = list(set(dep_set))
                    dep_set = new.copy()  #make sure no repetitions in depset
                    if len(dep_set) < 1:
                        continue
                if len(dep_set) < 1:
                    break
            except IndexError:
                if int(0) not in prog[:, 0]:
                    return np.array([])
                else:
                    raise IndexError
    atleastone = 0
    for i in range(0, last):
        if eff_i[i] == 1:
            eff_prog.append(prog[i])
    return np.array(eff_prog).astype(np.int32)


import graphviz as gv


def draw_graph(ind, i_a, i_b, fb_node=12, n_inp=1, max_d=4, n_bias=10, bias=np.arange(0, 10, 1).astype(np.int32),
               bank_string=("+", "-", "*", "/")):
    dot = gv.Digraph()
    #print(a)
    #print(b)
    ind = ind.astype(np.int32)
    used_nodes = []  #unused nodes will be removed
    for i in range(1):  #outputs
        dot.node(f'O_{i}', f'R_{i}', shape='square', rank='same', fillcolor='lightblue', style='filled')
    #	dot.edge("l_0", f"N_{i}", style='invisible')
    for i in range(1, n_inp + 1):  #inputs
        dot.node(f'I_{i}', f'R_{i}', shape='square', rank='same', fillcolor='orange', style='filled')
    for b in range(n_bias):  #biases
        dot.node(f'B_{b + n_inp + 1}', f"{bias[b]}", shape='square', rank='same', fillcolor='yellow', style='filled')
    #	dot.edge("l_0", f"N_{b}", style='invisible')
    dot.attr(rank='same')
    j = 1 + n_inp
    for i in range(j, max_d + j):  #calculation registers
        dot.node(f'D_{i}', f'R_{i}', shape='square', rank='same', fillcolor='darkseagreen1', style='filled')
    #operators
    for o in range(len(bank_string)):
        dot.node(f'Op_{o}', f'{bank_string[o]}')
    #instructions
    for i in range(ind.shape[0]):
        i_num = i + 1
        instruction = ind[i, :]
        #print(instruction)
        reg = instruction[0]
        op = int(instruction[1])
        ops = instruction[2:].astype(np.int32)
        #dot.node(f'N_{i}', f'{bank_string[op]}')
        op_node = f'Op_{op}'
        if reg == 0:
            dot.edge(op_node, f'O_0', f'i.{i_num}')
        else:
            used_nodes.append(f'D_{reg - fb_node + 2}')
            dot.edge(op_node, f'D_{reg - fb_node + 2}', f'i.{i_num}')
        for n_i in range(len(ops)):
            n = ops[n_i]
            n_num = n_i + 1
            if n == 0:
                dot.edge(f'O_{0}', op_node, f'(i.{i_num}|a.{n_num})')
            elif n > 0 and n <= n_inp:
                used_nodes.append(f'I_{n}')
                dot.edge(f'I_{n}', op_node, f'(i.{i_num}|a.{n_num})')
            elif n > n_inp and n < fb_node:
                used_nodes.append(f'B_{n}')
                dot.edge(f'B_{n}', op_node, f'(i.{i_num}|a.{n_num})')
            else:
                used_nodes.append(f'D_{n - fb_node + 2}')
                dot.edge(f'D_{n - fb_node + 2}', op_node, f'(i.{i_num}|a.{n_num})')
    dot.attr(rank='max')
    print(used_nodes)
    dot.node(f'A', f'*{np.round(i_a, 5)}', shape='diamond', fillcolor='green', style='filled')
    dot.node(f'B', f'+({np.round(i_b, 5)})', shape='diamond', fillcolor='green', style='filled')
    dot.edge(f'A', f'B')
    dot.edge(f'O_0', f'A')
    return dot


def get_thiccness(ind, fb_node=12, max_d=4, operators=4):
    length = fb_node + max_d
    to_operator = np.zeros((operators, length))  # rows are to, columns are from
    from_operator = np.zeros((operators, length))
    for i in range(ind.shape[0]):  #each instruction
        de = ind[i, 0]
        operator = ind[i, 1]
        from_operator[operator, de] += 1  #goes from the operator to the destination
        #print(list(range(2, ind.shape[1])))
        for a in range(2, ind.shape[1]):  # each operand
            op = ind[i, a]
            #print(op)
            to_operator[operator, op] += 1  #goes from operand to operator
    print(from_operator)
    print(to_operator)
    return from_operator, to_operator


def draw_graph_thicc(ind, i_a, i_b, fb_node=12, n_inp=1, max_d=4, n_bias=10, bias=np.arange(0, 10, 1).astype(np.int32),
                     bank_string=("+", "-", "*", "/")):
    f_mat, t_mat = get_thiccness(ind)
    dot = gv.Digraph()
    ind = ind.astype(np.int32)
    used_nodes = []  #unused nodes will be removed
    for i in range(1):  #outputs
        #print(i)
        dot.node(f'N_{i}', f'R_{i}', shape='square', rank='same', fillcolor='lightblue', style='filled')
    #	dot.edge("l_0", f"N_{i}", style='invisible')
    for i in range(1, n_inp + 1):  #inputs
        print(i)
        dot.node(f'N_{i}', f'R_{i}', shape='square', rank='same', fillcolor='orange', style='filled')
    for b in range(0, 10):  #biases
        #print(b+n_inp+1)
        dot.node(f'N_{b + n_inp + 1}', f"{bias[b]}", shape='square', rank='same', fillcolor='yellow', style='filled')
    #	dot.edge("l_0", f"N_{b}", style='invisible')
    dot.attr(rank='same')
    j = 1 + n_inp
    for i in range(max_d):  #calculation registers
        #print(fb_node+i)
        dot.node(f'N_{fb_node + i}', f'R_{j + i}', shape='square', rank='same', fillcolor='darkseagreen1',
                 style='filled')
    #operators
    for o in range(len(bank_string)):
        dot.node(f'Op_{o}', f'{bank_string[o]}')
    #instructions
    for i in range(f_mat.shape[0]):  #from operator to destination
        op_node = f'Op_{i}'
        for j in range(f_mat.shape[1]):
            t = f_mat[i, j]
            if t > 0:
                dot.edge(f'{op_node}', f'N_{j}', penwidth=f'{t}')  #, penwidth=t
    for i in range(t_mat.shape[0]):  #from operator to destination
        op_node = f'Op_{i}'
        for j in range(t_mat.shape[1]):
            t = t_mat[i, j]
            if t > 0:
                dot.edge(f'N_{j}', f'{op_node}', penwidth=f'{t}')  #, penwidth=t)

    dot.attr(rank='max')
    print(used_nodes)
    dot.node(f'A', f'*{np.round(i_a, 5)}', shape='diamond', fillcolor='green', style='filled')
    dot.node(f'B', f'+({np.round(i_b, 5)})', shape='diamond', fillcolor='green', style='filled')
    dot.edge(f'A', f'B')
    dot.edge(f'N_0', f'A')
    return dot


def percent_change(new, old):
    if np.isfinite(new) and np.isfinite(old):
        return new - old
    else:
        return np.nan


def cgp_active_nodes(ind_base, output_nodes, outputs=1, first_body_node=11, opt=0):
    active_nodes = []

    def plot_body_node(n_node, arity=2):
        node = ind_base[n_node - first_body_node]
        for a in range(arity):
            prev_node = node[a]
            if prev_node not in active_nodes:
                active_nodes.append(prev_node)  #count active nodes
            if prev_node >= first_body_node:  #inputs
                plot_body_node(prev_node)

    for o in range(outputs):
        node = output_nodes[o]
        if node not in active_nodes:
            active_nodes.append(node)
        if node >= first_body_node:  #bias:
            plot_body_node(node)
    active_node_num = len(active_nodes) + outputs  #all outputs are active by definition
    if opt == 0:
        return active_node_num
    elif opt == 1:
        return active_nodes
    elif opt == 2:
        return active_node_num / (ind_base.shape[0] + outputs + first_body_node)
    elif opt == 3:  #return indices, not node number
        active_nodes = np.array(active_nodes)
        return (active_nodes[active_nodes >= first_body_node] - first_body_node)


def cgp_graph(inputs, bias, ind_base, first_body_node, output_nodes, p_A, p_B, func_name, run_name, t, max_n=64,
              arity=2, biases=list(range(0, 10)), bank_string=['+', '-', '*', '/']):
    max_n = ind_base.shape[0]
    dot = gv.Digraph()
    for i in range(inputs):
        dot.node(f'N_{i}', f'I_{i}', shape='square', rank='same', fillcolor='orange', style='filled')
    #       dot.edge("l_0", f"N_{i}", style='invisible')
    for b in range(bias):
        dot.node(f'N_{b + inputs}', f"{biases[b]}", shape='square', rank='same', fillcolor='yellow', style='filled')
    #       dot.edge("l_0", f"N_{b}", style='invisible')
    dot.attr(rank='same')
    for n in range(first_body_node, max_n + first_body_node):

        node = ind_base[n - first_body_node]
        op = bank_string[node[-1]]
        dot.node(f'N_{n}', op)
        for a in range(arity):
            dot.edge(f'N_{node[a]}', f'N_{n}')

    dot.attr(rank='max')
    dot.node(f'A', f'*{p_A}', shape='diamond', fillcolor='green', style='filled')
    dot.node(f'B', f'+{p_B}', shape='diamond', fillcolor='green', style='filled')
    dot.edge(f'A', f'B')
    outputs = len(output_nodes)
    for o in range(outputs):
        node = output_nodes[o]
        dot.attr(rank='max')
        dot.node(f'O_{o}', f'O_{o}', shape='square', fillcolor='lightblue', style='filled')
        dot.edge(f'N_{node}', f'O_{o}')
        dot.edge(f'O_{o}', 'A')

    Path(f"../output/{run_name}/{func_name}/full_graphs/").mkdir(parents=True, exist_ok=True)
    dot.render(f"../output/{run_name}/{func_name}/full_graphs/graph_{t}", view=False)


def plot_active_nodes(ind_base, output_nodes, first_body_node, bank_string, biases, inputs, p_A, p_B, func_name,
                      run_name, t, arity=2, opt=0):
    outputs = len(output_nodes)
    active_graph = gv.Digraph(strict=True)
    active_nodes = []
    size = 0

    def plot_body_node(n_node):
        node = ind_base[n_node - first_body_node]
        op = bank_string[node[-1]]
        active_graph.node(f'N_{n_node}', op)
        for a in range(arity):
            prev_node = node[a]
            if prev_node not in active_nodes:
                active_nodes.append(prev_node)  #count active nodes
            if prev_node < inputs:  #inputs
                active_graph.node(f'N_{prev_node}', f'I_{prev_node}', shape='square', rank='same', fillcolor='orange',
                                  style='filled')
                active_graph.edge(f'N_{prev_node}', f'N_{n_node}')
            elif prev_node >= inputs and prev_node < first_body_node:  #bias:
                active_graph.node(f'N_{prev_node}', f"{biases[prev_node - inputs]}", shape='square', rank='same',
                                  fillcolor='yellow', style='filled')
                active_graph.edge(f'N_{prev_node}', f'N_{n_node}')
            else:
                plot_body_node(prev_node)
                active_graph.edge(f'N_{prev_node}', f'N_{n_node}')

    active_graph.node(f'A', f'*{np.round(p_A, 5)}', shape='diamond', fillcolor='green', style='filled')
    active_graph.node(f'B', f'+{np.round(p_B, 5)}', shape='diamond', fillcolor='green', style='filled')
    active_graph.edge(f'A', f'B')
    for o in range(outputs):
        node = output_nodes[o]
        active_graph.node(f'O_{o}', f'O_{o}', shape='square', fillcolor='lightblue', style='filled')
        if node < inputs:  #inputs
            active_graph.node(f'N_{node}', f'I_{node}', shape='square', rank='same', fillcolor='orange', style='filled')
            active_graph.edge(f'N_{node}', f'O_{o}')
            if node not in active_nodes:
                active_nodes.append(node)
        elif node >= inputs and node < first_body_node:  #bias:
            active_graph.node(f'N_{node}', f"{biases[node - inputs]}", shape='square', rank='same', fillcolor='yellow',
                              style='filled')
            active_graph.edge(f'N_{node}', f'O_{o}')
            if node not in active_nodes:
                active_nodes.append(node)
        else:
            plot_body_node(node)
            active_graph.edge(f'N_{node}', f'O_{o}')
            if node not in active_nodes:
                active_nodes.append(node)
        active_graph.edge(f'O_{o}', 'A')
    if opt == 0:
        Path(f"../output/{run_name}/{func_name}/active_nodes/").mkdir(parents=True, exist_ok=True)
        active_graph.render(f"../output/{run_name}/{func_name}/active_nodes/active_{t}", view=False)
    active_node_num = len(active_nodes) + outputs  #all outputs are active by definition
    return active_node_num


def lgp_print_individual(ind, a, b, fb_node, run_name, func_name, bank_string, t, bias, n_inp=1):
    Path(f"../output/{run_name}/{func_name}/best_program/").mkdir(parents=True, exist_ok=True)
    ind = np.atleast_2d(ind)

    def put_r(reg, fb_node=fb_node, n_inp=1):
        if reg == 0:
            return f'R_0'
        if reg > 0 and reg <= n_inp:
            return f'R_{reg}'
        return f'R_{int(reg - fb_node + 2)}'

    try:
        registers = ind[:, 0].astype(np.int32)
        operators = ind[:, 1].astype(np.int32)
        operands = ind[:, 2:].astype(np.int32)
    except IndexError:
        print(f'Individual:\n{ind}')
        try:
            registers = ind[0]
            operators = ind[1]
            operands = ind[2:]
        except IndexError:
            print('No Effective Instructions!')
            return

    registers = list(map(put_r, registers))
    operators = [bank_string[i] for i in operators]
    with open(f"../output/{run_name}/{func_name}/best_program/best_{t}.txt", 'a') as f:
        f.write('R\tOp\tI\n')
    print("R\tOp\tI")
    for i in range(len(registers)):
        reg = registers[i]
        op = operators[i]
        ops = operands[i, :]
        nums = []
        for n in ops:
            #print(n)
            if n > 0 and n <= n_inp:
                nums.append(f'R_{n}')
            elif n == 0:
                nums.append(f'R_0')
            elif n > n_inp and n < fb_node:
                nums.append(bias[n - n_inp - 1])
            else:
                nums.append(f'R_{n - fb_node + 2}')
        with open(f"../output/{run_name}/{func_name}/best_program/best_{t}.txt", 'a') as f:
            f.write(f"{reg}\t{op}\t{nums}\n")
        print(f"{reg}\t{op}\t{nums}")
    print("Scaling")
    print(f"R_0 = {a}*R_0+{b}")
    with open(f"../output/{run_name}/{func_name}/best_program/best_{t}.txt", 'a') as f:
        f.write(f"R_0 = {a}*R_0+{b}\n\n")
        f.write(f'{ind}')

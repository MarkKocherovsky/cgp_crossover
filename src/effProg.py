import numpy as np

def effProg(n_calc, prog_long, fb_node = 12): #Wolfgang gave this to me
	n_calc = n_calc + fb_node
	prog = prog_long.copy().astype(np.int32)
	last = len(prog)
	eff_i = np.zeros((last,)).astype(np.int32)
	eff_prog = []
	dep_set = [0] #assume only register 0 is output
	for i in range(last-1, -1, -1):
		ind = i #+1
		#print(f'ind {ind}')
		for j in range(0, len(dep_set)):
			#print(f'eff_i {eff_i}')
			if eff_i[ind] == 1:
				break
			#print(f'prog[ind][0] {prog[ind][0]} | dep_set[j] {dep_set[j]}')
			if prog[ind][0] ==dep_set[j]:
				eff_i[ind]=1 #Mark instruction as effective
				#x = 0
				a = dep_set.copy()
				#print(a)
				#print(dep_set[j])
				#print(list(filter(lambda x: x != dep_set[j], a)))
				a = [x for x in a if x != dep_set[j]]
				dep_set = a.copy()
				for k in [2, 3]:
					#print(f'prog[ind][{k}] {prog[ind][k]}')
					if prog[ind][k] < n_calc + 1: #add 1st source register if calculation
						#print(f'{prog[ind][k]} {prog[ind][k] not in dep_set}')
						if prog[ind][k] not in dep_set:
							dep_set.append(prog[ind][k])
				new = list(set(dep_set))
				dep_set = new.copy() #make sure no repetitions in depset
				if len(dep_set) < 1:
					continue
			if len(dep_set) < 1:
				break
	atleastone = 0
	for i in range(0, last):
		if eff_i[i] == 1:
			eff_prog.append(prog[i])
	return np.array(eff_prog).astype(np.int32)

import graphviz as gv
def draw_graph(ind, i_a, i_b, fb_node = 12, n_inp = 1, max_d = 4, n_bias = 10, bias = np.arange(0, 10, 1).astype(np.int32), bank_string = ("+", "-", "*", "/")):
	dot = gv.Digraph()
	#print(a)
	#print(b)
	ind = ind.astype(np.int32)
	used_nodes = [] #unused nodes will be removed
	for i in range(1): #outputs
		dot.node(f'O_{i}', f'R_{i}', shape='square', rank='same', fillcolor = 'lightblue', style='filled')
	#	dot.edge("l_0", f"N_{i}", style='invisible')
	for i in range(1, n_inp+1): #inputs
		dot.node(f'I_{i}', f'R_{i}', shape='square', rank='same', fillcolor='orange', style='filled')
	for b in range(n_bias): #biases
		dot.node(f'B_{b+n_inp+1}', f"{bias[b]}", shape='square', rank='same', fillcolor='yellow', style='filled')
	#	dot.edge("l_0", f"N_{b}", style='invisible')
	dot.attr(rank='same')
	j = 1+n_inp
	for i in range(j, max_d+j): #calculation registers
		dot.node(f'D_{i}', f'R_{i}', shape='square', rank='same', fillcolor='darkseagreen1', style='filled')
	#operators
	for o in range(len(bank_string)):
		dot.node(f'Op_{o}', f'{bank_string[o]}')
	#instructions
	for i in range(ind.shape[0]):
		i_num = i+1
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
			used_nodes.append(f'D_{reg-fb_node+2}')
			dot.edge(op_node, f'D_{reg-fb_node+2}', f'i.{i_num}')
		for n_i in range(len(ops)):
			n = ops[n_i]
			n_num = n_i+1
			if n == 0:
				dot.edge(f'O_{0}', op_node, f'(i.{i_num}|a.{n_num})')
			elif n > 0 and n <= n_inp:
				used_nodes.append(f'I_{n}')
				dot.edge(f'I_{n}', op_node, f'(i.{i_num}|a.{n_num})')
			elif n > n_inp and n < fb_node:
				used_nodes.append(f'B_{n}')
				dot.edge(f'B_{n}', op_node, f'(i.{i_num}|a.{n_num})')
			else:
				used_nodes.append(f'D_{n-fb_node+2}')
				dot.edge(f'D_{n-fb_node+2}', op_node, f'(i.{i_num}|a.{n_num})')
	dot.attr(rank = 'max')
	print(used_nodes)
	dot.node(f'A', f'*{np.round(i_a, 5)}', shape = 'diamond', fillcolor='green', style='filled')
	dot.node(f'B', f'+({np.round(i_b, 5)})', shape = 'diamond', fillcolor = 'green', style = 'filled')
	dot.edge(f'A', f'B')
	dot.edge(f'O_0', f'A')
	return dot

def get_thiccness(ind, fb_node = 12, max_d = 4, operators = 4):
	length = fb_node+max_d
	to_operator = np.zeros((operators, length)) # rows are to, columns are from
	from_operator = np.zeros((operators, length))
	for i in range(ind.shape[0]): #each instruction
		de = ind[i, 0]
		operator = ind[i, 1]
		from_operator[operator, de] += 1 #goes from the operator to the destination
		#print(list(range(2, ind.shape[1])))
		for a in range(2, ind.shape[1]): # each operand
			op = ind[i, a]
			#print(op)
			to_operator[operator, op] += 1 #goes from operand to operator
	print(from_operator)
	print(to_operator)
	return from_operator, to_operator

def draw_graph_thicc(ind, i_a, i_b, fb_node = 12, n_inp = 1, max_d = 4, n_bias = 10, bias = np.arange(0, 10, 1).astype(np.int32), bank_string = ("+", "-", "*", "/")):
	f_mat, t_mat = get_thiccness(ind)
	dot = gv.Digraph()
	ind = ind.astype(np.int32)
	used_nodes = [] #unused nodes will be removed
	for i in range(1): #outputs
		#print(i)
		dot.node(f'N_{i}', f'R_{i}', shape='square', rank='same', fillcolor = 'lightblue', style='filled')
	#	dot.edge("l_0", f"N_{i}", style='invisible')
	for i in range(1, n_inp+1): #inputs
		print(i)
		dot.node(f'N_{i}', f'R_{i}', shape='square', rank='same', fillcolor='orange', style='filled')
	for b in range(0, 10): #biases
		#print(b+n_inp+1)
		dot.node(f'N_{b+n_inp+1}', f"{bias[b]}", shape='square', rank='same', fillcolor='yellow', style='filled')
	#	dot.edge("l_0", f"N_{b}", style='invisible')
	dot.attr(rank='same')
	j = 1+n_inp
	for i in range(max_d): #calculation registers
		#print(fb_node+i)
		dot.node(f'N_{fb_node+i}', f'R_{j+i}', shape='square', rank='same', fillcolor='darkseagreen1', style='filled')
	#operators
	for o in range(len(bank_string)):
		dot.node(f'Op_{o}', f'{bank_string[o]}')
	#instructions
	for i in range(f_mat.shape[0]): #from operator to destination
		op_node = f'Op_{i}'
		for j in range(f_mat.shape[1]):
			t = f_mat[i,j]
			if t > 0:
				dot.edge(f'{op_node}', f'N_{j}', penwidth=f'{t}')#, penwidth=t
	for i in range(t_mat.shape[0]): #from operator to destination
		op_node = f'Op_{i}'
		for j in range(t_mat.shape[1]):
			t = t_mat[i,j]
			if t > 0:
				dot.edge(f'N_{j}',f'{op_node}', penwidth=f'{t}')#, penwidth=t)

	dot.attr(rank = 'max')
	print(used_nodes)
	dot.node(f'A', f'*{np.round(i_a, 5)}', shape = 'diamond', fillcolor='green', style='filled')
	dot.node(f'B', f'+({np.round(i_b, 5)})', shape = 'diamond', fillcolor = 'green', style = 'filled')
	dot.edge(f'A', f'B')
	dot.edge(f'N_0', f'A')
	return dot

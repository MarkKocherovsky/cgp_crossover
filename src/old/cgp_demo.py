#CGP demonstrate destruction
import numpy as np
import matplotlib.pyplot as plt
import warnings
from numpy import random, sin, cos, tan, sqrt, exp, log, abs, floor, ceil
from math import log, pi
from sys import path
from pathlib import Path
from functions import *
from sys import argv
warnings.filterwarnings('ignore')
print("started")
def add(x,y):
	return x+y
def sub(x,y):
	return x-y
def mul(x,y):
	return x*y
def div(x,y):
	if y == 0.0:
		return np.PINF
	return x/y
#test_x = np.arange(11, 30.1, 1)
#test_y = [func([y]) for y in test_x]
#print(train_x)
#print(train_y)
#print(powe(0)
t = 0 #trial
print(f'trial {t}')
max_g = 0  #max generations
print(f'generations {max_g}')
max_n = 64 #max body nodes
print(f'max body nodes {max_n}')
max_p = 4 #max parents
print(f'Parents {max_p}')
max_c = 4 #max children
print(f'children {max_c}')
outputs = 1
inputs = 1
biases = np.arange(0, 10, 1).astype(np.int32)
bias = biases.shape[0] #number of biases
print(f'biases {biases}')
arity = 2

#bank = (add, sub, mul, div, x, y, cos_x, cos_y, sin_x, sin_y, powe, sqrt_x_y, distance, abs_x, abs_y, floor_x, floor_y, ceil_x, ceil_y, max_f, min_f, midpoint)
#bank_string = ("+", "-", "*", "/", "x", "y", "cos(x)","cos(y)", "sin(x)", "sin(y)", "^", "$\sqrt{x+y}$", "$sqrt{x^2+y^2}$", "|x|", "|y|", "$\lfloor{x}\rfloor$", "$\lfloor{y}\rfloor$", "$\lceil{x}\rceil$", "$\lceil{y}\rceil$", "max", "min", "avg")

bank = (add, sub, mul, div) #, cos_x, cos_y, sin_x, sin_y, powe, sqrt_x_y, distance, abs_x, abs_y, midpoint)
bank_string = ("+", "-", "*", "/") #, "cos(x)","cos(y)", "sin(x)", "sin(y)", "^", "$\sqrt{x+y}$", "$sqrt{x^2+y^2}$", "|x|", "|y|", "avg")

func_bank = Collection()
func = func_bank.func_list[2]
func_name = func_bank.name_list[2]
train_x = func.x_dom
train_y = func.y_test
print(train_x)
from scipy.stats import pearsonr
def rmse(preds, reals):
	return np.sqrt(np.mean((preds-reals)**2)) #copied from stack overflow

def corr(preds, reals, x=train_x):
	if any(np.isnan(preds)) or any(np.isinf(preds)):
		return np.PINF
	r = pearsonr(preds, reals)[0]
	if np.isnan(r):
		r = 0
	return (1-r**2)
fit_bank = [rmse, corr]
fit_names = ["RMSE", "1-R^2"]
f = 1
fit = fit_bank[f]
fit_name  = fit_names[f]
print(fit_name)
def align(ind, out, preds, reals, x = train_x):
	if not all(np.isfinite(preds)):
		return 1.0, 0.0
	try:
		align = np.round(np.polyfit(preds, reals, 1, rcond=1e-16), decimals = 14)
	except:
		return 1.0, 0.0
	a = align[0]
	b = align[1]
	#print(f'align {align}')
	return (a,b)
def run(ind, cur_node, inp_nodes, arity = arity):
	inp_size = inp_nodes.shape[0]
	#print(f"inp_size: {inp_size}")
	args = []
	for j in range(arity):
		if cur_node[j] < inp_size:
			#print(cur_node[j])
			args.append(inp_nodes[cur_node[j]])
		else:
			args.append(run(ind, ind[cur_node[j]-inp_nodes.shape[0]], inp_nodes))
	function = bank[cur_node[-1]]
	#print(args[0], args[1], function(args[0], args[1]))
	return function(args[0], args[1]) # so far 2d only
			

def run_output(ind, out_nodes, inp_nodes, arity = arity):
	inp_nodes = np.array(inp_nodes)
	#print(inp_nodes)
	outs = np.zeros(out_nodes.shape,)
	#print(f'inputs + biases = {inputs+bias}')
	#print(inp_nodes)
	#print(inp_nodes.shape[0])
	for i in np.arange(0, outs.shape[0], 1, np.int32):
		#print(ind[out_nodes[i]-inp_nodes.shape[0]])
		if out_nodes[i] < (inputs+bias):
			#print(f'out_node = {out_nodes[i]} | input = {inp_nodes[out_nodes[i]]}')
			outs[i] = inp_nodes[out_nodes[i]]
		else:
			#print(f'out_node = {out_nodes[i]} | input = {ind[out_nodes[i]-inp_nodes.shape[0]]}')
			outs[i] = run(ind, ind[out_nodes[i]-inp_nodes.shape[0]], inp_nodes)
	#print(outs)
	return outs

def fitness(data, targ, ind_base, output_nodes, opt = 0):
	data = np.array(data)
	out_x = np.zeros(data.shape[0])
	for x in range(data.shape[0]):
		if len(data.shape) <= 1:
			in_val = [data[x]]
		else:
			in_val = data[x, :]
		#out_x[x] = run_output(ind_base, output_nodes, in_val)
		with np.errstate(invalid='raise'):
			try:
				out_x[x] = run_output(ind_base, output_nodes, in_val)
			except (OverflowError, FloatingPointError):
				out_x[x] = np.nan
	with np.errstate(invalid='raise'):
		try:
			(a,b) = align(ind_base, output_nodes, out_x, train_y)
		except (OverflowError, FloatingPointError):
			return np.nan, 1.0, 0.0
	#print('A, B')
	#print(a)
	#print(b)
	new_x = out_x*a+b
	if opt == 1:
		return new_x, a, b
	return fit(new_x, train_y), a, b

def xover(parents):
	children = []
	for i in range(0, len(parents), 2):
		ind1, ind2, out1, out2 = xover_aux(parents[i][0], parents[i+1][0], parents[i][1], parents[i+1][1])
		children.append((ind1, out1))
		children.append((ind2, out2))
	return children

def xover_aux(ind1, ind2, out1, out2):
	s = ind1.shape #preserve real shape
	ind1 = ind1.flatten()
	ind2 = ind2.flatten()
	i = random.randint(0, ind1.shape[0])
	front_1 = ind1[:i].copy()
	back_1 = ind1[i:].copy()
	front_2 = ind2[:i].copy()
	back_2 = ind2[i:].copy()
	ind1 = np.concatenate((front_1, back_2))
	ind2 = np.concatenate((front_2, back_1))
	
	if len(out1.shape) <=1 and out1.shape[0] == 1:
		t = out1.copy()
		out1 = out2.copy()
		out2 = t
	else:
		i = random.randint(0, out2.shape[0])
		front_1 = out1[:i].copy()
		back_1 = out1[i:].copy()
		front_2 = out2[:i].copy()
		back_2 = out2[i:].copy()
		out1 = np.concatenate((front_1, back_2))
		out2 = np.concatenate((front_2, back_1))
	return ind1.reshape(s), ind2.reshape(s), out1, out2
	
def mutate(subjects, arity = arity, in_size = inputs+bias):
	mut_id = np.random.randint(0, len(subjects), (1,))
	#mutants = parents[mut_id]
	for m in mut_id:
		mutant = subjects[m]
		ind = mutant[0]
		out = mutant[1]
		i = int((ind.shape[0]+out.shape[0])*random.random_sample())
		if i >= ind.shape[0]: #output node
			i = i-ind.shape[0]
			out[i] = random.randint(0, ind.shape[0]+in_size)
		else: #body node
			j = int(ind.shape[1]*random.random_sample())
			#print(i,j)
			if j < arity:
				ind[i, j] = random.randint(0, i+in_size)
			else:
				ind[i,j] = random.randint(0, len(bank))
		subjects[m] = (ind, out)
	return subjects

def select(pop, f_list, max_p = max_p, n_con = 4):
	new_p = []
	idx = np.array(range(0, len(pop)))
	idxs = []
	#keep best ind
	best_f_i = np.argmin(f_list)
	new_p.append(pop[best_f_i])
	idxs.append(best_f_i)
	while len(new_p) < max_p:
		c_id = random.choice(idx, (n_con,), replace = False) #get contestants id
		f_c = fitnesses[c_id]
		winner = np.argmin(f_c)
		w_id = c_id[winner]
		idxs.append(w_id)
		new_p.append(pop[w_id])
	return new_p, np.array(idxs)
final_fit = []
fit_track = []
ind_base = np.zeros(((arity+1)*max_n,), np.int32)

alignment = np.zeros((max_p+max_c, 2))
alignment[:, 0] = 1.0

ind_base = ind_base.reshape(-1, arity+1) #for my sanity
train_x_bias = np.zeros((train_x.shape[0], biases.shape[0]+1))
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = biases
print(train_x_bias)
#print(train_x_bias)
#print(inputs+bias+max_n)
#print("instantiating parent")
#instantiate parents
parents = []
for p in range(0, max_p):
	for i in range(0, max_n):
		#print(i < inputs+bias)
		for j in range(0, arity): #input
			if i < (inputs+bias):
				ind_base[i,j] = random.randint(0, (inputs+bias))
			else:
				ind_base[i,j] = random.randint(0, i+(inputs+bias))
		ind_base[i, -1] = random.randint(0, len(bank))
		output_nodes = random.randint(0, max_n+(inputs+bias), (outputs,), np.int32)
	parents.append((ind_base.copy(), output_nodes.copy()))
print(parents)
#test = run_output(ind_base, output_nodes, np.array([10.0]))
fitnesses = np.zeros((max_p+max_c,))
fit_temp = np.array([fitness(train_x_bias, train_y, ind_base, output_nodes) for ind_base, output_nodes in parents])
fitnesses[:max_p] = fit_temp[:, 0].copy().flatten()
alignment[:max_p, 0] = fit_temp[:, 1].copy() #a
alignment[:max_p, 1] = fit_temp[:, 2].copy() #b
#print`(np.round(fitnesses, 2))
#fit_track.append(p_fit)
#print(f"Pre-Run Parent Fitness: {p_fit}")
for g in range(1, max_g+1):
	#print(f'Gen {g}')
	children = xover(parents)
	#print('\txover')
	children = mutate(children)
	#print('\tmutate')
	fit_temp =  np.array([fitness(train_x_bias, train_y, child[0], child[1]) for child in children])
	fitnesses[max_p:] = fit_temp[:, 0].copy().flatten()
	alignment[max_p:, 0] = fit_temp[:, 1].copy()
	alignment[max_p:, 1] = fit_temp[:, 2].copy()
	#print('\teval children')
	#print(p_fit)
	#print(c_fit)
	if any(np.isnan(fitnesses)): #Replace nans with positive infinity to screen them out
		nans = np.isnan(fitnesses)
		fitnesses[nans] = np.PINF 
	pop = parents+children
	#print(len(pop))
	parents, p_idxs = select(pop, fitnesses)
	fitnesses[:max_p] = fitnesses.copy()[p_idxs]
	best_i = np.argmin(fitnesses)
	best_fit = fitnesses[best_i]
	if g % 10 == 0:
		print(f"Gen {g} Best Fitness: {best_fit}")
	#print(p_fit)
	fit_track.append(best_fit)
	#if(p_fit > 0.96):
	#	break

#print(f"Trial {t}: Best Fitness = {best_fit}")
#final_fit.append(p_fit)
#fig, ax = plt.subplots()
#ax = plt.plot(fit_track)
#print(fit_track)
#plt.show()
#print(final_fit)
#print('biases')
#print(biases)
p1 = parents[0]
p2 = parents[1]
#print(parents)
preds1, a1, b1 = fitness(train_x_bias, train_y, p1[0], p1[1], opt = 1)
preds2, a2, b2 = fitness(train_x_bias, train_y, p2[0], p2[1], opt = 1)
first_body_node = inputs+bias
import graphviz as gv
def plot_active_nodes(ind_base, a, b, name = "active_nodes", output_nodes = output_nodes, outputs=outputs, fb_node = first_body_node):
	active_graph = gv.Digraph(comment=f'trial {t} active nodes', strict = True)
	active_nodes = []
	size = 0
	def plot_body_node(n_node):
		node = ind_base[n_node-first_body_node]
		op = bank_string[node[-1]]
		active_graph.node(f'N_{n_node}', op)
		for a in range(arity):
			prev_node = node[a]
			if prev_node not in active_nodes:
				active_nodes.append(prev_node) #count active nodes
			if prev_node < inputs: #inputs
				active_graph.node(f'N_{prev_node}', f'I_{prev_node}', shape='square', rank='same', fillcolor = 'orange', style='filled')
				active_graph.edge(f'N_{prev_node}', f'N_{n_node}')
			elif prev_node >= inputs and prev_node < first_body_node: #bias:
				active_graph.node(f'N_{prev_node}', f"{biases[prev_node-inputs]}", shape='square', rank='same', fillcolor='yellow', style='filled')
				active_graph.edge(f'N_{prev_node}', f'N_{n_node}')
			else:
				plot_body_node(prev_node)
				active_graph.edge(f'N_{prev_node}', f'N_{n_node}')
	active_graph.node(f'A', f'*{np.round(a, 5)}', shape = 'diamond', fillcolor='green', style='filled')
	active_graph.node(f'B', f'+{np.round(b, 5)}', shape = 'diamond', fillcolor = 'green', style = 'filled')
	active_graph.edge(f'A', f'B')	
	for o in range(outputs):
		node = output_nodes[o]
		active_graph.node(f'O_{o}', f'O_{o}', shape='square', fillcolor='lightblue', style='filled')
		if node < inputs: #inputs
			active_graph.node(f'N_{node}', f'I_{node}', shape='square', rank='same', fillcolor = 'orange', style='filled')
			active_graph.edge(f'N_{node}', f'O_{o}')
			if node not in active_nodes:
				active_nodes.append(node)
		elif node >= inputs and node < first_body_node: #bias:
			active_graph.node(f'N_{node}', f"{biases[node-inputs]}", shape='square', rank='same', fillcolor='yellow', style='filled')
			active_graph.edge(f'N_{node}', f'O_{o}')
			if node not in active_nodes:
				active_nodes.append(node)
		else:
			plot_body_node(node)
			active_graph.edge(f'N_{node}', f'O_{o}')
			if node not in active_nodes:
				active_nodes.append(node)
		active_graph.edge(f'O_{o}', 'A')
	Path(f"../output/cgp_demo/{func_name}/active_nodes/").mkdir(parents=True, exist_ok=True)
	print(f'../output/cgp_demo/{func_name}/active_nodes/{name}')
	active_graph.render(f"../output/cgp_demo/{func_name}/active_nodes/{name}", view=False)
	active_node_num = len(active_nodes)+outputs #all outputs are active by definition
	return active_node_num

def get_expression(output_nodes = output_nodes, outputs = outputs, fb_node = first_body_node):
	expressions = []
	def get_body_expressions(n_node):		
		node = ind_base[n_node-fb_node]
		op = bank_string[node[-1]]
		tokens = []
		for a in range(arity):
			prev_node = node[a]
			if prev_node < inputs:
				tokens.append(f'I_{prev_node}')
			elif prev_node > inputs and prev_node < fb_node:
				tokens.append(f'{biases[prev_node-inputs]}')
			else:
				if arity == 0:
					tokens.append(f'{get_body_expressions(prev_node)}')
				else:
					tokens.append(f'{get_body_expressions(prev_node)}')
		sub_expression = f'{op}({tokens[0]}, {tokens[1]})'
		return sub_expression
	
	for o in range(outputs):
		expressions.append("")
		expressions[o] = f"O_{o} = "
		prev_node = output_nodes[o]
		if prev_node < inputs: 
			expressions = expressions[o] + f'I_{prev_node}'
		elif prev_node > inputs and prev_node < fb_node:
			expressions[o] = expressions[o] + f'{biases[prev_node-inputs]}'
		else:
			expressions[o] += get_body_expressions(prev_node)
	#print(expressions)
	return(expressions)

def plot_full_graph(ind_base, a, b, name = "active_nodes", output_nodes = output_nodes, outputs=outputs, fb_node = first_body_node):
	dot = gv.Digraph(comment=f'trial {t}')

	total_layers = max_n+2 #input and output layers
	#for l in range(total_layers):
	#	dot.node(f"l_{l}", style="invisible")
	#	if i > 0:
	#		dot.edge(f"l_{l-1}", f"l_{l}", style="invisible")

	first_body_node = inputs+bias
	#print(first_body_node)

	for i in range(inputs):
		dot.node(f'N_{i}', f'I_{i}', shape='square', rank='same', fillcolor = 'orange', style='filled')
	#	dot.edge("l_0", f"N_{i}", style='invisible')
	for b in range(bias):
		dot.node(f'N_{b+inputs}', f"{biases[b]}", shape='square', rank='same', fillcolor='yellow', style='filled')
	#	dot.edge("l_0", f"N_{b}", style='invisible')
	dot.attr(rank='same')
	for n in range(first_body_node, max_n+first_body_node):

		node = ind_base[n-first_body_node]
		op = bank_string[node[-1]]
		dot.node(f'N_{n}', op)
		for a in range(arity):
			dot.edge(f'N_{node[a]}', f'N_{n}')

	dot.attr(rank = 'max')
	dot.node(f'A', f'*{a}', shape = 'diamond', fillcolor='green', style='filled')
	dot.node(f'B', f'+{b}', shape = 'diamond', fillcolor = 'green', style = 'filled')
	dot.edge(f'A', f'B')
	for o in range(outputs):
		node = output_nodes[o]
		dot.attr(rank='max')
		dot.node(f'O_{o}', f'O_{o}', shape='square', fillcolor='lightblue', style='filled')
		dot.edge(f'N_{node}', f'O_{o}')
		dot.edge(f'O_{o}', 'A')
	Path(f"../output/cgp_demo/{func_name}/full_graphs/").mkdir(parents=True, exist_ok=True)
	dot.render(f"../output/cgp_demo/{func_name}/full_graphs/{name}", view=False)
	
_ = plot_active_nodes(p1[0], a1, b1, name='p1', output_nodes = p1[1])
_ = plot_active_nodes(p2[0], a2, b2, name='p2', output_nodes = p2[1])
plot_full_graph(p1[0], a1, b1, name='p1', output_nodes=p1[1])
plot_full_graph(p2[0], a2, b2, name='p2', output_nodes=p2[1])
#p = p1+p2
c = xover(parents)
c1 = c[0]
c2 = c[2]

preds3, a3, b3 = fitness(train_x_bias, train_y, c1[0], c1[1], opt = 1)
preds4, a4, b4 = fitness(train_x_bias, train_y, c2[0], c2[1], opt = 1)

_ = plot_active_nodes(c1[0], a3, b3, name='c1', output_nodes = c1[1])
_ = plot_active_nodes(c2[0], a4, b4, name='c2', output_nodes = c2[1])
plot_full_graph(c1[0], a3, b3, name='c1', output_nodes = c1[1])
plot_full_graph(c2[0], a4, b4, name='c2', output_nodes = c2[1])

print(p1)
print(p2)
print(c1)
print(c2)

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
def x(x,y):
	return x
def y(x, y):
	return y
def add(x,y):
	return x+y
def sub(x,y):
	return x-y
def mul(x,y):
	return x*y
def div(x,y):
	if y == 0.0:
		return 0
#test_x = np.arange(11, 30.1, 1)
#test_y = [func([y]) for y in test_x]
#print(train_x)
#print(train_y)
#print(powe(0)
t = 1 #trial
print(f'trial {t}')
max_g = 200 #max generations
print(f'generations {max_g}')
max_n = 12 #max body nodes
print(f'max body nodes {max_n}')
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
func = func_bank.func_list[0]
func_name = func_bank.name_list[0]
train_x = func.x_dom
train_y = func.y_test
print(train_x)
from scipy.stats import pearsonr
def rmse(preds, reals):
	return np.sqrt(np.mean((preds-reals)**2)) #copied from stack overflow

def align(ind, out, preds, reals, x = train_x):
	if not all(np.isfinite(preds)):
		return 1.0, 0.0
	try:
		align = np.round(np.polyfit(preds, reals, 1, rcond=1e-16), decimals = 14)
	except:
		return 1.0, 0.0
	a = align[0]
	b = align[1]
	print(f'align {align}')
	return (a,b)

def corr(preds, reals, x=train_x):
	if any(np.isnan(preds)) or any(np.isinf(preds)):
		return -1
	r = pearsonr(preds, reals)[0]
	if np.isnan(r):
		r = 0
	return (1-r**2)

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
	try:
		return function(args[0], args[1]) # so far 2d only
	except TypeError:
		print(args)
		print(function)
		print(cur_node)
			

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
		out_x[x] = run_output(ind_base, output_nodes, in_val)
	(a,b) = align(ind_base, output_nodes, out_x, train_y)
	#print('A, B')
	#print(a)
	#print(b)
	new_x = out_x*a+b
	if opt == 1:
		return new_x, a, b
	return rmse(new_x, train_y), a, b
def mutate(ind, out, arity = arity, in_size = inputs+bias):
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
	return ind, out
final_fit = []
fit_track = []
ind_base = np.zeros(((arity+1)*max_n,), np.int32)
ind_base = ind_base.reshape(-1, arity+1) #for my sanity
train_x_bias = np.zeros((train_x.shape[0], biases.shape[0]+1))
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = biases
print(train_x_bias)
#print(train_x_bias)
#print(inputs+bias+max_n)
print("instantiating parent")
#instantiate parent
for i in range(0, max_n):
	#print(i < inputs+bias)
	for j in range(0, arity): #input
		if i < (inputs+bias):
			ind_base[i,j] = random.randint(0, (inputs+bias))
		else:
			ind_base[i,j] = random.randint(0, i+(inputs+bias))
	ind_base[i, -1] = random.randint(0, len(bank))
#print("First Parent")
#print(ind_base)
output_nodes = random.randint(0, max_n+(inputs+bias), (outputs,), np.int32)

#test = run_output(ind_base, output_nodes, np.array([10.0]))
p_fit, p_A, p_B = fitness(train_x_bias, train_y, ind_base, output_nodes)
#fit_track.append(p_fit)
#print(f"Pre-Run Parent Fitness: {p_fit}")
for g in range(1, max_g+1):
	children = [mutate(ind_base.copy(), output_nodes.copy()) for x in range(0, max_c)]
	c_fit = np.array([fitness(train_x_bias, train_y, child[0], child[1]) for child in children])
	a = c_fit[:, 1].copy().flatten()
	b = c_fit[:, 2].copy().flatten()
	c_fit = c_fit[:, 0].flatten()
	#print(p_fit)
	#print(c_fit)
	if any(np.isnan(c_fit)): #Replace nans with positive infinity to screen them out
		nans = np.isnan(c_fit)
		c_fit[nans] = np.PINF 
	if any(c_fit <= p_fit):
		best = np.argmin(c_fit)
		ind_base = children[best][0].copy()
		output_nodes = children[best][1].copy()
		p_fi = np.argmin(c_fit)
		p_fit = np.min(c_fit)
		p_A = a[p_fi]
		p_B = b[p_fi]
	if g % 100 == 0:
		print(f"Gen {g} Best Fitness: {p_fit}")
	#print(p_fit)
	fit_track.append(p_fit)
	#if(p_fit > 0.96):
	#	break

print(f"Trial {t}: Best Fitness = {p_fit}")
#final_fit.append(p_fit)
#fig, ax = plt.subplots()
#ax = plt.plot(fit_track)
#print(fit_track)
#plt.show()
#print(final_fit)
print('biases')
print(biases)
print('best individual')
print(ind_base)
print('output nodes')
print(output_nodes)
print('preds')
preds, a, b = fitness(train_x_bias, train_y, ind_base, output_nodes, opt = 1)
print(preds)
#print(list(train_y))
Path(f"../output/align_test/{func_name}/log/").mkdir(parents=True, exist_ok=True)
import pickle

fig, ax = plt.subplots()
ax.scatter(train_x, train_y, label = 'Ground Truth')
ax.scatter(train_x, preds, label = 'Predicted')
fig.suptitle(f"{func_name} Trial {t}")
ax.set_title(f"RMSE = {np.round(p_fit, 2)}")
ax.legend()
Path(f"../output/align_test/{func_name}/scatter/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/align_test/{func_name}/scatter/plot_{t}.png")

fig, ax = plt.subplots()
ax.plot(fit_track)
ax.set_title(f'{func_name} Trial {t}')
Path(f"../output/align_test/{func_name}/plot/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/align_test/{func_name}/plot/plot_{t}.png")


#export graph
import graphviz as gv
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
dot.node(f'A', f'*{p_A}', shape = 'diamond', fillcolor='green', style='filled')
dot.node(f'B', f'+{p_B}', shape = 'diamond', fillcolor = 'green', style = 'filled')
dot.edge(f'A', f'B')
for o in range(outputs):
	node = output_nodes[o]
	dot.attr(rank='max')
	dot.node(f'O_{o}', f'O_{o}', shape='square', fillcolor='lightblue', style='filled')
	dot.edge(f'N_{node}', f'O_{o}')
	dot.edge(f'O_{o}', 'A')

#	dot.edge(f"l_{total_layers-1}", f'O_{o}')
Path(f"../output/align_test/{func_name}/full_graphs/").mkdir(parents=True, exist_ok=True)
dot.render(f"../output/align_test/{func_name}/full_graphs/graph_{t}", view=False)

#active nodes only

def plot_active_nodes(p_A = p_A, p_B = p_B, name = "active_nodes", output_nodes = output_nodes, outputs=outputs, fb_node = first_body_node):
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
	active_graph.node(f'A', f'*{np.round(p_A, 5)}', shape = 'diamond', fillcolor='green', style='filled')
	active_graph.node(f'B', f'+{np.round(p_B, 5)}', shape = 'diamond', fillcolor = 'green', style = 'filled')
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

	Path(f"../output/align_test/{func_name}/active_nodes/").mkdir(parents=True, exist_ok=True)
	active_graph.render(f"../output/align_test/{func_name}/active_nodes/active_{t}", view=False)
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
		expressions[o]= f'({expressions[o]})*{p_A}+{p_B}'
	return(expressions)
	
n = plot_active_nodes()
e = get_expression()
print(f'{e}')
print(f'Active Nodes = {n}')
print(f"../output/align_test/{func_name}/log/output_{t}.pkl")
with open(f"../output/align_test/{func_name}/log/output_{t}.pkl", "wb") as f:
	pickle.dump(biases, f)
	pickle.dump(ind_base, f)
	pickle.dump(output_nodes, f)
	pickle.dump(preds, f)
	pickle.dump(p_fit, f)
	pickle.dump(n, f)
	pickle.dump(fit_track, f)
	pickle.dump(e, f)
#expressions = get_expression()
#for expression in expressions:
#	print(expression)

#CGP 1 Point Crossover
import numpy as np
import warnings
import matplotlib.pyplot as plt
from numpy import random, sin, cos, tan, sqrt, exp, log, abs, floor, ceil
from math import log, pi
from sys import path
from pathlib import Path
from functions import *
from effProg import *
from similarity import *
from scipy.signal import savgol_filter
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
		return np.PINF
	return x/y
#test_x = np.arange(11, 30.1, 1)
#test_y = [func([y]) for y in test_x]
#print(train_x)
#print(train_y)
#print(powe(0)
t = int(argv[1]) #trial
print(f'trial {t}')
max_g = int(argv[2]) #max generations
print(f'generations {max_g}')
max_n = int(argv[3]) #max body nodes
print(f'max body nodes {max_n}')
max_p = int(argv[4]) #max parents
print(f'Parents {max_p}')
max_c = int(argv[5]) #max children
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
func = func_bank.func_list[int(argv[6])]
func_name = func_bank.name_list[int(argv[6])]
train_x = func.x_dom
train_y = func.y_test
print('train x and y')
print(train_x)
print(train_y)
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
f = int(argv[7])
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
	if opt == 1:
		new_x = out_x*a+b
		return new_x, a, b
	return fit(out_x, train_y), a, b

def mutate(subjects, arity = arity, in_size = inputs+bias, max_p = max_p, max_c = max_c):
	mut_id = list(range(0, max_p))
	children = []
	#mutants = parents[mut_id]
	for m in range(0, max_p):
		mutant = subjects[m]
		for j in range(0, max_c):
			ind = mutant[0].copy()
			out = mutant[1].copy()
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
			children.append((ind.copy(), out.copy()))
	return children

def select(pop, f_list, max_p = max_p, n_con = 4):
	#print("Fitnesses as given to selection")
	#print(np.round(f_list, 5))
	new_p = []
	new_p_f = []
	idx = np.array(range(0, len(pop)))
	idxs = []
	#keep best ind
	best_f_i = np.argmin(f_list)
	new_p.append(pop[best_f_i])
	new_p_f.append(f_list[best_f_i])
	#print(f'index {best_f_i}')
	#print(f'best fitness in selection {f_list[best_f_i]}\n')
	idxs.append(best_f_i)
	while len(new_p) < max_p:
		c_id = random.choice(idx, (n_con,), replace = False) #get contestants id
		f_c = fitnesses[c_id]
		winner = np.argmin(f_c)
		w_id = c_id[winner]
		new_p_f.append(f_c[winner])
		new_p.append(pop[w_id])
		idxs.append(w_id)
	return new_p, idxs, new_p_f
final_fit = []
fit_track = []
ind_base = np.zeros(((arity+1)*max_n,), np.int32)
ind_base = ind_base.reshape(-1, arity+1) #for my sanity

alignment = np.zeros((max_p+max_c*max_p, 2))
alignment[:, 0] = 1.0

train_x_bias = np.zeros((train_x.shape[0], biases.shape[0]+1))
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = biases
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

#test = run_output(ind_base, output_nodes, np.array([10.0]))
fitnesses = np.zeros((max_p+max_c*max_p,))
print(fitnesses.shape)
fit_temp = np.array([fitness(train_x_bias, train_y, ind_base, output_nodes) for ind_base, output_nodes in parents])
fitnesses[:max_p] = fit_temp[:, 0].copy().flatten()
alignment[:max_p, 0] = fit_temp[:, 1].copy() #a
alignment[:max_p, 1] = fit_temp[:, 2].copy() #b

ret_avg_list = []
ret_std_list = []
avg_change_list = []
std_change_list = []
avg_hist_list = []

best_i = np.argmin(fitnesses[:max_p])
p_size = [cgp_active_nodes(parents[best_i][0], parents[best_i][1], opt = 2)]
#fit_track.append(p_fit)
#print(f"Pre-Run Parent Fitness: {p_fit}")
for g in range(1, max_g+1):
	#print(f'Generation {g}')
	children = mutate(parents.copy())
	#print(parents[0])
	#print(children[0])

	if any(np.isnan(fitnesses)): #Replace nans with positive infinity to screen them out
		nans = np.isnan(fitnesses)
		fitnesses[nans] = np.PINF 

	pop = parents+children
	fit_temp =  np.array([fitness(train_x_bias, train_y, ind[0], ind[1]) for ind in pop])
	fitnesses = fit_temp[:, 0].copy().flatten()
	alignment = fit_temp[:, 1].copy()
	alignment = fit_temp[:, 2].copy()
	#print(f'fitnesses\n {np.round(fitnesses, 5)}')

	ret = []
	chg = []
	full_change_list = []
	for i in range(0, len(parents)-1):
		p1 = parents[i]
		p2 = parents[i+1]
		ps = [p1, p2]
		p_fits = [fitnesses[p], fitnesses[p+1]]
		best_p_idx = np.argmin(p_fits)
		best_p_fit = p_fits[best_p_idx]
		best_p = ps[best_p_idx]
		c_fits = []
		cs = []
		for j in range(i*max_c, i*max_c+max_c):
			c = children[j]
			cs.append(c)
			c_fit = fitnesses[max_p+j]
			c_fits.append(c_fit)
			full_change_list.append(percent_change(c_fit, best_p_fit))
			ret.append(find_similarity(c[0],best_p[0], c[1], best_p[1]))
		best_c_idx = np.argmin(c_fits)
		best_c_fit = c_fits[best_c_idx]
		best_c = cs[best_c_idx]
		chg.append(percent_change(best_c_fit, best_p_fit))

	ret_avg_list.append(np.nanmean(ret))
	ret_std_list.append(np.nanstd(ret))
	avg_change_list.append(np.nanmean(chg))
	std_change_list.append(np.nanstd(chg))
	full_change_list = np.array(full_change_list).flatten()
	full_change_list = full_change_list[np.isfinite(full_change_list)]
	cl_std = np.nanstd(full_change_list)
	if not all(cl == 0.0 for cl in full_change_list):
		avg_hist_list.append((g, np.histogram(full_change_list, bins = 10, range=(cl_std*-2, cl_std*2))))
	#print(len(pop))
	best_i = np.argmin(fitnesses)
	best_fit = fitnesses[best_i]
	#print(f'best_fit {best_fit}')
	fit_track.append(best_fit)
	p_size.append(cgp_active_nodes(pop[best_i][0], pop[best_i][1], opt = 2))

	parents, p_idxs, new_p_f = select(pop.copy(), fitnesses.copy())

	#print(max_p)
	if g % 100 == 0:
		print(f"Gen {g} Best Fitness: {best_fit}")
	#print("------")
import pickle
pop = parents+children
fit_temp =  np.array([fitness(train_x_bias, train_y, ind[0], ind[1]) for ind in pop])
fitnesses = fit_temp[:, 0].copy().flatten()
alignment = fit_temp[:, 1].copy()
alignment = fit_temp[:, 2].copy()
print('Fitnesses:')
print(np.round(fitnesses, 5))
print('Best Fitness Over Time')
print(np.round(fit_track, 5))
best_i = np.argmin(fitnesses)
print(f'Best Fitness Index: {best_i}')
best_fit = fitnesses[best_i]
ind_base = pop[best_i][0]
output_nodes = pop[best_i][1]
print(f"Trial {t}: Best Fitness = {best_fit}")
print('best individual')
print(ind_base)
print(output_nodes)
print('preds')
print(train_y)
preds, p_A, p_B = fitness(train_x_bias, train_y, ind_base, output_nodes, opt = 1)
print(preds)
print(preds.shape)
print(fitness(train_x_bias, train_y, ind_base, output_nodes, opt = 0))

#print(list(train_y))
Path(f"../output/cgp_40/{func_name}/log/").mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots()
ax.scatter(train_x, train_y, label = 'Ground Truth')
ax.scatter(train_x, preds, label = 'Predicted')
fig.suptitle(f"{func_name} Trial {t}")
ax.set_title(f"{fit_name} = {np.round(best_fit, 4)}")
ax.legend()
Path(f"../output/cgp_40/{func_name}/scatter/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/cgp_40/{func_name}/scatter/plot_{t}.png")

fig, ax = plt.subplots()
ax.plot(fit_track)
ax.set_yscale('log')
ax.set_ylabel("1-R^2")
ax.set_xlabel("Generations")
ax.set_title(f'{func_name} Trial {t}')
Path(f"../output/cgp_40/{func_name}/plot/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/cgp_40/{func_name}/plot/plot_{t}.png")


win_length = 100
from matplotlib import colormaps

fig, ax = plt.subplots(figsize = (10, 5))
hist_gens = np.array([hist_list[0] for hist_list in avg_hist_list])
avg_hist_list = [hist_list[1] for hist_list in avg_hist_list]
bin_edges = np.array([avg_hist_list[i][1] for i in range(len(avg_hist_list))])
hists = np.array([avg_hist_list[i][0] for i in range(len(avg_hist_list))])
bin_centers = []
for i in range(bin_edges.shape[0]):
	centers = []
	for j in range(0, bin_edges.shape[1]-1):
		centers.append((bin_edges[i][j]+bin_edges[i][j+1])/2)
	bin_centers.append(centers)
bin_centers = np.array(bin_centers)
for i in range(hist_gens.shape[0]):
	g = hist_gens[i]
	x = np.full((bin_centers.shape[1],), g)
	ax.scatter(x, bin_centers[i, :], c=hists[i], cmap = colormaps['Greys'], alpha = 0.33)
ax.set_yscale('log')
ax.set_ylabel('Frequency[Fit(Child)-Fit(Best Parent)]')
ax.set_xlabel('Generations')
ax.set_xlim(0, max_g)
fig.tight_layout()
Path(f"../output/cgp_40/{func_name}/change_hists/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/cgp_40/{func_name}/change_hists/change_hists{t}.png")


avg_change_list = np.array(avg_change_list)
std_change_list = np.array(std_change_list)
fig, ax = plt.subplots()
try:
	ax.plot(savgol_filter(avg_change_list, win_length, 3), c = 'blue')
	ax.fill_between(range(avg_change_list.shape[0]), savgol_filter((avg_change_list-std_change_list), win_length, 3), savgol_filter((avg_change_list+std_change_list), win_length, 3), color = 'blue', alpha = 0.1)
except:
	ax.plot(avg_change_list)
	ax.fill_between(range(avg_change_list.shape[0]), (avg_change_list-std_change_list), (avg_change_list+std_change_list))
ax.set_title(f'{func_name} Trial {t}')
ax.set_ylabel('Avg(Fitness(Best Child)-Fitness(Best Parent))')
ax.set_xlabel('Generations')
plt.tight_layout()
Path(f"../output/cgp_40/{func_name}/change_plot/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/cgp_40/{func_name}/change_plot/change_{t}.png")


ret_avg_list = np.array(ret_avg_list)
ret_std_list = np.array(ret_std_list)
fig, ax = plt.subplots()
ax.plot(savgol_filter(ret_avg_list, win_length, 3), c = 'blue')
ax.fill_between(range(ret_avg_list.shape[0]), savgol_filter((ret_avg_list-ret_std_list), win_length, 3), savgol_filter((ret_avg_list+ret_std_list), win_length, 3), color = 'blue', alpha = 0.1)
ax.set_title(f'{func_name} Trial {t}')
ax.set_ylabel('Average Similarity between Best Parents and Children')
ax.set_xlabel('Generations')
plt.tight_layout()
Path(f"../output/cgp_40/{func_name}/retention/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/cgp_40/{func_name}/retention/retention_{t}.png")

p_size = np.array(p_size)

fig, ax = plt.subplots()
ax.plot(p_size)
ax.set_title(f'{func_name} Trial {t}')
ax.set_ylabel("Proportion of Nodes Which Are Active")
ax.set_xlabel("Generations")
Path(f"../output/cgp_40/{func_name}/proportion_plot/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/cgp_40/{func_name}/proportion_plot/proportion_plot_{t}.png")


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

Path(f"../output/cgp_40/{func_name}/full_graphs/").mkdir(parents=True, exist_ok=True)
dot.render(f"../output/cgp_40/{func_name}/full_graphs/graph_{t}", view=False)

#active nodes only

def plot_active_nodes(name = "active_nodes", output_nodes = output_nodes, outputs=outputs, fb_node = first_body_node):
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
	Path(f"../output/cgp_40/{func_name}/active_nodes/").mkdir(parents=True, exist_ok=True)
	active_graph.render(f"../output/cgp_40/{func_name}/active_nodes/active_{t}", view=False)
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
	
n = plot_active_nodes()
print(f'Active Nodes = {n}')
print(f"../output/cgp_40/{func_name}/log/output_{t}.pkl")
with open(f"../output/cgp_40/{func_name}/log/output_{t}.pkl", "wb") as f:
	pickle.dump(biases, f)
	pickle.dump(ind_base, f)
	pickle.dump(output_nodes, f)
	pickle.dump(preds, f)
	pickle.dump(best_fit, f)
	pickle.dump(n, f)
	pickle.dump(fit_track, f)
	pickle.dump([avg_change_list], f)
	pickle.dump([ret_avg_list], f)
	pickle.dump(p_size, f)
	pickle.dump([bin_centers, hist_gens, avg_hist_list], f)
#expressions = get_expression()
#for expression in expressions:
#	print(expression)

#CGP Subgraph XOver - From Kalkreuth 2017 
import numpy as np
import warnings
import matplotlib.pyplot as plt
from numpy import random, sin, cos, tan, sqrt, exp, log, abs, floor, ceil
from math import log, pi
from sys import path
from pathlib import Path
from functions import *
from sys import argv
print("started")
warnings.filterwarnings('ignore')

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
first_body_node = inputs+bias
#bank = (add, sub, mul, div, x, y, cos_x, cos_y, sin_x, sin_y, powe, sqrt_x_y, distance, abs_x, abs_y, floor_x, floor_y, ceil_x, ceil_y, max_f, min_f, midpoint)
#bank_string = ("+", "-", "*", "/", "x", "y", "cos(x)","cos(y)", "sin(x)", "sin(y)", "^", "$\sqrt{x+y}$", "$sqrt{x^2+y^2}$", "|x|", "|y|", "$\lfloor{x}\rfloor$", "$\lfloor{y}\rfloor$", "$\lceil{x}\rceil$", "$\lceil{y}\rceil$", "max", "min", "avg")

bank = (add, sub, mul, div) #, cos_x, cos_y, sin_x, sin_y, powe, sqrt_x_y, distance, abs_x, abs_y, midpoint)
bank_string = ("+", "-", "*", "/") #, "cos(x)","cos(y)", "sin(x)", "sin(y)", "^", "$\sqrt{x+y}$", "$sqrt{x^2+y^2}$", "|x|", "|y|", "avg")

p_mut = float(argv[8])
p_xov = float(argv[9])

func_bank = Collection()
func = func_bank.func_list[int(argv[6])]
func_name = func_bank.name_list[int(argv[6])]
train_x = func.x_dom
train_y = func.y_test
#print(train_x)
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
	try:
		for j in range(arity):
			if cur_node[j] < inp_size:
				#print(cur_node[j])
				args.append(inp_nodes[cur_node[j]])
			else:
				args.append(run(ind, ind[cur_node[j]-inp_nodes.shape[0]], inp_nodes))
	except RecursionError:
		print(f'Recursion Error')
		print(f'Individual {ind}')
		print(f'Current Node: {cur_node}')
	function = bank[cur_node[-1]]
	#print(args[0], args[1], function(args[0], args[1]))
	try:
		return function(args[0], args[1]) # so far 2d only
	except IndexError:
		print("Index Error")
		print(f'{cur_node[-1]}')
		print(f'{bank[cur_node[-1]]}')
		print(f'cur node {cur_node}')
		print(f'function {function}')
		print(f'args {args}')
			

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
	(a,b) = align(ind_base, output_nodes, out_x, train_y)
	#print('A, B')
	#print(a)
	#print(b)
	new_x = out_x*a+b
	if opt == 1:
		return new_x, a, b
	return fit(new_x, train_y), a, b

#Kalkreuth 2017
#n_i: Number of inputs
#m: Upper node number limit, null by default
#I List of input nodes, null by default
#N_F List of number of body nodes [x, x+1,...,x_n]
def RandomNodeNumber(n_i = inputs, I = [], N_f = [], m = None):
	N_R = [] #Initialize an Empty List to store random input and function node numbers
	if len(N_f) > 0: #check if function node numbers have been passed as argument
		if m != None: #check if a node number limit has been passed to the function
			#determine a sublist of N_F where the list elements X of N_F are less or equal to m
			N_m = list(N_f[N_f <= m])
			if len(N_m) == 0: #if the sublist is empty, there are no function nodes before m
				i = random.randint(0, n_i) #generate a random input node
				N_R.append(i) #append the random input to the list
			else:
				if len(N_m)-1 <= 0:
					i = random.randint(0,1)
				else:
					i = random.randint(0, len(N_m)-1) #generate a random integer in the range from 0 to |N_m|-1 inclusive(?)
				N_R.append(N_m[i]) #use i as index and get the node number from N_F
		else: #otherwise, randomly select a node number in the range from 0 to |N_F|-1 inclusive
			i = random.randint(0, len(list(N_f))-1)
			N_R.append(N_F[i])
	if len(I) > 0:
		#Select a random input node number in the range from 0 to |I|-1 inclusive by chance
		if len(I)-1 <= 0:
			j = 0
		else:
			j = random.randint(0, len(I)-1)
		N_R.append(j)
	#select one node number from the list N_R by chance
	try:
		r = random.randint(0, len(N_R)-1)
	except ValueError:
		r = 0
	return N_R[r]

#Kalkreuth 2017
#P1: Genome of first parent
#P2: Genome of second parent
#M1: List of active nodes of the first parent
#M2: List of active nodes of the second parent
def DetermineCrossoverPoint(P1, P2, M1, M2):
	if len(M1) > 0 and len(M2) > 0: #The opposite shouldn't happen but just in case!
		if len(M1) > 0:
			a = np.min(M1) #Determine minimum node number of M1
			b = np.max(M1) #Determine maximum node number of M1
			if a == b:
				CP1 = a
			else:
				CP1 = random.randint(a,b) #Choose the first possible crossover point by chance
		else:
			CP1 = np.PINF
		if len(M2) > 0:
			a = np.min(M2) #Determine minimum node number of M2
			b = np.max(M2) #Determinem aximum node number of M2
			if a==b:
				CP2 = a
			else:
				CP2 = random.randint(a,b)
		else:
			CP2 = np.PINF
		return int(np.min([CP1, CP2])) #The crossover point is the inimum of the possible points
	else:
		return -1
def DetermineActiveNodes(ind, output_nodes, first_body_node = inputs+bias, arity = arity):
	active_nodes = []
	def get_body_node(n_node):
		node = ind_base[n_node-first_body_node]
		for a in range(arity):
			prev_node = node[a]
			if prev_node > first_body_node: #inputs
				if prev_node not in active_nodes:
					active_nodes.append(prev_node)
				get_body_node(prev_node)
	for o in range(outputs):
		node = output_nodes[o]
		if node >= first_body_node:
			get_body_node(node)
			if node not in active_nodes:
				active_nodes.append(node)
		else:
			if node not in active_nodes:
				active_nodes.append(node)
	return active_nodes


#Kalkreuth 2017
#P1 Genome of the first parent
#P2 Genome of the second parent
#n_i Number of Inputs
def SubgraphCrossover(P1, P2, n_i = inputs, first_body_node = first_body_node):
	G1 = P1[0].copy() #store the genome of parent p1 in g1
	G2 = P2[0].copy() #store the genome of parent p2 in g2
	O1 = P1[1].copy()
	O2 = P2[1].copy()

	M1 = DetermineActiveNodes(P1, O1)
	M2 = DetermineActiveNodes(P2, O2)
	#print(M1, M2)
	n_g = G1.shape[0] #Determine Number of Genes
	
	C_P = DetermineCrossoverPoint(G1, G2, M1, M2) #Determine Crossover Point
	if C_P < 0: #if neither have active nodes
		print("No Active Nodes?")
		print(G1, O1)
		print(G2, O2)
		return (G1, O1)
	p_c = C_P+1-first_body_node
	G0 = np.concatenate((G1[:p_c], G2[p_c:]), axis = 0) #Copy the parts before and after crossver from G1 and G2 respectively
	O = O2.copy() #back of the list so self explanatory really
	#Create the list of active function nodes of the offspring
	#Determine and store a sublist of M1 where the list elements of M are less or equal to CP
	#print(C_P)
	#print(M1)
	M1 = np.array(M1)
	M2 = np.array(M2)
	NA1 = M1[M1 <= C_P]
	NA2 = M2[M2 > C_P]
	
	if NA1.shape[0] > 0 and NA2.shape[0] > 0: #check if both lists contain active nodes
		#print(NA1[-1])
		#print(NA2[0])
		nF = NA1[-1] #Determine first active node before CP
		nB = NA2[0] #Determine the first active node after CP
		G0 = NeighborhoodConnect(nF, nB, G0)
		
	NA = np.concatenate((NA1, NA2)) #combine lists
	if NA.shape[0] > 0:
		G0, O = RandomActiveConnect(n_i, NA, C_P, G0, O)
	return (G0, O)

#Kalkreuth 2017
#nF: Number of the first active node before the crossover point
#nB: Number of the first active node behind the crossover point
#G0: Offspring Genome
def NeighborhoodConnect(nF, nB, G0):
	#print('neighborhood connect')
	#print(nF)
	#print(nB)
	#print(G0)
	"""
	if nB >= nF:
		print(f'nB {nB} >= nF {nF}')
	if nF >= nB:
		print(f'{nF} >= {nB} - {first_body_node}')
	if nB >= G0.shape[0]:
		print(f'nF {nF}')
		print(f'nB {nB}')
		print(f'G0 {G0}')
	"""
	#print(f'G0[{nB-first_body_node}, 0] = {nF}')
	G0[nB-first_body_node, 0] = nF
	return G0

#Kalkreuth 2017
# n_i: number of inputs
# NA: List of active function nodes
# CP: The Crossover Point
# G0: Genome of the Offspring
# O: Output Nodes
def RandomActiveConnect(n_i, NA, CP, G0, O):
	#print('random active connect')
	I = [] #get input nodes
	for n in G0:
		for a in range(0, arity):
			if n[a] < first_body_node and n[a] not in I:
				I.append(n[a])
	#print(f'I {I}')
	#print(f'Crossover Point {CP}')
	#print(f'First Body Node {first_body_node}')
	#print(f'NA {NA}')
	for n in NA: #iterate over the active nodes
		#print(f'n {n}')
		if n > CP: #if node is greater than xover point
			#print(f'\tn > CP: {n > CP}')
			node = G0[n-first_body_node] #get connection genes
			#print(f'\tnode {n} = G0[{n-first_body_node}] = {node}')
			GC = node[:arity]
			#print(f'\tGC = {GC}')
			for i in range(arity): #iterate over connection genes
				g = GC[i]
				#print(f'\t\tg = GC[{i}] = {g}')
				if g not in NA: #if the current connection gene is not connected to an active funciton node
					#print(f'\t\t\t{g} not in NA')
					#print(f'\t\t\t{node}')
					#print(f'\t\t\t{G0[node, i]}')
					G0[n-first_body_node, i] = RandomNodeNumber(n_i, I, NA, CP)
					#print(f'\t\t\t{G0[node, i]}')
	for o in O: #Adjust output genes
		if o not in NA: #if output is connected to an inactive nodes
			o = RandomNodeNumber(I, NA)
	return G0, O
	
def xover(parents, p_xov = p_xov):
	children = []
	#print(list(range(0, len(parents), 2)))
	for i in range(0, len(parents), 2):
	#print(i)
		P1 = parents[i]
		P2 = parents[i+1]
		if random.random() < p_xov:
			children.append(SubgraphCrossover(P1, P2, inputs))
			children.append(SubgraphCrossover(P2, P1, inputs))
		else:
			children.append(P1)
			children.append(P2)
	#print(len(children))
	return children

def mutate(subjects, arity = arity, in_size = inputs+bias, p_mut = p_mut):
	mut_id = np.random.randint(0, len(subjects), (1,))
	#mutants = parents[mut_id]
	for m in range(len(subjects)):
		if random.random() >= p_mut:
			continue
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
ind_base = ind_base.reshape(-1, arity+1) #for my sanity

alignment = np.zeros((max_p+max_c, 2))
alignment[:, 0] = 1.0

train_x_bias = np.zeros((train_x.shape[0], biases.shape[0]+1))
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = biases
#print(train_x_bias)
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

#test = run_output(ind_base, output_nodes, np.array([10.0]))
fitnesses = np.zeros((max_p+max_c,))
fit_temp = np.array([fitness(train_x_bias, train_y, ind_base, output_nodes) for ind_base, output_nodes in parents])
fitnesses[:max_p] = fit_temp[:, 0].copy().flatten()
alignment[:max_p, 0] = fit_temp[:, 1].copy() #a
alignment[:max_p, 1] = fit_temp[:, 2].copy() #b
print(np.round(fitnesses, 4))
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
	#print(len(parents))
	#print(len(children)) 
	pop = parents+children
	#print(len(pop))
	parents, idxs = select(pop, fitnesses)
	fitnesses[:max_p] = fitnesses.copy()[idxs]
	alignment[:max_p, 0] = alignment[:,0].copy()[idxs]
	alignment[:max_p, 1] = alignment[:,1].copy()[idxs]
	best_i = np.argmin(fitnesses)
	best_fit = fitnesses[best_i]
	if g % 100 == 0:
		print(f"Gen {g} Best Fitness: {best_fit}")
	#print(p_fit)
	fit_track.append(best_fit)
	#if(p_fit > 0.96):
	#	break

print(f"Trial {t}: Best Fitness = {best_fit}")
#final_fit.append(p_fit)
#fig, ax = plt.subplots()
#ax = plt.plot(fit_track)
#print(fit_track)
#plt.show()
#print(final_fit)
#print('biases')
#print(biases)
print('best individual')
print(pop[best_i])
print('preds')
preds, p_A, p_B = fitness(train_x_bias, train_y, pop[best_i][0], pop[best_i][1], opt = 1)
print(preds)
#print(list(train_y))
Path(f"../output/cgp_sgx/{func_name}/log/").mkdir(parents=True, exist_ok=True)
import pickle

ind_base = pop[best_i][0]
output_nodes = pop[best_i][1]

fig, ax = plt.subplots()
ax.scatter(train_x, train_y, label = 'Ground Truth')
ax.scatter(train_x, preds, label = 'Predicted')
fig.suptitle(f"{func_name} Trial {t}")
ax.set_title(f"{fit_name} = {np.round(best_fit, 5)}")
ax.legend()
Path(f"../output/cgp_sgx/{func_name}/scatter/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/cgp_sgx/{func_name}/scatter/plot_{t}.png")

fig, ax = plt.subplots()
ax.plot(fit_track)
ax.set_title(f'{func_name} Trial {t}')
Path(f"../output/cgp_sgx/{func_name}/plot/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/cgp_sgx/{func_name}/plot/plot_{t}.png")


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
Path(f"../output/cgp_sgx/{func_name}/full_graphs/").mkdir(parents=True, exist_ok=True)
dot.render(f"../output/cgp_sgx/{func_name}/full_graphs/graph_{t}", view=False)

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
	Path(f"../output/cgp_sgx/{func_name}/active_nodes/").mkdir(parents=True, exist_ok=True)
	active_graph.render(f"../output/cgp_sgx/{func_name}/active_nodes/active_{t}", view=False)
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
print(f"../output/cgp_sgx/{func_name}/log/output_{t}.pkl")
with open(f"../output/cgp_sgx/{func_name}/log/output_{t}.pkl", "wb") as f:
	pickle.dump(biases, f)
	pickle.dump(ind_base, f)
	pickle.dump(output_nodes, f)
	pickle.dump(preds, f)
	pickle.dump(best_fit, f)
	pickle.dump(n, f)
	pickle.dump(fit_track, f)
#expressions = get_expression()
#for expression in expressions:
#	print(expression)

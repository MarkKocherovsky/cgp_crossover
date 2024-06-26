import numpy as np
import warnings
import matplotlib.pyplot as plt
from numpy import random, sin, exp, cos, sqrt, pi
from sys import path, argv
from pathlib import Path
from functions import *
from effProg import *
warnings.filterwarnings('ignore')

def first(x):
	return x[0]
def last(x):
	return x[-1]
def add(x):
	return sum(x)
def sub(x):
	y = x[0]
	for i in x[1:]:
		y -= i
	return y
def mul(x):
	y = x[0]
	for i in x[1:]:
		y *= i
	return y
def div(x):
	if any(x[1:] == 0.0):
		return np.PINF
	y = x[0]
	for i in x[1:]:
		y /= i
	return y
t = 0 #trials
max_g = 200 #generations
max_r = 64 #rules
max_d = 4 #destinations (other than output)
if max_r < 1:
	print("Number of rules too small, setting to 10")
	max_r = 10
max_p = 10 #parents
max_c = 10 #children
arity = 2 #sources
n_inp = 1 #number of inputs
bias = np.arange(0, 10, 1)
n_bias = bias.shape[0] #number of bias inputs

p_mut = 1/max_p #mutation probability

func_bank = Collection()
func = func_bank.func_list[0]
func_name = func_bank.name_list[0]
train_x = func.x_dom
train_y = func.y_test

output_index = 0
input_indices = np.arange(1, n_inp+1, 1)
#print(input_indices)

#bank = (add, add)
bank = (add, sub, mul, div) #, cos_x, cos_y, sin_x, sin_y, powe, sqrt_x_y, distance, abs_x, abs_y, midpoint)
bank_string = ("+", "-", "*", "/") #, "cos(x)","cos(y)", "sin(x)", "sin(y)", "^", "$\sqrt{x+y}$", "$sqrt{x^2+y^2}$", "|x|", "|y|", "avg")

def rmse(preds, reals):
	return np.sqrt(np.mean((preds-reals)**2)) #copied from stack overflow
from scipy.stats import pearsonr

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
def align(preds, reals, x = train_x):
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
#, output_index = output_index, input_indices = input_indices
def generate_instruction(sources, destinations, arity=arity, bank = bank):
	instruction = np.zeros(((2+arity),), dtype=np.int32)
	instruction[0] = random.choice(destinations) #destination index
	instruction[1] = random.randint(0, len(bank))
	instruction[2:] = random.choice(sources, (arity,))
	return instruction
	
def get_registers(arity = arity, max_d = max_d, n_bias = n_bias, inputs = n_inp):
	possible_indices = np.arange(0, max_d+n_bias+inputs+1)
	possible_destinations = np.delete(possible_indices, np.arange(1, n_bias+inputs+1)) #removes inputs for destination
	possible_sources = possible_indices
	#print(possible_sources)
	return possible_destinations, possible_sources
	

def generate_ind(max_r = max_r, inputs = n_inp, n_bias = n_bias, max_d = max_d, arity = arity):
	#instruction_count = 5
	instruction_count = random.randint(2, max_r)
	#print(instruction_count)
	instructions = np.zeros((instruction_count, 2+arity))
	possible_destinations, possible_sources = get_registers()
	for i in range(instruction_count):
		instructions[i, :] = generate_instruction(possible_sources, possible_destinations)
	#print(instructions)
	return instructions
	
def generate_single_ind(max_r = max_r, inputs = n_inp, n_bias = n_bias, max_d = max_d, arity = arity):
	#instruction_count = 5
	instruction_count = 1
	#print(instruction_count)
	instructions = np.zeros((instruction_count, 2+arity))
	possible_destinations, possible_sources = get_registers()
	for i in range(instruction_count):
		instructions[i, :] = generate_instruction(possible_sources, possible_destinations)
	#print(instructions)
	return instructions
	
def run(individual, train_x = train_x, train_y = train_y, max_d = max_d, n_inp = n_inp, n_bias = n_bias, bias = bias, bank = bank):
	preds = np.zeros((len(train_y),))
	train_x_bias = np.zeros((train_x.shape[0], bias.shape[0]+1))
	train_x_bias[:, 0] = train_x
	train_x_bias[:, 1:] = bias
	for i in range(len(train_x)):
		registers = np.zeros((1+n_inp+n_bias+max_d,))
		registers[1:n_inp+n_bias+1] = train_x_bias[i, :]
		#registers[n_inp+1, n_bias+1] = bias
		for j in range(len(individual)):
			operation = individual[j].astype(int)
			destination = operation[0]
			operator = bank[operation[1]]
			sources = operation[2:]
			registers[destination] = operator(registers[sources])
		preds[i] = registers[0]
	#print(train_y
	(a,b) = align(preds, train_y)
	preds = preds*a+b
	return fit(preds, train_y), a, b
	
def predict(individual, a, b, test, max_d = max_d, n_inp = n_inp, n_bias = n_bias, bias = bias, bank = bank):
	preds = np.zeros((len(test),))
	train_x_bias = np.zeros((test.shape[0], bias.shape[0]+1))
	train_x_bias[:, 0] = test
	train_x_bias[:, 1:] = bias
	for i in range(len(train_x)):
		registers = np.zeros((1+n_inp+n_bias+max_d,))
		registers[1:n_inp+n_bias+1] = train_x_bias[i, :]
		#registers[n_inp+1, n_bias+1] = bias
		for j in range(len(individual)):
			operation = individual[j].astype(int)
			destination = operation[0]
			operator = bank[operation[1]]
			sources = operation[2:]
			registers[destination] = operator(registers[sources])
		preds[i] = registers[0]
	return preds*a+b
			

def mass_eval(pop, func = func):
	fitnesses = []
	A = []
	B = []
	for ind in pop:
		with np.errstate(invalid='raise'):
			try:
				f = run(ind)
				v = f[0]
				a = f[1]
				b = f[2]
				fitnesses.append(v)
				A.append(a)
				B.append(b)			
			except (OverflowError, FloatingPointError):
				fitnesses.append(np.nan)
				A.append(1.0)
				B.append(0.0)
	return fitnesses, A, B

import operator as opt
def xover(parents, max_r = max_r):
	children = []
	for i in range(0, len(parents), 2):
		for j in [1,2]: #two children
			p1 = parents[i].copy()
			p2 = parents[i+1].copy()
			
			inst_counts = [len(p1), len(p2)]
			samples = [random.choice(inst_counts[0], size = (int(len(p1)/2),), replace = False), random.choice(inst_counts[1], size = (int(len(p2)/2),), replace = False)]
			p1_list = samples[0]
			p2_list = samples[1]
			fu_list = np.concatenate((p1_list,p2_list))
			c1 = np.concatenate((p1[p1_list],p2[p2_list]), axis = 0)
			#c2 = np.concatenate((p1[-p1_list], p2[-p2_list]), axis = 0)
			#https://stackoverflow.com/questions/9007877/sort-arrays-rows-by-another-array-in-python
			fu_list = fu_list.argsort()
			c = c1[fu_list, :]
			children.append(c)
	return children
	
def mutate(parents, max_c = max_c, p_mut = p_mut, bank = bank):
	children = []
	while len(children) < max_c:
		for p in range(0, len(parents)): #go through each parent
			if random.random() <= p_mut:
				parent = parents[p]
				if len(parent < 2):
					mutation = random.randint(1, 3)
				else:
					mutation = random.randint(0, 3)
				
				if mutation == 0: #remove instruction
					inst = random.randint(0, len(parent))
					children.append(np.delete(parent, inst, axis = 0))
				elif mutation == 1: #change instruction
					inst = random.randint(0, len(parent))
					part = random.randint(0, len(parent[inst]))
					possible_destinations, possible_sources = get_registers()
					if part == 0: #destination
						parent[inst, part] = random.choice(possible_destinations) #destination index
					elif part == 1: #operator
						parent[inst, part] = random.randint(0, len(bank))
					else: #source
						parent[inst, part] = random.choice(possible_sources)
					children.append(parent)
				elif mutation == 2:
					inst = random.randint(0, len(parent))
					new_inst = generate_single_ind() #easiest just to generate a new individual with one instruction
					np.insert(parent, inst, new_inst, axis = 0) #parent.insert(inst, new_inst)
					children.append(parent)
			if len(children) >= max_c:
				break
	return children

def fight(contestants, c_fitnesses):
	winner = np.argmin(c_fitnesses)
	return contestants[winner], winner
	
def select(pop, fitnesses, max_p = max_p):
	n_tour = int(len(pop)/10)
	if n_tour <=1:
		n_tour = 2
	new_parents = []
	idxs = []
	#new_fitnesses = []
	#print(fitnesses)
	#print(np.argmin(fitnesses)
	best_fit_id = np.argmin(fitnesses)
	best_fit = np.argmin(fitnesses)
	#print(pop[best_fit_id])
	new_parents.append(pop[best_fit_id])
	idxs.append(best_fit_id)
	while len(new_parents) < max_p:
		contestant_indices = random.choice(range(len(pop)), n_tour, replace = False)
		#print(pop)
		contestants = []
		for i in contestant_indices:
			contestants.append(pop[i])
		c_fitnesses = fitnesses[contestant_indices]
		candidate, winner = fight(contestants, c_fitnesses)
		#if winner not in idxs:
		idxs.append(contestant_indices[winner])
		new_parents.append(candidate)	
	return new_parents, idxs

from numpy import unique
def clean(pop): # remove consecutive duplicate rules
	new = []
	for p in pop:
		current_parent = p.copy()
		c, idx = unique(current_parent, return_index=True, axis = 0)
		#print(c)
		#print(idx)
		c = c[np.argsort(idx)]
		new.append(c.copy())
	return new
					
print(f"#####Trial {t}#####")
parents = []
fit_track = []
alignment = np.zeros((max_p+max_c, 2))
alignment[:, 0] = 1.0
for i in range(0, max_p):
	parents.append(generate_ind())
fitnesses = np.zeros((max_p+max_c),)
fitnesses[:max_p], alignment[:max_p, 0], alignment[:max_p, 1] = mass_eval(parents)
#print(f'starting fitnesses')
#print(fitnesses)
#print(f'starting scaling')
#print(alignment)
#fitnesses[:max_p] = fit_temp[:, 0].copy().flatten()
#alignment[:max_p, 0] = fit_temp[:, 1].copy() #a
#alignment[:max_p, 1] = fit_temp[:, 2].copy() #b
#fit_track.append(np.argmin(fitnesses))
#sort parents before xover and mutation?
#select before or after xover?
for g in range(1, max_g+1):
	children = xover(parents)
	#print(f'p[0] shape {parents[0].shape}')
	#parents = clean(parents.copy())
	#print(f'after cleaning: {parents[0].shape}')
	children = mutate(children.copy())
	#children = clean(children.copy())
	fitnesses[max_p:], alignment[max_p:, 0], alignment[max_p:, 1] = mass_eval(children)
	#fit_temp =  mass_eval(children)
	#fitnesses[max_p:] = fit_temp[:, 0].copy().flatten()
	#alignment[max_p:, 0] = fit_temp[:, 1].copy()
	#alignment[max_p:, 1] = fit_temp[:, 2].copy()
	pop = parents+children
	if any(np.isnan(fitnesses)): #screen out nan values
		nans = np.isnan(fitnesses)
		fitnesses[nans] = np.PINF	
	parents, p_idxs = select(pop, fitnesses)
	fitnesses[:max_p] = fitnesses.copy()[p_idxs]
	alignment[:max_p, :] = alignment.copy()[p_idxs, :]
	pop = parents+children
	best_i = np.argmin(fitnesses)
	best_pop = pop[best_i]
	best_a = alignment[best_i, 0]
	best_b = alignment[best_i, 1]
	best_fit = fitnesses[best_i]
	fit_track.append(best_fit)
	if g % 100 == 0:
		print(f'Generation {g}: Best Fit {best_fit}')
	
#fig, ax = plt.subplots()
#ax = plt.plot(fit_track)
#print(fit_track)
#plt.show()
print("Fitnesses")	
print(np.round(fitnesses, 5))
print("Alignment")
print(alignment)
print(f"Best Pop:\n{best_pop}")
print(f"Best Fit: {np.round(best_fit, 4)}")
#elif np.isnan(best_fit):
#	print("Redoing Trial")
#	t = t-1
preds = predict(best_pop, best_a, best_b, train_x)
print('preds')
print(preds)
print(f"../output/lgp_draw/{func_name}/log") 

import pickle

first_body_node = n_inp+n_bias+1
print(n_inp)
print(n_bias)
print(f'fb node {first_body_node}')

def print_individual(ind, a, b, fb_node = first_body_node):
	def put_r(reg, fb_node = first_body_node):
		if reg == 0:
			return f'R_0'
		if reg > 0 and reg <= n_inp:
			return f'R_{reg}'
		return f'R_{int(reg-fb_node+2)}'
	registers = ind[:, 0].astype(np.int32)
	operators = ind[:, 1].astype(np.int32)
	operands = ind[:, 2:].astype(np.int32)
	
	registers = list(map(put_r, registers))
	operators = [bank_string[i] for i in operators]
	
	print("R\tOp\tI")
	for i in range(len(registers)):
		reg = registers[i]
		op = operators[i]
		ops = operands[i, :]
		nums = []
		for n in ops:
			#print(n)
			if n > 0 and n<=n_inp:
				nums.append(f'R_{n}')
			elif n == 0:
				nums.append(f'R_0')
			elif n > n_inp and n < fb_node:
				nums.append(bias[n-n_inp-1])
			else:
				nums.append(f'R_{n-fb_node+2}')	
		print(f"{reg}\t{op}\t{nums}")
	print("Scaling")
	print(f"R_0 = {a}*R_0+{b}")

import graphviz as gv
	
def get_reg(reg, fb_node = first_body_node):
	if reg == 0:
		return 'R_0'
	elif reg > 0 and reg<=n_inp:
		return f'R_{reg}'
	elif reg > n_inp and reg < fb_node:
		return f'{bias[reg-2]}'
	else:
		return f'R_{reg-fb_node+2}'

def map_individual(ind): #this is ass
	ind_new = []
	for i in ind:
		i_new = []
		i_new.append(get_reg(i[0]))
		i_new.append(bank_string[1])
		i_new+=list(map(get_reg, i[2:]))
		ind_new.append(i_new)
	return ind_new

#bramier pg 36-37


class Node:
	def __init__(self, name):
		self.name = name
		self.connections = []
		self.label =''
	def set_connections(self, connections):
		self.connections = []
	def set_name(self, name):
		self.name = name
	def set_label(self, label):
		self.label = label

def effProg(n_calc, prog_long, fb_node = first_body_node): #Wolfgang gave this to me
	n_calc = n_calc + first_body_node
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

			
	
	
	
	


print_individual(best_pop, best_a, best_b)
#draw_graph(best_pop, best_a, best_b)
#draw_graph_active(best_pop, best_a, best_b)
#s = convert_to_graph(best_pop)
#print(s)
p = effProg(4, best_pop)
dot = draw_graph_thicc(p, best_a, best_b)
print(p)
print_individual(p, best_a, best_b)
Path(f"../output/lgp_draw/{func_name}/full_graphs/").mkdir(parents=True, exist_ok=True)
dot.render(f"../output/lgp_draw/{func_name}/full_graphs/graph_{t}", view=False)


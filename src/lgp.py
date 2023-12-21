import numpy as np
import matplotlib.pyplot as plt
from numpy import random, sin, exp, cos, sqrt, pi
from sys import path, argv
from pathlib import Path
from functions import *

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
	if any(x == 0.0):
		return 0
	y = x[0]
	for i in x[1:]:
		y /= i
	return y
def sin_x(x):
	return sin(x[0])
def sin_y(x):
	return sin(x[-1])
def cos_x(x):
	return cos(x[0])
def cos_y(x):
	return cos(x[-1])
def exp_x(x):
	return exp(x[0])
def exp_y(x):
	return exp(x[-1])


t = int(argv[1]) #trials
max_g = int(argv[2]) #generations
max_r = int(argv[3]) #rules
max_d = int(argv[4]) #destinations (other than output)
if max_r < 1:
	print("Number of rules too small, setting to 10")
	max_r = 10
max_p = int(argv[5]) #parents
max_c = int(argv[6]) #children
arity = 2 #sources
n_inp = 1 #number of inputs
bias = np.arange(0, 11, 1)
n_bias = bias.shape[0] #number of bias inputs

p_mut = 1/max_p #mutation probability

func_bank = Collection()
func = func_bank.func_list[int(argv[7])]
func_name = func_bank.name_list[int(argv[7])]
train_x = func.x_dom
train_y = func.y_test

output_index = 0
input_indices = np.arange(1, n_inp+1, 1)
#print(input_indices)

#bank = (add, add)
bank = (first, last, add, sub, mul, div, sin_x, sin_y, cos_x, cos_y, exp_x, exp_y)
bank_string = ('first', 'last', '+', '-', '*', '/', 'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'exp(x)', 'exp(y)')
def rmse(preds, reals):
	return np.sqrt(np.mean((preds-reals)**2)) #copied from stack overflow

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
	possible_sources = possible_indices[1:]
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
	#print(train_y)
	return rmse(preds, train_y)
	
def predict(individual, test, max_d = max_d, n_inp = n_inp, n_bias = n_bias, bias = bias, bank = bank):
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
	return preds
			

def mass_eval(pop, func = func):
	fitnesses = []
	for ind in pop:
		fitnesses.append(run(ind))
	return fitnesses

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
				
				if mutation == 0: #remove instfuction
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
	return contestants[winner]
	
def select(pop, fitnesses, max_p = max_p):
	n_tour = int(len(pop)/10)
	if n_tour <=1:
		n_tour = 2
	new_parents = []
	while len(new_parents) < max_p:
		contestant_indices = random.choice(range(len(pop)), n_tour, replace = False)
		#print(pop)
		contestants = []
		for i in contestant_indices:
			contestants.append(pop[i])
		c_fitnesses = fitnesses[contestant_indices]
		candidate = fight(contestants, c_fitnesses)
		new_parents.append(candidate)
	
	return new_parents

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
for i in range(0, max_p):
	parents.append(generate_ind())
fitnesses = np.zeros((max_p+max_c),)
#sort parents before xover and mutation?
#select before or after xover?
for g in range(1, max_g+1):
	fitnesses[0:max_p] = mass_eval(parents)
	parents = xover(parents)
	#print(f'p[0] shape {parents[0].shape}')
	#parents = clean(parents.copy())
	#print(f'after cleaning: {parents[0].shape}')
	children = mutate(parents.copy())
	#children = clean(children.copy())
	fitnesses[max_p:] = mass_eval(children)
	pop = parents+children
	
	best_i = np.argmin(fitnesses)
	best_pop = pop[i]
	best_fit = fitnesses[i]
	fit_track.append(best_fit)
	
	parents = select(pop, fitnesses)
	
#fig, ax = plt.subplots()
#ax = plt.plot(fit_track)
#print(fit_track)
#plt.show()
	
print(np.round(fitnesses, 5))
print(f"Best Pop:\n{best_pop}")
print(f"Best Fit: {np.round(best_fit, 4)}")
#elif np.isnan(best_fit):
#	print("Redoing Trial")
#	t = t-1
preds = predict(best_pop, train_x)
print('preds')
print(preds)

Path(f"../output/lgp/{func_name}/log/").mkdir(parents=True, exist_ok=True)
import pickle
with open(f"../output/lgp/{func_name}/log/output_{t}.pkl", "wb") as f:
	pickle.dump(bias, f)
	pickle.dump(best_pop, f)
	pickle.dump(preds, f)
	pickle.dump(np.round(best_fit, 4), f)


fig, ax = plt.subplots()
ax.scatter(train_x, train_y, label = 'Ground Truth')
ax.scatter(train_x, preds, label = 'Predicted')
fig.suptitle(f"{func_name} Trial {t}")
ax.set_title(f"RMSE = {np.round(best_fit, 2)}")
ax.legend()
Path(f"../output/lgp/{func_name}/scatter/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/lgp/{func_name}/scatter/comp_{t}.png")

first_body_node = n_inp+n_bias
print(first_body_node)

def print_individual(ind, fb_node = first_body_node):
	def put_r(reg, fb_node = first_body_node):
		if reg == 0:
			return f'R_0'
		return f'R_{int(reg-fb_node)}'
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
			print(n)
			if n > 0 and n<n_inp:
				nums.append(f'I_{n}')
			elif n > n_inp and n < fb_node:
				nums.append(bias[n-n_inp])
			else:
				nums.append(f'R_{n-fb_node+1}')
		
		print(f"{reg}\t{op}\t{nums}")
print_individual(best_pop)


import numpy as np
import warnings
import matplotlib.pyplot as plt
from numpy import random, sin, exp, cos, sqrt, pi
from sys import path, argv
from pathlib import Path
from functions import *
from effProg import *
from similarity import *
from scipy.signal import savgol_filter
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
bias = np.arange(0, 10, 1)
n_bias = bias.shape[0] #number of bias inputs
print("bias nodes")
print(bias)
p_mut = float(argv[9])  #mutation probability
p_xov = float(argv[10]) #xover probability

func_bank = Collection()
func = func_bank.func_list[int(argv[7])]
func_name = func_bank.name_list[int(argv[7])]
train_x = func.x_dom
train_y = func.y_test

output_index = 0
input_indices = np.arange(1, n_inp+1, 1)
print(f'Input Indices {input_indices}')
#print(input_indices)

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
f = int(argv[8])
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
def xover(parents, max_r = max_r, p_xov = p_xov): # 1 point crossover
	children = []
	retention = []
	for i in range(0, len(parents), 2):
		p1 = parents[i].copy()
		p2 = parents[i+1].copy()
		if random.random() > p_xov:
			children.append(p1)
			children.append(p2)
			continue
		retention.append(i)
		inst_counts = [len(p1), len(p2)]
		s = [random.randint(0, p1.shape[0]), random.randint(0, p2.shape[0])]

		p1_list_front = p1[:s[0]].copy()
		p1_list_back = p1[s[0]:].copy()

		p2_list_front = p2[:s[1]].copy()
		p2_list_back = p2[s[1]:].copy()
	
		c1 = np.concatenate((p1_list_front, p2_list_back), axis = 0)
		c2 = np.concatenate((p2_list_front, p1_list_back), axis = 0)
		
		if c1.shape[0] > max_r: #keep to maximimum rule size!
			idxs = np.array(range(0, max_r))
			to_del = random.choice(idxs, ((c1.shape[0]-max_r),), replace=False)
			c1 = np.delete(c1, to_del, axis = 0)
		if c2.shape[0] > max_r:
			idxs = np.array(range(0, max_r))
			to_del = random.choice(idxs, ((c2.shape[0]-max_r),), replace=False)
			c2 = np.delete(c2, to_del, axis = 0)
		children.append(c1)
		children.append(c2)
		#children.append(np.concatenate((p1_list_front, p2_list_back), axis = 0))
		#children.append(np.concatenate((p2_list_front, p1_list_back), axis = 0))
	return children, np.array(retention).astype(np.int32)
	
def mutate(parents, max_c = max_c, p_mut = p_mut, bank = bank):
	children = []
	#while len(children) < max_c:
	for p in range(0, len(parents)): #go through each parent
		if random.random() <= p_mut:
			parent = parents[p]
			if len(parent) < 2:
				mutation = random.randint(1, 3)
			elif len(parent) >= max_r:
				mutation = random.randint(0, 2)
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
		else:
			children.append(parents[p])
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
	#pop.pop(best_fit_id) #always select best individual as a parent
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
		#pop.pop(winner)
	
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
print(f'starting fitnesses')
print(fitnesses)
#print(f'starting scaling')
#print(alignment)

ret_1_1_avg_list = [] #first parent first child
ret_1_2_avg_list = [] #first parent second child
ret_1_1_std_list = []
ret_1_2_std_list = []
ret_2_2_avg_list = [] #second parent second child
ret_2_1_avg_list = [] #second parent first child
ret_2_2_std_list = []
ret_2_1_std_list = []

avg_1_1_change_list = []
avg_1_2_change_list = []
avg_2_1_change_list = []
avg_2_2_change_list = []
std_1_1_change_list = []
std_1_2_change_list = []
std_2_1_change_list = []
std_2_2_change_list = []
best_i = np.argmin(fitnesses[:max_p])
p_size = [len(effProg(4, parents[best_i]))/len(parents[best_i])]

for g in range(1, max_g+1):
	children, retention = xover(parents)
	children = mutate(children.copy())
	fitnesses[max_p:], alignment[max_p:, 0], alignment[max_p:, 1] = mass_eval(children)
	
	pop = parents+children

	if any(np.isnan(fitnesses)): #screen out nan values
		nans = np.isnan(fitnesses)
		fitnesses[nans] = np.PINF	
	
	change_1_1_list = []
	change_1_2_list = []
	change_2_1_list = []
	change_2_2_list = []
	ret_1_1_list = []
	ret_1_2_list = []
	ret_2_1_list = []
	ret_2_2_list = []
	for p in retention:
		change_1_1_list.append(percent_change(fitnesses[p+max_p], fitnesses[p]))
		change_1_2_list.append(percent_change(fitnesses[p+1+max_p], fitnesses[p]))
		change_2_1_list.append(percent_change(fitnesses[p+max_p], fitnesses[p+1]))
		change_2_2_list.append(percent_change(fitnesses[p+1+max_p], fitnesses[p+1]))
		ret_1_1_list.append(find_similarity(pop[p+max_p], pop[p], mode = 'lgp'))
		ret_1_2_list.append(find_similarity(pop[p+max_p+1], pop[p], mode = 'lgp'))
		ret_2_2_list.append(find_similarity(pop[p+1+max_p], pop[p+1], mode = 'lgp'))
		ret_2_1_list.append(find_similarity(pop[p+max_p], pop[p+1], mode = 'lgp'))
	
	avg_1_1_change_list.append(np.nanmean(change_1_1_list))
	avg_1_2_change_list.append(np.nanmean(change_1_2_list))
	avg_2_1_change_list.append(np.nanmean(change_2_1_list))
	avg_2_2_change_list.append(np.nanmean(change_2_2_list))
	std_1_1_change_list.append(np.nanstd(change_1_1_list))
	std_1_2_change_list.append(np.nanstd(change_1_2_list))
	std_2_1_change_list.append(np.nanstd(change_2_1_list))
	std_2_2_change_list.append(np.nanstd(change_2_2_list))

	ret_1_1_avg_list.append(np.nanmean(ret_1_1_list))
	ret_1_2_avg_list.append(np.nanmean(ret_1_2_list))
	ret_1_1_std_list.append(np.nanstd(ret_1_1_list))
	ret_1_2_std_list.append(np.nanstd(ret_1_2_list))
	ret_2_1_avg_list.append(np.nanmean(ret_2_1_list))
	ret_2_2_avg_list.append(np.nanmean(ret_2_2_list))
	ret_2_1_std_list.append(np.nanstd(ret_2_1_list))
	ret_2_2_std_list.append(np.nanstd(ret_2_2_list))

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
	p_size.append(len(effProg(4, parents[best_i]))/len(parents[best_i]))
	if g % 100 == 0:
		print(f'Generation {g}: Best Fit {best_fit}')	
	
print(np.round(fitnesses, 5))
print(f"Best Pop:\n{best_pop}")
print(f"Best Fit: {np.round(best_fit, 4)}")
preds = predict(best_pop, best_a, best_b, train_x)
print('preds')
print(preds)

Path(f"../output/lgp_1x/{func_name}/log/").mkdir(parents=True, exist_ok=True)
import pickle
fig, ax = plt.subplots()
ax.scatter(train_x, train_y, label = 'Ground Truth')
ax.scatter(train_x, preds, label = 'Predicted')
fig.suptitle(f"{func_name} Trial {t}")
ax.set_title(f"{fit_name} = {np.round(best_fit, 4)}")
ax.legend()
Path(f"../output/lgp_1x/{func_name}/scatter/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/lgp_1x/{func_name}/scatter/comp_{t}.png")

fig, ax = plt.subplots()
ax.plot(fit_track)
ax.set_title(f"{func_name} Trial {t}")
Path(f"../output/lgp_1x/{func_name}/plot/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/lgp_1x/{func_name}/plot/plot_{t}.png")

win_length = 100
ret_1_1_avg_list = np.array(ret_1_1_avg_list)
ret_1_2_avg_list = np.array(ret_1_2_avg_list)
ret_2_1_avg_list = np.array(ret_2_1_avg_list)
ret_2_2_avg_list = np.array(ret_2_2_avg_list)
ret_1_1_std_list = np.array(ret_1_1_std_list)
ret_1_2_std_list = np.array(ret_1_2_std_list)
ret_2_1_std_list = np.array(ret_2_1_std_list)
ret_2_2_std_list = np.array(ret_2_2_std_list)
fig, ax = plt.subplots()
ax.plot(savgol_filter(ret_1_1_avg_list, win_length, 3), c = 'blue', label = 'First Parent | First Child')
ax.plot(savgol_filter(ret_1_2_avg_list, win_length, 3), c = 'blue', label = 'First Parent | Second Child', linestyle = 'dashed')
ax.plot(savgol_filter(ret_2_1_avg_list, win_length, 3), c = 'red', label = 'Second Parent | First Child', linestyle='dashed')
ax.plot(savgol_filter(ret_2_2_avg_list, win_length, 3), c = 'red', label = 'Second Parent | Second Child')
ax.fill_between(range(ret_1_1_avg_list.shape[0]), savgol_filter((ret_1_1_avg_list-ret_1_1_std_list), win_length, 3), savgol_filter((ret_1_1_avg_list+ret_1_1_std_list), win_length, 3), color = 'blue', alpha = 0.1)
ax.fill_between(range(ret_1_2_avg_list.shape[0]), savgol_filter((ret_1_2_avg_list-ret_1_2_std_list), win_length, 3), savgol_filter((ret_1_2_avg_list+ret_1_2_std_list), win_length, 3), color = 'blue', alpha = 0.1)
ax.fill_between(range(ret_2_1_avg_list.shape[0]), savgol_filter((ret_2_1_avg_list-ret_2_1_std_list), win_length, 3), savgol_filter((ret_2_1_avg_list+ret_2_1_std_list), win_length, 3), color = 'red', alpha = 0.1)
ax.fill_between(range(ret_2_2_avg_list.shape[0]), savgol_filter((ret_2_2_avg_list-ret_2_2_std_list), win_length, 3), savgol_filter((ret_2_2_avg_list+ret_2_2_std_list), win_length, 3), color = 'red', alpha = 0.1)
ax.set_title(f'{func_name} Trial {t}')
ax.set_ylabel('Average Retention Proportion from Parent to Child after Crossover')
ax.set_xlabel('Generations')
fig.legend()
Path(f"../output/lgp_1x/{func_name}/retention/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/lgp_1x/{func_name}/retention/retention_{t}.png")

avg_1_1_change_list = np.array(avg_1_1_change_list)
avg_1_2_change_list = np.array(avg_1_2_change_list)
avg_2_1_change_list = np.array(avg_2_1_change_list)
avg_2_2_change_list = np.array(avg_2_2_change_list)
std_1_1_change_list = np.array(std_1_1_change_list)
std_1_2_change_list = np.array(std_1_2_change_list)
std_2_1_change_list = np.array(std_2_1_change_list)
std_2_2_change_list = np.array(std_2_2_change_list)
fig, ax = plt.subplots()
ax.plot(savgol_filter(avg_1_1_change_list, win_length, 3), c = 'blue', label = 'First Parent | First Child')
ax.plot(savgol_filter(avg_1_2_change_list, win_length, 3), c = 'blue', label = 'First Parent | Second Child', linestyle='dashed')
ax.plot(savgol_filter(avg_2_1_change_list, win_length, 3), c = 'red', label = 'Second Parent | First Child', linestyle='dashed')
ax.plot(savgol_filter(avg_2_2_change_list, win_length, 3), c = 'red', label = 'Second Parent | Second Child')
ax.fill_between(range(avg_1_1_change_list.shape[0]), savgol_filter((avg_1_1_change_list-std_1_1_change_list), win_length, 3), savgol_filter((avg_1_1_change_list+std_1_1_change_list), win_length, 3), color = 'blue', alpha = 0.1)
ax.fill_between(range(avg_1_2_change_list.shape[0]), savgol_filter((avg_1_2_change_list-std_1_2_change_list), win_length, 3), savgol_filter((avg_1_2_change_list+std_1_2_change_list), win_length, 3), color = 'blue', alpha = 0.1)
ax.fill_between(range(avg_2_1_change_list.shape[0]), savgol_filter((avg_2_1_change_list-std_2_1_change_list), win_length, 3), savgol_filter((avg_2_1_change_list+std_2_1_change_list), win_length, 3), color = 'red', alpha = 0.1)
ax.fill_between(range(avg_2_2_change_list.shape[0]), savgol_filter((avg_2_2_change_list-std_2_2_change_list), win_length, 3), savgol_filter((avg_2_2_change_list+std_2_1_change_list), win_length, 3), color = 'red', alpha = 0.1)
ax.set_title(f'{func_name} Trial {t}')
ax.set_ylabel('Percent Change between parents and children')
ax.set_xlabel('Generations')
fig.legend()
plt.tight_layout()
Path(f"../output/lgp_1x/{func_name}/change_plot/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/lgp_1x/{func_name}/change_plot/change_{t}.png")

p_size = np.array(p_size)

fig, ax = plt.subplots()
ax.plot(p_size)
ax.set_title(f'{func_name} Trial {t}')
ax.set_ylabel("Proportion of Nodes Which Are Active")
ax.set_xlabel("Generations")
Path(f"../output/lgp_1x/{func_name}/proportion_plot/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"../output/lgp_1x/{func_name}/proportion_plot/proportion_plot_{t}.png")


first_body_node = n_inp+n_bias+1 #1 output + 1 input + 10 bias registers = 12 is the first body register
print(first_body_node)

Path(f"../output/lgp_1x/{func_name}/best_program/").mkdir(parents=True, exist_ok=True)
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
	with open(f"../output/lgp_1x/{func_name}/best_program/best_{t}.txt", 'w') as f:
		f.write('R\tOp\tI\n')
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
		with open(f"../output/lgp_1x/{func_name}/best_program/best_{t}.txt", 'a') as f:
			f.write(f"{reg}\t{op}\t{nums}\n")	
		print(f"{reg}\t{op}\t{nums}")
	print("Scaling")
	print(f"R_0 = {a}*R_0+{b}")
	with open(f"../output/lgp_1x/{func_name}/best_program/best_{t}.txt", 'a') as f:
		f.write(f"R_0 = {a}*R_0+{b}\n\n")
		f.write(f'{ind}')   

import graphviz as gv
Path(f"../output/lgp_1x/{func_name}/full_graphs/").mkdir(parents=True, exist_ok=True)


print_individual(best_pop, best_a, best_b)
p = effProg(max_d, best_pop, first_body_node)
with open(f"../output/lgp_1x/{func_name}/best_program/best_{t}.txt", 'a') as f:
	f.write(f"\nEffective Instructions\n\n")  
print(print_individual(p, best_a, best_b))
dot = draw_graph_thicc(p, best_a, best_b)

dot.render(f"../output/lgp_1x/{func_name}/full_graphs/graph_{t}", view=False)


with open(f"../output/lgp_1x/{func_name}/log/output_{t}.pkl", "wb") as f:
	pickle.dump(bias, f)
	pickle.dump(best_pop, f)
	pickle.dump(preds, f)
	pickle.dump(np.round(best_fit, 4), f)
	pickle.dump(len(p), f)
	pickle.dump(fit_track, f)
	pickle.dump(p, f)
	pickle.dump([avg_1_1_change_list, avg_1_2_change_list, avg_2_1_change_list, avg_2_2_change_list], f)
	pickle.dump([ret_1_1_avg_list, ret_1_2_avg_list, ret_2_1_avg_list, ret_2_2_avg_list], f)
	pickle.dump(p_size, f)




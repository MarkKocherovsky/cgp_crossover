import numpy as np
import matplotlib.pyplot as plt
import warnings
from numpy import random, sin, cos, tan, sqrt, exp, log, abs, floor, ceil
from math import log, pi
from sys import path
from pathlib import Path
from functions import *
from effProg import *
from similarity import *
from cgp_selection import *
from cgp_mutation import *
from cgp_xover import *
from cgp_fitness import *
from cgp_operators import *
from cgp_parents import *
from cgp_impact import *
from sys import argv
from math import isnan
from copy import deepcopy
from collections import Counter
from selection_impact import *
warnings.filterwarnings('ignore')

print("started")
t = int(argv[1]) #trial
print(f'trial {t}')
max_g = int(argv[2]) #max generations
print(f'generations {max_g}')
max_n = int(argv[3]) #max body nodes
print(f'max body nodes {max_n}')
max_c = int(argv[4]) #max children
print(f'children {max_c}')
outputs = 1
inputs = 1
biases = np.arange(0, 10, 1).astype(np.int32)
bias = biases.shape[0] #number of biases
print(f'biases {biases}')
arity = 2
random.seed(t+50)
print(f'Seed = {sqrt(t)}')


bank = (add, sub, mul, div) #, cos_x, cos_y, sin_x, sin_y, powe, sqrt_x_y, distance, abs_x, abs_y, midpoint)
bank_string = ("+", "-", "*", "/") #, "cos(x)","cos(y)", "sin(x)", "sin(y)", "^", "$\sqrt{x+y}$", "$sqrt{x^2+y^2}$", "|x|", "|y|", "avg")

func_bank = Collection()
func = func_bank.func_list[int(argv[5])]
func_name = func_bank.name_list[int(argv[5])]

train_x = func.x_dom
train_y = func.y_test
#print(train_x)

f = int(argv[6])
fits = FitCollection()
fit = fits.fit_list[f]
fit_name  = fits.name_list[f]
print(fit)
print(fit_name)

#No Crossover!
mutate = mutate_1_plus_4

final_fit = []
fit_track = []
ind_base = np.zeros(((arity+1)*max_n,), np.int32)
ind_base = ind_base.reshape(-1, arity+1) #for my sanity - Mark
train_x_bias = np.zeros((train_x.shape[0], biases.shape[0]+1)) #attach constants to input
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = biases
print("instantiating parent")
#instantiate parent
parent = generate_parents(1, max_n, bank, first_body_node = 11, outputs = 1, arity = 2)

fitness = Fitness()
p_fit, p_A, p_B = fitness(train_x_bias, train_y, parent)

f_change = np.zeros((max_c,)) # % difference from p_fit
avg_change_list = []
avg_hist_list = []
std_change_list = []
p_size = [cgp_active_nodes(parent[0], parent[1], opt = 2)]#/ind_base.shape[0]]
ret_avg_list = []
ret_std_list = []
fitness_objects = [Fitness() for i in range(0, max_c)]

drift_list = []
#deleterious, drift, beneficial
drift_cum = np.array([0, 0, 0])

#N1 = 1
#N2 = max_c
#P = 1
#s_impact = SelectionImpact(N1, N2, P)
#impact_list = []

for g in range(1, max_g+1):
	children = [mutate(deepcopy(parent)) for x in range(0, max_c)]
	#print(children)
	c_fit = np.array([fitness_objects[x](train_x_bias, train_y, child) for child, x in zip(children, list(range(0, max_c)))])
	best_child_index = np.argmin(c_fit[:, 0])
	best_c_fit = c_fit[best_child_index, 0]
	best_child = children[best_child_index]
	avg_change_list.append(percent_change(best_c_fit, p_fit))
	change_list = np.array([percent_change(c, p_fit) for c in c_fit[:, 0]])
	change_list = change_list[np.isfinite(change_list)]
	cl_std = np.nanstd(change_list)
	if not all(cl == 0.0 for cl in change_list):
		avg_hist_list.append((g, np.histogram(change_list, bins = 5, range=(cl_std*-2, cl_std*2))))
	ret_avg_list.append(find_similarity(best_child[0], parent[0], best_child[1], parent[1], mode = 'cgp', method = 'distance'))
	ret_std_list.append(0.0)
	std_change_list.append(0)
	a = c_fit[:, 1].copy().flatten()
	b = c_fit[:, 2].copy().flatten()
	c_fit = c_fit[:, 0].flatten()
	#print(p_fit)
	#print(c_fit)
	if any(np.isnan(c_fit)): #Replace nans with positive infinity to screen them out
		nans = np.isnan(c_fit)
		c_fit[nans] = np.PINF 
	drift = np.array([0, 0, 0])
	for c in c_fit:
		if change(c, p_fit) > 0.1:
			drift[0] +=1
		elif change(c, p_fit) < -0.1:
			drift[2] += 1
		else:
			drift[1] += 1
	drift_cum += np.copy(drift)
	drift_list.append(np.copy(drift_cum))
	#parent_distro = np.zeros((N1,))
	if any(c_fit <= p_fit):
		best = np.argmin(c_fit)
		parent = deepcopy(children[best])
		p_fi = np.argmin(c_fit)
		#parent_distro[0] += 1
		p_fit = np.min(c_fit)
		p_A = a[p_fi]
		p_B = b[p_fi]
	#selection_impact = s_impact(parent_distro)
	#impact_list.append(selection_impact)
	if g % 100 == 0:
		print(f"Gen {g} Best Fitness: {p_fit}")
	#print(p_fit)
	fit_track.append(p_fit)
	p_size.append(cgp_active_nodes(parent[0], parent[1], opt = 2))#/ind_base.shape[0])
	#if(p_fit > 0.96):
	#	break
avg_change_list = np.array(avg_change_list)
std_change_list = np.array(std_change_list)
p_size = np.array(p_size)
print(cgp_active_nodes(parent[0], parent[1], opt = 1))
print(cgp_active_nodes(parent[0], parent[1]))
print(f"Trial {t}: Best Fitness = {p_fit}")
print(f"Mutations:\tDeleterious\tNeutral\tBeneficial")
print(f"\t{drift_cum[0]}\t{drift_cum[1]}\t{drift_cum[2]}")
print('biases')
print(biases)
print('best individual')
print(parent)
print('preds')
preds, p_A, p_b = fitness(train_x_bias, train_y, parent, opt = 1)
print(preds)
print(fitness(train_x_bias, train_y, parent, opt = 0))

Path(f"../output/cgp/{func_name}/log/").mkdir(parents=True, exist_ok=True)
import pickle

#Write Plots
from scipy.signal import savgol_filter
from cgp_plots import *
run_name = 'cgp'
scatter(train_x, train_y, preds, func_name, run_name, t, fit_name, p_fit)
fit_plot(fit_track, func_name, run_name, t)
proportion_plot(p_size, func_name, run_name, t)
bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g)
change_avg_plot(avg_change_list, std_change_list, func_name, run_name, t, win_length = 100, order = 4)
retention_plot(ret_avg_list, ret_std_list, func_name, run_name, t, win_length = 100, order = 2)
drift_list_sum = np.sum(drift_list, axis = 1).flatten()
drift_list = np.divide(drift_list, drift_list_sum[:, np.newaxis])
drift_plot(drift_list, drift_cum / np.sum(drift_cum), func_name, run_name, t)
#impact_plot(impact_list, func_name, run_name, t)
#export graph
first_body_node = inputs+bias
cgp_graph(inputs, bias, parent[0], parent[1], p_A, p_B, func_name, run_name, t,  max_n = max_n, first_body_node = first_body_node, arity = arity)

#active nodes only

n = plot_active_nodes(parent[0], parent[1], first_body_node, bank_string, biases, inputs, p_A, p_B, func_name, run_name, t)
print(f'Active Nodes = {n}')
print(f"../output/cgp/{func_name}/log/output_{t}.pkl")
with open(f"../output/cgp/{func_name}/log/output_{t}.pkl", "wb") as f:
	pickle.dump(biases, f)
	pickle.dump(parent[0], f)
	pickle.dump(parent[1], f)
	pickle.dump(preds, f)
	pickle.dump(p_fit, f)
	pickle.dump(n, f)
	pickle.dump(fit_track, f)
	pickle.dump([avg_change_list], f)
	#pickle.dump(std_change_list, f)
	pickle.dump([ret_avg_list], f)
	pickle.dump(p_size, f)
	pickle.dump([bin_centers, hist_gens, avg_hist_list], f)
	pickle.dump(drift_list, f)
	pickle.dump(drift_cum, f)
	#pickle.dump(impact_list, f)
	#pickle.dump(e, f)
#expressions = get_expression()
#for expression in expressions:
#	print(expression)

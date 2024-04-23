#CGP 2 Point Crossover
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
from copy import deepcopy
from scipy.signal import savgol_filter
from sys import argv
import selection_methods
warnings.filterwarnings('ignore')
print("started")
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
p_mut = float(argv[8])
p_xov = float(argv[9])

bank = (add, sub, mul, div) #, cos_x, cos_y, sin_x, sin_y, powe, sqrt_x_y, distance, abs_x, abs_y, midpoint)
bank_string = ("+", "-", "*", "/") #, "cos(x)","cos(y)", "sin(x)", "sin(y)", "^", "$\sqrt{x+y}$", "$sqrt{x^2+y^2}$", "|x|", "|y|", "avg")

func_bank = Collection()
func = func_bank.func_list[int(argv[6])]
func_name = func_bank.name_list[int(argv[6])]
train_x = func.x_dom
train_y = func.y_test
print(train_x)

f = int(argv[7])
fits = FitCollection()
fit = fits.fit_list[f]
print(f)
print(fits.fit_list)
fit_name  = fits.name_list[f]
print('Fitness Function')
print(fit)
print(fit_name)

alignment = np.zeros((max_p+max_c, 2))
alignment[:, 0] = 1.0

train_x_bias = np.zeros((train_x.shape[0], biases.shape[0]+1))
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = biases
print(train_x_bias)

mutate = basic_mutation
select = selection_methods.roulette_wheel # IMPORTANT
parents = generate_parents(max_p, max_n, bank, first_body_node = 11, outputs = 1, arity = 2)

fitness_objects = [Fitness() for i in range(0, max_p+max_c)]
fitnesses = np.zeros((max_p+max_c),)
fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, parent) for i, parent in zip(range(0, max_p), parents)])
#print(*zip(range(0, max_p), parents))
fitnesses[:max_p] = fit_temp[:, 0].copy().flatten()
alignment[:max_p, 0] = fit_temp[:, 1].copy() #a
alignment[:max_p, 1] = fit_temp[:, 2].copy() #b
print(np.round(fitnesses, 4))

fit_track = []
ret_avg_list = [] #best parent best child
ret_std_list = []

avg_change_list = [] #best parent best child
avg_hist_list = []
std_change_list = []

best_i = np.argmin(fitnesses[:max_p])
p_size = [cgp_active_nodes(parents[best_i][0], parents[best_i][1], opt = 2)]

for g in range(1, max_g+1):
	children, retention = xover(deepcopy(parents), method = 'TwoPoint') 
	children = mutate(deepcopy(children))
	pop = parents+children
	fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(list(range(0, max_p+max_c)), pop)])
	
	# IMPORTANT LEXICASE
	# testcase_scores = np.array([fitness_objects[i].testcases(train_x_bias, train_y, ind) for i, ind in zip(list(range(0, max_p+max_c)), pop)])

	fitnesses = fit_temp[:, 0].copy().flatten()
	alignment = fit_temp[:, 1].copy()
	alignment = fit_temp[:, 2].copy()
	if any(np.isnan(fitnesses)): #Replace nans with positive infinity to screen them out
		nans = np.isnan(fitnesses)
		fitnesses[nans] = np.PINF

	change_list = []
	full_change_list = []
	ret_list = []
	for p in retention:
		ps = [pop[p], pop[p+1]]
		p_fits = np.array([fitnesses[p],fitnesses[p+1]])
		cs = [pop[p+max_p], pop[p+max_p+1]]
		c_fits = np.array([fitnesses[p+max_p], fitnesses[p+max_p+1]])
		best_p = np.argmin(p_fits)
		best_c = np.argmin(c_fits)

		change_list.append(percent_change(c_fits[best_c], p_fits[best_p]))
		ret_list.append(find_similarity(cs[best_c][0], ps[best_p][0], cs[best_c][1], ps[best_p][1], 'cgp'))
		
		full_change_list.append([percent_change(c, best_p) for c in c_fits])
	full_change_list = np.array(full_change_list).flatten()
	full_change_list = full_change_list[np.isfinite(full_change_list)]
	cl_std = np.nanstd(full_change_list)
	if not all(cl == 0.0 for cl in full_change_list):
		avg_hist_list.append((g, np.histogram(full_change_list, bins = 10, range=(cl_std*-2, cl_std*2))))
	avg_change_list.append(np.nanmean(change_list))
	std_change_list.append(np.nanstd(change_list))
	ret_avg_list.append(np.nanmean(ret_list))
	ret_std_list.append(np.nanstd(ret_list))	

	best_i = np.argmin(fitnesses)
	best_fit = fitnesses[best_i]
	if g % 100 == 0:
		print(f"Gen {g} Best Fitness: {best_fit}")
	fit_track.append(best_fit)
	p_size.append(cgp_active_nodes(pop[best_i][0], pop[best_i][1], opt = 2))
	parents = select(pop, fitnesses, max_p)
	# parents = select(pop, testcase_scores, max_p) # IMPORTANT LEXICASE

pop = parents+children
fit_temp =  np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(range(0, max_p+max_c), pop)])
fitnesses = fit_temp[:, 0].copy().flatten()
best_i = np.argmin(fitnesses)
best_fit = fitnesses[best_i]
best_pop = pop[best_i]
print(f"Trial {t}: Best Fitness = {best_fit}")
print('best individual')
print(pop[best_i])
print('preds')
pred_fitness = Fitness()
preds, p_A, p_B = pred_fitness(train_x_bias, train_y, best_pop, opt = 1)
print(preds)
print(pred_fitness(train_x_bias, train_y, best_pop, opt = 0))

#print(list(train_y))
Path(f"../output/cgp_2x/{func_name}/log/").mkdir(parents=True, exist_ok=True)
import pickle

win_length = 100
#Write Plots
from scipy.signal import savgol_filter
from cgp_plots import *
run_name = 'cgp_2x'
scatter(train_x, train_y, preds, func_name, run_name, t, fit_name, best_fit)
fit_plot(fit_track, func_name, run_name, t)
proportion_plot(p_size, func_name, run_name, t)
bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g)
change_avg_plot(avg_change_list, std_change_list, func_name, run_name, t, win_length = 100, order = 4)
retention_plot(ret_avg_list, ret_std_list, func_name, run_name, t, win_length = 100, order = 2)

#export graph
first_body_node = inputs+bias
cgp_graph(inputs, bias, best_pop[0], best_pop[1], p_A, p_B, func_name, run_name, t,  max_n = max_n, first_body_node = first_body_node, arity = arity)

#active nodes only

n = plot_active_nodes(best_pop[0], best_pop[1], first_body_node, bank_string, biases, inputs, p_A, p_B, func_name, run_name, t)

print(f'Active Nodes = {n}')
print(f"../output/cgp_2x/{func_name}/log/output_{t}.pkl")
with open(f"../output/cgp_2x/{func_name}/log/output_{t}.pkl", "wb") as f:
	pickle.dump(biases, f)
	pickle.dump(best_pop[0], f)
	pickle.dump(best_pop[1], f)
	pickle.dump(preds, f)
	pickle.dump(best_fit, f)
	pickle.dump(n, f)
	pickle.dump(fit_track, f)
	pickle.dump([avg_change_list], f)
	pickle.dump([ret_avg_list], f)
	pickle.dump(p_size, f)
	pickle.dump([bin_centers, hist_gens, avg_hist_list], f)

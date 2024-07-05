#CGP 16+64 No Crossover
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
from cgp_plots import *
from sharpness import *
from cgp_parents import *
from copy import deepcopy
from scipy.signal import savgol_filter
from sys import argv
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
run_name = "cgp_40"
num_elites = 7
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
#print(train_x_bias)
#print(inputs+bias+max_n)
#print("instantiating parent")
#instantiate parents
#test = run_output(ind_base, output_nodes, np.array([10.0]))

mutate = mutate_1_plus_4
select = tournament_elitism
parents = generate_parents(max_p, max_n, bank, first_body_node = 11, outputs = 1, arity = 2)

fitness_objects = [Fitness() for i in range(0, max_p+max_c*max_p)]
fitnesses = np.zeros((max_p+max_c*max_p),)
fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, parent) for i, parent in zip(range(0, max_p), parents)])
#print(*zip(range(0, max_p), parents))
fitnesses[:max_p] = fit_temp[:, 0].copy().flatten()
alignment[:max_p, 0] = fit_temp[:, 1].copy() #a
alignment[:max_p, 1] = fit_temp[:, 2].copy() #b
print(np.round(fitnesses, 4))

sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()

def getNoise(shape, pop_size = max_p+max_p*max_c, inputs = inputs, func = func, opt = 0):
	x = []
	y = []
	if opt == 1:
		fixed_inputs = sharp_in_manager.perturb_data()[:, :inputs]
	for p in range(pop_size):
		noisy_x = np.zeros((shape))
		if opt == 1:
			noisy_x[:, :inputs] = deepcopy(fixed_inputs)
		else:
			noisy_x[:, :inputs] = sharp_in_manager.perturb_data()[:, :inputs]
		noisy_x[:, inputs:] = sharp_in_manager.perturb_constants()[:, inputs:]
		noisy_y = np.fromiter(map(func.func, list(noisy_x[:, :inputs].flatten())), dtype=np.float32)
		x.append(noisy_x)
		y.append(noisy_y)
	return np.array(x), np.array(y)

#SAM-IN
noisy_x, noisy_y = getNoise(train_x_bias.shape)
sharpness = np.array([fitness_objects[i](noisy_x[i], noisy_y[i], parent)[0] for i, parent in zip(range(0, max_p), parents)])
sharp_in_list = [np.mean(sharpness)]
sharp_in_std = [np.std(sharpness)]

#SAM-OUT

preds = [fitness_objects[i](train_x_bias, train_y, parent, opt = 1)[0] for i, parent in zip(range(0, max_p), parents)]
sharp_out_manager = SAM_OUT()
def get_neighbor_map(preds, sharp_out_manager, fitness, train_y = train_y):
	neighborhood = sharp_out_manager.perturb(preds)
	return [fitness.fit(neighbor, train_y) for neighbor in neighborhood]
neighbor_map = np.array([get_neighbor_map(pred, sharp_out_manager, fitness_objects[i]) for i, pred in zip(range(0, max_p), preds)])
print(neighbor_map.shape)
sharp_out_list = [np.mean(np.std(neighbor_map, axis = 1)**2)] #variance
sharp_out_std = [np.std(np.std(neighbor_map, axis = 1))]

print(np.round(sharpness, 4))
print(np.round(np.std(neighbor_map, axis = 1)**2, 4))

fit_track = []
ret_avg_list = [] #best parent best child
ret_std_list = []

avg_change_list = [] #best parent best child
avg_hist_list = []
std_change_list = []

best_i = np.argmin(fitnesses[:max_p])
p_size = [cgp_active_nodes(parents[best_i][0], parents[best_i][1], opt = 2)]

mut_impact = MutationImpact(neutral_limit = 0.1)

for g in range(1, max_g+1):
	children = [mutate(deepcopy(parent)) for parent in parents for _ in range(0, max_c)]
	pop = parents+children
	fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(list(range(0, max_p+max_c*max_p)), pop)])

	#fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(list(range(0, max_p+max_c)), pop)])
	fitnesses = fit_temp[:, 0].copy().flatten()
	#print(f"Fitnesses after mutation")
	#print(np.round(fitnesses, 4))
	alignment = fit_temp[:, 1].copy()
	alignment = fit_temp[:, 2].copy()
	if any(np.isnan(fitnesses)): #Replace nans with positive infinity to screen them out
		nans = np.isnan(fitnesses)
		fitnesses[nans] = np.PINF 
	mut_impact(fitnesses, max_p, option = 'OneParent', children = 4)
	ret = []
	chg = []
	full_change_list = []
	for i in range(0, len(parents)):
		best_p = parents[i]
		best_p_fit = fitnesses[i]
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
	noisy_x, noisy_y = getNoise(train_x_bias.shape, opt = 1)
	sharpness = np.array([fitness_objects[i](noisy_x[i], noisy_y[i], individual)[0] for i, individual in zip(range(0, max_p+max_p*max_c), pop)])
	sharp_in_list.append(np.mean(sharpness))
	sharp_in_std.append(np.std(sharpness))

	#SAM-OUT

	preds = [fitness_objects[i](train_x_bias, train_y, individual, opt = 1)[0] for i, individual in zip(range(0, max_p+max_p*max_c), pop)]
	neighbor_map = np.array([get_neighbor_map(pred, sharp_out_manager, fitness_objects[i]) for i, pred in zip(range(0, max_p+max_p*max_c), preds)])
	out_sharpness = np.std(neighbor_map, axis = 1)**2
	sharp_out_list.append(np.mean(out_sharpness)) #variance
	sharp_out_std.append(np.std(out_sharpness))

	best_i = np.argmin(fitnesses)
	best_fit = fitnesses[best_i]
	#print(f'best_fit {best_fit}')
	fit_track.append(best_fit)
	p_size.append(cgp_active_nodes(pop[best_i][0], pop[best_i][1], opt = 2))
	if g % 100 == 0:
		print(f"Gen {g} Best Fitness: {best_fit}\tMean SAM-In: {sharp_in_list[-1]}\tMean SAM-Out: {sharp_out_list[-1]}")
		#sort sharpness
		indices = np.argsort(fitnesses)[:num_elites]
		elites = [pop[i] for i in indices]
		sharpness = np.array(sharpness)
		out_sharpness = np.array(out_sharpness)
		in_sharp = sharpness[indices]
		out_sharp = out_sharpness[indices]
		#sharp_bar_plot(in_sharp, out_sharp, func_name, run_name, t, g)
		#scatter_elites(elites, func, func_name, run_name, t, g, fit_name, best_fit)
	parents = select(pop, fitnesses, max_p)
	#print("Fitnesses at end of generation, should not have changed")
	#print(np.round(fitnesses, 4))
	#print('----')

pop = parents+children
fit_temp =  np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(range(0, max_p+max_c), pop)])
fitnesses = fit_temp[:, 0].copy().flatten()
best_i = np.argmin(fitnesses)
best_fit = fitnesses[best_i]
best_pop = pop[best_i]
print(f"Trial {t}: Best Fitness = {best_fit}")
drift_cum, drift_list = mut_impact.return_lists(option = 1)
print(f"Operators:\tDeleterious\tNeutral\tBeneficial")
print(f"\t{drift_cum[0]}\t{drift_cum[1]}\t{drift_cum[2]}")
print(pop[best_i])
print('preds')
pred_fitness = Fitness()
preds, p_A, p_B = pred_fitness(train_x_bias, train_y, best_pop, opt = 1)
print(preds)
print(pred_fitness(train_x_bias, train_y, best_pop, opt = 0))

#print(list(train_y))
Path(f"../output/cgp_40/{func_name}/log/").mkdir(parents=True, exist_ok=True)
import pickle

win_length = 100
#Write Plots
from scipy.signal import savgol_filter
from cgp_plots import *
run_name = 'cgp_40'

"""
scatter(train_x, train_y, preds, func_name, run_name, t, fit_name, best_fit)
fit_plot(fit_track, func_name, run_name, t)
proportion_plot(p_size, func_name, run_name, t)
bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g)
change_avg_plot(avg_change_list, std_change_list, func_name, run_name, t, win_length = 100, order = 4)
retention_plot(ret_avg_list, ret_std_list, func_name, run_name, t, win_length = 100, order = 2)
drift_plot(drift_list, drift_cum, func_name, run_name, t, win_length = 100)
indices = np.argsort(fitnesses)[:num_elites]
elites = [pop[i] for i in indices]
sharpness = np.array(sharpness)
out_sharpness = np.array(out_sharpness)
in_sharp = sharpness[indices]
out_sharp = out_sharpness[indices]

sharp_bar_plot(in_sharp, out_sharp, func_name, run_name, t)
sharp_plot(sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std, func_name, run_name, t)
#export graph
first_body_node = inputs+bias
cgp_graph(inputs, bias, best_pop[0], best_pop[1], p_A, p_B, func_name, run_name, t,  max_n = max_n, first_body_node = first_body_node, arity = arity)

#active nodes only


print(f'Active Nodes = {n}')
"""
first_body_node = inputs+bias
print(f"../output/{run_name}/{func_name}/log/output_{t}.pkl")
n = plot_active_nodes(best_pop[0], best_pop[1], first_body_node, bank_string, biases, inputs, p_A, p_B, func_name, run_name, t)
with open(f"../output/{run_name}/{func_name}/log/output_{t}.pkl", "wb") as f:
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
	pickle.dump(drift_list, f)
	pickle.dump(drift_cum, f)
	pickle.dump([sharp_in_list, sharp_out_list], f)
	pickle.dump([sharp_in_std, sharp_out_std], f)
#expressions = get_expression()
#for expression in expressions:
#	print(expression)

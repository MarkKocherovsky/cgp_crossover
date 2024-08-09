import numpy as np
import warnings
import matplotlib.pyplot as plt
from numpy import random, sin, exp, cos, sqrt, pi
from sys import path, argv
from pathlib import Path
from copy import deepcopy
from functions import *
from effProg import *
from similarity import *
from lgp_parents import *
from lgp_fitness import *
from lgp_functions import *
from lgp_xover import *
from lgp_mutation import *
from lgp_select import *
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')


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

try:
	p_mut = float(argv[9])
except:
	p_mut = 1.0
try:
	p_xov = float(argv[10])
except:
	p_xov = 1.0

func_bank = Collection()
func = func_bank.func_list[int(argv[7])]
func_name = func_bank.name_list[int(argv[7])]
train_x = func.x_dom
train_y = func.y_test

f = int(argv[8])
fits = FitCollection()
fit = fits.fit_list[f]
print(f)
print(fits.fit_list)
fit_name  = fits.name_list[f]
print('Fitness Function')
print(fit)
print(fit_name)

output_index = 0
input_indices = np.arange(1, n_inp+1, 1)
#print(input_indices)
Path(f"../output/lgp_mut/{func_name}/best_program/").mkdir(parents=True, exist_ok=True)
with open(f"../output/lgp_mut/{func_name}/best_program/best_{t}.txt", 'w') as f:
	f.write(f"Problem {func_name}\n")
	f.write(f'Trial {t}\n')
	f.write(f'----------\n\n')

bank = get_functions()
bank_string = ("+", "-", "*", "/") #, "cos(x)","cos(y)", "sin(x)", "sin(y)", "^", "$\sqrt{x+y}$", "$sqrt{x^2+y^2}$", "|x|", "|y|", "avg")

mutate = macromicro_mutation

select = lgp_tournament_elitism_selection
n_tour = 4
print(f"#####Trial {t}#####")
fit_track = []
alignment = np.zeros((max_c, 2))
alignment[:, 0] = 1.0

parent_generator = lgpParentGenerator(max_p, max_r, max_d, bank, n_inp, n_bias, arity)
parent = parent_generator()
if max_p <=1:
	parent = np.array([parent])
c_fits = np.zeros((max_c),)
fitness_evaluator = Fitness(train_x, bias, train_y, parent, func, bank, n_inp, max_d, fit, arity)
p_fit, p_A, p_B = fitness_evaluator()
print(f'starting fitnesses')
print(p_fit)
#print(f'starting scaling')
#print(alignment)
ret_avg_list = [] #best parent best child
ret_std_list = []

avg_change_list = [] #best parent best child
avg_hist_list = []
std_change_list = []
print(parent)
print(parent.shape)
p_size = [len(effProg(4, parent[0]))/len(parent[0])]

for g in range(1, max_g+1):
	#children, retention = xover(deepcopy(parents), max_r, p_xov, 'uniform')
	retention = np.arange(0, max_p/2+1, 1).astype(np.int32)
	children = lgp_1pc_mut(deepcopy(parent), max_c, max_r, max_d, bank, inputs = 1, n_bias = 10, arity = 2, p_mut = 1.0)
	#print(len(pop))
	#print(children.shape)
	#print([child.shape for child in children])
	children = [child[0] for child in children]
	#print([child.shape for child in children])
	fitness_evaluator = Fitness(train_x, bias, train_y, children, func, bank, n_inp, max_d, fit, arity)
	c_fits, alignment[:, 0], alignment[:, 1] = fitness_evaluator()

	if any(np.isnan(c_fits)): #screen out nan values
		nans = np.isnan(fitnesses)
		c_fits[nans] = np.PINF
	
	best_c_i = np.argmin(c_fits)
	best_c = children[best_c_i]
	best_c_f = c_fits[best_c_i]
	best_c_A = alignment[best_c_i, 0]
	best_c_B = alignment[best_c_i, 1]
	change_list = []
	full_change_list = []
	ret_list = []
	change_list.append(percent_change(best_c_f, p_fit))
	ret_list.append(find_similarity(best_c, parent[0], [], [], 'lgp'))
	
	full_change_list.append([percent_change(c, p_fit) for c in c_fits])
	full_change_list = np.array(full_change_list).flatten()
	full_change_list = full_change_list[np.isfinite(full_change_list)]
	cl_std = np.nanstd(full_change_list)
	if not all(cl == 0.0 for cl in full_change_list):
		avg_hist_list.append((g, np.histogram(full_change_list, bins = 10, range=(cl_std*-2, cl_std*2))))
	avg_change_list.append(np.nanmean(change_list))
	std_change_list.append(np.nanstd(change_list))
	ret_avg_list.append(np.nanmean(ret_list))
	ret_std_list.append(np.nanstd(ret_list))	
	if best_c_f <= p_fit:
		parent = np.array([deepcopy(best_c)])
		p_fit = deepcopy(best_c_f)
	fit_track.append(p_fit)
	p_size.append(len(effProg(4, parent[0]))/len(parent[0]))
	#parents = select(pop, fitnesses, max_p, n_tour)

	if g % 100 == 0:
		print(f'Generation {g}: Best Fit {p_fit}')

fitness_evaluator = Fitness(train_x, bias, train_y, parent, func, bank, n_inp, max_d, fit, arity)
p_fit, p_A, p_B = fitness_evaluator()
print(f"Trial {t}: Best Fitness = {p_fit}")
#final_fit.append(p_fit)
#fig, ax = plt.subplots()
#ax = plt.plot(fit_track)
#print(fit_track)
#plt.show()
#print(final_fit)
#print('biases')
#print(biases)
print('best individual')
print(parent)
print('preds')
print(parent.shape)
preds = fitness_evaluator.predict(parent[0], p_A, p_B, train_x)
print(preds)

print(f"../output/lgp_mut/{func_name}/log/output_{t}") 
Path(f"../output/lgp_mut/{func_name}/log/").mkdir(parents=True, exist_ok=True)
import pickle
from cgp_plots import *

first_body_node = n_inp+n_bias+1
print(first_body_node)
run_name = 'lgp_mut'
win_length = 100
#Write Plots
from scipy.signal import savgol_filter
from cgp_plots import *
scatter(train_x, train_y, preds, func_name, run_name, t, fit_name, p_fit)
fit_plot(fit_track, func_name, run_name, t)
proportion_plot(p_size, func_name, run_name, t)
bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g)
change_avg_plot(avg_change_list, std_change_list, func_name, run_name, t, win_length = 100, order = 4)
retention_plot(ret_avg_list, ret_std_list, func_name, run_name, t, win_length = 100, order = 2)

import graphviz as gv
p = effProg(max_d, parent[0], first_body_node)
lgp_print_individual(p, p_A, p_B, 'lgp', func_name, bank_string, t, bias, n_inp, first_body_node)
with open(f"../output/lgp_mut/{func_name}/best_program/best_{t}.txt", 'a') as f:
	f.write(f"\nEffective Instructions\n\n")  
	f.write(f'{p}')
print('effective program')
print(p)
with open(f"../output/lgp_mut/{func_name}/log/output_{t}.pkl", "wb") as f:
	pickle.dump(bias, f)
	pickle.dump(parent, f)
	pickle.dump(preds, f)
	pickle.dump(np.round(p_fit, 4), f)
	pickle.dump(len(p), f)
	pickle.dump(fit_track, f)
	pickle.dump([avg_change_list], f)
	pickle.dump([ret_avg_list], f)
	pickle.dump(p_size, f)
	pickle.dump([bin_centers, hist_gens, avg_hist_list], f)
	pickle.dump(p_size, f)

dot = draw_graph_thicc(p, p_A, p_B)
Path(f"../output/lgp_mut/{func_name}/full_graphs/").mkdir(parents=True, exist_ok=True)
dot.render(f"../output/lgp_mut/{func_name}/full_graphs/graph_{t}", view=False)

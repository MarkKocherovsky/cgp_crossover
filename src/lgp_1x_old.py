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
from lgp_functions import *
from lgp_xover import *
from lgp_mutation import *
from lgp_select import *
from sharpness import *
from cgp_plots import *
from scipy.stats import skew, kurtosis
from cgp_fitness import MutationImpact
from lgp_fitness import *
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')

print(argv)
t = int(argv[1]) #trials
print(f"Trial {t}")
max_g = int(argv[2]) #generations
print(f'Generations {max_g}')
max_r = int(argv[3]) #rules
print(f'# of instructions {max_r}')
try:
	fixed_length = True if int(argv[12]) == 1 else False
except:
	fixed_length = False
print(f'Fixed Length? {fixed_length}')
max_d = int(argv[4]) #destinations (other than output)
print(f'Calculation Registerts {max_d}')
if max_r < 1:
	print("Number of rules too small, setting to 10")
	max_r = 10
max_p = int(argv[5]) #parents
max_c = int(argv[6]) #children
arity = 2 #sources
n_inp = 1 #number of inputs
bias = np.arange(0, 10, 1)
n_bias = bias.shape[0] #number of bias inputs
random.seed(t+350)
try:
	p_mut = float(argv[9])
except:
	p_mut = 1/max_p #mutation probability
try:
	p_xov = float(argv[10])
except:
	p_xov = 0.5

try:
	run_name = f'lgp_1x{str(argv[11])}'
	print(run_name)
except:
	run_name = 'lgp_1x'

#run_name = 'lgp_1x'
num_elites = 7

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
Path(f"../output/{run_name}/{func_name}/log").mkdir(parents=True, exist_ok=True)
Path(f"../output/{run_name}/{func_name}/best_program").mkdir(parents=True, exist_ok=True)
with open(f"../output/{run_name}/{func_name}/best_program/best_{t}.txt", 'w') as f:
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
alignment = np.zeros((max_p+max_c, 2))
alignment[:, 0] = 1.0

parent_generator = lgpParentGenerator(max_p, max_r, max_d, bank, n_inp, n_bias, arity, fixed_length)
parents = parent_generator()

train_x_bias = np.zeros((train_x.shape[0], bias.shape[0]+1))
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = bias

density_distro = np.zeros((max_r,), dtype = np.int32)


fitnesses = np.zeros((max_p+max_c),)
fitness_evaluator = Fitness(train_x, bias, train_y, parents, func, bank, n_inp, max_d, fit, arity) 
fitnesses[:max_p], alignment[:max_p, 0], alignment[:max_p, 1] = fitness_evaluator()
print(f'starting fitnesses')
print(fitnesses)

sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()

def getNoise(shape, pop_size = max_p+max_c, inputs = n_inp, func = func, opt = 0):
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

def get_fitness_evaluators(x, b, y, pr):
	return [Fitness(x[i], b[i], y[i], [p], func, bank, n_inp, max_d, fit, arity) for p, i in zip(pr, range(len(pr)))]

def get_sam_in(noisy_x, noisy_y, pop):
	sharpness_evaluators = get_fitness_evaluators(noisy_x[:, :, :n_inp], noisy_x[:, :, n_inp:], noisy_y, pop)
	sharpness = np.array([evaluator()[0] for evaluator in sharpness_evaluators]).flatten()
	return sharpness
def get_neighbor_map(preds, sharp_out_manager, parent, train_y = train_y):
	neighborhood = sharp_out_manager.perturb(preds)
	return [corr(neighbor, train_y) for neighbor in neighborhood]


noisy_x, noisy_y = getNoise(train_x_bias.shape)
sharpness = get_sam_in(noisy_x, noisy_y, parents) #np.array([fitness_objects[i](noisy_x[i], noisy_y[i], parent)[0] for i, parent in zip(range(0, max_p), parents)]) 
sharp_in_list = [np.mean(sharpness)]
sharp_in_std = [np.std(sharpness)]

preds = [fitness_evaluator.predict(parent, A, B, train_x) for parent, A, B in zip(parents, alignment[:, 0], alignment[:, 1])]
sharp_out_manager = SAM_OUT()

neighbor_map = np.array([get_neighbor_map(pred, sharp_out_manager, p) for i, pred, p in zip(range(0, max_p), preds, parents)])
sharp_out_list = [np.mean(np.std(neighbor_map, axis = 1)**2)] #variance
sharp_out_std = [np.std(np.std(neighbor_map, axis = 1))]

print(sharp_in_list)
print(sharpness)
print(sharp_out_list)
print(np.var(neighbor_map, axis = 1))

#print(f'starting scaling')
#print(alignment)
ret_avg_list = [] #best parent best child
ret_std_list = []

avg_change_list = [] #best parent best child
avg_hist_list = []
std_change_list = []
best_i = np.argmin(fitnesses[:max_p])
p_size = [len(effProg(4, parents[best_i]))/len(parents[best_i])]
mut_impact = MutationImpact(neutral_limit = 0.1)


for g in range(1, max_g+1):
	children, retention, density_distro = xover(deepcopy(parents), max_r, p_xov, 'OnePoint', density_distro, fixed_length = fixed_length)
	children = mutate(deepcopy(children), max_c, max_r, max_d, bank, inputs = 1, n_bias = 10, arity = 2)
	pop = parents+children
	fitness_evaluator = Fitness(train_x, bias, train_y, pop, func, bank, n_inp, max_d, fit, arity)
	fitnesses, alignment[:, 0], alignment[:, 1] = fitness_evaluator()
	pop = parents+children
	if any(np.isnan(fitnesses)): #screen out nan values
		nans = np.isnan(fitnesses)
		fitnesses[nans] = np.PINF
	mut_impact(fitnesses, max_p)	
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
		ret_list.append(find_similarity(cs[best_c], ps[best_p], [], [], 'lgp'))
		
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

	noisy_x, noisy_y = getNoise(train_x_bias.shape)
	sharpness = get_sam_in(noisy_x, noisy_y, pop) #np.array([fitness_objects[i](noisy_x[i], noisy_y[i], parent)[0] for i, parent in zip(range(0, max_p), parents)]) 
	sharp_in_list.append(np.mean(sharpness))
	sharp_in_std.append(np.std(sharpness))

	preds = [fitness_evaluator.predict(p, A, B, train_x) for p, A, B in zip(pop, alignment[:, 0], alignment[:, 1])]

	neighbor_map = np.array([get_neighbor_map(pred, sharp_out_manager, p) for i, pred, p in zip(range(0, max_p+max_c), preds, pop)])
	out_sharpness = np.std(neighbor_map, axis = 1)**2
	sharp_out_list.append(np.mean(out_sharpness)) #variance
	sharp_out_std.append(np.std(out_sharpness))

	best_i = np.argmin(fitnesses)
	best_pop = pop[best_i]
	best_a = alignment[best_i, 0]
	best_b = alignment[best_i, 1]
	best_fit = fitnesses[best_i]
	fit_track.append(best_fit)
	p_size.append(len(effProg(4, pop[best_i]))/len(pop[best_i]))
	parents = select(pop, fitnesses, max_p, n_tour)

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
		#scatter_elites(elites, func, func_name, run_name, t, g, fit_name, best_fit, mode = 'lgp')
	
pop = parents+children
fitness_evaluator = Fitness(train_x, bias, train_y, pop, func, bank, n_inp, max_d, fit, arity)
fitnesses, alignment[:, 0], alignment[:, 1] = fitness_evaluator()
best_i = np.argmin(fitnesses)
best_fit = fitnesses[best_i]
best_pop = pop[best_i]
p_A = alignment[best_i, 0]
p_B = alignment[best_i, 1]
print(f"Trial {t}: Best Fitness = {best_fit}")
drift_cum, drift_list = mut_impact.returnLists(option=1)
print(f"Operators:\tDeleterious\tNeutral\tBeneficial")
print(f"\t{drift_cum[0]}\t{drift_cum[1]}\t{drift_cum[2]}")
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
preds = fitness_evaluator.predict(best_pop, p_A, p_B, train_x)
print(preds)
print(density_distro)
print('Probability Density Fuction')
density_distro = density_distro/np.sum(density_distro)
print(density_distro)
print(skew(density_distro, bias = True), kurtosis(density_distro, bias = True))
print(f"../output/{run_name}/{func_name}/log/output_{t}") 
Path(f"../output/{run_name}/{func_name}/log/").mkdir(parents=True, exist_ok=True)
import pickle

first_body_node = n_inp+n_bias+1
print(first_body_node)
win_length = 100
#Write Plots
"""
from scipy.signal import savgol_filter
from cgp_plots import *
scatter(train_x, train_y, preds, func_name, run_name, t, fit_name, best_fit)
fit_plot(fit_track, func_name, run_name, t)
proportion_plot(p_size, func_name, run_name, t)
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
"""
bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g, opt = 1)
import graphviz as gv
p = effProg(max_d, best_pop, first_body_node)
lgp_print_individual(p, p_A, p_B, 'lgp', func_name, bank_string, t, bias, n_inp, first_body_node)
with open(f"../output/{run_name}/{func_name}/best_program/best_{t}.txt", 'a') as f:
	f.write(f"\nEffective Instructions\n\n")  
	f.write(f'{p}')
print('effective program')
print(p)
with open(f"../output/{run_name}/{func_name}/log/output_{t}.pkl", "wb") as f:
	pickle.dump(bias, f)
	pickle.dump(best_pop, f)
	pickle.dump(preds, f)
	pickle.dump(np.round(best_fit, 4), f)
	pickle.dump(len(p), f)
	pickle.dump(fit_track, f)
	pickle.dump([avg_change_list], f)
	pickle.dump([ret_avg_list], f)
	pickle.dump(p_size, f)
	pickle.dump([bin_centers, hist_gens, avg_hist_list], f)
	pickle.dump(p_size, f)
	pickle.dump(drift_list, f)
	pickle.dump(drift_cum, f)
	pickle.dump([sharp_in_list, sharp_out_list], f)
	pickle.dump([sharp_in_std, sharp_out_std], f)
	pickle.dump(density_distro, f)
dot = draw_graph_thicc(p, p_A, p_B, max_d=max_d)
Path(f"../output/{run_name}/{func_name}/full_graphs/").mkdir(parents=True, exist_ok=True)
dot.render(f"../output/{run_name}/{func_name}/full_graphs/graph_{t}", view=False)

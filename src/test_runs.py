import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
import cgp_fitness
import lgp_fitness
import matplotlib.pyplot as plt
from numpy import random, sin, cos, tan, sqrt, exp, log, abs, floor, ceil
from math import log, pi
from sys import path
from pathlib import Path
from functions import *
from cgp_fitness import *
from lgp_fitness import *

max_e = 50
c = Collection()
f_list = c.func_list
f_name = c.name_list

def load_individuals(base_path, mode,max_e = max_e, f_name = f_name):
	#max_e = 10 #number of trials
	population = []
	for name in f_name:
		print(f"Loading {base_path}{name}")
		pop_temp = []
		for e in range(1, max_e+1):
			p = f'{base_path}{name}/log/output_{e}.pkl'
			if mode == 'cgp':
				with open(p, "rb") as f:
					bias = pickle.load(f)
					ind = pickle.load(f)
					out = pickle.load(f)
				pop_temp.append((ind, out))
			elif mode == 'lgp':
				with open(p, "rb") as f:
					bias = pickle.load(f)
					ind = pickle.load(f)
				pop_temp.append(ind)
		population.append(pop_temp)
	return population

cgp_base_path = "../output/cgp/"
lgp_mut_path = "../output/lgp_mut/"
lgp_base_path = "../output/lgp/"
cgp_1x_path = "../output/cgp_1x/"
lgp_1x_path = "../output/lgp_1x/"
cgp_2x_path = "../output/cgp_2x/"
lgp_2x_path = "../output/lgp_2x/"
cgp_sgx_path = "../output/cgp_sgx/"
cgp_40_path = "../output/cgp_40/"

data_path = "test_set.pkl"
with open(data_path, "rb") as f:
	data = pickle.load(f)

x_temp = data[:, 0]
y = data[:, 1]

biases = np.arange(0, 10, 1).astype(np.int32)
bias = biases.shape[0] #number of biases
x = np.zeros((x_temp.shape[0], x_temp.shape[1], biases.shape[0]+1)) #attach constants to input
x[:, :, 0] = x_temp
x[:, :, 1:] = biases
all_individuals = []
all_individuals.append(load_individuals(cgp_base_path, 'cgp'))
#all_individuals.append(load_individuals(lgp_mut_path, 'lgp'))
all_individuals.append(load_individuals(cgp_40_path, 'cgp'))
all_individuals.append(load_individuals(cgp_1x_path, 'cgp'))
all_individuals.append(load_individuals(lgp_1x_path, 'lgp'))
all_individuals.append(load_individuals(cgp_2x_path, 'cgp'))
all_individuals.append(load_individuals(lgp_2x_path, 'lgp'))
all_individuals.append(load_individuals(cgp_sgx_path, 'cgp'))
all_individuals.append(load_individuals(lgp_base_path, 'lgp'))


method_modes = ['cgp', 'cgp', 'cgp', 'lgp', 'cgp', 'lgp', 'cgp', 'lgp']
#method_modes = ['cgp', 'lgp']
f = 1 #correlation, rmse would = 0
fits = FitCollection()
fit = fits.fit_list[f]
fit_name  = fits.name_list[f]

preds = []
fits = []
a = []
b = []
print(len(all_individuals))
print(x.shape)
import lgp_functions
for m in range(0, len(all_individuals)):
	fit_temp = []
	pre_temp = []
	a_temp = []
	b_temp = []
	method = all_individuals[m]
	for p in range(len(method)):
		problem = method[p]
		x_p = x[p]
		y_p = y[p]
		fit_pro = []
		pre_pro = []
		a_pro = []
		b_pro = []
		for i in range(0, max_e):
			if method_modes[m] == 'cgp':
				fit = cgp_fitness.Fitness()
				p_fit, p_A, p_B = fit(x_p, y_p, problem[i])
				fit_pro.append(p_fit)
				a_pro.append(p_A)
				b_pro.append(p_B)
				pre_pro.append(fit(x_p, y_p, problem[i], opt = 1)[0])
				del fit
			else:
				#def __init__(self, data, bias, target, pop, func, bank, n_inp = 1, max_d = 4, fit_function = corr, arity = 2)
				fit = lgp_fitness.Fitness(x_p[:, 0], x_p[0, 1:], y_p, [problem[i]], lgp_fitness.corr, lgp_functions.get_functions())
				p_fit, p_A, p_B = fit()
				fit_pro.append(p_fit[0])
				a_pro.append(p_A[0])
				b_pro.append(p_B[0])
				pre_pro.append(fit.predict(problem[i], p_A, p_B, x_p[:, 0]))
				del fit
		fit_temp.append(fit_pro)
		pre_temp.append(pre_pro)
		a_temp.append(a_pro)
		b_temp.append(b_pro)
	preds.append(pre_temp)
	fits.append(fit_temp)
	a.append(a_temp)
	b.append(b_temp)
fits = np.array(fits) #method -> problem -> trial
fits[np.isnan(fits)] = 1.0
print(fits.shape)
preds = np.array(preds)
print(preds.shape)

color_order = ['blue', 'royalblue', 'skyblue', 'lightgreen', 'steelblue', 'mediumseagreen', 'indigo', 'green']#, 'cadetblue', 'olive']
method_names = ["CGP(1+4)", "CGP(16+64)", "CGP-1x(40+40)","LGP-1x(40+40)", "CGP-2x(40+40)","LGP-2x(40+40)", "CGP-SGx(40+40)", "LGP-Ux(40+40)"]#, "CGP-Nx(40+40)", "LGP-Fx(40+40)"]
method_names_long = ["CGP(1+4)", "CGP(16+64)", "CGP-OnePoint(40+40)","LGP-OnePoint(40+40)", "CGP-TwoPoint(40+40)","LGP-TwoPoint(40+40)", "CGP-Subgraph(40+40)", "LGP-Uniform(40+40)"]#, "CGP-NodeOnePoint(40+40)", "LGP-FlattenedOnePoint(40+40)"]  

fits_med = np.median(fits, axis = 2)
print(fits_med.shape)

fig, axs = plt.subplots(len(f_name), 1, figsize = (9.5*1.1, 11*1.1))
fig.subplots_adjust(hspace=0)
from copy import deepcopy
for n in range(len(f_name)):
	#print(n)
	boxes = axs[n].boxplot([fits[0, n, :], fits[1, n, :], fits[2, n, :], fits[3, n, :], fits[4, n, :], fits[5, n, :], fits[6, n, :], fits[7, n, :]], patch_artist = True, showfliers = True)
	print(fits[5, n, :])
	box_list = boxes['boxes']
	axs[n].set_yscale('log')
	for box, color in zip(box_list, color_order):
        	box.set_facecolor(color)
	axs[n].set_xticks(list(range(1, len(method_names)+1)),method_names, rotation=0, fontsize=11)
	axs[n].set_title(f"{f_name[n]}", fontsize=12)
	axs[n].set_ylabel("1-r^2", fontsize=11)
	axs[n].set_ylim(bottom=1e-6, top = 1e-1)
	axs[n].tick_params(axis='y', labelsize=10)
	axs[n].tick_params(axis='x', labelsize=10)
fig.suptitle("Fitness Evaluation on SR Problems", fontsize=16)
axs[0].set_ylim(bottom = 1e-4, top = 1e-1)
axs[1].set_ylim(bottom = 1e-4, top = 1e0)
axs[2].set_ylim(bottom = 1e-4, top = 1e0)
axs[3].set_ylim(bottom = 1e-4, top = 1e0)
axs[5].set_ylim(bottom = 1e-4, top = 1e-1)
axs[6].set_ylim(bottom = 1e-7, top = 1e-2)
legend_objects = [box for box in box_list]
fig.legend(legend_objects, method_names_long, fontsize=10, ncol = 2, bbox_to_anchor = (0.5, 0.965), loc='upper center')
fig.tight_layout(rect=[0, 0, 1, 0.920])
plt.show()
plt.savefig("../output/test_fitness.png")

fit_name = '1-r^2'
print(preds[0, 3, 17])
print(all_individuals[0][3][17])
from effProg import effProg, lgp_print_individual, cgp_active_nodes
print(cgp_active_nodes(all_individuals[0][3][17][0], all_individuals[0][3][17][1]))
#e_prog = effProg(4, all_individuals[1][3][17])
print(f'test/{method_names[0]} {f_name[3]} trial 18')
#lgp_print_individual(e_prog, a[1][3][17], b[1][3][17], f'test/{method_names[5]}', f_name[3], ("+", "-", "*", "/"), 20, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_inp = 1, fb_node = 12)
"""
from cgp_plots import scatter
for m in range(len(method_names)):
	run_name = f'test/{method_names[m]}/'
	for p in range(len(f_name)):
		func_name = f_name[p]
		for e in range(max_e):
			pred_inst = preds[m, p, e]
			p_fit = fits[m, p, e]
			print(m, p, e, pred_inst, p_fit)
			scatter(x_temp[p], y[p], pred_inst, func_name, run_name, e, fit_name, p_fit)
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from functions import *
from pathlib import Path
from sys import argv

c = Collection()
f_list = c.func_list
f_name = c.name_list

cgp_base_path = "../output/cgp/"
cgp_base = [] #method -> [function name -> trial number -> stats]
cgp_base_p_fits = []

lgp_base_path = "../output/lgp/"
lgp_base = [] #method -> [function name -> trial number -> stats]
lgp_base_p_fits = []

method_names = ["CGP - Base", "LGP - Base"]

max_e = 10
def get_logs_cgp(base_path, max_e = 10):
	full_logs = [] #full logs
	full_fits = []
	#max_e = 10 #number of trials
	for name in f_name:
		print(f"Loading {base_path}{name}")
		logs = [] #idk what to call it, I just want everything easily accessible from a few lists
		p_log = []
		for e in range(1, max_e+1):
			p = f'{base_path}{name}/log/output_{e}.pkl'
			with open(p, "rb") as f:
				bias = pickle.load(f)
				ind = pickle.load(f)
				out = pickle.load(f)
				preds = pickle.load(f)
				p_fit = pickle.load(f)
				if np.isnan(p_fit):
					p_fit = np.PINF
			logs.append((bias, ind, out, preds, p_fit))
			p_log.append(p_fit)
		full_logs.append(logs)
		full_fits.append(p_log)
	return logs, np.array(full_fits)
	
def get_logs_lgp(base_path, max_e = 10):
	full_logs = [] #full logs
	full_fits = []
	#max_e = 10 #number of trials
	for name in f_name:
		print(f"Loading {base_path} {name}")
		logs = [] #idk what to call it, I just want everything easily accessible from a few lists
		p_log = []
		for e in range(1, max_e+1):
			p = f'{base_path}{name}/log/output_{e}.pkl'
			with open(p, "rb") as f:
				bias = pickle.load(f)
				ind = pickle.load(f)
				preds = pickle.load(f)
				best_fit = pickle.load(f)
				if np.isnan(best_fit):
					best_fit = np.PINF
			logs.append((bias, ind, preds, best_fit))
			p_log.append(best_fit)
		full_logs.append(logs)
		full_fits.append(p_log)
	return logs, np.array(full_fits)


#plot boxes
_ , cgp_base_p_fits = get_logs_cgp(cgp_base_path)
_ , lgp_base_p_fits = get_logs_lgp(lgp_base_path)
#print(cgp_base_p_fits)
fig, axs = plt.subplots(1,len(f_name), figsize = (20, 5))
for n in range(len(f_name)):
	print(n)
	axs[n].boxplot([cgp_base_p_fits[n], lgp_base_p_fits[n]], showfliers = False)
	axs[n].set_xticks(list(range(1, len(method_names)+1)),method_names)
	axs[n].set_title(f"{f_name[n]}")
fig.suptitle("Fitness Evaluation on SR Problems")
fig.tight_layout()
plt.show()
plt.savefig("../output/rmse.png")

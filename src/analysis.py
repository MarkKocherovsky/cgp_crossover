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
	fit_tracks = []
	active_nodes = []
	#max_e = 10 #number of trials
	for name in f_name:
		print(f"Loading {base_path}{name}")
		logs = [] #idk what to call it, I just want everything easily accessible from a few lists
		p_log = []
		track_log = []
		node_log = []
		for e in range(1, max_e+1):
			p = f'{base_path}{name}/log/output_{e}.pkl'
			with open(p, "rb") as f:
				bias = pickle.load(f)
				ind = pickle.load(f)
				out = pickle.load(f)
				preds = pickle.load(f)
				p_fit = pickle.load(f)
				n = pickle.load(f)
				fit_track = pickle.load(f)
				if np.isnan(p_fit):
					p_fit = np.PINF
			logs.append((bias, ind, out, preds, p_fit))
			p_log.append(p_fit)
			track_log.append(fit_track)
			node_log.append(n)
		full_logs.append(logs)
		full_fits.append(p_log)
		fit_tracks.append(track_log)
		#print(track_log)
		active_nodes.append(node_log)
	return logs, np.array(full_fits), np.array(fit_tracks), np.array(active_nodes)
	
def get_logs_lgp(base_path, max_e = 10):
	full_logs = [] #full logs
	full_fits = []
	prog_length = []
	fit_tracks = []
	#max_e = 10 #number of trials
	for name in f_name:
		print(f"Loading {base_path} {name}")
		logs = [] #idk what to call it, I just want everything easily accessible from a few lists
		p_log = []
		track_log = []
		node_log = []
		for e in range(1, max_e+1):
			p = f'{base_path}{name}/log/output_{e}.pkl'
			with open(p, "rb") as f:
				bias = pickle.load(f)
				ind = pickle.load(f)
				preds = pickle.load(f)
				best_fit = pickle.load(f)
				if np.isnan(best_fit):
					best_fit = np.PINF
				n = pickle.load(f)
				fit_track = pickle.load(f)

			logs.append((bias, ind, preds, best_fit))
			p_log.append(best_fit)
			track_log.append(fit_track)
			node_log.append(n)
		full_logs.append(logs)
		full_fits.append(p_log)
		fit_tracks.append(track_log)
		prog_length.append(node_log)
	return logs, np.array(full_fits), np.array(fit_tracks), np.array(prog_length)


#plot boxes
_ , cgp_base_p_fits, cgp_base_fit_tracks, cgp_base_nodes = get_logs_cgp(cgp_base_path)
_ , lgp_base_p_fits, lgp_base_fit_tracks, lgp_base_nodes = get_logs_lgp(lgp_base_path) #I use nodes for LGP just for standards
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

#program size, I'll do the averaging at home
fig, axs = plt.subplots(1, len(f_name), figsize = (20,5))
for n in range(len(f_name)):
	axs[n].boxplot([cgp_base_nodes[n], lgp_base_nodes[n]], showfliers = False)
	axs[n].set_xticks(list(range(1,len(method_names)+1)), method_names)
	axs[n].set_title(f"{f_name[n]}")
fig.suptitle("Program Size for SR problems")
fig.tight_layout()
plt.show()
plt.savefig("../output/prog_size.png")

#perform averaging
def get_avg_gens(f):
	avgs = []
	std_devs = []
	for p in f: #problem in fit lists
		#print(p.shape)
		avgs.append(np.average(p, axis = 0))
		std_devs.append(np.std(p, axis = 0))
	return np.array(avgs), np.array(std_devs)

def get_err_ribbon(avgs, stds):
	return avgs + stds, avgs - stds
cgp_avgs, cgp_std_devs = get_avg_gens(cgp_base_fit_tracks)
lgp_avgs, lgp_std_devs = get_avg_gens(lgp_base_fit_tracks)
print(cgp_avgs.shape)
print(lgp_avgs.shape)
fig, axs = plt.subplots(1, len(f_name), figsize = (20, 5))
for n in range(len(f_name)):
	axs[n].plot(cgp_avgs[n])
	axs[n].plot(lgp_avgs[n])
	axs[n].fill_between(range(cgp_avgs[n].shape[0]), get_err_ribbon(cgp_avgs[n], cgp_std_devs[n]), )
	axs[n].set_title(f'{f_name[n]}')
axs[0].set_ylabel("RMSE")
fig.suptitle("Fitness over generations")
fig.tight_layout()
plt.show()
plt.savefig("../output/fitness.png")

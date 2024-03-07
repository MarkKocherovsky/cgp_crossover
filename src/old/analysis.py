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

cgp_1x_path = "../output/cgp_1x/"
cgp_1x = []
cgp_1x_p_fits = []

lgp_1x_path = "../output/lgp_1x/"
lgp_1x = []
lgp_1x_p_fits = []

lgp_2x_path = "../output/lgp_2x/"
lgp_2x = []
lgp_2x_p_fits = []

lgp_mut_path = "../output/lgp_mut/"
lgp_mut = []
lgp_mut_p_fits = []

cgp_2x_path = "../output/cgp_2x/"
cgp_2x = []
cgp_2x_p_fits = []

cgp_40_path = "../output/cgp_40/"
cgp_40 = []
cgp_40_p_fits = []

cgp_sgx_path = "../output/cgp_sgx/"
cgp_sgx = []
cgp_sgx_p_fits = []

method_names = ["CGP(1+4)", "CGP(16+64)", "CGP-1x(40+40)", "CGP-2x(40+40)", "CGP-SGx(40+40)", "LGP-Ux(40+40)", "LGP-1x(40+40)", "LGP-2x(40+40)"]

max_e = 50
def get_logs_cgp(base_path, max_e = max_e):
	full_logs = [] #full logs
	full_fits = []
	fit_tracks = []
	active_nodes = []
	node_prop = []
	#max_e = 10 #number of trials
	for name in f_name:
		print(f"Loading {base_path}{name}")
		logs = [] #idk what to call it, I just want everything easily accessible from a few lists
		p_log = []
		track_log = []
		node_log = []
		prop_log = []
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
			prop_log.append(n/len(ind))
		full_logs.append(logs)
		full_fits.append(p_log)
		fit_tracks.append(track_log)
		node_prop.append(prop_log)
		#print(len(track_log))
		#print(track_log)
		#print(len(track_log), len(track_log[0]))
		#print(track_log[4])
		active_nodes.append(node_log)
	#print(fit_tracks[4][0])
	return logs, np.array(full_fits), np.array(fit_tracks), np.array(active_nodes), np.array(node_prop)
	
def get_logs_lgp(base_path, max_e = max_e):
	full_logs = [] #full logs
	full_fits = []
	prog_length = []
	fit_tracks = []
	node_prop = []
	#max_e = 10 #number of trials
	for name in f_name:
		print(f"Loading {base_path} {name}")
		logs = [] #idk what to call it, I just want everything easily accessible from a few lists
		p_log = []
		track_log = []
		node_log = []
		prop_log = []
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
			prop_log.append(n/len(ind))
		full_logs.append(logs)
		full_fits.append(p_log)
		fit_tracks.append(track_log)
		#print(len(fit_tracks))
		prog_length.append(node_log)
		node_prop.append(prop_log)
	return logs, np.array(full_fits), np.array(fit_tracks), np.array(prog_length), np.array(node_prop)


#plot boxes
_ , cgp_base_p_fits, cgp_base_fit_tracks, cgp_base_nodes, cgp_base_prop = get_logs_cgp(cgp_base_path)
_ , cgp_1x_p_fits, cgp_1x_fit_tracks, cgp_1x_nodes, cgp_1x_prop = get_logs_cgp(cgp_1x_path)
_ , cgp_2x_p_fits, cgp_2x_fit_tracks, cgp_2x_nodes, cgp_2x_prop = get_logs_cgp(cgp_2x_path)
_ , cgp_40_p_fits, cgp_40_fit_tracks, cgp_40_nodes, cgp_40_prop = get_logs_cgp(cgp_40_path)
_ , cgp_sgx_p_fits, cgp_sgx_fit_tracks, cgp_sgx_nodes, cgp_sgx_prop = get_logs_cgp(cgp_sgx_path)
_ , lgp_base_p_fits, lgp_base_fit_tracks, lgp_base_nodes, lgp_base_prop = get_logs_lgp(lgp_base_path) #I use nodes for LGP just for standards
_ , lgp_1x_p_fits, lgp_1x_fit_tracks, lgp_1x_nodes, lgp_1x_prop = get_logs_lgp(lgp_1x_path)
_ , lgp_2x_p_fits, lgp_2x_fit_tracks, lgp_2x_nodes, lgp_2x_prop = get_logs_lgp(lgp_2x_path)
#_ , lgp_mut_p_fits, lgp_mut_fit_tracks, lgp_mut_nodes = get_logs_lgp(lgp_mut_path)
print(len(lgp_mut_p_fits))
print(list(range(1, len(method_names)+1)))
print(method_names)
#print(cgp_base_p_fits)
fig, axs = plt.subplots(len(f_name), 1, figsize = (12, 20))
fig.subplots_adjust(hspace=0)
for n in range(len(f_name)):
	#print(n)
	axs[n].boxplot([cgp_base_p_fits[n], cgp_40_p_fits[n], cgp_1x_p_fits[n], cgp_2x_p_fits[n], cgp_sgx_p_fits[n], lgp_base_p_fits[n], lgp_1x_p_fits[n], lgp_2x_p_fits[n]], showfliers = False)
	axs[n].set_xticks(list(range(1, len(method_names)+1)),method_names)
	axs[n].set_title(f"{f_name[n]}", fontsize=18)
	axs[n].set_ylabel("1-r^2", fontsize=12)
fig.suptitle("Fitness Evaluation on SR Problems", fontsize=24)
fig.tight_layout(rect=[0, 0, 1, 0.985])
plt.show()
plt.savefig("../output/rmse.png")
print('fitness')

#program size, I'll do the averaging at home
fig, axs = plt.subplots(len(f_name), 1, figsize = (10,20))
for n in range(len(f_name)):
	axs[n].boxplot([cgp_base_prop[n], cgp_40_prop[n], cgp_1x_prop[n], cgp_2x_prop[n], cgp_sgx_prop[n], lgp_base_prop[n], lgp_1x_prop[n], lgp_2x_prop[n]], showfliers = False)
	axs[n].set_xticks(list(range(1,len(method_names)+1)), method_names)
	axs[n].set_title(f"{f_name[n]}")
fig.suptitle("Proportion of active nodes for SR problems")
fig.tight_layout()
plt.show()
plt.savefig("../output/prog_prop.png")

fig, axs = plt.subplots(len(f_name), 1, figsize = (10,20))
for n in range(len(f_name)):
	axs[n].boxplot([cgp_base_nodes[n], cgp_40_nodes[n], cgp_1x_nodes[n], cgp_2x_nodes[n], cgp_sgx_nodes[n], lgp_base_nodes[n], lgp_1x_nodes[n], lgp_2x_nodes[n]], showfliers = False)
	axs[n].set_xticks(list(range(1,len(method_names)+1)), method_names)
	axs[n].set_title(f"{f_name[n]}")
fig.suptitle("Program Size for SR problems")
fig.tight_layout()
plt.show()
plt.savefig("../output/prog_size.png")

print('prog size')
#fit vs size
fig, axs = plt.subplots(1, len(f_name), figsize = (25, 6), sharey=False)
fig.subplots_adjust(hspace=0)
for n in range(len(f_name)):
	axs[n].set_yscale('log')
	axs[n].scatter(cgp_base_nodes[n], cgp_base_p_fits[n], label='CGP(1+4)', c = 'blue', marker = 'o', s = 8)
	axs[n].scatter(cgp_1x_nodes[n], cgp_1x_p_fits[n], label='CGP-1x(40+40)', c = 'orange', marker = 'v',s = 8)
	axs[n].scatter(cgp_2x_nodes[n], cgp_2x_p_fits[n], label='CGP-2x(40+40)', c = 'green', marker = 's', s = 8)
	axs[n].scatter(cgp_40_nodes[n], cgp_40_p_fits[n], label='CGP(16+64)', c = 'olive', marker = 'P', s = 8)
	axs[n].scatter(cgp_sgx_nodes[n], cgp_sgx_p_fits[n], label='CGP-SGx(40+40)', c = 'slategray', marker = '*', s = 8)
	axs[n].scatter(lgp_base_nodes[n], lgp_base_p_fits[n], label='LGP-Ux(40+40)', c = 'red', marker = '^', s = 8)
	axs[n].scatter(lgp_1x_nodes[n], lgp_1x_p_fits[n], label = 'LGP-1x(40+40)', c = 'purple', marker = 'X', s = 8)
	axs[n].scatter(lgp_2x_nodes[n], lgp_2x_p_fits[n], label = 'LGP-2x(40+40)', c = 'brown', marker = 'P', s = 8)
	axs[n].set_title(f'{f_name[n]}')
	axs[n].set_xlabel('Active Instructions')
axs[0].set_ylabel('1-R^2')
fig.suptitle('Fitness vs Nodes')
axs[0].legend()
plt.tight_layout()
plt.show()
plt.savefig('../output/fit_size.png')

fig, axs = plt.subplots(1, len(f_name), figsize = (25, 6), sharey=False)
fig.subplots_adjust(hspace=0)
for n in range(len(f_name)):
	axs[n].set_yscale('log')
	axs[n].scatter(cgp_base_prop[n], cgp_base_p_fits[n], label='CGP(1+4)', c = 'blue', marker = 'o', s = 8)
	axs[n].scatter(cgp_1x_prop[n], cgp_1x_p_fits[n], label='CGP-1x(40+40)', c = 'orange', marker = 'v',s = 8)
	axs[n].scatter(cgp_2x_prop[n], cgp_2x_p_fits[n], label='CGP-2x(40+40)', c = 'green', marker = 's', s = 8)
	axs[n].scatter(cgp_40_prop[n], cgp_40_p_fits[n], label='CGP(16+64)', c = 'olive', marker = 'P', s = 8)
	axs[n].scatter(cgp_sgx_prop[n], cgp_sgx_p_fits[n], label='CGP-SGx(40+40)', c = 'slategray', marker = '*', s = 8)
	axs[n].scatter(lgp_base_prop[n], lgp_base_p_fits[n], label='LGP-Ux(40+40)', c = 'red', marker = '^', s = 8)
	axs[n].scatter(lgp_1x_prop[n], lgp_1x_p_fits[n], label = 'LGP-1x(40+40)', c = 'purple', marker = 'X', s = 8)
	axs[n].scatter(lgp_2x_prop[n], lgp_2x_p_fits[n], label = 'LGP-2x(40+40)', c = 'brown', marker = 'P', s = 8)
	axs[n].set_title(f'{f_name[n]}')
	axs[n].set_xlabel('Proportion of Active Instructions')
axs[0].set_ylabel('1-R^2')
fig.suptitle('Fitness vs Active Node Proportions')
axs[0].legend()
plt.tight_layout()
plt.show()
plt.savefig('../output/fit_prop.png')

print('fit size')
#perform averaging
def get_avg_gens(f):
	avgs = []
	std_devs = []
	for p in f: #problem in fit lists
		#print(p.shape)
		avgs.append(np.average(p, axis = 0))
		std_devs.append(np.std(p, axis = 0))
	avgs = np.array(avgs)
	std_devs = np.array(std_devs)
	return avgs, avgs-std_devs, avgs+std_devs

def get_err_ribbon(avgs, stds):
	return avgs + stds, avgs - stds
cgp_avgs, cgp_s_p, cgp_s_m = get_avg_gens(cgp_base_fit_tracks)
cgp_1x_avgs, cgp_1x_s_p, cgp_1x_s_m = get_avg_gens(cgp_1x_fit_tracks)
cgp_2x_avgs, cgp_2x_s_p, cgp_2x_s_m = get_avg_gens(cgp_2x_fit_tracks)
cgp_40_avgs, cgp_40_s_p, cgp_40_s_m = get_avg_gens(cgp_40_fit_tracks)
cgp_sgx_avgs, cgp_sgx_s_p, cgp_sgx_s_m = get_avg_gens(cgp_sgx_fit_tracks)
lgp_avgs, lgp_s_p, lgp_s_m = get_avg_gens(lgp_base_fit_tracks)
lgp_1x_avgs, lgp_1x_s_p, lgp_1x_s_m = get_avg_gens(lgp_1x_fit_tracks)
lgp_2x_avgs, lgp_2x_s_p, lgp_2x_s_m = get_avg_gens(lgp_2x_fit_tracks)
#lgp_mut_avgs, lgp_mut_s_p, lgp_mut_s_m = get_avg_gens(lgp_mut_fit_tracks)
print(cgp_avgs.shape)
print(lgp_avgs.shape)
print(lgp_1x_avgs.shape)
fig, axs = plt.subplots(len(f_name), 1, figsize = (10, 20))
for n in range(len(f_name)):
	axs[n].set_yscale('log')
	axs[n].plot(cgp_avgs[n], label='CGP (1+4)', c = 'blue')
	axs[n].plot(cgp_1x_avgs[n], label='CGP-1x (40+40)', c = 'orange')
	axs[n].plot(cgp_2x_avgs[n], label='CGP-2x (40+40)', c = 'green')
	axs[n].plot(cgp_40_avgs[n], label='CGP(16+64)', c = 'olive')
	axs[n].plot(cgp_sgx_avgs[n], label="CGP-SGx(40+40)", c = 'slategray')
	axs[n].plot(lgp_avgs[n], label='LGP-Ux (40+40)', c = 'red')
	axs[n].plot(lgp_1x_avgs[n], label='LGP-1x (40+40)',c = 'purple')
	axs[n].plot(lgp_2x_avgs[n], label='LGP-2x (40+40)',c = 'brown')
	#axs[n].plot(lgp_mut_avgs[n], label='LGP (40+40)', c = 'brown')
	axs[n].fill_between(range(cgp_avgs[n].shape[0]), cgp_s_m[n], cgp_s_p[n], color='blue', alpha = 0.10)
	axs[n].fill_between(range(cgp_1x_avgs[n].shape[0]), cgp_1x_s_m[n], cgp_1x_s_p[n], color='orange', alpha = 0.10)
	axs[n].fill_between(range(cgp_2x_avgs[n].shape[0]), cgp_2x_s_m[n], cgp_2x_s_p[n], color='green', alpha = 0.10)
	axs[n].fill_between(range(cgp_40_avgs[n].shape[0]), cgp_40_s_m[n], cgp_40_s_p[n], color='olive', alpha = 0.10)
	axs[n].fill_between(range(cgp_sgx_avgs[n].shape[0]), cgp_sgx_s_m[n], cgp_sgx_s_p[n], color='slategray', alpha = 0.10)
	axs[n].fill_between(range(lgp_avgs[n].shape[0]), lgp_s_m[n], lgp_s_p[n], color='red', alpha = 0.10)
	axs[n].fill_between(range(lgp_1x_avgs[n].shape[0]), lgp_1x_s_m[n], lgp_1x_s_p[n], color='purple', alpha = 0.10)
	axs[n].fill_between(range(lgp_2x_avgs[n].shape[0]), lgp_2x_s_m[n], lgp_2x_s_p[n], color='brown', alpha = 0.10)
	#axs[n].fill_between(range(lgp_mut_avgs[n].shape[0]), lgp_mut_s_m[n], lgp_mut_s_p[n], color = "brown", alpha = 0.25)
	axs[n].set_ylim(0.001, 1)
	axs[n].set_title(f'{f_name[n]}', fontsize=18)
	axs[n].set_ylabel("1-r^2", fontsize=12)
axs[-1].set_xlabel("Generations", fontsize=12)
axs[0].set_ylim(0.0001, 0.1)
axs[2].set_ylim(0.01, 1)
axs[3].set_ylim(0.0001, 0.1)
axs[4].set_ylim(0.00001, 0.1)
axs[5].set_ylim(0.0001, 0.05)
axs[6].set_ylim(0.00001, 0.001)
#axs[8].set_ylim(0, 0.01)
axs[0].legend()
fig.suptitle("Fitness over generations", fontsize=24)
fig.tight_layout(rect=[0, 0, 1, 0.985]) #https://stackoverflow.com/a/45161551
plt.show()
plt.savefig("../output/fitness.png")
print("fitness over generations")

#axs[n].boxplot([cgp_base_p_fits[n], cgp_1x_p_fits[n], cgp_2x_p_fits[n], cgp_40_p_fits[n], cgp_sgx_p_fits[n], lgp_base_p_fits[n], lgp_1x_p_fits[n], lgp_2x_p_fits[n]], showfliers = False)
#alg_out = [m+'\t' for m in f_name]
alg_out = ['Algorithm']+f_name
alg_out = ','.join(map(str, alg_out)) #https://stackoverflow.com/questions/2399112/print-all-items-in-a-list-with-a-delimiter
with open('../output/medians.txt', 'w') as f:
	f.write(f'{alg_out}')
print(alg_out, sep=',')
all_meds = []
#print("medians")
for n in range(len(f_name)):
	cgp_base_med = np.median(cgp_base_p_fits[n])
	cgp_40_med = np.median(cgp_40_p_fits[n])
	cgp_1x_med = np.median(cgp_1x_p_fits[n])
	cgp_2x_med = np.median(cgp_2x_p_fits[n])
	cgp_sgx_med = np.median(cgp_sgx_p_fits[n])
	lgp_base_med = np.median(lgp_base_p_fits[n])
	lgp_1x_med = np.median(lgp_1x_p_fits[n])
	lgp_2x_med = np.median(lgp_2x_p_fits[n])	
	all_meds.append([cgp_base_med, cgp_40_med, cgp_1x_med, cgp_2x_med, cgp_sgx_med, lgp_base_med, lgp_1x_med, lgp_2x_med])
#print(np.array(all_meds))
all_meds = np.array(all_meds).T
for m in range(len(method_names)):
	print(f'{method_names[m]},', end='')
	m_out = ','.join(map(str, all_meds[m, :]))
	print(m_out)
	with open('../output/medians.txt', 'a') as f:
		f.write(m_out)
		f.write('\n')

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
cgp_base_avg_change_list = []
cgp_base_ret_avg_list = []
cgp_base_p_size_list = []
cgp_base_hist_list = []

lgp_base_path = "../output/lgp/"
lgp_base = [] #method -> [function name -> trial number -> stats]
lgp_base_p_fits = []
lgp_base_avg_change_list = []
lgp_base_ret_avg_list = []
lgp_base_p_size_list = []
lgp_base_hist_list = []

cgp_1x_path = "../output/cgp_1x/"
cgp_1x = []
cgp_1x_p_fits = []
cgp_1x_avg_change_list = []
cgp_1x_ret_avg_list = []
cgp_1x_p_size_list = []
cgp_1x_hist_list = []

lgp_1x_path = "../output/lgp_1x/"
lgp_1x = []
lgp_1x_p_fits = []
lgp_1x_avg_change_list = []
lgp_1x_ret_avg_list = []
lgp_1x_p_size_list = []
lgp_1x_hist_list = []

lgp_2x_path = "../output/lgp_2x/"
lgp_2x = []
lgp_2x_p_fits = []
lgp_2x_avg_change_list = []
lgp_2x_ret_avg_list = []
lgp_2x_p_size_list = []
lgp_2x_hist_list = []

lgp_mut_path = "../output/lgp_mut/"
lgp_mut = []
lgp_mut_p_fits = []

cgp_2x_path = "../output/cgp_2x/"
cgp_2x = []
cgp_2x_p_fits = []
cgp_2x_avg_change_list = []
cgp_2x_ret_avg_list = []
cgp_2x_p_size_list = []
cgp_2x_hist_list = []

cgp_40_path = "../output/cgp_40/"
cgp_40 = []
cgp_40_p_fits = []
cgp_40_avg_change_list = []
cgp_40_ret_avg_list = []
cgp_40_p_size_list = []
cgp_40_hist_list = []

cgp_sgx_path = "../output/cgp_sgx/"
cgp_sgx = []
cgp_sgx_p_fits = []
cgp_sgx_avg_change_list = []
cgp_sgx_ret_avg_list = []
cgp_sgx_p_size_list = []
cgp_sgx_hist_list = []

cgp_nx_path = "../output/cgp_nx/"
cgp_nx = []
cgp_nx_p_fits = []
cgp_nx_avg_change_list = []
cgp_nx_ret_avg_list = []
cgp_nx_p_size_list = []
cgp_nx_hist_list = []

lgp_fx_path = "../output/lgp_fx/"
lgp_fx = []
lgp_fx_p_fits = []
lgp_fx_avg_change_list = []
lgp_fx_ret_avg_list = []
lgp_fx_p_size_list = []
lgp_fx_hist_list = []

color_order = ['blue', 'royalblue', 'skyblue', 'lightgreen', 'steelblue', 'mediumseagreen', 'indigo', 'green']#, 'cadetblue', 'olive']
method_names = ["CGP(1+4)", "CGP(16+64)", "CGP-1x(40+40)","LGP-1x(40+40)", "CGP-2x(40+40)","LGP-2x(40+40)", "CGP-SGx(40+40)", "LGP-Ux(40+40)"]#, "CGP-Nx(40+40)", "LGP-Fx(40+40)"]
method_names_long = ["CGP(1+4)", "CGP(16+64)", "CGP-OnePoint(40+40)","LGP-OnePoint(40+40)", "CGP-TwoPoint(40+40)","LGP-TwoPoint(40+40)", "CGP-Subgraph(40+40)", "LGP-Uniform(40+40)"]#, "CGP-NodeOnePoint(40+40)", "LGP-FlattenedOnePoint(40+40)"]  

max_e = 50
def get_logs_cgp(base_path, max_e = max_e, f_name = f_name):
	full_logs = [] #full logs
	full_fits = []
	fit_tracks = []
	active_nodes = []
	node_prop = []
	avg_chg = []
	avg_ret = []
	p_sz_li = []
	hist_li = []
	#max_e = 10 #number of trials
	for name in f_name:
		print(f"Loading {base_path}{name}")
		logs = [] #idk what to call it, I just want everything easily accessible from a few lists
		p_log = []
		track_log = []
		node_log = []
		prop_log = []
		change_log = []
		retent_log = []
		p_size_log = []
		histog_log = []
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
				average_change = pickle.load(f)
				average_retention = pickle.load(f)
				p_size = pickle.load(f)
				histograms = pickle.load(f)
				if np.isnan(p_fit):
					p_fit = np.PINF
			logs.append((bias, ind, out, preds, p_fit))
			p_log.append(p_fit)
			track_log.append(fit_track)
			node_log.append(n)
			prop_log.append(n/len(ind))
			change_log.append(average_change)
			retent_log.append(average_retention)
			p_size_log.append(p_size)
			histog_log.append(histograms)
		full_logs.append(logs)
		full_fits.append(p_log)
		fit_tracks.append(track_log)
		node_prop.append(prop_log)
		avg_chg.append(change_log)
		avg_ret.append(retent_log)
		p_sz_li.append(p_size_log)
		hist_li.append(histog_log)
		#print(len(track_log))
		#print(track_log)
		#print(len(track_log), len(track_log[0]))
		#print(track_log[4])
		active_nodes.append(node_log)
	#print(fit_tracks[4][0])
	return [logs, np.array(full_fits), np.array(fit_tracks), np.array(active_nodes), np.array(node_prop), np.array(avg_chg), np.array(avg_ret), np.array(p_sz_li), hist_li]
	
def get_logs_lgp(base_path, max_e = max_e, f_name = f_name):
	full_logs = [] #full logs
	full_fits = []
	prog_length = []
	fit_tracks = []
	node_prop = []
	avg_chg = []
	avg_ret = []
	p_sz_li = []
	hist_li = []
	#max_e = 10 #number of trials
	for name in f_name:
		print(f"Loading {base_path}{name}")
		logs = [] #idk what to call it, I just want everything easily accessible from a few lists
		p_log = []
		track_log = []
		node_log = []
		prop_log = []
		change_log = []
		retent_log = []
		p_size_log = []
		histog_log = []
		for e in range(1, max_e+1):
			p = f'{base_path}{name}/log/output_{e}.pkl'
			try:
				with open(p, "rb") as f:
					bias = pickle.load(f)
					ind = pickle.load(f)
					preds = pickle.load(f)
					best_fit = pickle.load(f)
					if np.isnan(best_fit):
						best_fit = np.PINF
					n = pickle.load(f)
					fit_track = pickle.load(f)
					average_change = pickle.load(f)
					average_retention = pickle.load(f)
					p_size = pickle.load(f)
					histograms = pickle.load(f)
			except:
				print(f'{p} | NOT FOUND')
				continue
			logs.append((bias, ind, preds, best_fit))
			p_log.append(best_fit)
			track_log.append(fit_track)
			node_log.append(n)
			prop_log.append(n/len(ind))
			change_log.append(average_change)
			retent_log.append(average_retention)
			p_size_log.append(p_size)
			histog_log.append(histograms)
		full_logs.append(logs)
		full_fits.append(p_log)
		fit_tracks.append(track_log)
		#print(len(fit_tracks))
		prog_length.append(node_log)
		node_prop.append(prop_log)
		avg_chg.append(change_log)
		avg_ret.append(retent_log)
		p_sz_li.append(p_size_log)
		hist_li.append(histog_log)
	return [logs, np.array(full_fits), np.array(fit_tracks), np.array(prog_length), np.array(node_prop), np.array(avg_chg), np.array(avg_ret), np.array(p_sz_li), hist_li]

def dataDict(data):
	return {'logs': data[0], 'p_fits': data[1], 'fit_track': data[2], 'nodes': data[3], 'prop': data[4], 'average_change': data[5], 'average_retention': data[6], 'p_size': data[7], 'histograms': data[8]}

cgp_base_data = dataDict(get_logs_cgp(cgp_base_path))
#lgp_fx_data = dataDict(get_logs_lgp(lgp_fx_path))
cgp_1x_data = dataDict(get_logs_cgp(cgp_1x_path))
cgp_2x_data = dataDict(get_logs_cgp(cgp_2x_path))
cgp_40_data = dataDict(get_logs_cgp(cgp_40_path))
cgp_sgx_data = dataDict(get_logs_cgp(cgp_sgx_path))
#cgp_nx_data = dataDict(get_logs_cgp(cgp_nx_path))
lgp_base_data = dataDict(get_logs_lgp(lgp_base_path))
lgp_1x_data = dataDict(get_logs_lgp(lgp_1x_path))
lgp_2x_data = dataDict(get_logs_lgp(lgp_2x_path))

print(list(range(1, len(method_names)+1)))
print(method_names)
#print(cgp_base_p_fits)
fig, axs = plt.subplots(len(f_name), 1, figsize = (9.5*1.1, 11*1.1))
fig.subplots_adjust(hspace=0)
from copy import deepcopy
for n in range(len(f_name)):
	#print(n)
	boxes = axs[n].boxplot([cgp_base_data['p_fits'][n], cgp_40_data['p_fits'][n], cgp_1x_data['p_fits'][n], lgp_1x_data['p_fits'][n], cgp_2x_data['p_fits'][n], lgp_2x_data['p_fits'][n], cgp_sgx_data['p_fits'][n], lgp_base_data['p_fits'][n]], showfliers = False, patch_artist = True)
	box_list = boxes['boxes']
	axs[n].set_yscale('log')
	for box, color in zip(box_list, color_order):
        	box.set_facecolor(color)
	axs[n].set_xticks(list(range(1, len(method_names)+1)),method_names, rotation=0, fontsize=11)
	axs[n].set_title(f"{f_name[n]}", fontsize=12)
	axs[n].set_ylabel("1-r^2", fontsize=11)
	axs[n].set_ylim(bottom=1e-6)
	axs[n].tick_params(axis='y', labelsize=10)
	axs[n].tick_params(axis='x', labelsize=10)
fig.suptitle("Fitness Evaluation on SR Problems", fontsize=16)
legend_objects = [box for box in box_list]
fig.legend(legend_objects, method_names_long, fontsize=10, ncol = 2, bbox_to_anchor = (0.5, 0.965), loc='upper center')
fig.tight_layout(rect=[0, 0, 1, 0.920])
plt.show()
plt.savefig("../output/rmse.png")

#program size, I'll do the averaging at home
fig, axs = plt.subplots(len(f_name), 1, figsize = (10,20))
for n in range(len(f_name)):
	boxes = axs[n].boxplot([cgp_base_data['prop'][n], cgp_40_data['prop'][n], cgp_1x_data['prop'][n], lgp_1x_data['prop'][n], cgp_2x_data['prop'][n], lgp_2x_data['prop'][n], cgp_sgx_data['prop'][n], lgp_base_data['prop'][n]], showfliers = False, patch_artist = True)
	axs[n].set_xticks(list(range(1,len(method_names)+1)), method_names)
	box_list = boxes['boxes']
	for box, color in zip(box_list, color_order):
        	box.set_facecolor(color)
	axs[n].set_title(f"{f_name[n]}", fontsize=12)
	axs[n].set_ylabel("Active Nodes / Total Nodes", fontsize=10)
fig.suptitle("Proportion of Active Nodes for SR problems", fontsize = 24)
legend_objects = [box for box in box_list]
fig.legend(legend_objects, method_names_long)
fig.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()
plt.savefig("../output/prog_prop.png")


print('prog size')
#fit vs size
fig, axs = plt.subplots(1, len(f_name), figsize = (25, 6), sharey=False)
fig.subplots_adjust(hspace=0)
for n in range(len(f_name)):
	axs[n].set_yscale('log')
	axs[n].scatter(cgp_base_data['nodes'][n], cgp_base_data['p_fits'][n], label='CGP(1+4)', c = color_order[0], marker = 'o', s = 8)
	axs[n].scatter(cgp_1x_data['nodes'][n], cgp_1x_data['p_fits'][n], label='CGP-OnePoint(40+40)', c = color_order[2], marker = 'v',s = 8)
	axs[n].scatter(cgp_2x_data['nodes'][n], cgp_2x_data['p_fits'][n], label='CGP-TwoPoint(40+40)', c = color_order[4], marker = 's', s = 8)
	axs[n].scatter(cgp_40_data['nodes'][n], cgp_40_data['p_fits'][n], label='CGP(16+64)', c = color_order[1], marker = 'P', s = 8)
	axs[n].scatter(cgp_sgx_data['nodes'][n], cgp_sgx_data['p_fits'][n], label='CGP-Subgraph(40+40)', c = color_order[6], marker = '*', s = 8)
	#axs[n].scatter(cgp_nx_data['nodes'][n], cgp_nx_data['p_fits'][n], label='CGP-NodeOnePoint(40+40)', c = color_order[8], marker = '<', s = 8)
	axs[n].scatter(lgp_base_data['nodes'][n], lgp_base_data['p_fits'][n], label='LGP-Uniform(40+40)', c = color_order[7], marker = '^', s = 8)
	axs[n].scatter(lgp_1x_data['nodes'][n], lgp_1x_data['p_fits'][n], label = 'LGP-OnePoint(40+40)', c = color_order[3], marker = 'X', s = 8)
	axs[n].scatter(lgp_2x_data['nodes'][n], lgp_2x_data['p_fits'][n], label = 'LGP-TwoPoint(40+40)', c = color_order[5], marker = 'P', s = 8)
	#axs[n].scatter(lgp_fx_data['nodes'][n], lgp_fx_data['p_fits'][n], label = 'LGP-FlattenedOnePoint(40+40)', c = color_order[9], marker = '>', s = 8)
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
	axs[n].scatter(cgp_base_data['prop'][n], cgp_base_data['p_fits'][n], label='CGP(1+4)', c = color_order[0], marker = 'o', s = 8)
	axs[n].scatter(cgp_1x_data['prop'][n], cgp_1x_data['p_fits'][n], label='CGP-OnePoint(40+40)', c = color_order[2], marker = 'v',s = 8)
	axs[n].scatter(cgp_2x_data['prop'][n], cgp_2x_data['p_fits'][n], label='CGP-TwoPoint(40+40)', c = color_order[4], marker = 's', s = 8)
	axs[n].scatter(cgp_40_data['prop'][n], cgp_40_data['p_fits'][n], label='CGP(16+64)', c = color_order[1], marker = 'P', s = 8)
	axs[n].scatter(cgp_sgx_data['prop'][n], cgp_sgx_data['p_fits'][n], label='CGP-Subgraph(40+40)', c = color_order[6], marker = '*', s = 8)
	#axs[n].scatter(cgp_nx_data['prop'][n], cgp_nx_data['p_fits'][n], label='CGP-NodeOnePoint(40+40)', c = color_order[8], marker = '<', s = 8)
	axs[n].scatter(lgp_base_data['prop'][n], lgp_base_data['p_fits'][n], label='LGP-Uniform(40+40)', c = color_order[7], marker = '^', s = 8)
	axs[n].scatter(lgp_1x_data['prop'][n], lgp_1x_data['p_fits'][n], label = 'LGP-OnePoint(40+40)', c = color_order[3], marker = 'X', s = 8)
	axs[n].scatter(lgp_2x_data['prop'][n], lgp_2x_data['p_fits'][n], label = 'LGP-TwoPoint(40+40)', c = color_order[5], marker = 'P', s = 8)
	#axs[n].scatter(lgp_fx_data['prop'][n], lgp_fx_data['p_fits'][n], label = 'LGP-FlattenedOnePoint(40+40)', c = color_order[9], marker = '>', s = 8)
	axs[n].set_title(f'{f_name[n]}')
	axs[n].set_xlabel('Active Instructions')
axs[0].set_ylabel('1-R^2')
fig.suptitle('Fitness vs Nodes')
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
cgp_avgs, cgp_s_p, cgp_s_m = get_avg_gens(cgp_base_data['fit_track'])
cgp_1x_avgs, cgp_1x_s_p, cgp_1x_s_m = get_avg_gens(cgp_1x_data['fit_track'])
cgp_2x_avgs, cgp_2x_s_p, cgp_2x_s_m = get_avg_gens(cgp_2x_data['fit_track'])
cgp_40_avgs, cgp_40_s_p, cgp_40_s_m = get_avg_gens(cgp_40_data['fit_track'])
cgp_sgx_avgs, cgp_sgx_s_p, cgp_sgx_s_m = get_avg_gens(cgp_sgx_data['fit_track'])
#cgp_nx_avgs, cgp_nx_s_p, cgp_nx_s_m = get_avg_gens(cgp_nx_data['fit_track'])
#lgp_fx_avgs, lgp_fx_s_p, lgp_fx_s_m = get_avg_gens(lgp_fx_data['fit_track'])
lgp_avgs, lgp_s_p, lgp_s_m = get_avg_gens(lgp_base_data['fit_track'])
lgp_1x_avgs, lgp_1x_s_p, lgp_1x_s_m = get_avg_gens(lgp_1x_data['fit_track'])
lgp_2x_avgs, lgp_2x_s_p, lgp_2x_s_m = get_avg_gens(lgp_2x_data['fit_track'])
#lgp_mut_avgs, lgp_mut_s_p, lgp_mut_s_m = get_avg_gens(lgp_mut_fit_tracks)
print(cgp_avgs.shape)
print(lgp_avgs.shape)
print(lgp_1x_avgs.shape)
fig, axs = plt.subplots(len(f_name), 1, figsize = (10, 20))
for n in range(len(f_name)):
	axs[n].set_yscale('log')
	axs[n].plot(cgp_avgs[n], label='CGP (1+4)', c = color_order[0])
	axs[n].plot(cgp_40_avgs[n], label='CGP(16+64)', c = color_order[1])
	axs[n].plot(cgp_1x_avgs[n], label='CGP-OnePoint (40+40)', c = color_order[2])
	axs[n].plot(lgp_1x_avgs[n], label='LGP-OnePoint (40+40)',c = color_order[3])
	axs[n].plot(cgp_2x_avgs[n], label='CGP-TwoPoint (40+40)', c = color_order[4])
	axs[n].plot(lgp_2x_avgs[n], label='LGP-TwoPoint (40+40)',c = color_order[5])
	axs[n].plot(cgp_sgx_avgs[n], label="CGP-Subgraph(40+40)", c = color_order[6])
	axs[n].plot(lgp_avgs[n], label='LGP-Uniform(40+40)', c =  color_order[7])
	#axs[n].plot(cgp_nx_avgs[n], label="CGP-NodeOnePoint(40+40)", c = color_order[8])
	#axs[n].plot(lgp_fx_avgs[n], label='LGP-FlattenedOnepOInt(40+40)', c =  color_order[9])

	#axs[n].plot(lgp_mut_avgs[n], label='LGP (40+40)', c = 'brown')
	alpha = 0.05
	axs[n].fill_between(range(cgp_avgs[n].shape[0]), cgp_s_m[n], cgp_s_p[n], color=color_order[0], alpha = alpha)
	axs[n].fill_between(range(cgp_40_avgs[n].shape[0]), cgp_40_s_m[n], cgp_40_s_p[n], color=color_order[1], alpha = alpha)
	axs[n].fill_between(range(cgp_1x_avgs[n].shape[0]), cgp_1x_s_m[n], cgp_1x_s_p[n], color=color_order[2], alpha = alpha)
	axs[n].fill_between(range(lgp_1x_avgs[n].shape[0]), lgp_1x_s_m[n], lgp_1x_s_p[n], color=color_order[3], alpha = alpha)
	axs[n].fill_between(range(cgp_2x_avgs[n].shape[0]), cgp_2x_s_m[n], cgp_2x_s_p[n], color=color_order[4], alpha = alpha)
	axs[n].fill_between(range(lgp_2x_avgs[n].shape[0]), lgp_2x_s_m[n], lgp_2x_s_p[n], color=color_order[5], alpha = alpha)
	axs[n].fill_between(range(cgp_sgx_avgs[n].shape[0]), cgp_sgx_s_m[n], cgp_sgx_s_p[n], color=color_order[6], alpha = alpha)
	axs[n].fill_between(range(lgp_avgs[n].shape[0]), lgp_s_m[n], lgp_s_p[n], color=color_order[7], alpha = 0.10)
	#axs[n].fill_between(range(cgp_nx_avgs[n].shape[0]), cgp_nx_s_m[n], cgp_nx_s_p[n], color=color_order[8], alpha = alpha)
	#axs[n].fill_between(range(lgp_fx_avgs[n].shape[0]), lgp_fx_s_m[n], lgp_fx_s_p[n], color=color_order[9], alpha = 0.10)
	#axs[n].fill_between(range(lgp_mut_avgs[n].shape[0]), lgp_mut_s_m[n], lgp_mut_s_p[n], color = "brown", alpha = 0.25)
	axs[n].set_ylim(1e-5, 1)
	axs[n].set_title(f'{f_name[n]}', fontsize=24)
	axs[n].set_ylabel("1-r^2", fontsize=18)
axs[-1].set_xlabel("Generations", fontsize=18)
#axs[0].set_ylim(0.0001, 0.1)
#axs[2].set_ylim(0.01, 1)
#axs[3].set_ylim(0.0001, 0.1)
#axs[4].set_ylim(0.00001, 0.1)
#axs[5].set_ylim(0.0001, 0.05)
#axs[6].set_ylim(0.00001, 0.001)
#axs[8].set_ylim(0, 0.01)
axs[0].legend()
fig.suptitle("Fitness over generations", fontsize=30)
fig.tight_layout(rect=[0, 0, 1, 0.95]) #https://stackoverflow.com/a/45161551
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
	cgp_base_med = np.median(cgp_base_data['p_fits'][n])
	cgp_40_med = np.median(cgp_40_data['p_fits'][n])
	cgp_1x_med = np.median(cgp_1x_data['p_fits'][n])
	cgp_2x_med = np.median(cgp_2x_data['p_fits'][n])
	cgp_sgx_med = np.median(cgp_sgx_data['p_fits'][n])
	lgp_base_med = np.median(lgp_base_data['p_fits'][n])
	lgp_1x_med = np.median(lgp_1x_data['p_fits'][n])
	lgp_2x_med = np.median(lgp_2x_data['p_fits'][n])
	#lgp_fx_med = np.median(lgp_fx_data['p_fits'][n])
	#cgp_nx_med = np.median(cgp_nx_data['p_fits'][n])
	all_meds.append(np.round([cgp_base_med, cgp_40_med, cgp_1x_med, lgp_1x_med, cgp_2x_med, lgp_2x_med, cgp_sgx_med, lgp_base_med], 5))
#print(np.array(all_meds))
all_meds = np.array(all_meds).T
for m in range(len(method_names)):
	print(f'{method_names[m]},', end='')
	m_out = ','.join(map(str, all_meds[m, :]))
	print(m_out)
	with open('../output/medians.txt', 'a') as f:
		f.write(m_out)
		f.write('\n')

p = 2 #Koza 3

fig, axs = plt.subplots(1, 3)
axs[0].hist(cgp_base_data['p_fits'][p])
axs[0].set_title('CGP(1+4) best fit distribution')
axs[1].hist(lgp_base_data['p_fits'][p])
axs[1].set_title('LGP-Ux(40+40) best fit distribution')
axs[2].hist(cgp_1x_data['p_fits'][p])
axs[2].set_title('CGP-1x(40+40) best fit distribution')
plt.savefig("../output/BestFitDistribution.png")

#significance

from scipy.stats import f_oneway, mannwhitneyu
fig, axs = plt.subplots(4, 2, figsize=(16, 20))
for p, ax in enumerate(axs.flat[:len(f_name)]):
	data = [cgp_base_data['p_fits'][p],cgp_40_data['p_fits'][p], cgp_1x_data['p_fits'][p], lgp_1x_data['p_fits'][p],cgp_2x_data['p_fits'][p],lgp_2x_data['p_fits'][p],cgp_sgx_data['p_fits'][p],lgp_base_data['p_fits'][p]]
	for y in range(len(data)):
		data[y] = data[y][~np.isnan(data[y])]
	p_mat = np.zeros((len(data), len(data)))
	for y in range(len(data)):
		for x in range(y, len(data)):
			p_mat[y,x] = mannwhitneyu(data[y], data[x]).pvalue
	p_mat = np.round(p_mat, 5)
	print(f'{f_name[p]}')
	print(method_names, sep=',')
	for m in range(len(method_names)):
		print(f'{method_names[m]},', end='')
		m_out = ','.join(map(str, p_mat[m, :]))
		print(m_out)
	print('--------')
	im = ax.imshow(p_mat)
	for y in range(len(data)):
		for x in range(len(data)):
			if x <= y:
				txt = '-'
			else:
				txt = p_mat[y, x]
			text = ax.text(x, y, txt, ha="center", va="center", color="w")
	ax.set_xticks(range(len(method_names)), method_names, rotation=0.45)
	ax.set_yticks(range(len(method_names)), method_names)
	ax.set_title(f_name[p])
	#ax.colorbar()
	
fig.tight_layout()
plt.savefig('../output/significance.png')


#LGP 1x different registers



lgp_1x_4regs_path = "../output/lgp_1x_4_regs/"
lgp_1x_2regs_path = "../output/lgp_1x_2_regs/"
lgp_1x_2regv_path = "../output/lgp_1x_2_regs_var/"
lgp_1x_0regv_path = "../output/lgp_1x_0_regs_var/"
lgp_1x_0regs_path = "../output/lgp_1x_0_regs/"
regs_f_name = ['Koza 3']

print(lgp_1x_4regs_path)
print(lgp_1x_2regs_path)
print(lgp_1x_0regs_path)
lgp_1x_4regs_data = dataDict(get_logs_lgp(lgp_1x_4regs_path, f_name = regs_f_name ))
lgp_1x_2regs_data = dataDict(get_logs_lgp(lgp_1x_2regs_path, f_name = regs_f_name ))
lgp_1x_2regv_data = dataDict(get_logs_lgp(lgp_1x_2regv_path, f_name = regs_f_name ))
lgp_1x_0regs_data = dataDict(get_logs_lgp(lgp_1x_0regs_path, f_name = regs_f_name ))
lgp_1x_0regv_data = dataDict(get_logs_lgp(lgp_1x_0regv_path, f_name = regs_f_name ))

regs_method_names = ['CGP(1+4)', 'CGP-1x', 'LGP-1x\n4 Registers\nFixed Length', 'LGP-1x\n4 Registers', 'LGP-1x\n2 Registers\nFixed Length', 'LGP-1x\n2 Registers', 'LGP-1x\n0 Registers\nFixed Length', 'LGP-1x\n0 Registers']
regs_color_order = ['blue', 'skyblue', 'lightseagreen', 'lightgreen', 'chartreuse', 'springgreen', 'mediumseagreen', 'limegreen']#, 'cadetblue', 'olive']

fig, axs = plt.subplots(1, 2, figsize=(6.5, 5))
fig.subplots_adjust(hspace=0)
#print(n)
p = 2
regs_boxes = axs[0].boxplot([cgp_base_data['p_fits'][p], cgp_1x_data['p_fits'][p], lgp_1x_4regs_data['p_fits'][0], lgp_1x_data['p_fits'][p], lgp_1x_2regs_data['p_fits'][0], lgp_1x_2regv_data['p_fits'][0], lgp_1x_0regs_data['p_fits'][0], lgp_1x_0regv_data['p_fits'][0]], showfliers = False, patch_artist = True, vert = False)
regs_box_list = regs_boxes['boxes']
axs[0].set_xscale('log')
for box, color in zip(regs_box_list, regs_color_order):
	box.set_facecolor(color)
#axs.set_yticks(list(range(1, len(regs_method_names)+1)),regs_method_names, rotation = 45)
axs[0].set_yticks([])
fig.suptitle("Register Ablation on Koza 3")
axs[0].set_title(f"{f_name[2]}", fontsize=12)
axs[0].set_xlabel("1-r^2", fontsize=10)
axs[0].set_xlim(left=1e-3)
legend_objects = [box for box in reversed(regs_box_list)]
fig.legend(legend_objects, reversed(regs_method_names), loc = 'upper center', bbox_to_anchor=(0.5, 0.96), ncol=len(legend_objects)/2)
axs[0].set_title('Fitness')
#register ablation similarity
cgp_avgs, cgp_s_p, cgp_s_m = get_avg_gens(cgp_base_data['average_retention'][:, :, 0, :])
cgp_1x_avgs, cgp_1x_s_p, cgp_1x_s_m = get_avg_gens(cgp_1x_data['average_retention'][:, :, 0, :])
lgp_1x_avgs, lgp_1x_s_p, lgp_1x_s_m = get_avg_gens(lgp_1x_data['average_retention'][:, :, 0, :])
lgp_1x_4_regs_avgs, lgp_1x_4_regs_s_p, lgp_1x_4_regs_s_m = get_avg_gens(lgp_1x_4regs_data['average_retention'][:, :, 0, :])
lgp_1x_2_regs_avgs, lgp_1x_2_regs_s_p, lgp_1x_2_regs_s_m = get_avg_gens(lgp_1x_2regs_data['average_retention'][:, :, 0, :])
lgp_1x_0_regs_avgs, lgp_1x_0_regs_s_p, lgp_1x_0_regs_s_m = get_avg_gens(lgp_1x_0regs_data['average_retention'][:, :, 0, :])
lgp_1x_2_regv_avgs, lgp_1x_2_regv_s_p, lgp_1x_2_regv_s_m = get_avg_gens(lgp_1x_2regv_data['average_retention'][:, :, 0, :])
lgp_1x_0_regv_avgs, lgp_1x_0_regv_s_p, lgp_1x_0_regv_s_m = get_avg_gens(lgp_1x_0regv_data['average_retention'][:, :, 0, :])
print(cgp_avgs.shape)
print(cgp_1x_avgs.shape)
print(lgp_1x_avgs.shape)
print(lgp_1x_0_regv_avgs.shape)

axs[1].plot(lgp_1x_0_regv_avgs[0], label = regs_method_names[-1], c = regs_color_order[-1])
axs[1].plot(lgp_1x_0_regs_avgs[0], label = regs_method_names[-2], c = regs_color_order[-2])
axs[1].plot(lgp_1x_2_regv_avgs[0], label = regs_method_names[-3], c = regs_color_order[-3])
axs[1].plot(lgp_1x_2_regs_avgs[0],label = regs_method_names[-4], c = regs_color_order[-4])
axs[1].plot(lgp_1x_avgs[2], label = regs_method_names[-5], c = regs_color_order[-5])
axs[1].plot(lgp_1x_4_regs_avgs[0], label = regs_method_names[-6], c = regs_color_order[-6])
axs[1].plot(cgp_1x_avgs[2], label = regs_method_names[-7], c = regs_color_order[-7])
axs[1].plot(cgp_avgs[2], label = regs_method_names[-8], c = regs_color_order[-8])

axs[1].fill_between(range(lgp_1x_0_regv_avgs[0].shape[0]), lgp_1x_0_regv_s_m[0], lgp_1x_0_regv_s_p[0], color=color_order[-1], alpha = alpha)
axs[1].fill_between(range(lgp_1x_0_regs_avgs[0].shape[0]), lgp_1x_0_regs_s_m[0], lgp_1x_0_regs_s_p[0], color=color_order[-2], alpha = alpha)
axs[1].fill_between(range(lgp_1x_2_regv_avgs[0].shape[0]), lgp_1x_2_regv_s_m[0], lgp_1x_2_regv_s_p[0], color=color_order[-3], alpha = alpha)
axs[1].fill_between(range(lgp_1x_2_regs_avgs[0].shape[0]), lgp_1x_2_regs_s_m[0], lgp_1x_2_regs_s_p[0], color=color_order[-4], alpha = alpha)
axs[1].fill_between(range(lgp_1x_avgs[2].shape[0]), lgp_1x_s_m[2], cgp_1x_s_p[2], color=color_order[-5], alpha = alpha)
axs[1].fill_between(range(lgp_1x_4_regs_avgs[0].shape[0]), lgp_1x_4_regs_s_m[0], lgp_1x_4_regs_s_p[0], color=color_order[-6], alpha = alpha)
axs[1].fill_between(range(cgp_1x_avgs[2].shape[0]), cgp_1x_s_m[2], cgp_1x_s_p[2], color=color_order[-7], alpha = alpha)
axs[1].fill_between(range(cgp_avgs[2].shape[0]), cgp_s_m[2], cgp_s_p[2], color=color_order[-8], alpha = alpha)
#axs[1].legend()
axs[1].set_ylim(bottom=0.0, top = 50.0)
axs[1].set_title('Average Retention')
axs[1].set_xlabel('Generations')
axs[1].set_ylabel('Similarity')

fig.tight_layout(rect=[0, 0, 1, 0.84])
plt.show()
plt.savefig("../output/regs.png")


data = [cgp_base_data['p_fits'][p], cgp_1x_data['p_fits'][p], lgp_1x_4regs_data['p_fits'][0], lgp_1x_data['p_fits'][p], lgp_1x_2regs_data['p_fits'][0], lgp_1x_2regv_data['p_fits'][0], lgp_1x_0regs_data['p_fits'][0], lgp_1x_0regv_data['p_fits'][0]]

for y in range(len(data)):
	for x in range(y, len(data)):
		p_mat[y,x] = mannwhitneyu(data[y], data[x]).pvalue
print("Register Ablation Significance")
p_mat = np.round(p_mat, 5)
print(regs_method_names, sep=',')
for m in range(len(regs_method_names)):
	print(f'{regs_method_names[m]},', end='')
	m_out = ','.join(map(str, p_mat[m, :]))
	print(m_out)


#similarity
print(cgp_base_data['average_retention'].shape)
cgp_avgs, cgp_s_p, cgp_s_m = get_avg_gens(cgp_base_data['average_retention'][:, :, 0, :])
cgp_1x_avgs, cgp_1x_s_p, cgp_1x_s_m = get_avg_gens(cgp_1x_data['average_retention'][:, :, 0, :])
cgp_2x_avgs, cgp_2x_s_p, cgp_2x_s_m = get_avg_gens(cgp_2x_data['average_retention'][:, :, 0, :])
cgp_40_avgs, cgp_40_s_p, cgp_40_s_m = get_avg_gens(cgp_40_data['average_retention'][:, :, 0, :])
cgp_sgx_avgs, cgp_sgx_s_p, cgp_sgx_s_m = get_avg_gens(cgp_sgx_data['average_retention'][:, :, 0, :])
lgp_avgs, lgp_s_p, lgp_s_m = get_avg_gens(lgp_base_data['average_retention'][:, :, 0, :])
lgp_1x_avgs, lgp_1x_s_p, lgp_1x_s_m = get_avg_gens(lgp_1x_data['average_retention'][:, :, 0, :])
lgp_2x_avgs, lgp_2x_s_p, lgp_2x_s_m = get_avg_gens(lgp_2x_data['average_retention'][:, :, 0, :])
print(cgp_avgs.shape)
print(cgp_s_p.shape)
print(cgp_s_m.shape)
ratio = 1.5
fig, axs = plt.subplots(4, 2, figsize = (6.5*ratio, 4.5*ratio))
#for n in range(len(f_name)):
for n, ax in enumerate(axs.flat[:len(f_name)]):
	ax.plot(cgp_avgs[n], label='CGP (1+4)', c = color_order[0])
	ax.plot(cgp_40_avgs[n], label='CGP(16+64)', c = color_order[1])
	ax.plot(cgp_1x_avgs[n], label='CGP-OnePoint (40+40)', c = color_order[2])
	ax.plot(lgp_1x_avgs[n], label='LGP-OnePoint (40+40)',c = color_order[3])
	ax.plot(cgp_2x_avgs[n], label='CGP-TwoPoint (40+40)', c = color_order[4])
	ax.plot(lgp_2x_avgs[n], label='LGP-TwoPoint (40+40)',c = color_order[5])
	ax.plot(cgp_sgx_avgs[n], label="CGP-Subgraph(40+40)", c = color_order[6])
	ax.plot(lgp_avgs[n], label='LGP-Uniform(40+40)', c =  color_order[7])

	alpha = 0.10
	ax.fill_between(range(cgp_avgs[n].shape[0]), cgp_s_m[n], cgp_s_p[n], color=color_order[0], alpha = alpha)
	ax.fill_between(range(cgp_40_avgs[n].shape[0]), cgp_40_s_m[n], cgp_40_s_p[n], color=color_order[1], alpha = alpha)
	ax.fill_between(range(cgp_1x_avgs[n].shape[0]), cgp_1x_s_m[n], cgp_1x_s_p[n], color=color_order[2], alpha = alpha)
	ax.fill_between(range(lgp_1x_avgs[n].shape[0]), lgp_1x_s_m[n], lgp_1x_s_p[n], color=color_order[3], alpha = alpha)
	ax.fill_between(range(cgp_2x_avgs[n].shape[0]), cgp_2x_s_m[n], cgp_2x_s_p[n], color=color_order[4], alpha = alpha)
	ax.fill_between(range(lgp_2x_avgs[n].shape[0]), lgp_2x_s_m[n], lgp_2x_s_p[n], color=color_order[5], alpha = alpha)
	ax.fill_between(range(cgp_sgx_avgs[n].shape[0]), cgp_sgx_s_m[n], cgp_sgx_s_p[n], color=color_order[6], alpha = alpha)
	ax.fill_between(range(lgp_avgs[n].shape[0]), lgp_s_m[n], lgp_s_p[n], color=color_order[7], alpha = 0.10)
	ax.set_title(f'{f_name[n]}', fontsize=12)
	if n % 2 == 0:
		ax.set_ylabel("Average Similarity", fontsize=10)
	ax.set_ylim(bottom = 0)
	if n > 5:
		ax.set_xlabel("Generations", fontsize=10)
axs.flat[-1].set_visible(False) #https://stackoverflow.com/questions/44980658/remove-the-extra-plot-in-the-matplotlib-subplot
#fig.suptitle("Average Retention of Active Instructions\nbetween Best Parents and Best Children", fontsize=20)
fig.suptitle("Average Instruction Retention", fontsize=16)
legend_objects = [box for box in box_list]
print(box_list)
print(legend_objects)
fig.legend(legend_objects, method_names_long, fontsize=10, loc = 'lower right', bbox_to_anchor = (1, 0.05), ncol=2)

#fig.tight_layout()
fig.tight_layout(rect=[0, 0, 1, 0.975]) #https://stackoverflow.com/a/45161551
#fig.subplots_adjust(top=0.9)

plt.show()
plt.savefig("../output/retention.png")
print("retention over generations")

print(cgp_base_data['average_change'].shape)
cgp_avgs, cgp_s_p, cgp_s_m = get_avg_gens(cgp_base_data['average_change'][:, :, 0, :])
cgp_1x_avgs, cgp_1x_s_p, cgp_1x_s_m = get_avg_gens(cgp_1x_data['average_change'][:, :, 0, :])
cgp_2x_avgs, cgp_2x_s_p, cgp_2x_s_m = get_avg_gens(cgp_2x_data['average_change'][:, :, 0, :])
cgp_40_avgs, cgp_40_s_p, cgp_40_s_m = get_avg_gens(cgp_40_data['average_change'][:, :, 0, :])
cgp_sgx_avgs, cgp_sgx_s_p, cgp_sgx_s_m = get_avg_gens(cgp_sgx_data['average_change'][:, :, 0, :])
lgp_avgs, lgp_s_p, lgp_s_m = get_avg_gens(lgp_base_data['average_change'][:, :, 0, :])
lgp_1x_avgs, lgp_1x_s_p, lgp_1x_s_m = get_avg_gens(lgp_1x_data['average_change'][:, :, 0, :])
lgp_2x_avgs, lgp_2x_s_p, lgp_2x_s_m = get_avg_gens(lgp_2x_data['average_change'][:, :, 0, :])
print(cgp_avgs.shape)
print(cgp_s_p.shape)
print(cgp_s_m.shape)
fig, axs = plt.subplots(len(f_name), 1, figsize = (10, 20))
for n in range(len(f_name)):
	axs[n].plot(cgp_avgs[n], label='CGP (1+4)', c = color_order[0])
	axs[n].plot(cgp_40_avgs[n], label='CGP(16+64)', c = color_order[1])
	axs[n].plot(cgp_1x_avgs[n], label='CGP-OnePoint (40+40)', c = color_order[2])
	axs[n].plot(lgp_1x_avgs[n], label='LGP-OnePoint (40+40)',c = color_order[3])
	axs[n].plot(cgp_2x_avgs[n], label='CGP-TwoPoint (40+40)', c = color_order[4])
	axs[n].plot(lgp_2x_avgs[n], label='LGP-TwoPoint (40+40)',c = color_order[5])
	axs[n].plot(cgp_sgx_avgs[n], label="CGP-Subgraph(40+40)", c = color_order[6])
	axs[n].plot(lgp_avgs[n], label='LGP-Uniform(40+40)', c =  color_order[7])

	alpha = 0.10
	axs[n].fill_between(range(cgp_avgs[n].shape[0]), cgp_s_m[n], cgp_s_p[n], color=color_order[0], alpha = alpha)
	axs[n].fill_between(range(cgp_40_avgs[n].shape[0]), cgp_40_s_m[n], cgp_40_s_p[n], color=color_order[1], alpha = alpha)
	axs[n].fill_between(range(cgp_1x_avgs[n].shape[0]), cgp_1x_s_m[n], cgp_1x_s_p[n], color=color_order[2], alpha = alpha)
	axs[n].fill_between(range(lgp_1x_avgs[n].shape[0]), lgp_1x_s_m[n], lgp_1x_s_p[n], color=color_order[3], alpha = alpha)
	axs[n].fill_between(range(cgp_2x_avgs[n].shape[0]), cgp_2x_s_m[n], cgp_2x_s_p[n], color=color_order[4], alpha = alpha)
	axs[n].fill_between(range(lgp_2x_avgs[n].shape[0]), lgp_2x_s_m[n], lgp_2x_s_p[n], color=color_order[5], alpha = alpha)
	axs[n].fill_between(range(cgp_sgx_avgs[n].shape[0]), cgp_sgx_s_m[n], cgp_sgx_s_p[n], color=color_order[6], alpha = alpha)
	axs[n].fill_between(range(lgp_avgs[n].shape[0]), lgp_s_m[n], lgp_s_p[n], color=color_order[7], alpha = 0.10)
	axs[n].set_title(f'{f_name[n]}', fontsize=24)
	axs[n].set_ylabel("Average Difference", fontsize=12)
axs[-1].set_xlabel("Generations", fontsize=18)
axs[0].legend()
fig.suptitle("Average Change between Best Parents and their Best Children", fontsize=30)
fig.tight_layout(rect=[0, 0, 1, 0.95]) #https://stackoverflow.com/a/45161551
plt.show()
plt.savefig("../output/change.png")
print("retention over generations")



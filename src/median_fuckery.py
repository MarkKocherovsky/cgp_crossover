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

color_order = ['blue', 'cyan', 'skyblue', 'lightgreen', 'steelblue', 'mediumseagreen', 'indigo', 'green', 'cadetblue', 'olive']
method_names = ["CGP(1+4)", "CGP(16+64)", "CGP-1x(40+40)","LGP-1x(40+40)", "CGP-2x(40+40)","LGP-2x(40+40)", "CGP-SGx(40+40)", "LGP-Ux(40+40)", "CGP-Nx(40+40)", "LGP-Fx(40+40)"]
method_names_long = ["CGP(1+4)", "CGP(16+64)", "CGP-OnePoint(40+40)","LGP-OnePoint(40+40)", "CGP-TwoPoint(40+40)","LGP-TwoPoint(40+40)", "CGP-Subgraph(40+40)", "LGP-Uniform(40+40)", "CGP-NodeOnePoint(40+40)", "LGP-FlattenedOnePoint(40+40)"]  

max_e = 50
def get_logs_cgp(base_path, max_e = max_e):
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
	
def get_logs_lgp(base_path, max_e = max_e):
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

lgp_1x_data = dataDict(get_logs_lgp(lgp_1x_path))
#plot boxes
print('fitness')
print(lgp_1x_data['p_fits'][0])
fig, ax = plt.subplots()
boxes = ax.boxplot(lgp_1x_data['p_fits'][0])
print('lgp_1x koza 1 median according to the box')
print(boxes['medians'][0])
print(np.quantile(lgp_1x_data['p_fits'][0], [0.25, 0.5, 0.75]))
print('lgp_1x koza 1 median according to numpy')
print(np.median(lgp_1x_data['p_fits'][0]))

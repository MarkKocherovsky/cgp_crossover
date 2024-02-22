import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sys import path
from pathlib import Path
def scatter(train_x, train_y, preds, func_name, run_name, t, fit_name, p_fit):
	fig, ax = plt.subplots()
	ax.scatter(train_x, train_y, label = 'Ground Truth')
	ax.scatter(train_x, preds, label = 'Predicted')
	fig.suptitle(f"{func_name} Trial {t}")
	ax.set_title(f"{fit_name} = {np.round(p_fit, 4)}")
	ax.legend()
	Path(f"../output/{run_name}/{func_name}/scatter/").mkdir(parents=True, exist_ok=True)
	plt.savefig(f"../output/{run_name}/{func_name}/scatter/plot_{t}.png")
	print(f"../output/{run_name}/{func_name}/scatter/plot_{t}.png")

def fit_plot(fit_track, func_name, run_name, t):
	fig, ax = plt.subplots()
	ax.plot(fit_track)
	ax.set_yscale('log')
	ax.set_title(f'{func_name} Trial {t}')
	ax.set_ylabel("1-R^2")
	ax.set_xlabel("Generations")
	Path(f"../output/{run_name}/{func_name}/plot/").mkdir(parents=True, exist_ok=True)
	plt.savefig(f"../output/{run_name}/{func_name}/plot/plot_{t}.png")

def proportion_plot(p_size, func_name, run_name, t):
	fig, ax = plt.subplots()
	ax.plot(p_size)
	ax.set_title(f'{func_name} Trial {t}')
	ax.set_ylabel("Proportion of Active Nodes")
	ax.set_xlabel("Generations")
	Path(f"../output/{run_name}/{func_name}/proportion_plot/").mkdir(parents=True, exist_ok=True)
	plt.savefig(f"../output/{run_name}/{func_name}/proportion_plot/proportion_plot_{t}.png")

from matplotlib import colormaps
def change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g):
	fig, ax = plt.subplots(figsize = (10, 5))
	hist_gens = np.array([hist_list[0] for hist_list in avg_hist_list])
	avg_hist_list = [hist_list[1] for hist_list in avg_hist_list]
	bin_edges = np.array([avg_hist_list[i][1] for i in range(len(avg_hist_list))])
	hists = np.array([avg_hist_list[i][0] for i in range(len(avg_hist_list))])
	bin_centers = []
	for i in range(bin_edges.shape[0]):
		centers = []
		for j in range(0, bin_edges.shape[1]-1):
			centers.append((bin_edges[i][j]+bin_edges[i][j+1])/2)
		bin_centers.append(centers)
	bin_centers = np.array(bin_centers)
	for i in range(hist_gens.shape[0]):
		g = hist_gens[i]
		x = np.full((bin_centers.shape[1],), g)
		ax.scatter(x, bin_centers[i, :], c=hists[i], cmap = colormaps['Greys'], alpha = 0.33)
	ax.set_yscale('log')
	ax.set_ylabel('Frequency[Fit(Child)-Fit(Parent)]')
	ax.set_xlabel('Generations')
	ax.set_xlim(0, max_g)
	fig.tight_layout()
	Path(f"../output/{run_name}/{func_name}/change_hists/").mkdir(parents=True, exist_ok=True)
	plt.savefig(f"../output/{run_name}/{func_name}/change_hists/change_hists{t}.png")
	return bin_centers, hist_gens, avg_hist_list
def change_avg_plot(avg_change_list, std_change_list, func_name, run_name, t, win_length = 100, order = 4):
	avg_change_list = np.array(avg_change_list)
	std_change_list = np.array(std_change_list)
	fig, ax = plt.subplots(figsize = (10, 5))
	try:
		ax.plot(savgol_filter(avg_change_list, win_length, order), c = 'blue', label = 'f(Best Child) - f(Parent)')
		ax.fill_between(range(len(avg_change_list)), savgol_filter(avg_change_list-std_change_list, win_length, order), savgol_filter(avg_change_list+std_change_list, win_length, order), color = 'blue', alpha = 0.1)
	except (np.linalg.LinAlgError, ValueError):
		ax.plot(avg_change_list, label = 'f(Best Child) - f(Parent)')
		ax.fill_between(range(len(avg_change_list)), (avg_change_list-std_change_list), (avg_change_list+std_change_list), color = 'blue', alpha = 0.1)
		
	change_stddev = np.nanstd(avg_change_list)
	fig.suptitle(f'{func_name} Trial {t}')
	ax.set_title(f'Darker Color = Higher Frequency')
	#ax.set_yscale('log')
	ax.legend()
	ax.set_ylabel('Fit(Best Child) - Fit(Parent)')
	ax.set_xlabel('Generations')
	fig.tight_layout()
	Path(f"../output/{run_name}/{func_name}/change_plot/").mkdir(parents=True, exist_ok=True)
	plt.savefig(f"../output/{run_name}/{func_name}/change_plot/change_{t}.png")

def retention_plot(ret_avg_list, ret_std_list, func_name, run_name, t, win_length = 100, order = 2):
	ret_avg_list = np.array(ret_avg_list)
	ret_std_list = np.array(ret_std_list)
	fig, ax = plt.subplots()
	try:
		ax.plot(savgol_filter(ret_avg_list, win_length, 3))
		ax.fill_between(range(len(ret_avg_list)), savgol_filter(ret_avg_list-ret_std_list, win_length, order), savgol_filter(ret_avg_list+ret_std_list, win_length, order), color = 'blue', alpha = 0.1)
	except:
		ax.plot(ret_avg_list)
		ax.fill_between(range(len(ret_avg_list)), (ret_avg_list-ret_std_list), (ret_avg_list+ret_std_list), color = 'blue', alpha = 0.1)
	ax.set_title(f'{func_name} Trial {t}')
	ax.set_ylabel('Similarity between Parent and Best Child')
	ax.set_xlabel('Generations')
	Path(f"../output/{run_name}/{func_name}/retention/").mkdir(parents=True, exist_ok=True)
	plt.savefig(f"../output/{run_name}/{func_name}/retention/retention_{t}.png")

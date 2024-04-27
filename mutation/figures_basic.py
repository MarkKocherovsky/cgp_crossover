import numpy as np
import pickle
import matplotlib.pyplot as plt
from function import *
from pathlib import Path
from sys import argv

c = Collection()
f_list = c.func_list
f_name = c.name_list

cgp_base_path = "../output/cgp_withoutcrossover/"
cgp_base = []
cgp_base_p_fits = []
cgp_1x_path = "../output/cgp_one/"
cgp_1x = []
cgp_1x_p_fits = []
cgp_2x_path = "../output/cgp_two/"
cgp_2x = []
cgp_2x_p_fits = []
color_order = ['blue', 'lightgreen', 'red']
method_names = ["CGP", "CGP_1X", "CGP_2X"]
method_names_long = ["CGP", "CGP_1X", "CGP_2X"]
max_e = 50

Path(f"../output/figures/").mkdir(parents=True, exist_ok=True)


def get_logs_cgp(base_path, max_e=max_e, f_name=f_name):
    full_fits = []
    fit_tracks = []
    for name in f_name:
        print(f"Loading {base_path}{name}")
        p_log = []
        track_log = []
        for e in range(1, max_e + 1):
            p = f'{base_path}{name}/log/output_{e}.pkl'
            with open(p, "rb") as f:
                fit_track = pickle.load(f)
                p_fit = pickle.load(f)
                if np.isnan(p_fit):
                    p_fit = np.PINF
            track_log.append(fit_track)
            p_log.append(p_fit)
        fit_tracks.append(track_log)
        full_fits.append(p_log)
    return [np.array(full_fits), np.array(fit_tracks)]


def datadict(data):
    return {'p_fits': data[0], 'fit_track': data[1], }


cgp_base_data = datadict(get_logs_cgp(cgp_base_path))
cgp_1x_data = datadict(get_logs_cgp(cgp_1x_path))
cgp_2x_data = datadict(get_logs_cgp(cgp_2x_path))
fig, axs = plt.subplots(len(f_name), 1, figsize=(9.5 * 1.1, 11 * 1.1))
fig.subplots_adjust(hspace=0)
from copy import deepcopy

for n in range(len(f_name)):
    boxes = axs[n].boxplot([cgp_base_data['p_fits'][n], cgp_1x_data['p_fits'][n], cgp_2x_data['p_fits'][n]],
                           showfliers=False, patch_artist=True)
    box_list = boxes['boxes']
    axs[n].set_yscale('log')
    for box, color in zip(box_list, color_order):
        box.set_facecolor(color)
    axs[n].set_xticks(list(range(1, len(method_names) + 1)), method_names, rotation=0, fontsize=11)
    axs[n].set_title(f"{f_name[n]}", fontsize=12)
    axs[n].set_ylabel("1-r^2", fontsize=11)
    axs[n].set_ylim(bottom=1e-6)
    axs[n].tick_params(axis='y', labelsize=10)
    axs[n].tick_params(axis='x', labelsize=10)


fig.suptitle("Fitness Evaluation", fontsize=16)
legend_objects = [box for box in box_list]
fig.legend(legend_objects, method_names_long, fontsize=10, ncol=2, bbox_to_anchor=(0.5, 0.965), loc='upper center')
fig.tight_layout(rect=[0, 0, 1, 0.920])
plt.show()
plt.savefig("../output/figures/fit1.png")



def get_avg_gens(f):
    avgs = []
    std_devs = []
    for p in f:
        avgs.append(np.average(p, axis=0))
        std_devs.append(np.std(p, axis=0))
    avgs = np.array(avgs)
    std_devs = np.array(std_devs)
    return avgs, avgs - std_devs, avgs + std_devs


def get_err_ribbon(avgs, stds):
    return avgs + stds, avgs - stds


cgp_avgs, cgp_s_p, cgp_s_m = get_avg_gens(cgp_base_data['fit_track'])
cgp_1x_avgs, cgp_1x_s_p, cgp_1x_s_m = get_avg_gens(cgp_1x_data['fit_track'])
cgp_2x_avgs, cgp_2x_s_p, cgp_2x_s_m = get_avg_gens(cgp_2x_data['fit_track'])
fig, axs = plt.subplots(len(f_name), 1, figsize=(10, 20))
for n in range(len(f_name)):
    axs[n].set_yscale('log')
    axs[n].plot(cgp_avgs[n], label='CGP', c=color_order[0])
    axs[n].plot(cgp_1x_avgs[n], label='CGP-OnePoint', c=color_order[1])
    axs[n].plot(cgp_2x_avgs[n], label='CGP-TwoPoint', c=color_order[2])
    alpha = 0.05
    axs[n].fill_between(range(cgp_avgs[n].shape[0]), cgp_s_m[n], cgp_s_p[n], color=color_order[0], alpha=alpha)
    axs[n].fill_between(range(cgp_1x_avgs[n].shape[0]), cgp_1x_s_m[n], cgp_1x_s_p[n], color=color_order[1], alpha=alpha)
    axs[n].fill_between(range(cgp_2x_avgs[n].shape[0]), cgp_2x_s_m[n], cgp_2x_s_p[n], color=color_order[2], alpha=alpha)
    axs[n].set_ylim(1e-5, 1)
    axs[n].set_title(f'{f_name[n]}', fontsize=24)
    axs[n].set_ylabel("1-r^2", fontsize=18)
axs[-1].set_xlabel("Generations", fontsize=18)
axs[0].legend()
fig.suptitle("Fitness over generations", fontsize=30)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
plt.savefig("../output/figures/fitness1.png")


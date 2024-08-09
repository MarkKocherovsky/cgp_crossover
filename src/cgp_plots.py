from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import lgp_fitness
from cgp_fitness import *
from matplotlib import colormaps


def scatter_elites(elites, func, func_name, run_name, t, g, fit_name, p_fit, bias=None,
                   points=250, mode='cgp'):
    if not None:
        bias = bias
    else:
        bias = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def createInputVector(x_in, c, num_inputs=1):
        vec = np.zeros((x_in.shape[0], c.shape[0] + 1))
        x_in = x_in.reshape(-1, num_inputs)
        vec[:, :num_inputs] = x_in
        vec[:, num_inputs:] = c
        return vec

    #create input x
    x_dom = func.x_rng
    grain = (x_dom[1] - x_dom[0]) / points
    x = np.arange(x_dom[0], x_dom[1] + grain, grain)
    # map to y
    try:
        y = func.func(x)
    except:
        y = np.array([func.func(x_1) for x_1 in x])
    if mode == 'cgp':
        # create input vector
        train_x = createInputVector(x, np.array(bias))
        #make predictions
        fitness = Fitness()
        preds = np.array([fitness(train_x, y, elite, opt=1)[0] for elite in elites])
    elif mode == 'lgp':
        fitness = lgp_fitness.Fitness(x, bias, y, elites, func, (add, sub, mul, div))
        _, a, b = fitness()
        preds = np.array([fitness.predict(elite, a, b, x) for elite, a, b in zip(elites, a, b)])
    else:
        raise NameError(f"cgp_plots::scatter_elites: asked for mode {mode}, only 'cgp' and 'lgp' are valid!")
    #plot figure
    fig, ax = plt.subplots()
    ax.plot(x, y, label='Ground Truth', color='black')
    colormap = cm.get_cmap('coolwarm', preds.shape[0])
    for p in range(preds.shape[0]):
        ax.plot(x, preds[p], label=f'Elite {p}', color=colormap(p))
    fig.suptitle(f"{func_name} Trial {t} Generation {g}")
    ax.set_title(f"{fit_name} = {np.round(p_fit, 4)}")
    ax.legend()
    Path(f"../output/{run_name}/{func_name}/plot_elites/trial_{t}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"../output/{run_name}/{func_name}/plot_elites/trial_{t}/generation_{g}.png")


def scatter(train_x, train_y, preds, func_name, run_name, t, fit_name, p_fit):
    fig, ax = plt.subplots()
    ax.scatter(train_x, train_y, label='Ground Truth')
    ax.scatter(train_x, preds, label='Predicted')
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


def sharp_plot(sharp_list, sharp_std, sharp_out_list, sharp_out_std, func_name, run_name, t, window_length=100,
               order=2):
    fig, ax = plt.subplots()
    sharp_list = np.array(sharp_list)
    sharp_std = np.array(sharp_std)
    sharp_out_list = np.array(sharp_out_list)
    sharp_out_std = np.array(sharp_out_std)
    try:
        ax.plot(savgol_filter(sharp_list, window_length, order), label="SAM-In", color='b')
    except:
        ax.plot(sharp_list, label="SAM-In", color='b')
    ax2 = ax.twinx()
    ax2.set_ylabel('Avg(SAM-Out)')
    try:
        ax2.plot(savgol_filter(sharp_out_list, window_length, order), label="SAM-Out", color='orange')
    except:
        ax2.plot(sharp_out_list, label="SAM-Out", color='orange')
    try:
        ax.fill_between(range(sharp_list.shape[0]), savgol_filter((sharp_list - sharp_std), window_length, order),
                        savgol_filter((sharp_list + sharp_std), window_length, order), alpha=0.15, color='b')
    except:
        ax.fill_between(range(sharp_list.shape[0]), (sharp_list - sharp_std), (sharp_list + sharp_std), alpha=0.15,
                        color='b')
    try:
        ax2.fill_between(range(sharp_out_list.shape[0]),
                         savgol_filter((sharp_out_list - sharp_out_std), window_length, order),
                         savgol_filter((sharp_out_list + sharp_out_std), window_length, order), alpha=0.15,
                         color='orange')
    except:
        ax.fill_between(range(sharp_out_list.shape[0]), (sharp_out_list - sharp_out_std),
                        (sharp_out_list + sharp_out_std), alpha=0.15, color='orange')
    #ax.set_yscale('log')
    ax.set_title(f'{func_name} Trial {t}')
    ax.set_ylabel("Avg(SAM-In)")
    ax.set_xlabel("Generations")
    #from ChatGPT
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2)
    ###
    plt.tight_layout()
    Path(f"../output/{run_name}/{func_name}/sharpness/").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"../output/{run_name}/{func_name}/sharpness/sharpness_{t}.png")


def sharp_bar_plot(sam_in, sam_out, func_name, run_name, t, g=None):
    bar_width = 0.35
    assert len(sam_in) == len(sam_out)
    index = np.arange(len(sam_in))  #partially from chatgpt
    fig, ax = plt.subplots()
    bar1 = ax.bar(index, sam_in, bar_width, color='blue', label='SAM-In')
    ax2 = ax.twinx()
    bar2 = ax2.bar(index + bar_width, sam_out, bar_width, color='orange', label='SAM-In')
    ax.set_xlabel("Elites (Descending Order)")
    ax.set_ylabel("SAM-In")
    ax2.set_ylabel("SAM-Out")
    ax.set_title(f"{func_name} trial {t} generation {g}")
    if g is not None:
        path = f"../output/{run_name}/{func_name}/sharp_compare/trial_{t}/"
        gen = f'_{g}'
    else:
        path = f"../output/{run_name}/{func_name}/sharp_compare/"
        gen = ''
    Path(path).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{path}sharp_compare_{t}{gen}.png")


def proportion_plot(p_size, func_name, run_name, t):
    fig, ax = plt.subplots()
    ax.plot(p_size)
    ax.set_title(f'{func_name} Trial {t}')
    ax.set_ylabel("Proportion of Active Nodes")
    ax.set_xlabel("Generations")
    Path(f"../output/{run_name}/{func_name}/proportion_plot/").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"../output/{run_name}/{func_name}/proportion_plot/proportion_plot_{t}.png")


def change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g, opt=1):
    fig, ax = plt.subplots(figsize=(10, 5))
    hist_gens = np.array([hist_list[0] for hist_list in avg_hist_list])
    avg_hist_list = [hist_list[1] for hist_list in avg_hist_list]
    bin_edges = np.array([avg_hist_list[i][1] for i in range(len(avg_hist_list))])
    hists = np.array([avg_hist_list[i][0] for i in range(len(avg_hist_list))])
    bin_centers = []
    for i in range(bin_edges.shape[0]):
        centers = []
        for j in range(0, bin_edges.shape[1] - 1):
            centers.append((bin_edges[i][j] + bin_edges[i][j + 1]) / 2)
        bin_centers.append(centers)
    bin_centers = np.array(bin_centers)
    for i in range(hist_gens.shape[0]):
        g = hist_gens[i]
        x = np.full((bin_centers.shape[1],), g)
        ax.scatter(x, bin_centers[i, :], c=hists[i], cmap=colormaps['Greys'], alpha=0.33)
    ax.set_yscale('log')
    ax.set_ylabel('Frequency[Fit(Child)-Fit(Parent)]')
    ax.set_xlabel('Generations')
    ax.set_xlim(0, max_g)
    fig.tight_layout()
    Path(f"../output/{run_name}/{func_name}/change_hists/").mkdir(parents=True, exist_ok=True)
    if opt == 0:
        plt.savefig(f"../output/{run_name}/{func_name}/change_hists/change_hists{t}.png")
    return bin_centers, hist_gens, avg_hist_list


def change_avg_plot(avg_change_list, std_change_list, func_name, run_name, t, win_length=100, order=4):
    avg_change_list = np.array(avg_change_list)
    std_change_list = np.array(std_change_list)
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        ax.plot(savgol_filter(avg_change_list, win_length, order), c='blue', label='f(Best Child) - f(Parent)')
        ax.fill_between(range(len(avg_change_list)),
                        savgol_filter(avg_change_list - std_change_list, win_length, order),
                        savgol_filter(avg_change_list + std_change_list, win_length, order), color='blue', alpha=0.1)
    except (np.linalg.LinAlgError, ValueError):
        ax.plot(avg_change_list, label='f(Best Child) - f(Parent)')
        ax.fill_between(range(len(avg_change_list)), (avg_change_list - std_change_list),
                        (avg_change_list + std_change_list), color='blue', alpha=0.1)

    fig.suptitle(f'{func_name} Trial {t}')
    ax.set_title(f'Darker Color = Higher Frequency')
    #ax.set_yscale('log')
    ax.legend()
    ax.set_ylabel('Fit(Best Child) - Fit(Parent)')
    ax.set_xlabel('Generations')
    fig.tight_layout()
    Path(f"../output/{run_name}/{func_name}/change_plot/").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"../output/{run_name}/{func_name}/change_plot/change_{t}.png")


def retention_plot(ret_avg_list, ret_std_list, func_name, run_name, t, win_length=100, order=2):
    ret_avg_list = np.array(ret_avg_list)
    ret_std_list = np.array(ret_std_list)
    fig, ax = plt.subplots()
    try:
        ax.plot(savgol_filter(ret_avg_list, win_length, 3))
        ax.fill_between(range(len(ret_avg_list)), savgol_filter(ret_avg_list - ret_std_list, win_length, order),
                        savgol_filter(ret_avg_list + ret_std_list, win_length, order), color='blue', alpha=0.1)
    except:
        ax.plot(ret_avg_list)
        ax.fill_between(range(len(ret_avg_list)), (ret_avg_list - ret_std_list), (ret_avg_list + ret_std_list),
                        color='blue', alpha=0.1)
    ax.set_title(f'{func_name} Trial {t}')
    ax.set_ylabel('Similarity between Parent and Best Child')
    ax.set_xlabel('Generations')
    Path(f"../output/{run_name}/{func_name}/retention/").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"../output/{run_name}/{func_name}/retention/retention_{t}.png")


def drift_plot(drift_list, drift_cum, func_name, run_name, t):
    drift_list = np.array(drift_list)
    drift_cum = np.round(drift_cum, 5)
    fig, ax = plt.subplots()
    #plt.yscale('log')
    ax.plot(drift_list[:, 0], label=f"Deleterious, Total {drift_cum[0]}", color='red')
    ax.plot(drift_list[:, 1], label=f'Near-Neutral, Total {drift_cum[1]}', color='blue')
    ax.plot(drift_list[:, 2], label=f'Beneficial, Total {drift_cum[2]}', color='green')
    ax.set_title(f'{func_name} Trial {t}')
    ax.set_ylabel('Proportion of Cumulative')
    ax.set_xlabel('Generations')
    ax.legend()
    Path(f"../output/{run_name}/{func_name}/mutations/").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"../output/{run_name}/{func_name}/mutations/mutations_{t}.png")


def impact_plot(impact_list, func_name, run_name, t):
    impact_list = np.array(impact_list)
    fig, ax = plt.subplots()
    ax.plot(impact_list)
    ax.set_title(f'{func_name} Trial {t}')
    ax.set_ylabel('Selection Impact')
    ax.set_xlabel('Generations')
    Path(f"../output/{run_name}/{func_name}/selection_impact/").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"../output/{run_name}/{func_name}/selection_impact/selection_impact_{t}.png")

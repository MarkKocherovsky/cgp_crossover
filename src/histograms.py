import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
import numpy as np
import seaborn as sns

# Load data
fitness = pd.read_csv('../output/graphs/median_end_test_fitnesses_elite_tournament.csv')
sizes = pd.read_csv('../output/graphs/median_end_sizes_elite_tournament.csv')

# Set 'Problem' as index
try:
    fitness.set_index('Problem', inplace=True)
except:
    fitness.set_index('Unnamed', inplace=True)
sizes.set_index('Problem', inplace=True)

method_handles = {}
# color_list = ['blue', 'red', 'orange', 'gold', 'green', 'cyan', 'violet',  'brown', 'chartreuse', 'deepskyblue', 'indigo', 'deeppink', 'grey','peru', 'olivedrab']
color_list = ['blue', 'lightblue', 'green', 'deeppink', 'orange', 'red', 'lightcoral', 'grey', 'olivedrab', 'purple',
              'dodgerblue', 'chartreuse', 'indigo', 'peru', 'gold', 'tomato']
name_list = ['CGP(1+4)', 'CGP(1+4)-F', 'CGP(40+40)-1x', 'CGP(40+40)-1x1d', 'CGP(40+40)-Ux', 'CGP(40+40)-Ux1d',
             'CGP(40+40)-SGx']  # , 'CGP(40+40)-S1x',
#             'CGP(40+40)-AIS1x', 'CGP(40+40)-ASUx', 'CGP(40+40)-AISUx', 'CGP(40+40)-ISUx']
# Plot each problem
print(fitness.head)


def plot_histogram(metric, table, y_label, full_title, name, color_list, name_list, log=False):
    # Setup subplot grid
    n_problems = len(table.index)
    # n_problems = 4
    ncols = 2
    nrows = (n_problems + 1) // ncols  # round up

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 10), squeeze=False)
    for idx, (problem, ax) in enumerate(zip(table.index, axes.flatten())):

        median_lines = []
        for j, method in enumerate(table.columns):
            # Path(f'../output/intermediate_results/{problem}_{xover}_{selection}_{metric}.csv')
            if j < 2:
                selection = 'elite'
            else:
                selection = 'elite_tournament'
            path = f'../output/intermediate_results/{problem.replace(" ", "")}_1d_{method}_{selection}_{metric}.csv'
            print(path)
            data = pd.read_csv(path, header=None).squeeze()
            if log:
                data = data[data > 0]
                data = np.log10(data)
            median = data.median()
            print(f"[{method}] Median: {median}, Data size: {len(data)}")

            sns.kdeplot(data=data, ax=ax, color=color_list[j], label=name_list[j])
            lines = ax.lines
            if len(lines) > 0:
                kde_data = lines[-1].get_xydata()
                if kde_data.size > 0:
                    kde_bound = np.interp(median, kde_data[:, 0], kde_data[:, 1])
                    # Now you can use kde_bound safely
                else:
                    print(f"[{method}] Warning: KDE line exists but has no data.")
                    print(f'kde_data:\n{kde_data}')
            else:
                print(f"[{method}] Warning: No KDE line was plotted.")
                print(f'kde_data:\n{kde_data}')
            # kde_data = ax.lines[j].get_xydata()
            # kde_bound = np.interp(median, kde_data[:, 0], kde_data[:, 1])
            median_lines.append(([median, median], [0, kde_bound], color_list[j]))
            ##fill 1st to 3rd quartile
            x_start, x_end = np.quantile(data, [0.25, 0.75])
            x_vals, y_vals = kde_data[:, 0], kde_data[:, 1]

            # Interpolate y-values at the exact endpoints
            y_start = np.interp(x_start, x_vals, y_vals)
            y_end = np.interp(x_end, x_vals, y_vals)

            # Mask x within the desired range
            mask = (x_vals >= x_start) & (x_vals <= x_end)
            x_segment = x_vals[mask]
            y_segment = y_vals[mask]

            # Insert interpolated endpoints
            x_fill = np.concatenate([[x_start], x_segment, [x_end]])
            y_fill = np.concatenate([[y_start], y_segment, [y_end]])

            # Fill under the curve
            ax.fill_between(x_fill, y_fill, color=color_list[j], alpha=0.075)
            # Save handle for legend
            if method not in method_handles:
                method_handles[name_list[j]], = ax.plot([], [], color=color_list[j], label=name_list[j])  # dummy line

            # Save one handle per method (for legend)

        ax.set_title(problem, fontsize=12)
        ax.set_xlabel('')
        if idx % 2 == 0:
            ax.set_ylabel('Kernel Density', fontsize=11)
        else:
            ax.set_ylabel('')

        for j, method in enumerate(table.columns):
            x, y, c = median_lines[j]
            ax.plot(x, y, c, alpha=0.85, linestyle='dashed')
    # Hide unused subplots
    for ax in axes.flatten()[-2:]:
        ax.set_xlabel(y_label, fontsize=11)
    for ax in axes.flatten()[n_problems:]:
        ax.axis('off')
    fig.suptitle(full_title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.legend(method_handles.values(), method_handles.keys(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.95))
    plt.savefig(f'../output/graphs/{name}.pdf')


plot_histogram('min_fitnesses', fitness, "Fitness Magnitude", 'Fitness Distributions', 'fit_histogram', color_list,
               name_list, log=True)

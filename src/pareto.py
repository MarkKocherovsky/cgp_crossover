import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
import numpy as np

# Load data
fitness = pd.read_csv('../output/graphs/median_end_fitnesses_elite_tournament.csv')
sizes = pd.read_csv('../output/graphs/median_end_sizes_elite_tournament.csv')

# Set 'Problem' as index
fitness.set_index('Problem', inplace=True)
sizes.set_index('Problem', inplace=True)

# Setup subplot grid
n_problems = len(fitness.index)
#n_problems = 4
ncols = 2
nrows = (n_problems + 1) // ncols  # round up

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 10), squeeze=False)
method_handles = {}
color_list = ['blue', 'green', 'deeppink', 'orange', 'red', 'lightcoral', 'grey', 'olivedrab', 'purple', 'dodgerblue', 'chartreuse', 'indigo', 'peru', 'gold', 'tomato']
name_list = ['CGP(1+4)', 'CGP(40+40)-1x', 'CGP(40+40)-Ux', 'CGP(40+40) - SGx', 'CGP(40+40)-S1x', 
             'CGP(40+40)-AIS1x', 'CGP(40+40)-ASUx', 'CGP(40+40)-AISUx', 'CGP(40+40)-ISUx'] 

# Plot each problem
for idx, (problem, ax) in enumerate(zip(fitness.index, axes.flatten())):
    #if problem not in ['Koza 3', 'Nguyen 5', 'Nguyen 7', 'Rastrigin']:
    #    continue
    x = []
    y = []
    for j, method in enumerate(fitness.columns):
        #Path(f'../output/intermediate_results/{problem}_{xover}_{selection}_{metric}.csv')
        if j < 1:
           selection='elite'
        else:
           selection='elite_tournament'
        fit_data = pd.read_csv(f'../output/intermediate_results/{problem.replace(" ", "")}_1d_{method}_{selection}_min_fitnesses.csv', header=None)
        size_data = pd.read_csv(f'../output/intermediate_results/{problem.replace(" ", "")}_1d_{method}_{selection}_best_sizes.csv', header=None)
        f_data = fit_data.to_numpy(copy=True)
        sb = ax.scatter(size_data, fit_data, c=color_list[j], s=5)
        #median
        sx = sizes.loc[problem, method]
        fy = fitness.loc[problem, method]
        sc = ax.scatter(sx, fy, s=20, label=method, c = color_list[j], edgecolors='black')
        # Save one handle per method (for legend)
        if method not in method_handles:
            method_handles[name_list[j]] = sc
        x = x + size_data.to_numpy().flatten().tolist()
        y = y + fit_data.to_numpy().flatten().tolist()
    ax.set_title(problem, fontsize=13)
    ax.set_yscale('log')
    ax.set_xlabel('')
    if idx % 2 == 0:
        ax.set_ylabel('Fitness',fontsize=11)
    else:
        ax.set_ylabel('')

     # Convert to numpy arrays for regression
    x = np.array(x)
    y = np.array(y)

    # Compute linear regression on original or log scale
    valid_mask = y > 0
    y_valid = y[valid_mask]
    x_valid = x[valid_mask]
    slope, intercept, r_value, _, _ = linregress(x_valid, np.log(y_valid))

    # Plot the trendline
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = np.exp(intercept + slope * x_fit)
    ax.plot(x_fit, y_fit, color='black', linestyle='--', linewidth=1)

    # Annotate R²
    ax.text(0.05, 0.95, f"$R^2$ = {r_value**2:.3f}", transform=ax.transAxes,
            ha='left', va='top', fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
# Hide unused subplots
for ax in axes.flatten()[n_problems:]:
    ax.axis('off')
for ax in axes.flatten()[-2:]:
    ax.set_xlabel('Size',fontsize=11)
fig.suptitle('Fitness vs. Size', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.legend(method_handles.values(), method_handles.keys(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.95))
plt.savefig('../output/graphs/pareto_per_problem.pdf')


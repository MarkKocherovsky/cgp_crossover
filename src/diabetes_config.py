import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

# Dictionary to store loaded figures
figures = {}

# List of keys for the paths
keys = [
    'fitness_diabetes', 'sharpness_diabetes', 'density_distro_diabetes', 'fitness_gen_diabetes', 'fitness_gen_dnc_diabetes', 'mut_density_distro_gens_diabetes', 
    'xov_density_distro_gens_diabetes', 'xov_density_distro_gens_short_diabetes', 'mut_drift_diabetes', 'xov_drift_diabetes', 'prog_prop_diabetes', 
    'similarity_diabetes', 'semantic_diversity_diabetes', 'fitness_mean_diabetes', 'nodes_diabetes'
]

main_path = '../output/graphs_raw/'
out_path = '../output/'
image_filetype = '.png'
store_filetype = '.pkl'

# Use a dictionary comprehension to create the graph_paths dictionary
graph_paths = {key: f'{main_path}{key}{store_filetype}' for key in keys}

# Iterate over each graph path and load the corresponding pickle file
for key, file_path in graph_paths.items():
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            fig = pickle.load(f)  # Load the figure object
            figures[key] = fig  # Store the figure in the dictionary
    else:
        print(f"File not found: {file_path}")

# Set global font sizes
plt.rcParams.update({
    'font.size': 32,               # Default text size
    'axes.titlesize': 26,          # Font size for axes titles
    'axes.labelsize': 24,          # Font size for x and y labels
    'xtick.labelsize': 22,         # Font size for x-axis tick labels
    'ytick.labelsize': 22,         # Font size for y-axis tick labels
    'legend.fontsize': 22,         # Font size for legend text
    'figure.titlesize': 28         # Font size for figure title
})

fig = figures['fitness_diabetes']

scale = 3
fig.set_size_inches(7.5 * scale, 2 * scale)
axs = fig.get_axes()
axs[0].set_ylim(4e-1, 1)
#axs[2].set_ylim(1e-4, 1)
#axs[3].set_ylim(1e-4, 1e-1)
#axs[4].set_ylim(1e-5, 1e-1)
#axs[5].set_ylim(1e-4, 1e-2)
fig.suptitle(fig.get_suptitle(), fontsize=28)
for ax in axs:
    ax.set_title(ax.get_title(), fontsize=24)         # Set title font size
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)       # Set x-axis label font size
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)       # Set y-axis label font size
    ax.tick_params(axis='y', which='major', labelsize=18)  # Set tick label font size
    ax.tick_params(axis='x', which='major', labelsize=15)  # Set tick label font size

legend = fig.legend()
legend.set_ncols(5)  # Set the number of columns
# Directly access the legend's text properties
for text in legend.get_texts():
    text.set_fontsize(20)  # Update the font size for each text in the legend

fig.tight_layout(rect=[0, 0, 1, 0.97])

"""
fig = figures['sharpness_diabetes']
fig.set_size_inches(7.5 * scale, 2 * scale)
axs = fig.get_axes()
#axs[0].set_ylim(1e-9, 1e-1)
#axs[2].set_ylim(1e-4, 1)
#axs[4].set_ylim(1e-5, 1e-1)
#axs[8].set_ylim(1e-2, 1)
#axs[9].set_ylim(1e-3, 1)
axs[10].set_ylim(1e-6, 1e-3)
fig.suptitle(fig.get_suptitle(), fontsize=28)
for ax in axs:
    ax.set_title(ax.get_title(), fontsize=24)         # Set title font size
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)       # Set x-axis label font size
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)       # Set y-axis label font size
    ax.tick_params(axis='y', which='major', labelsize=18)  # Set tick label font size
    ax.tick_params(axis='x', which='major', labelsize=15)  # Set tick label font size

legend = fig.legend()
legend.set_ncols(5)  # Set the number of columns
# Directly access the legend's text properties
for text in legend.get_texts():
    text.set_fontsize(20)  # Update the font size for each text in the legend

fig.tight_layout(rect=[0, 0, 1, 0.97])
"""


fig = figures['fitness_mean_diabetes']
axs = fig.get_axes()
fig.suptitle(fig.get_suptitle(), fontsize=20)
for ax in axs:
    ax.set_ylim(5e-1, 1)
    ax.set_xlim(10, 3000)
    ax.set_title(ax.get_title(), fontsize=18)         # Set title font size
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)       # Set x-axis label font size
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)       # Set y-axis label font size
    ax.tick_params(axis='y', which='major', labelsize=16)  # Set tick label font size
    ax.tick_params(axis='y', which='minor', labelsize=16)  # Set tick label font size
    ax.tick_params(axis='x', which='major', labelsize=16)  # Set tick label font size

# regular mode
"""
axs[0].set_ylim(1e-4, 1e-1)
axs[1].set_ylim(1e-3, 1)
axs[2].set_ylim(1e-2, 1)
axs[3].set_ylim(1e-4, 1e-1)
axs[4].set_ylim(1e-5, 1e-1)
axs[5].set_ylim(1e-4, 1e-1)
axs[6].set_ylim(8e-6, 1e-3)
axs[7].set_ylim(2e-2, 1e-1)
axs[8].set_ylim(3e-1, 5e-1)
axs[9].set_ylim(1e-2, 1)
axs[10].set_ylim(5e-4, 1e-3)
"""
#axs[0].set_ylim(1e-2, 1) # koza 3
#axs[1].set_ylim(3e-1, 5e-1) # rastrigin
fig.set_size_inches(scale*3, scale*3)
#legend = fig.legend()
#legend.set_ncols(3)  # Set the number of columns
# Directly access the legend's text properties
#for text in legend.get_texts():
#    text.set_fontsize(16)  # Update the font size for each text in the legend


#fig.set_size_inches(20, 40)
fig.tight_layout(rect=[0, 0, 1, 0.90])


fig = figures['fitness_gen_diabetes']
axs = fig.get_axes()
fig.suptitle(fig.get_suptitle(), fontsize=20)
for ax in axs:
    ax.set_ylim(5e-1, 7e-1)
    ax.set_xlim(10, 3000)
    ax.set_title(ax.get_title(), fontsize=18)         # Set title font size
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)       # Set x-axis label font size
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)       # Set y-axis label font size
    ax.tick_params(axis='y', which='major', labelsize=16)  # Set tick label font size
    ax.tick_params(axis='y', which='minor', labelsize=16)  # Set tick label font size
    ax.tick_params(axis='x', which='major', labelsize=16)  # Set tick label font size

# regular mode
"""
axs[0].set_ylim(1e-4, 1e-1)
axs[1].set_ylim(1e-3, 1)
axs[2].set_ylim(1e-2, 1)
axs[3].set_ylim(1e-4, 1e-1)
axs[4].set_ylim(1e-5, 1e-1)
axs[5].set_ylim(1e-4, 1e-1)
axs[6].set_ylim(8e-6, 1e-3)
axs[7].set_ylim(2e-2, 1e-1)
axs[8].set_ylim(3e-1, 5e-1)
axs[9].set_ylim(1e-2, 1)
axs[10].set_ylim(5e-4, 1e-3)
"""
#axs[0].set_ylim(1e-2, 1) # koza 3
#axs[1].set_ylim(3e-1, 5e-1) # rastrigin
fig.set_size_inches(scale*3, scale*3)
#legend = fig.legend()
#legend.set_ncols(3)  # Set the number of columns
# Directly access the legend's text properties
#for text in legend.get_texts():
#    text.set_fontsize(16)  # Update the font size for each text in the legend


#fig.set_size_inches(20, 40)
fig.tight_layout(rect=[0, 0, 1, 0.90])

fig = figures['similarity_diabetes']
axs = fig.get_axes()
fig.suptitle(fig.get_suptitle(), fontsize=28)
for ax in axs:
    ax.set_ylim(0, 20)
    ax.set_title(ax.get_title(), fontsize=24)         # Set title font size
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)       # Set x-axis label font size
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)       # Set y-axis label font size
    ax.tick_params(axis='y', which='major', labelsize=18)  # Set tick label font size
    ax.tick_params(axis='y', which='minor', labelsize=18)  # Set tick label font size
    ax.tick_params(axis='x', which='major', labelsize=18)  # Set tick label font size


fig.set_size_inches(scale*3, scale*3)
#fig.set_size_inches(20, 40)
fig.tight_layout(rect=[0, 0, 1, 0.90])

fig = figures['semantic_diversity_diabetes']
axs = fig.get_axes()
fig.suptitle(fig.get_suptitle(), fontsize=28)
for ax in axs:
    ax.set_ylim(1e-5, 1)
    ax.set_title(ax.get_title(), fontsize=24)         # Set title font size
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)       # Set x-axis label font size
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)       # Set y-axis label font size
    ax.tick_params(axis='y', which='major', labelsize=18)  # Set tick label font size
    ax.tick_params(axis='y', which='minor', labelsize=18)  # Set tick label font size
    ax.tick_params(axis='x', which='major', labelsize=18)  # Set tick label font size


fig.set_size_inches(scale*3, scale*3)
#fig.set_size_inches(20, 40)
fig.tight_layout(rect=[0, 0, 1, 0.90])

fig = figures['nodes_diabetes']
axs = fig.get_axes()
fig.suptitle(fig.get_suptitle(), fontsize=28)
for ax in axs:
    ax.set_ylim(0, 20)
    ax.set_title(ax.get_title(), fontsize=24)         # Set title font size
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)       # Set x-axis label font size
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)       # Set y-axis label font size
    ax.tick_params(axis='y', which='major', labelsize=18)  # Set tick label font size
    ax.tick_params(axis='y', which='minor', labelsize=18)  # Set tick label font size
    ax.tick_params(axis='x', which='major', labelsize=18)  # Set tick label font size


fig.set_size_inches(scale*3, scale*3)
#fig.set_size_inches(20, 40)
fig.tight_layout(rect=[0, 0, 1, 0.90])



fig = figures['mut_density_distro_gens_diabetes']
fig.set_size_inches(scale*7.5, scale*9.5)
#fig.set_size_inches(160, 50)
fig.tight_layout()
fig = figures['xov_density_distro_gens_diabetes']
fig.set_size_inches(scale*7.5, scale*9.5)
fig.tight_layout()
#fig.set_size_inches(160, 50)
axs = fig.get_axes()
fig = figures['xov_density_distro_gens_short_diabetes']
fig.set_size_inches(scale*7.5, scale*9.5)
#fig.set_size_inches(160, 50)
axs = fig.get_axes()
"""
for i, ax in enumerate(axs):
    if i % 3 == 0:
        ax.set_ylabel('Node w/Xover Index')
    else:
        ax.set_ylabel('')
"""
fig.tight_layout()
"""
fig = figures['fitness_gen']
fig.set_size_inches(20, 60)
axs = fig.get_axes()
axs[0].set_ylim(1e-3, 1)
axs[1].set_ylim(1e-2, 1)
axs[2].set_ylim(3e-1, 1)
axs[2].set_xlim(0, 100)
axs[3].set_ylim(1e-3, 1)
axs[4].set_ylim(1e-4, 1e-1)
axs[5].set_ylim(1e-3, 1e-1)
axs[6].set_ylim(1e-5, 1e-3)
axs[7].set_ylim(3e-2, 1e-1)
axs[8].set_ylim(4e-1, 6e-1)
axs[8].set_xlim(0, 1000)
axs[9].set_ylim(1.5e-1, 5e-1)
axs[10].set_ylim(1e-4, 1)
fig.tight_layout()
fig = figures['sharp_in']
fig.set_size_inches(20, 60)
fig.tight_layout()

fig = figures['sharp_out']
fig.set_size_inches(20, 40)
axs = fig.get_axes()
for i in range(0, 7):
    axs[i].set_ylim(1.4e-2, 2e-2)
for i in range(7, 11):
    axs[i].set_ylim(5e-3, 1e-2)
fig.tight_layout()

fig = figures['nodes']
fig.set_size_inches(20, 60)
axs = fig.get_axes()
for ax in axs:
    ax.set_ylim(0, 30)
fig.tight_layout()
"""

# Now `figures` contains the figures loaded from the pickle files
# You can save these figures as images if needed
for key, fig in figures.items():
    image_path = f"{out_path}{key}{image_filetype}"
    fig.savefig(image_path)
    print(f"Saved figure for {key} to {image_path}")
   

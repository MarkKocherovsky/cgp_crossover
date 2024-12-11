from functions import *
from analysis_helper import *
import pickle
import psutil
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
#c = Collection()
#f_list = c.func_list
#f_name = c.name_list

cgp_base_path = "../output/cgp_real_world/"
cgp_uniform_path = "../output/cgp_uniform_real_world/"
cgp_1x_path = "../output/cgp_1x_real_world/"
cgp_2x_path = "../output/cgp_2x_real_world/"
cgp_dnc_path = "output/cgp_dnc_Uniform_real_world_lr_1.0e-04/"
cgp_dnc_1x_path = "output/cgp_dnc_OnePoint_real_world_lr_1.0e-04/"
cgp_diversity_path = "../output/cgp_diversity_real_world/"
cgp_vlen_path = "../output/cgp_vlen_real_world/"
cgp_sgx_path = "../output/cgp_sgx_real_world/"
#color_order = ['blue', 'green', 'red', '', 'skyblue', 'crimson', 'green', 'brown', 'purple',
#               'slategray', 'goldenrod']
#method_names = ["CGP(1+4)", "CGP(16+64)", "CGP-1x(40+40)", "CGP-DNC(40+40)", "CGP-1xD", "CGP-2x(40+40)", "CGP-SGx(40+40)", "LGP-Ux(40+40)"]
#method_names_long = ["CGP(1+4)", "CGP(16+64)", "CGP-OnePoint(40+40)", "CGP-DeepNeuralXover(40+40)", "CGP-OnePointDiversity(40+40)",
                     #"CGP-TwoPoint(40+40)", "CGP-Subgraph(40+40)", "LGP-Uniform(40+40)"]
max_e = 50

colors = {
    'cgp_base': 'blue',
    'cgp_ep': 'darkgray',
    'cgp_dnc': 'red',
    'cgp_dnc_1x': 'crimson',
    'cgp_dnc_1x_lr3': 'hotpink',
    'cgp_dnc_1x_lr5': 'firebrick',
    'cgp_dnc_parents': 'sandybrown',
    'cgp_dnc_1x_parents': "sienna",
    'cgp_1x': 'green',
    'cgp_div': 'mediumseagreen',
    'cgp_vlen': 'lightseagreen',
    'cgp_2x': 'limegreen',
    'cgp_unx': 'orange',
    'cgp_sgx': 'goldenrod'
}
# Usage
base_paths = {
    'cgp_base': cgp_base_path,
    'cgp_1x': cgp_1x_path,
    'cgp_dnc': cgp_dnc_path,
    'cgp_dnc_1x': cgp_dnc_1x_path,
    'cgp_vlen': cgp_vlen_path,
    'cgp_2x': cgp_2x_path,
    'cgp_sgx': cgp_sgx_path,
    'cgp_unx': cgp_uniform_path,
    'cgp_div': cgp_diversity_path
}

f_names = {
    'cgp_base': "CGP(1+4)",
    #'cgp_ep': "CGP(1+4)-Smaller Epsilon",
    #'cgp_40': "CGP(16+64)",
    'cgp_1x': "CGP-OnePoint(40+40)",
    'cgp_dnc': "CGP-DeepNeuralCrossover(40+40)",
    'cgp_dnc_1x': "CGP-DeepNeuralCrossover-OnePoint(40+40)",
    #'cgp_dnc_1x_lr3': "CGP-DeepNeuralCrossover-OnePoint(40+40)-Lr3",
    # 'cgp_dnc_1x_lr5': "CGP-DeepNeuralCrossover-OnePoint(40+40)-Lr5",
    # 'cgp_dnc_parents': "CGP-DeepNeuralCrossover-FixedParents(40+40)",
    # 'cgp_dnc_1x_parents': "CGP-DeepNeuralCrossover-OnePoint-FixedParents(40+40)",
    'cgp_div': "CGP-OnePointDiversity(40+40)",
    'cgp_vlen': "CGP-OnePointVL(40+40)",
    'cgp_2x': "CGP-TwoPoint(40+40)",
    'cgp_sgx': "CGP-Subgraph(40+40)",
    # 'lgp_1x': "LGP-OnePoint(40+40)",
    # 'lgp_2x': "LGP-TwoPoint(40+40)",
    'cgp_unx': "CGP-Uniform(40+40)",
    # 'lgp_base': "LGP-Uniform(40+40)"
}

short_names = {
    'cgp_base': "CGP(1+4)",
    #'cgp_ep': "CGPe(1+4)",
    #'cgp_40': "CGP(16+64)",
    'cgp_1x': "CGP-1x(40+40)",
    'cgp_dnc': "CGP-DNC(40+40)",
    'cgp_dnc_1x': "CGP-DNC-1x(40+40)",
    # 'cgp_dnc_1x_lr3': "CGP-DNC-1xLr3(40+40)",
    # 'cgp_dnc_1x_lr5': "CGP-DNC-1xLr5(40+40)",
    #'cgp_dnc_parents': "CGP-DNC-P(40+40)",
    #'cgp_dnc_1x_parents': "CGP-DNC-1x-P(40+40)",
    'cgp_div': "CGP-1xD(40+40)",
    'cgp_vlen': "CGP-1xVL(40+40)",
    'cgp_2x': "CGP-2x(40+40)",
    'cgp_sgx': "CGP-SGx(40+40)",
    # 'lgp_1x': "LGP-OnePoint(40+40)",
    # 'lgp_2x': "LGP-TwoPoint(40+40)",
    'cgp_unx': 'CGP-Ux(40+40)',
    # 'lgp_base': "LGP-Uniform(40+40)"
}

print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
method_names = [short_names[key] for key in short_names]
method_names_long = [f_names[key] for key in f_names]
color_order = [colors[key] for key in f_names]
#problem_names = ['Ackley', 'Rastrigin']
#problem_names = ['Koza 3', 'Rastrigin', 'Koza 1', 'Koza 2', 'Nguyen 4', 'Nguyen 5', 'Nguyen 6', 'Nguyen 7', 'Ackley', 'Levy', 'Griewank']
problem_names = ['Diabetes'] #, 'Rastrigin']
drift_colors = ['red', 'blue', 'green']
drift_names = ['Deleterious', 'Near-Neutral', 'Beneficial']
drift_categories = ['d', 'n', 'b']

load_data = True
if load_data:
    load_all_data(base_paths, max_e, f_names, problem_names)

from functions import *
sharpness_fnames = {key: f_names[key] for key in ['cgp_base', 'cgp_sgx', 'cgp_dnc']}
sharpness_problem_names = ['Koza 3', 'Rastrigin', 'Levy']
sharpness_problems = [koza3(), rastrigin(), levy()]
sharpness_colors = [colors[key] for key in sharpness_fnames]
plot_sharpness(sharpness_fnames, sharpness_problem_names, sharpness_problems, sharpness_colors, [sharpness_fnames[key] for key in sharpness_fnames], n_points=1000, n_replicates=10)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
medians = get_medians(f_names, problem_names, 'p_fits', name='medians')

print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
significance = get_significance(f_names, problem_names, 'p_fits')
#exit(1)

print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
plot_box_plots(f_names, problem_names, method_names, method_names_long, color_order, 'p_fits', 'fitness',
               'Fitness of Best Models', r'$1 - r^2$', log=True)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
plot_box_plots(f_names, problem_names, method_names, method_names_long, color_order, 'prop', 'prog_prop',
               'Proportion of Active Nodes for SR problems', 'Active Nodes / Total Nodes', log=False)

plot_over_generations(f_names, problem_names, 'p_size', 0, 64, method_names_long, color_order, 'nodes',
                      'Active Nodes', r'$\mathrm{Average}$(nodes)')

print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
# 'Density Distribution for Xover Operators Across Generations',
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
#save_density_csv(den_list_avgs, 'xover', f_names, problem_names)
#save_density_csv(mut_list_avgs, 'mutation', f_names, problem_names)
plot_density_heatmap_split(f_names, short_names, problem_names, max_e, 'density_list',
                     'xov_density_distro_gens', 'Density Distribution for Crossover Operators Across Generations',
                     'Node with Xover Index', drift_names, 3, 'Generation')
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
#exit(0)
plot_density_heatmap_split(f_names, short_names, problem_names, max_e, 'mut_density_list', #check data for strange lines
                     'mut_density_distro_gens', 'Density Distribution for Mutation Operators Across Generations',
                     'Mutation Index', drift_names,1,'Generation')
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
#exit(0)
plot_multiple_series(f_names, problem_names, drift_colors, drift_names, drift_categories, 'density_distro', 'density_distro',
                     'Density Distribution for xover operators', 'Frequency', x_label="Crossover Index", histogram=True)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
plot_over_generations(f_names, problem_names, 'average_retention', 0, 60, method_names_long, color_order, 'similarity',
                      'Average Similarity of Parents and Children', 'Average Similarity')
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
plot_over_generations(f_names, problem_names, 'fit_track', 1e-3, 1, method_names_long, color_order, 'fitness_gen',
                      'Fitness over Generations',
                      '1-r^2', log = True, logx=True)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
plot_over_generations(f_names, problem_names, 'div_list', 1e-10, 1, method_names_long, color_order, 'semantic_diversity',
                      'Average Semantic Diversity of Population', r'$\mathrm{Average} \ \sigma (f)$')
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
plot_over_generations(f_names, problem_names, 'fit_mean', 1e-3, 1, method_names_long, color_order, 'fitness_mean',
                      'Average Fitness of Whole Population', r'$\mathrm{Average}(f)$', log=True)
# these should be (#methods) x (#problems), so 9 methods * 11 problems = 99 boxes (8|)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

#plot_multiple_series(f_names, problem_names, drift_colors, drift_names, drift_categories, 'mut_cumul_drift', 'mut_drift',
#                     'Effectiveness of Mutation Operator', 'Frequency')
#print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

#plot_multiple_series(f_names, problem_names, drift_colors, drift_names, drift_categories, 'xov_cumul_drift', 'xov_drift',
#                     'Effectiveness of Crossover Operator', 'Frequency')
#print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

plot_over_generations(f_names, problem_names, 'sharp_in_mean', 1e-1, 1, method_names_long, color_order, 'sharp_in',
                      'SAM-In over Generations',
                      'SAM-In', log = True)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
plot_over_generations(f_names, problem_names, 'sharp_out_mean', 1e-2, 1, method_names_long, color_order, 'sharp_out',
                      'SAM-Out over Generations',
                      'SAM-Out', log = True)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)


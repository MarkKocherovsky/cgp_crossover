from functions import *
from analysis_helper import *

c = Collection()
f_list = c.func_list
f_name = c.name_list

cgp_base_path = "../output/cgp/"
lgp_base_path = "../output/lgp/"
cgp_1x_path = "../output/cgp_1x/"
cgp_vlen_path = "../output/cgp_vlen/"
lgp_1x_path = "../output/lgp_1x0/"
lgp_2x_path = "../output/lgp_2x/"
lgp_mut_path = "../output/lgp_mut/"
cgp_2x_path = "../output/cgp_2x/"
cgp_40_path = "../output/cgp_40/"
cgp_sgx_path = "../output/cgp_sgx/"
cgp_nx_path = "../output/cgp_nx/"
lgp_fx_path = "../output/lgp_fx/"

color_order = ['blue', 'royalblue', 'deeppink', 'lightgreen', 'skyblue', 'crimson', 'green', 'brown', 'purple',
               'slategray', 'goldenrod']
method_names = ["CGP(1+4)", "CGP(16+64)", "CGP-1x(40+40)", "LGP-1x(40+40)", "CGP-2x(40+40)", "LGP-2x(40+40)",
                "CGP-SGx(40+40)", "LGP-Ux(40+40)"]
method_names_long = ["CGP(1+4)", "CGP(16+64)", "CGP-OnePoint(40+40)", "CGP-OnePointVL(40+40)", "LGP-OnePoint(40+40)",
                     "CGP-TwoPoint(40+40)",
                     "LGP-TwoPoint(40+40)", "CGP-Subgraph(40+40)",
                     "LGP-Uniform(40+40)"]

max_e = 50

# Usage
base_paths = {
    'cgp_base': cgp_base_path,
    'cgp_1x': cgp_1x_path,
    'cgp_vlen': cgp_vlen_path,
    'cgp_2x': cgp_2x_path,
    'cgp_40': cgp_40_path,
    'cgp_sgx': cgp_sgx_path,
    'lgp_1x': lgp_1x_path,
    'lgp_2x': lgp_2x_path,
    'lgp_base': lgp_base_path
}

f_names = {
    'cgp_base': "CGP(1+4)",
    'cgp_40': "CGP(16+64)",
    'cgp_1x': "CGP-OnePoint(40+40)",
    'cgp_vlen': "CGP-OnePointVL(40+40)",
    'cgp_2x': "CGP-TwoPoint(40+40)",
    'cgp_sgx': "CGP-Subgraph(40+40)",
    'lgp_1x': "LGP-OnePoint(40+40)",
    'lgp_2x': "LGP-TwoPoint(40+40)",
    'lgp_base': "LGP-Uniform(40+40)"
}

all_data = load_all_data(base_paths, max_e, f_names)

plot_box_plots(f_names, all_data, method_names, method_names_long, color_order, 'p_fits', 'fitness',
               'Fitness of Best Models', '1-r^2')
plot_box_plots(f_names, all_data, method_names, method_names_long, color_order, 'prop', 'prog_prop',
               'Proportion of Active Nodes for SR problems', 'Active Nodes / Total Nodes')

sim_avgs = prepare_avgs(all_data, 'average_retention')
fit_avgs = prepare_avgs(all_data, 'fit_track')

# these should be (#methods) x (#problems), so 9 methods * 11 problems = 99 boxes (8|)
mut_avgs = prepare_avgs_multi(all_data, 'mut_cumul_drift')
xov_avgs = prepare_avgs_multi(all_data, 'xov_cumul_drift')
shi_avgs = prepare_avgs_multi(all_data, 'sharp_in_mean')
sho_avgs = prepare_avgs_multi(all_data, 'sharp_out_mean')
den_avgs = prepare_avgs_multi(all_data, 'density_distro')
plot_over_generations(f_names, sim_avgs, method_names_long, color_order, 'similarity',
                      'Average Similarity of Parents and Children', 'Average Similarity')
plot_over_generations(f_names, fit_avgs, method_names_long, color_order, 'fitness_gen', 'Fitness over Generations',
                      '1-r^2')

# these have to be different but one at a time >:(
plot_over_generations(f_names, mut_avgs, method_names_long, color_order, 'mut_drift',
                      'Effectiveness of Mutation Operator', 'Frequency')
plot_over_generations(f_names, xov_avgs, method_names_long, color_order, 'xov_drift',
                      'Effectiveness of Crossover Operator', 'Frequency')
# this should be an entirely different plot, I've just put the stuff here for space
plot_over_generations(f_names, den_avgs, method_names_long, color_order, 'density',
                      'Density Distribution grouped by Xover Effectiveness', 'Frequency')

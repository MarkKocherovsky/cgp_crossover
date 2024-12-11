from sys import argv
import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from cgp_mutation import *
from cgp_parents import *
from cgp_plots import *
from cgp_selection import selectElite
from functions import *
from helper import *
from functions import Diabetes
warnings.filterwarnings('ignore')

print("started")
t = int(argv[1])  # trial
print(f'trial {t}')
max_g = int(argv[2])  # max generations
print(f'generations {max_g}')
max_n = int(argv[3])  # max body nodes
print(f'max body nodes {max_n}')
max_c = int(argv[4])  # max children
print(f'children {max_c}')
outputs = 1
biases = np.arange(0, 10, 1).astype(np.int32)
bias = biases.shape[0]  # number of biases
print(f'biases {biases}')
arity = 2
random.seed(t + 50)
print(f'Seed = {sqrt(t)}')
np.seterr(divide='raise')

run_name = 'cgp_real_world'
func_name = 'Diabetes'
bank, bank_string = loadBank()

max_p = 1
bank = (add, sub, mul, div)
bank_string = ("+", "-", "*", "/")

"""
dims = int(argv[6])
n_points = int(argv[7])
func, func_name, func_dims = getFunction(int(argv[5]), dims)
inputs = func_dims
"""

func = Diabetes()
print(len(func()))
train_x, test_x, train_y, test_y = func()
dims = train_x.shape[1]
inputs = dims

first_body_node = inputs + bias
#train_x, train_y = getXY(n_points, dims, func)

fits = FitCollection()
fit = fits.fit_list[1]
fit_name = fits.name_list[1]

# No Crossover!
mutate = mutate_1_plus_4

final_fit = []
ind_base = np.zeros(((arity + 1) * max_n,), np.int32)
ind_base = ind_base.reshape(-1, arity + 1)  # for my sanity - Mark
print(train_x)
print(train_y)
train_x_bias = prepareConstants(train_x, biases)
print("instantiating parent")
# instantiate parent
parent = generate_parents(1, max_n, bank, inputs=dims, n_constants=bias, outputs=1, arity=2)
operators = 1
density_distro, density_distro_list = initDensityDistro(max_n, operators, arity, max_g)
mut_density_distro, mut_distro_list = initDensityDistro(max_n, operators, arity, max_g, outputs = 1)
mut_impact = DriftImpact(neutral_limit=1e-3)

fitness = Fitness()
sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()
p_fit, p_A, p_B = fitness(train_x_bias, train_y, parent, arity)

sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std = [0], [0], [0], [0]
"""
sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std = processSharpness(train_x_bias, 1, max_c,
                                                                                  inputs, func, sharp_in_manager,
                                                                                  [Fitness()], [p_fit],
                                                                                  [parent], train_y,
                                                                                  sharp_in_list, sharp_in_std,
                                                                                  sharp_out_manager, sharp_out_list,
                                                                                  sharp_out_std, arity)
"""
f_change = np.zeros((max_c,))  # % difference from p_fit
p_size = [cgp_active_nodes(parent[0], parent[1], first_body_node)]  # /ind_base.shape[0]]
fit_track, ret_avg_list, ret_std_list, avg_change_list, avg_hist_list, std_change_list, div_list, fit_mean = initTrackers()
fitness_objects = [Fitness() for i in range(0, max_c)]

for g in range(1, max_g + 1):
    children, mutation_list = zip(*[mutate(deepcopy(parent), first_body_node) for _ in range(max_c)])
    mutated_inds = list(range(max_p))
    children = list(children)
    c_fit, alignment = processFitness(fitness_objects, train_x_bias, train_y, children, 0, max_c, arity, opt = 0)
    best_child_index = np.argmin(c_fit)
    best_c_fit = c_fit[best_child_index]
    best_child = children[best_child_index]
    avg_change_list.append(percent_change(best_c_fit, p_fit))
    change_list = np.array([percent_change(c, p_fit) for c in c_fit])
    change_list = change_list[np.isfinite(change_list)]
    cl_std = np.nanstd(change_list)
    if not all(cl == 0.0 for cl in change_list):
        avg_hist_list.append((g, np.histogram(change_list, bins=5, range=(cl_std * -2, cl_std * 2))))
    ret_avg_list.append(
        find_similarity(best_child[0], parent[0], first_body_node, best_child[1], parent[1], mode='cgp', method='distance'))
    ret_std_list.append(0.0)
    std_change_list.append(0)
    a = alignment[:, 0].copy().flatten()
    b = alignment[:, 1].copy().flatten()
    c_fit = c_fit.flatten()
    # print(p_fit)
    # print(c_fit)
    if any(np.isnan(c_fit)):  # Replace nans with positive infinity to screen them out
        nans = np.isnan(c_fit)
        c_fit[nans] = np.PINF
    full_fit = np.insert(c_fit, 0, p_fit)
    drift_per_parent_mut, drift_per_parent_xov = mut_impact(np.insert(c_fit, 0, p_fit), [], 1, [], mutated_inds, opt=1,
                                                            option='OneParent')
    drift_per_parent_mut = drift_per_parent_mut.flatten()
    """
    # get average sharpness
    sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std = processSharpness(train_x_bias, 1, max_c,
                                                                                  inputs, func, sharp_in_manager,
                                                                                  [Fitness()] + fitness_objects, full_fit,
                                                                                  [parent] + children, train_y,
                                                                                  sharp_in_list, sharp_in_std,
                                                                                  sharp_out_manager, sharp_out_list,
                                                                                  sharp_out_std, arity)
    """
    mut_density_distro, mut_distro_list = associateDistro(drift_per_parent_mut, list(range(max_c)), mutation_list,
                                                          mut_density_distro,
                                                          mut_distro_list, g)
    div_list.append(semantic_diversity(full_fit))
    fit_mean.append(np.nanmean(full_fit))
    parent, p_fit, p_A, p_B = selectElite(parent, children, p_fit, c_fit, p_A, p_B, a, b)
    if g % 100 == 0:
        logAndProcessSharpness(g, p_fit, fit_mean,  sharp_in_list, sharp_out_list)
    fit_track.append(p_fit)
    # sharp_list.append(np.abs(p_fit-p_sharp))
    p_size.append(cgp_active_nodes(parent[0], parent[1], first_body_node))  # /ind_base.shape[0])
# if(p_fit > 0.96):
#	break
avg_change_list = np.array(avg_change_list)
std_change_list = np.array(std_change_list)
p_size = np.array(p_size)

pop = [parent] + children
fitness_objects = [Fitness()] + fitness_objects
fitnesses, alignment = processFitness(fitness_objects, train_x_bias, train_y, pop, max_p, max_c, arity, opt=0)
print(mut_impact)
best_i, best_fit, best_pop, mut_list, mut_cum, xov_list, xov_cum, density_distro, mut_Density_distro, preds, p_a, p_b = processAndPrintResults(
    t, fitnesses, pop, mut_impact, density_distro, mut_density_distro, train_x_bias=train_x_bias, train_y=train_y, mode='cgp')
# print(list(train_y))
path = f"../../output/{run_name}/{func_name}/log/"
Path(path).mkdir(parents=True, exist_ok=True)

bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g)
n = plot_active_nodes(best_pop[0], best_pop[1], first_body_node, bank_string, biases, inputs, p_a, p_b, func_name,
                      run_name, t, arity, opt=1)
saveResults(run_name, func_name, t, biases, best_pop, preds, best_fit, n, fit_track, avg_change_list, ret_avg_list,
            p_size, bin_centers, hist_gens, avg_hist_list, mut_list, mut_cum,
            xov_list, xov_cum, sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std, density_distro,
            density_distro_list, mut_density_distro, mut_distro_list, div_list, fit_mean, path=path)

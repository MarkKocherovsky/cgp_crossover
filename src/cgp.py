from cgp_mutation import *
from cgp_parents import *
from cgp_plots import *
from cgp_selection import selectElite
from functions import *
from helper import *

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
inputs = 1
biases = np.arange(0, 10, 1).astype(np.int32)
bias = biases.shape[0]  # number of biases
print(f'biases {biases}')
arity = 2
random.seed(t + 50)
print(f'Seed = {sqrt(t)}')

max_p = 1

bank = (add, sub, mul, div)
bank_string = ("+", "-", "*", "/")

func, func_name, func_dims = getFunction(int(argv[6]))
train_x, train_y = getXY(func)

f = int(argv[6])
fits = FitCollection()
fit = fits.fit_list[f]
fit_name = fits.name_list[f]

# No Crossover!
mutate = mutate_1_plus_4

final_fit = []
ind_base = np.zeros(((arity + 1) * max_n,), np.int32)
ind_base = ind_base.reshape(-1, arity + 1)  # for my sanity - Mark
print(train_x)
train_x_bias = prepareConstants(train_x, biases)
print("instantiating parent")
# instantiate parent
parent = generate_parents(1, max_n, bank, inputs=func_dims, n_constants = len(biases), outputs=1, arity=2)
density_distro = initDensityDistro(max_n, outputs, arity)
mut_impact = DriftImpact(neutral_limit=1e-3)

fitness = Fitness()
sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()
print(train_x_bias)
p_fit, p_A, p_B = fitness(train_x_bias, train_y, parent)

# SAM-In
noisy_x, noisy_y = getNoise(train_x_bias.shape, 1, max_c, inputs, func, sharp_in_manager)
p_sharp, _, _ = fitness(noisy_x[0], noisy_y[0], parent)
sharp_in_list = [np.abs(p_fit - p_sharp)]
sharp_in_std = [0]

# SAM-Out
preds, _, _ = fitness(train_x_bias, train_y, parent, opt=1)

neighbor_map = getNeighborMap(preds, sharp_out_manager, fitness, train_y)
sharp_out_list = [np.std(neighbor_map) ** 2]  # variance
sharp_out_std = [0]

f_change = np.zeros((max_c,))  # % difference from p_fit
p_size = [cgp_active_nodes(parent[0], parent[1])]  # /ind_base.shape[0]]
fit_track, ret_avg_list, ret_std_list, avg_change_list, avg_hist_list, std_change_list = initTrackers()
fitness_objects = [Fitness() for i in range(0, max_c)]
run_name = 'cgp'

for g in range(1, max_g + 1):
    children, mutated_inds = zip(*[mutate(deepcopy(parent)) for _ in range(max_c)])
    children = list(children)
    c_fit, alignment = processFitness(fitness_objects, train_x_bias, train_y, children, 0, max_c)
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
        find_similarity(best_child[0], parent[0], best_child[1], parent[1], mode='cgp', method='distance'))
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
    drift_per_parent_mut, drift_per_parent_xov = mut_impact(np.insert(c_fit, 0, p_fit), 1, [], mutated_inds, opt=1,
                                                            option='OneParent')
    # get average sharpness
    sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std = processSharpness(train_x_bias, 1, max_c,
                                                                                  inputs, func, sharp_in_manager,
                                                                                  [Fitness()] + fitness_objects,
                                                                                  [parent] + children, train_y,
                                                                                  sharp_in_list, sharp_in_std,
                                                                                  sharp_out_manager, sharp_out_list,
                                                                                  sharp_out_std)

    parent, p_fit, p_A, p_B = selectElite(parent, children, p_fit, c_fit, p_A, p_B, a, b)

    if g % 100 == 0:
        logAndProcessSharpness(g, p_fit, sharp_in_list, sharp_out_list)
    fit_track.append(p_fit)
    # sharp_list.append(np.abs(p_fit-p_sharp))
    p_size.append(cgp_active_nodes(parent[0], parent[1]))  # /ind_base.shape[0])
# if(p_fit > 0.96):
#	break
avg_change_list = np.array(avg_change_list)
std_change_list = np.array(std_change_list)
p_size = np.array(p_size)

pop = [parent] + children
fitness_objects = [Fitness()] + fitness_objects
fitnesses, alignment = processFitness(fitness_objects, train_x_bias, train_y, pop, max_p, max_c)
best_i, best_fit, best_pop, mut_list, mut_cum, xov_list, xov_cum, density_distro, preds, p_a, p_b = processAndPrintResults(
    t, fitnesses, pop, mut_impact, density_distro, train_x_bias = train_x_bias, train_y=train_y, mode='cgp')

run_name = 'cgp'
# print(list(train_y))
Path(f"../output/{run_name}/{func_name}/log/").mkdir(parents=True, exist_ok=True)

first_body_node = inputs + bias
bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g)
n = plot_active_nodes(best_pop[0], best_pop[1], first_body_node, bank_string, biases, inputs, p_a, p_b, func_name,
                      run_name, t, opt=1)
saveResults(run_name, func_name, t, biases, best_pop, preds, best_fit, n, fit_track, avg_change_list, ret_avg_list,
            p_size, bin_centers, hist_gens, avg_hist_list, mut_list, mut_cum,
            xov_list, xov_cum, sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std, density_distro)

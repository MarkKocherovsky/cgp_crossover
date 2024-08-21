#CGP 16+64 No Crossover
from sys import argv

from cgp_mutation import *
from cgp_parents import *
from cgp_plots import *
from cgp_selection import *
from helper import *

warnings.filterwarnings('ignore')
print("started")
t = int(argv[1])  #trial
print(f'trial {t}')
max_g = int(argv[2])  #max generations
print(f'generations {max_g}')
max_n = int(argv[3])  #max body nodes
print(f'max body nodes {max_n}')
max_p = int(argv[4])  #max parents
print(f'Parents {max_p}')
max_c = int(argv[5])  #max children
print(f'children {max_c}')
outputs = 1
inputs = 1
biases = np.arange(0, 10, 1).astype(np.int32)
bias = biases.shape[0]  #number of biases
print(f'biases {biases}')
arity = 2
run_name = "cgp_40"
num_elites = 7
bank = (add, sub, mul, div)
bank_string = ("+", "-", "*", "/")

func_bank = Collection()
func = func_bank.func_list[int(argv[6])]
func_name = func_bank.name_list[int(argv[6])]
train_x = func.x_dom
train_y = func.y_test
print(train_x)

f = int(argv[7])
fits = FitCollection()
fit = fits.fit_list[f]
print(f)
print(fits.fit_list)
fit_name = fits.name_list[f]
print('Fitness Function')
print(fit)
print(fit_name)

alignment = np.zeros((max_p + max_c, 2))
alignment[:, 0] = 1.0

train_x_bias = np.zeros((train_x.shape[0], biases.shape[0] + 1))
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = biases
print(train_x_bias)
#print(train_x_bias)
#print(inputs+bias+max_n)
#print("instantiating parent")
#instantiate parents
#test = run_output(ind_base, output_nodes, np.array([10.0]))

mutate = mutate_1_plus_4
select = tournament_elitism
parents = generate_parents(max_p, max_n, bank, first_body_node=11, outputs=1, arity=2)
density_distro = initDensityDistro(max_n, outputs, arity)

fitness_objects = [Fitness() for i in range(0, max_p + max_c * max_p)]
fitnesses = np.zeros((max_p + max_c * max_p), )
fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, parent) for i, parent in zip(range(0, max_p), parents)])
#print(*zip(range(0, max_p), parents))
fitnesses[:max_p] = fit_temp[:, 0].copy().flatten()
alignment[:max_p, 0] = fit_temp[:, 1].copy()  #a
alignment[:max_p, 1] = fit_temp[:, 2].copy()  #b
print(np.round(fitnesses, 4))

sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()

#SAM-IN
noisy_x, noisy_y = getNoise(train_x_bias.shape, max_p, max_c, inputs, func, sharp_in_manager)
sharpness = np.array(
    [fitness_objects[i](noisy_x[i], noisy_y[i], parent)[0] for i, parent in zip(range(0, max_p), parents)])
sharp_in_list = [np.mean(sharpness)]
sharp_in_std = [np.std(sharpness)]

#SAM-OUT

preds = [fitness_objects[i](train_x_bias, train_y, parent, opt=1)[0] for i, parent in zip(range(0, max_p), parents)]

neighbor_map = np.array(
    [getNeighborMap(pred, sharp_out_manager, fitness_objects[i], train_y) for i, pred in zip(range(0, max_p), preds)])
sharp_out_list = [np.mean(np.std(neighbor_map, axis=1) ** 2)]  #variance
sharp_out_std = [np.std(np.std(neighbor_map, axis=1))]

print(np.round(sharpness, 4))
print(np.round(np.std(neighbor_map, axis=1) ** 2, 4))

fit_track = []
ret_avg_list = []  #best parent best child
ret_std_list = []

avg_change_list = []  #best parent best child
avg_hist_list = []
std_change_list = []

best_i = np.argmin(fitnesses[:max_p])
p_size = [cgp_active_nodes(parents[best_i][0], parents[best_i][1])]

mut_impact = DriftImpact(neutral_limit=1e-3)

for g in range(1, max_g + 1):
    children, mutated_inds = zip(*[mutate(deepcopy(parent)) for parent in parents for _ in range(0, max_c)])
    pop = parents + list(children)
    fit_temp = np.array(
        [fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(list(range(0, max_p + max_c * max_p)), pop)])
    fitnesses = fit_temp[:, 0].copy().flatten()

    if any(np.isnan(fitnesses)):  #Replace nans with positive infinity to screen them out
        nans = np.isnan(fitnesses)
        fitnesses[nans] = np.PINF
    drift_per_parent_mut, drift_per_parent_xov = mut_impact(fitnesses, max_p, [], mutated_inds, opt=1,
                                                            option='OneParent')
    ret = []
    chg = []
    full_change_list = []
    for i in range(0, len(parents)):
        best_p = parents[i]
        best_p_fit = fitnesses[i]
        c_fits = []
        cs = []
        for j in range(i * max_c, i * max_c + max_c):
            c = children[j]
            cs.append(c)
            c_fit = fitnesses[max_p + j]
            c_fits.append(c_fit)
            full_change_list.append(percent_change(c_fit, best_p_fit))
            ret.append(find_similarity(c[0], best_p[0], c[1], best_p[1]))
        best_c_idx = np.argmin(c_fits)
        best_c_fit = c_fits[best_c_idx]
        best_c = cs[best_c_idx]
        chg.append(percent_change(best_c_fit, best_p_fit))

    ret_avg_list.append(np.nanmean(ret))
    ret_std_list.append(np.nanstd(ret))
    avg_change_list.append(np.nanmean(chg))
    std_change_list.append(np.nanstd(chg))
    full_change_list = np.array(full_change_list).flatten()
    full_change_list = full_change_list[np.isfinite(full_change_list)]
    cl_std = np.nanstd(full_change_list)
    if not all(cl == 0.0 for cl in full_change_list):
        avg_hist_list.append((g, np.histogram(full_change_list, bins=10, range=(cl_std * -2, cl_std * 2))))
    #print(len(pop))
    noisy_x, noisy_y = getNoise(train_x_bias.shape, max_p, max_c*max_p, inputs, func, sharp_in_manager, opt=1)
    sharpness = np.array([fitness_objects[i](noisy_x[i], noisy_y[i], individual)[0] for i, individual in
                          zip(range(0, max_p + max_p * max_c), pop)])
    sharp_in_list.append(np.mean(sharpness))
    sharp_in_std.append(np.std(sharpness))

    #SAM-OUT

    preds = [fitness_objects[i](train_x_bias, train_y, individual, opt=1)[0] for i, individual in
             zip(range(0, max_p + max_p * max_c), pop)]
    neighbor_map = np.array([getNeighborMap(pred, sharp_out_manager, fitness_objects[i], train_y) for i, pred in
                             zip(range(0, max_p + max_p * max_c), preds)])
    out_sharpness = np.std(neighbor_map, axis=1) ** 2
    sharp_out_list.append(np.mean(out_sharpness))  #variance
    sharp_out_std.append(np.std(out_sharpness))

    best_i = np.argmin(fitnesses)
    best_fit = fitnesses[best_i]
    fit_track.append(best_fit)
    p_size.append(cgp_active_nodes(pop[best_i][0], pop[best_i][1]))
    if g % 100 == 0:
        logAndProcessSharpness(g, best_fit, sharp_in_list, sharp_out_list)
    parents = select(pop, fitnesses, max_p)
#print("Fitnesses at end of generation, should not have changed")
#print(np.round(fitnesses, 4))
#print('----')

pop = parents + list(children)
fitnesses, alignment = processFitness(fitness_objects, train_x_bias, train_y, pop, max_p, max_c)
best_i, best_fit, best_pop, mut_list, mut_cum, xov_list, xov_cum, density_distro, preds, p_a, p_b = processAndPrintResults(
    t, fitnesses, pop, mut_impact, density_distro, train_x_bias = train_x_bias, train_y = train_y, mode = 'cgp')

run_name = 'cgp_40'
# print(list(train_y))
Path(f"../output/{run_name}/{func_name}/log/").mkdir(parents=True, exist_ok=True)

first_body_node = inputs + bias
bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g)
n = plot_active_nodes(best_pop[0], best_pop[1], first_body_node, bank_string, biases, inputs, p_a, p_b, func_name,
                      run_name, t, opt=1)
saveResults(run_name, func_name, t, biases, best_pop, preds, best_fit, n, fit_track, avg_change_list, ret_avg_list,
            p_size, bin_centers, hist_gens, avg_hist_list, mut_list, mut_cum,
            xov_list, xov_cum, sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std, density_distro)

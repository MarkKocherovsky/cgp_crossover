#CGP subgraph Crossover
from sys import argv

from cgp_mutation import *
from cgp_parents import *
from cgp_plots import *
from cgp_selection import *
from cgp_xover import *
from helper import *
from similarity import *

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
p_mut = float(argv[8])
p_xov = float(argv[9])
random.seed(t + 100)
print(f'Seed = {t + 420}')

run_name = 'cgp_sgx'

bank, bank_string = loadBank()

func, func_name = getFunction(int(argv[6]))

train_x, train_y = getXY(func)
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

alignment = initAlignment(max_p, max_c)

train_x_bias = prepareConstants(train_x, biases)

mutate = basic_mutation
select = tournament_elitism

parents = generate_parents(max_p, max_n, bank, first_body_node=11, outputs=1, arity=2)
density_distro = initDensityDistro(max_n, outputs, arity)

fitness_objects, fitnesses = initFitness(max_p, max_c)

sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()

fitnesses, alignment = processFitness(fitness_objects, train_x_bias, train_y, parents, max_p, max_c, opt=1)

print(np.round(fitnesses, 4))

#SAM-IN
noisy_x, noisy_y = getNoise(train_x_bias.shape, max_p, max_c, inputs, func, sharp_in_manager)
sharpness = np.array(
    [fitness_objects[i](noisy_x[i], noisy_y[i], parent)[0] for i, parent in zip(range(0, max_p), parents)])
print(noisy_x.shape)
print(train_x_bias.shape)
sharp_in_list = [np.mean(sharpness)]
sharp_in_std = [np.std(sharpness)]
#SAM-OUT

preds = [fitness_objects[i](train_x_bias, train_y, parent, opt=1)[0] for i, parent in zip(range(0, max_p), parents)]

neighbor_map = np.array(
    [getNeighborMap(pred, sharp_out_manager, fitness_objects[i], train_y) for i, pred in zip(range(0, max_p), preds)])
print(neighbor_map.shape)
sharp_out_list = [np.mean(np.std(neighbor_map, axis=1) ** 2)]  #variance
sharp_out_std = [np.std(np.std(neighbor_map, axis=1))]

print(np.round(sharpness, 4))
print(np.round(np.std(neighbor_map, axis=1) ** 2, 4))

fit_track, ret_avg_list, ret_std_list, avg_change_list, avg_hist_list, std_change_list = initTrackers()

best_i = getBestInd(fitnesses, max_p)
p_size = [cgp_active_nodes(parents[best_i][0], parents[best_i][1])]

mut_impact = DriftImpact(neutral_limit=1e-3)
num_elites = 7  #for elite graph plotting

for g in range(1, max_g + 1):
    children, retention, d_distro = xover(deepcopy(parents), max_n, method='Subgraph')
    children, mutated_inds = mutate(deepcopy(children))
    pop = parents + children
    fitnesses, alignment = processFitness(fitness_objects, train_x_bias, train_y, pop, max_p, max_c)
    drift_per_parent_mut, drift_per_parent_xov = mut_impact(fitnesses, max_p, retention, mutated_inds, opt=1)
    avg_hist_list, avg_change_list, std_change_list, ret_avg_list, ret_std_list = processRetention(retention, pop,
                                                                                                   fitnesses, max_p,
                                                                                                   avg_hist_list,
                                                                                                   avg_change_list,
                                                                                                   std_change_list,
                                                                                                   ret_avg_list,
                                                                                                   ret_std_list, g)

    sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std = processSharpness(train_x_bias, max_p, max_c,
                                                                                  inputs, func, sharp_in_manager,
                                                                                  fitness_objects, pop, train_y,
                                                                                  sharp_in_list, sharp_in_std,
                                                                                  sharp_out_manager, sharp_out_list,
                                                                                  sharp_out_std)
    density_distro = associateDistro(drift_per_parent_xov, retention, d_distro, density_distro)

    best_i = getBestInd(fitnesses)
    best_fit = fitnesses[best_i]
    if g % 100 == 0:
        logAndProcessSharpness(g, best_fit, sharp_in_list, sharp_out_list)

    fit_track.append(best_fit)
    p_size.append(cgp_active_nodes(pop[best_i][0], pop[best_i][1]))
    parents = select(pop, fitnesses, max_p)

pop = parents + children
fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(range(0, max_p + max_c), pop)])
fitnesses, alignment = processFitness(fitness_objects, train_x_bias, train_y, pop, max_p, max_c)
best_i, best_fit, best_pop, mut_list, mut_cum, xov_list, xov_cum, density_distro, preds, p_a, p_b = processAndPrintResults(
    t, fitnesses, pop, mut_impact, density_distro, train_x_bias = train_x_bias, train_y = train_y, mode = 'cgp')
Path(f"../output/{run_name}/{func_name}/log/").mkdir(parents=True, exist_ok=True)

first_body_node = inputs + bias
bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g)
n = plot_active_nodes(best_pop[0], best_pop[1], first_body_node, bank_string, biases, inputs, p_a, p_b, func_name,
                      run_name, t, opt=1)
saveResults(run_name, func_name, t, biases, best_pop, preds, best_fit, n, fit_track, avg_change_list, ret_avg_list,
            p_size, bin_centers, hist_gens, avg_hist_list, mut_list, mut_cum,
            xov_list, xov_cum, sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std, density_distro)

# CGP Uniform Crossover
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
t = int(argv[1])  # trial
print(f'trial {t}')
max_g = int(argv[2])  # max generations
print(f'generations {max_g}')
max_n = int(argv[3])  # max body nodes
print(f'max body nodes {max_n}')
max_p = int(argv[4])  # max parents
print(f'Parents {max_p}')
max_c = int(argv[5])  # max children
print(f'children {max_c}')
outputs = 1
inputs = 1
biases = np.arange(0, 10, 1).astype(np.int32)
bias = biases.shape[0]  # number of biases
print(f'biases {biases}')
arity = 2
p_mut = float(argv[8])
p_xov = float(argv[9])
random.seed(t + 100)
print(f'Seed = {t + 100}')

run_name = 'cgp_uniform'

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
operators = 1  # because univariate
print((max_g, (max_n * (operators + arity) + outputs)))
density_distro, density_distro_list = initDensityDistro(max_n, operators, arity, max_g, outputs=outputs,
                                                        shape_override=(
                                                            max_g, (max_n * (operators + arity) + outputs)))
mut_density_distro, mut_distro_list = initDensityDistro(max_n, operators, arity, max_g, outputs=outputs, 
                                                        shape_override=(
                                                            max_g, (max_n*(operators+arity)+outputs)))
fitness_objects, fitnesses = initFitness(max_p, max_c)

sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()

fitnesses, alignment = processFitness(fitness_objects, train_x_bias, train_y, parents, max_p, max_c, opt=1)

print(np.round(fitnesses, 4))


# Sharpness
sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std = [], [], [], []
sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std = processSharpness(train_x_bias, max_p, max_c,
                                                                                  inputs, func, sharp_in_manager,
                                                                                  fitness_objects, fitnesses[:max_p], parents, train_y,
                                                                                  sharp_in_list, sharp_in_std,
                                                                                  sharp_out_manager, sharp_out_list,
                                                                                  sharp_out_std)


fit_track, ret_avg_list, ret_std_list, avg_change_list, avg_hist_list, std_change_list, div_list, fit_mean = initTrackers()

best_i = getBestInd(fitnesses, max_p)
p_size = [cgp_active_nodes(parents[best_i][0], parents[best_i][1])]

mut_impact = DriftImpact(neutral_limit=1e-3)
num_elites = 7  # for elite graph plotting

for g in range(1, max_g + 1):
    children, retention, d_distro = xover(deepcopy(parents), max_n, method='Uniform',
                                          shape_override=(len(parents), (max_n * (operators + arity) + outputs)))
    xov_fitnesses, xov_alignment = processFitness(fitness_objects, train_x_bias, train_y, children, max_p, max_c)
    children, mutated_inds, mutation_list = mutate(deepcopy(children))
    pop = parents + children
    fitnesses, alignment = processFitness(fitness_objects, train_x_bias, train_y, pop, max_p, max_c)
    drift_per_parent_mut, drift_per_parent_xov = mut_impact(fitnesses, xov_fitnesses, max_p, retention, mutated_inds,
                                                            opt=1)
    avg_hist_list, avg_change_list, std_change_list, ret_avg_list, ret_std_list = processRetention(retention, pop,
                                                                                                   fitnesses, xov_fitnesses, max_p,
                                                                                                   avg_hist_list,
                                                                                                   avg_change_list,
                                                                                                   std_change_list,
                                                                                                   ret_avg_list,
                                                                                                   ret_std_list, g)

    sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std = processSharpness(train_x_bias, max_p, max_c,
                                                                                  inputs, func, sharp_in_manager,
                                                                                  fitness_objects, fitnesses, pop, train_y,
                                                                                  sharp_in_list, sharp_in_std,
                                                                                  sharp_out_manager, sharp_out_list,
                                                                                  sharp_out_std)
    density_distro, density_distro_list = associateDistro(drift_per_parent_xov, retention, d_distro, density_distro,
                                                          density_distro_list, g)
    mut_density_distro, mut_distro_list = associateDistro(drift_per_parent_mut, mutated_inds, mutation_list,
                                                          mut_density_distro,
                                                          mut_distro_list, g)
    div_list.append(semantic_diversity(fitnesses))
    fit_mean.append(np.nanmean(fitnesses))
    best_i = getBestInd(fitnesses)
    best_fit = fitnesses[best_i]
    if g % 100 == 0:
        logAndProcessSharpness(g, best_fit, fit_mean, sharp_in_list, sharp_out_list)

    fit_track.append(best_fit)
    p_size.append(cgp_active_nodes(pop[best_i][0], pop[best_i][1]))
    parents = select(pop, fitnesses, max_p)

pop = parents + children
fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(range(0, max_p + max_c), pop)])
fitnesses, alignment = processFitness(fitness_objects, train_x_bias, train_y, pop, max_p, max_c)
best_i, best_fit, best_pop, mut_list, mut_cum, xov_list, xov_cum, density_distro, mut_density_distro, preds, p_a, p_b = processAndPrintResults(
    t, fitnesses, pop, mut_impact, density_distro, mut_density_distro, train_x_bias=train_x_bias, train_y=train_y, mode='cgp')
run_name = 'cgp_uniform'
Path(f"../output/{run_name}/{func_name}/log/").mkdir(parents=True, exist_ok=True)

first_body_node = inputs + bias
bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g)
n = plot_active_nodes(best_pop[0], best_pop[1], first_body_node, bank_string, biases, inputs, p_a, p_b, func_name,
                      run_name, t, opt=1)

saveResults(run_name, func_name, t, biases, best_pop, preds, best_fit, n, fit_track, avg_change_list, ret_avg_list,
            p_size, bin_centers, hist_gens, avg_hist_list, mut_list, mut_cum,
            xov_list, xov_cum, sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std, density_distro,
            density_distro_list, mut_density_distro, mut_distro_list, div_list, fit_mean)

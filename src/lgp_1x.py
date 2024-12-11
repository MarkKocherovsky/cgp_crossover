from sys import argv

from cgp_plots import *
from helper import *
from helper_lgp import *
from lgp_fitness import *
from lgp_mutation import *
from lgp_select import *
from lgp_xover import *

warnings.filterwarnings('ignore')


t = get_param(1, 0, int)  # trial
max_g = get_param(2, 100, int)  # generations
max_r = get_param(3, 10, int)  # instructions
max_d = get_param(4, 4, int)  # destinations (other than output)

if max_r < 1:
    print("Number of rules too small, setting to 10")
    max_r = 10

max_p = get_param(5, 50, int)  # parents
max_c = get_param(6, 100, int)  # children
arity = 2  # sources
n_inp = 1  # number of inputs
n_out = 1
bias = np.arange(0, 10, 1)
n_bias = bias.shape[0]  # number of bias inputs
random.seed(t + 300)

p_mut = get_param(9, 0.025)
p_xov = get_param(10, 0.5)
run_name = f'lgp_1x{get_param(11, "", str)}'
print(run_name)
fixed_length = get_param(12, False, lambda x: x == '1')
print(f'Fixed Length? {fixed_length}')


f = int(argv[8])
fits = FitCollection()
fit = fits.fit_list[f]
print(f)
print(fits.fit_list)
fit_name = fits.name_list[f]
print('Fitness Function')
print(fit)
print(fit_name)
num_elites = 7

bank, bank_string = loadBank()

func, func_name = getFunction(int(argv[7]))

train_x, train_y = getXY(func)

output_index = 0
input_indices = np.arange(1, n_inp + 1, 1)
# print(input_indices)
Path(f"../output/{run_name}/{func_name}/log/").mkdir(parents=True, exist_ok=True)
Path(f"../output/{run_name}/{func_name}/best_program/").mkdir(parents=True, exist_ok=True)
with open(f"../output/{run_name}/{func_name}/best_program/best_{t}.txt", 'w') as f:
    f.write(f"Problem {func_name}\n")
    f.write(f'Trial {t}\n')
    f.write(f'----------\n\n')

mutate = macromicro_mutation

select = lgp_tournament_elitism_selection
n_tour = 4
print(f"#####Trial {t}#####")
alignment = np.zeros((max_p + max_c, 2))
alignment[:, 0] = 1.0

parent_generator = lgpParentGenerator(max_p, max_r, max_d, bank, n_inp, n_bias, arity)
parents = parent_generator()

train_x_bias = prepareConstants(train_x, bias)

density_distro, density_distro_list = initDensityDistro(max_r, n_out, arity, max_g, mode='lgp')

fitnesses = np.zeros((max_p + max_c), )
fitness_evaluator = Fitness(train_x, bias, train_y, parents, func, bank, n_inp, max_d, fit, arity)
fitnesses[:max_p], alignment[:max_p, 0], alignment[:max_p, 1] = fitness_evaluator()
print(f'starting fitnesses')
print(fitnesses)

sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()

# SAM-IN
noisy_x, noisy_y = getNoise(train_x_bias.shape, max_p, max_c, n_inp, func, sharp_in_manager)
sharpness = get_sam_in(noisy_x[:, :, 0], noisy_x[:, :, 1:], noisy_y, parents, func, bank, n_inp, max_d, fit, arity)
sharp_in_list = [np.mean(sharpness)]
sharp_in_std = [np.std(sharpness)]

preds = [fitness_evaluator.predict(parent, A, B, train_x) for parent, A, B in
         zip(parents, alignment[:, 0], alignment[:, 1])]

sharp_out_manager = SAM_OUT()

neighbor_map = np.array(
    [getNeighborMap(pred, sharp_out_manager, train_y) for i, pred in zip(range(0, max_p), preds)])
sharp_out_list = [np.mean(np.std(neighbor_map, axis=1) ** 2)]  # variance
sharp_out_std = [np.std(np.std(neighbor_map, axis=1))]

fit_track, ret_avg_list, ret_std_list, avg_change_list, avg_hist_list, std_change_list = initTrackers()

best_i = getBestInd(fitnesses, max_p)
p_size = [len(effProg(4, parents[best_i])) / len(parents[best_i])]
mut_impact = DriftImpact(neutral_limit=1e-3)

for g in range(1, max_g + 1):
    children, retention, d_distro = xover(deepcopy(parents), max_r, p_xov, 'OnePoint', fixed_length = fixed_length)
    xov_fitness_evaluator = Fitness(train_x, bias, train_y, children, func, bank, n_inp, max_d, fit, arity)
    children, mutated_inds = mutate(deepcopy(children), max_c, max_r, max_d, bank, inputs=1, n_bias=10, arity=2)
    pop = parents + children
    fitness_evaluator = Fitness(train_x, bias, train_y, pop, func, bank, n_inp, max_d, fit, arity)
    fitnesses, alignment[:, 0], alignment[:, 1] = fitness_evaluator()
    xov_fitnesses, _, _ = xov_fitness_evaluator()
    drift_per_parent_mut, drift_per_parent_xov = mut_impact(fitnesses, xov_fitnesses, max_p, retention, mutated_inds, opt=1)
    avg_hist_list, avg_change_list, std_change_list, ret_avg_list, ret_std_list = processRetention(retention, pop,
                                                                                                   fitnesses, xov_fitnesses, max_p,
                                                                                                   avg_hist_list,
                                                                                                   avg_change_list,
                                                                                                   std_change_list,
                                                                                                   ret_avg_list,
                                                                                                   ret_std_list, g, mode='lgp')
 
    sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std = processSharpnessLGP(train_x, train_x_bias, alignment,
                                                                                     max_p,
                                                                                     max_c, n_inp, func, bank, n_inp,
                                                                                     fit, arity, max_d,
                                                                                     sharp_in_manager,
                                                                                     fitness_evaluator, pop,
                                                                                     sharp_in_list, sharp_in_std,
                                                                                     sharp_out_manager,
                                                                                     sharp_out_list, sharp_out_std,
                                                                                     train_y)
    density_distro, density_distro_list = associateDistro(drift_per_parent_xov, retention, d_distro, density_distro, density_distro_list, g)

    best_i = getBestInd(fitnesses)
    best_fit = fitnesses[best_i]
    if g % 100 == 0:
        logAndProcessSharpness(g, best_fit, sharp_in_list, sharp_out_list)

    fit_track.append(best_fit)
    p_size.append(len(effProg(max_d, pop[best_i])))
    parents = select(pop, fitnesses, max_p)

pop = parents + children
fitness_evaluator = Fitness(train_x, bias, train_y, pop, func, bank, n_inp, max_d, fit, arity)
fitnesses, alignment[:, 0], alignment[:, 1] = fitness_evaluator()
best_i, best_fit, best_pop, mut_list, mut_cum, xov_list, xov_cum, density_distro = processAndPrintResults(
    t, fitnesses, pop, mut_impact, density_distro, mode='lgp')
# print(list(train_y))
Path(f"../output/{run_name}/{func_name}/log/").mkdir(parents=True, exist_ok=True)

first_body_node = n_inp + bias.shape[0]
bin_centers, hist_gens, avg_hist_list = change_histogram_plot(avg_hist_list, func_name, run_name, t, max_g)

p_A = alignment[best_i, 0]
p_B = alignment[best_i, 1]

p = effProg(max_d, best_pop, first_body_node)
lgp_print_individual(p, p_A, p_B, 'lgp_1x', func_name, bank_string, t, bias, n_inp, first_body_node)
with open(f"../output/{run_name}/{func_name}/best_program/best_{t}.txt", 'a') as f:
    f.write(f"\nEffective Instructions\n\n")
    f.write(f'{p}')
print('effective program')
print(p)

saveResults(run_name, func_name, t, bias, best_pop, preds, best_fit, len(p), fit_track, avg_change_list, ret_avg_list,
            p_size, bin_centers, hist_gens, avg_hist_list, mut_list, mut_cum,
            xov_list, xov_cum, sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std, density_distro, density_distro_list)

print('run finished')
dot = draw_graph_thicc(p, p_A, p_B)
Path(f"../output/{run_name}/{func_name}/full_graphs/").mkdir(parents=True, exist_ok=True)
dot.render(f"../output/{run_name}/{func_name}/full_graphs/graph_{t}", view=False)

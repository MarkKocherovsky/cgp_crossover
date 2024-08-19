import pickle
from typing import Any, Generator
from sys import argv
from cgp_fitness import *
from functions import Function, Collection
from sharpness import *
from similarity import *


def get_param(index, default, cast=float): 
    try: 
        return cast(argv[index]) 
    except (IndexError, ValueError): 
        return default 

def getNeighborMap(true_predictions: np.ndarray, sharp_out: SAM_OUT, fitness_list: Fitness, target: np.ndarray):
    neighborhood = sharp_out.perturb(true_predictions)
    return [fitness_list.fit(neighbor, target) for neighbor in neighborhood]


def loadBank() -> tuple:
    return (add, sub, mul, div), ("+", "-", "*", "/")


def getXY(func: Function) -> tuple:
    return func.x_dom, func.y_test


def getFunction(func_index: int) -> tuple:
    func_bank = Collection()
    return func_bank.func_list[func_index], func_bank.name_list[func_index], func_bank.func_list[func_index].dimensions


def initAlignment(max_p: int, max_c: int) -> np.ndarray:
    alignment = np.ones((max_p + max_c, 2))
    alignment[:, 1] = 0.0
    return alignment


def prepareConstants(train_x: np.ndarray, biases: np.ndarray) -> np.ndarray:
    train_x_bias = np.zeros((train_x.shape[0], biases.shape[0] + 1))
    train_x_bias[:, 0] = train_x
    train_x_bias[:, 1:] = biases
    return train_x_bias


def initDensityDistro(max_n: int, operators: int, arity: int, mode: str = 'cgp', outputs:int = 0) -> dict:
    if mode == 'cgp':
        m = operators + arity
    else:
        m = 1
    return {
        'd': np.zeros(max_n * m+outputs, dtype=np.int32),
        'n': np.zeros(max_n * m+outputs, dtype=np.int32),
        'b': np.zeros(max_n * m+outputs, dtype=np.int32)
    }


def initFitness(max_p: int, max_c: int) -> (Fitness, np.ndarray):
    fitness_objects = [Fitness() for _ in range(0, max_p + max_c)]
    fitnesses = np.zeros((max_p + max_c), )
    return fitness_objects, fitnesses


def getNoise(shape, max_p, max_c, noise_inputs, noise_func, sharp_in_manager: SAM_IN, opt=0):
    pop_size = max_p + max_c
    x = []
    y = []

    if opt == 1:
        fixed_inputs = sharp_in_manager.perturb_data()[:, :noise_inputs]

    for _ in range(pop_size):
        noisy_x_preproc = np.zeros(shape)
        if opt == 1:
            # noinspection PyUnboundLocalVariable
            noisy_x_preproc[:, :noise_inputs] = fixed_inputs
        else:
            noisy_x_preproc[:, :noise_inputs] = sharp_in_manager.perturb_data()[:, :noise_inputs]

        noisy_x_preproc[:, noise_inputs:] = sharp_in_manager.perturb_constants()[:, noise_inputs:]

        noisy_y_preproc = np.fromiter(
            map(noise_func.func, noisy_x_preproc[:, :noise_inputs].flatten()),
            dtype=np.float32
        )

        x.append(noisy_x_preproc)
        y.append(noisy_y_preproc)

    return np.array(x), np.array(y)


def initTrackers() -> Generator[list[Any], Any, None]:
    return ([] for _ in range(6))


def getBestInd(fitnesses: np.ndarray, max_p: int = None) -> int:
    if max_p is not None:
        fitnesses = fitnesses[:max_p]
    return np.argmin(fitnesses)


def processFitness(fitness_objects, train_x_bias, train_y, pop, max_p, max_c, opt=0):
    fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(range(max_p + max_c), pop)])
    alignment = np.zeros((fit_temp.shape[0], 2))
    if opt == 0:
        fitnesses = fit_temp[:, 0].copy().flatten()
        alignment[:, 0] = fit_temp[:, 1].copy()
        alignment[:, 1] = fit_temp[:, 2].copy()
    elif opt == 1:
        fitnesses = fit_temp[:max_p, 0].copy().flatten()
        alignment[:max_p, 0] = fit_temp[:, 1].copy()
        alignment[:max_p, 1] = fit_temp[:, 2].copy()
    else:
        raise KeyError(f'helper.py::processFitness: asked for option {opt}, only 1 and 0 are valid.')

    if any(np.isnan(fitnesses)):  # Replace NaNs with positive infinity to screen them out
        fitnesses[np.isnan(fitnesses)] = np.PINF

    return fitnesses, alignment


def processRetention(retention, pop, fitnesses, max_p, avg_hist_list, avg_change_list, std_change_list, ret_avg_list,
                     ret_std_list, g, mode='cgp'):
    change_list = []
    full_change_list = []
    ret_list = []

    for p in retention:
        ps = [pop[p], pop[p + 1]]
        p_fits = np.array([fitnesses[p], fitnesses[p + 1]])
        cs = [pop[p + max_p], pop[p + max_p + 1]]
        c_fits = np.array([fitnesses[p + max_p], fitnesses[p + max_p + 1]])

        best_p = np.argmin(p_fits)
        best_c = np.argmin(c_fits)

        change_list.append(percent_change(c_fits[best_c], p_fits[best_p]))
        if mode == 'cgp':
            ret_list.append(find_similarity(cs[best_c][0], ps[best_p][0], cs[best_c][1], ps[best_p][1], mode))
        elif mode == 'lgp':
            ret_list.append(find_similarity(cs[best_c], ps[best_p], cs[best_c], ps[best_p], mode))
        full_change_list.extend([percent_change(c, p_fits[best_p]) for c in c_fits])

    full_change_list = np.array(full_change_list).flatten()
    full_change_list = full_change_list[np.isfinite(full_change_list)]

    cl_std = np.nanstd(full_change_list)

    if not all(cl == 0.0 for cl in full_change_list):
        avg_hist_list.append((g, np.histogram(full_change_list, bins=10, range=(cl_std * -2, cl_std * 2))))

    avg_change_list.append(np.nanmean(change_list))
    std_change_list.append(np.nanstd(change_list))
    ret_avg_list.append(np.nanmean(ret_list))
    ret_std_list.append(np.nanstd(ret_list))

    return avg_hist_list, avg_change_list, std_change_list, ret_avg_list, ret_std_list


def processSharpness(train_x_bias, max_p, max_c, inputs, func, sharp_in_manager, fitness_objects, pop, train_y,
                     sharp_in_list, sharp_in_std, sharp_out_manager, sharp_out_list, sharp_out_std):
    # Get noisy data
    noisy_x, noisy_y = getNoise(train_x_bias.shape, max_p, max_c, inputs, func, sharp_in_manager, opt=1)
    # Calculate sharpness for SAM-IN
    sharpness = np.array(
        [fitness_objects[i](noisy_x[i], noisy_y[i], individual)[0] for i, individual in zip(range(max_p + max_c), pop)])
    sharp_in_list.append(np.mean(sharpness))
    sharp_in_std.append(np.std(sharpness))

    # Calculate predictions
    preds = [fitness_objects[i](train_x_bias, train_y, individual, opt=1)[0] for i, individual in
             zip(range(max_p + max_c), pop)]

    # Calculate neighbor map and sharpness for SAM-OUT
    neighbor_map = np.array([getNeighborMap(pred, sharp_out_manager, fitness_objects[i], train_y) for i, pred in
                             zip(range(max_p + max_c), preds)])
    out_sharpness = np.std(neighbor_map, axis=1) ** 2
    sharp_out_list.append(np.mean(out_sharpness))
    sharp_out_std.append(np.std(out_sharpness))

    return sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std


def logAndProcessSharpness(g, best_fit, sharp_in_list, sharp_out_list):
    # Log the current generation's best fitness and sharpness statistics
    print(f"Gen {g} Best Fitness: {best_fit}\tMean SAM-In: {sharp_in_list[-1]}\tMean SAM-Out: {sharp_out_list[-1]}")


def processAndPrintResults(t, fitnesses, pop, mut_impact, density_distro, train_x_bias=None, train_y=None, mode='cgp'):
    # Get the best individual
    best_i = getBestInd(fitnesses)
    best_fit = fitnesses[best_i]
    best_pop = pop[best_i]

    # Print trial best fitness
    print(f"Trial {t}: Best Fitness = {best_fit}")

    # Get drift cumulative and list
    mut_cum, mut_list, xov_cum, xov_list = mut_impact.returnLists(option=0)

    # Print drift information
    print(f"mutation:\tDeleterious\tNeutral\tBeneficial")
    print(f"\t{mut_cum[0]}\t{mut_cum[1]}\t{mut_cum[2]}")
    print(f"crossover:\tDeleterious\tNeutral\tBeneficial")
    print(f"\t{xov_cum[0]}\t{xov_cum[1]}\t{xov_cum[2]}")

    print('best individual')
    print(best_pop)

    print('Density Distribution')
    print(f'Deleterious\n{density_distro["d"]}')
    print(f'Neutral\n{density_distro["n"]}')
    print(f'Beneficial\n{density_distro["b"]}')

    # Calculate and print Probability Density Function
    density_distro['d'] = density_distro['d'] / np.sum(density_distro['d'])
    density_distro['n'] = density_distro['n'] / np.sum(density_distro['n'])
    density_distro['b'] = density_distro['b'] / np.sum(density_distro['b'])
    print(density_distro)

    if mode == 'cgp':  #this is weird for some reason
        pred_fitness = Fitness()
        preds, p_a, p_b = pred_fitness(train_x_bias, train_y, best_pop, opt=1)
        return best_i, best_fit, best_pop, mut_list, mut_cum, xov_list, xov_cum, density_distro, preds, p_a, p_b

    return best_i, best_fit, best_pop, mut_list, mut_cum, xov_list, xov_cum, density_distro


def saveResults(run_name, func_name, t, biases, best_pop, preds, best_fit, n, fit_track, avg_change_list, ret_avg_list,
                p_size, bin_centers, hist_gens, avg_hist_list, mut_list, mut_cum,
                xov_list, xov_cum, sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std, density_distro):
    with open(f"../output/{run_name}/{func_name}/log/output_{t}.pkl", "wb") as f:
        pickle.dump(biases, f)
        pickle.dump(best_pop, f)
        pickle.dump(preds, f)
        pickle.dump(best_fit, f)
        pickle.dump(n, f)
        pickle.dump(fit_track, f)
        pickle.dump([avg_change_list], f)
        pickle.dump([ret_avg_list], f)
        pickle.dump(p_size, f)
        pickle.dump([bin_centers, hist_gens, avg_hist_list], f)
        pickle.dump([mut_list, mut_cum], f)
        pickle.dump([xov_list, xov_cum], f)
        pickle.dump([sharp_in_list, sharp_out_list], f)
        pickle.dump([sharp_in_std, sharp_out_std], f)
        pickle.dump(density_distro, f)


def associateDistro(drift_per_parent, retention, d_distro, density_distro, mode='normal'):
    drift_per_parent = np.array(drift_per_parent)

    d_distro = np.array(d_distro).astype(np.int32)
    if len(retention) < 1:
        return density_distro
    if mode == 'dnc':
        d_distro = np.sum(d_distro, axis = 1)
    for i, p_index in zip(range(drift_per_parent.shape[0]), list(retention)):
        if drift_per_parent[i] == 0:
            density_distro['d'] += d_distro[p_index, :]
        elif drift_per_parent[i] == 2:
            density_distro['b'] += d_distro[p_index, :]
        else:
            density_distro['n'] += d_distro[p_index, :]
    return density_distro

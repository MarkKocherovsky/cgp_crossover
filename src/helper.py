import pickle
from typing import Any, Generator
from sys import argv
from cgp_fitness import *
from functions import Function, Collection
from sharpness import *
from similarity import *
from numpy import random
from pathlib import Path

def get_param(index, default, cast=float): 
    try: 
        return cast(argv[index]) 
    except (IndexError, ValueError): 
        return default 

def getNeighborMap(true_predictions: np.ndarray, sharp_out: SAM_OUT, fitness_list: Fitness, target: np.ndarray, n_neighbors = 25):
    neighborhood = sharp_out.perturb(true_predictions, n_neighbors)
    return [fitness_list.fit(neighbor, target) for neighbor in neighborhood]


def loadBank() -> tuple:
    return (add, sub, mul, div), ("+", "-", "*", "/")

def getXY(n_dims:int, n_points:int, func: 'Function') -> tuple:
    low = func.x_dom[0]
    high = func.x_dom[-1]
    xs = np.sort(np.random.uniform(low, high, size=(n_points, n_dims)), axis=0)  # Corrected
    if n_dims == 1:
        xs = xs.flatten()
    
    ys = np.array([func(x) for x in list(xs)])  # Now will process each point independently
    return xs, ys


def getFunction(func_index: int, dimensions: int = 1) -> tuple:
    c = Collection('standard')
    return c(func_index, dimensions)

def initAlignment(max_p: int, max_c: int) -> np.ndarray:
    alignment = np.ones((max_p + max_c, 2))
    alignment[:, 1] = 0.0
    return alignment

def prepareConstants(train_x: np.ndarray, biases: np.ndarray) -> np.ndarray:
    if train_x.ndim == 1:
        train_x = train_x.reshape(len(train_x), -1)
    n_inputs = train_x.shape[-1]
    train_x_bias = np.zeros((train_x.shape[0], biases.shape[0] + n_inputs))
    train_x_bias[:, :n_inputs] = train_x
    train_x_bias[:, n_inputs:] = biases
    return train_x_bias

def initDensityDistro(max_n: int, operators: int, arity: int, max_g: int, mode: str = 'cgp', outputs: int = 1,
                      shape_override=None) -> dict:
    m = operators + arity if mode == 'cgp' else 1
    shape = shape_override if shape_override is not None else (max_g, (max_n * m + outputs))

    density_distro_list = {
        'd': np.zeros(shape, dtype=np.int32),
        'n': np.zeros(shape, dtype=np.int32),
        'b': np.zeros(shape, dtype=np.int32)
    }

    density_distro = {
        'd': np.zeros(shape[1], dtype=np.int32),
        'n': np.zeros(shape[1], dtype=np.int32),
        'b': np.zeros(shape[1], dtype=np.int32)
    }
    print(f'density_distro_list[d].shape: {density_distro_list["d"].shape}')
    print(f'density_distro[d].shape: {density_distro["d"].shape}')
    return density_distro, density_distro_list


def initFitness(max_p: int, max_c: int) -> (Fitness, np.ndarray):
    fitness_objects = [Fitness() for _ in range(0, max_p + max_c)]
    fitnesses = np.zeros((max_p + max_c), )
    return fitness_objects, fitnesses

"""
def getNoise(shape, max_p, max_c, noise_inputs, noise_func, sharp_in_manager: SAM_IN, opt=1, choice_prop = 1.00):
    pop_size = max_p + max_c
    x = []
    y = []
    for _ in range(pop_size):
        noisy_x_preproc = np.zeros(shape)
        
        noisy_x_preproc[:, :noise_inputs] = sharp_in_manager.perturb_data()[:, :noise_inputs]
        noisy_x_preproc[:, noise_inputs:] = sharp_in_manager.perturb_constants()[:, noise_inputs:]
        noisy_y_preproc = np.fromiter(
            map(noise_func.func, noisy_x_preproc[:, :noise_inputs]),
            dtype=np.float32
        )

        x.append(noisy_x_preproc)
        y.append(noisy_y_preproc)
    return np.atleast_2d(x), np.atleast_2d(y)
"""

def getNoise(shape, max_p, max_c, noise_inputs, noise_func, sharp_in_manager: SAM_IN, opt=0):
    pop_size = max_p + max_c
    x = []
    y = []

    if opt == 1: #Only perturb inputs once per generation
        fixed_inputs = sharp_in_manager.perturb_data()[:, :noise_inputs]
    for _ in range(pop_size):
        noisy_x_preproc = np.zeros(shape)
        if opt == 1:
            # noinspection PyUnboundLocalVariable
            noisy_x_preproc[:, :noise_inputs] = fixed_inputs
        else:
            noisy_x_preproc[:, :noise_inputs] = sharp_in_manager.perturb_data()[:, :noise_inputs]
        #perturb constants for each individual
        noisy_x_preproc[:, noise_inputs:] = sharp_in_manager.perturb_constants()[:, noise_inputs:]

        noisy_y_preproc = np.fromiter(
            map(noise_func.func, noisy_x_preproc[:, :noise_inputs]),
            dtype=np.float32
        )
        x.append(noisy_x_preproc)
        y.append(noisy_y_preproc)

    return np.array(x), np.array(y)

def initTrackers() -> Generator[list[Any], Any, None]:
    return ([] for _ in range(8))


def getBestInd(fitnesses: np.ndarray, max_p: int = None) -> int:
    if max_p is not None:
        fitnesses = fitnesses[:max_p]
    return np.argmin(fitnesses)


def processFitness(fitness_objects, train_x_bias, train_y, pop, max_p, max_c, arity, opt=1):
    fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind, arity) for i, ind in zip(range(max_p + max_c), pop)])
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


def processRetention(retention, pop, old_fits, fitnesses, max_p, avg_hist_list, avg_change_list, std_change_list, ret_avg_list,
                     ret_std_list, g, first_body_node, mode='cgp'):
    change_list = []
    full_change_list = []
    ret_list = []

    for p in retention:
        ps = [pop[p], pop[p + 1]]
        p_fits = np.array([old_fits[p], old_fits[p + 1]])
        cs = [pop[p + max_p], pop[p + max_p + 1]]
        c_fits = np.array([fitnesses[p], fitnesses[p+1]])

        best_p = np.argmin(p_fits)
        best_c = np.argmin(c_fits)

        change_list.append(percent_change(c_fits[best_c], p_fits[best_p]))
        if mode == 'cgp':
            ret_list.append(find_similarity(cs[best_c][0], ps[best_p][0], first_body_node, cs[best_c][1], ps[best_p][1], mode))
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


def processSharpness(train_x_bias, max_p, max_c, inputs, func, sharp_in_manager, fitness_objects, real_fitnesses, pop, train_y,
                     sharp_in_list, sharp_in_std, sharp_out_manager, sharp_out_list, sharp_out_std, arity, n_neighbors = 25):
    # Get noisy data
    noisy_x, noisy_y = getNoise(train_x_bias.shape, max_p, max_c, inputs, func, sharp_in_manager, opt=1)
    # Calculate sharpness for SAM-IN
    fitnesses = np.array( #perturbed fitnesses
        [fitness_objects[i](noisy_x[i], noisy_y[i], individual, arity)[0] for i, individual in zip(range(max_p + max_c), pop)])
    #print(f'SAM-In: {real_fitnesses} - {fitnesses} = {real_fitnesses - fitnesses}')
    sharpness = np.abs(real_fitnesses-fitnesses)
    sharp_in_list.append(np.mean(sharpness))
    sharp_in_std.append(np.std(sharpness))

    # Calculate predictions
    preds = [fitness_objects[i](train_x_bias, train_y, individual, arity, opt=1)[0] for i, individual in
             zip(range(max_p + max_c), pop)]

    # Calculate neighbor map and sharpness for SAM-OUT
    neighbor_map = np.array([getNeighborMap(pred, sharp_out_manager, fitness_objects[i], train_y, n_neighbors) for i, pred in
                             zip(range(max_p + max_c), preds)])
    out_sharpness = np.std(neighbor_map, axis=1) ** 2
    sharp_out_list.append(np.mean(out_sharpness))
    sharp_out_std.append(np.std(out_sharpness))

    return sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std


def logAndProcessSharpness(g, best_fit, fit_mean, sharp_in_list, sharp_out_list):
    # Log the current generation's best fitness and sharpness statistics
    print(f"Gen {g} Best Fitness: {best_fit}\tMean Fitness: {fit_mean[-1]}\tMean SAM-In: {sharp_in_list[-1]}\tMean SAM-Out: {sharp_out_list[-1]}")

def processAndPrintResults(t, fitnesses, pop, mut_impact, density_distro, mut_density_distro, train_x_bias=None,
                           train_y=None, mode='cgp', arity=2):
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

    print('Xover Density Distribution')
    print(f'Deleterious\n{density_distro["d"]}')
    print(f'Neutral\n{density_distro["n"]}')
    print(f'Beneficial\n{density_distro["b"]}')

    def calculate_distro(d_distro):
        d_distro['d'] = d_distro['d'] / np.sum(d_distro['d'])
        d_distro['n'] = d_distro['n'] / np.sum(d_distro['n'])
        d_distro['b'] = d_distro['b'] / np.sum(d_distro['b'])
        return d_distro

    print('Mutation Density Distribution')
    print(f'Deleterious\n{mut_density_distro["d"]}')
    print(f'Neutral\n{mut_density_distro["n"]}')
    print(f'Beneficial\n{mut_density_distro["b"]}')

    # Calculate and print Probability Density Function
    density_distro = calculate_distro(density_distro)
    mut_density_distro = calculate_distro(mut_density_distro)
    # print(density_distro)

    if mode == 'cgp':  # this is weird for some reason
        pred_fitness = Fitness()
        preds, p_a, p_b = pred_fitness(train_x_bias, train_y, best_pop, arity, opt=1)
        return best_i, best_fit, best_pop, mut_list, mut_cum, xov_list, xov_cum, density_distro, mut_density_distro, preds, p_a, p_b

    return best_i, best_fit, best_pop, mut_list, mut_cum, xov_list, xov_cum, density_distro, mut_density_distro


def saveResults(run_name, func_name, t, biases, best_pop, preds, best_fit, n, fit_track, avg_change_list, ret_avg_list,
                p_size, bin_centers, hist_gens, avg_hist_list, mut_list, mut_cum,
                xov_list, xov_cum, sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std, density_distro, dd_list,
                mut_distro, mut_distro_list, div_list, fit_mean, path = None):
    if path is None:
        path = f"../output/{run_name}/{func_name}/log/"
    print(f'{path}output_{t}')
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(f'{path}output_{t}.pkl', "wb") as f:
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
        pickle.dump(dd_list, f)
        pickle.dump(mut_distro, f)
        pickle.dump(mut_distro_list, f)
        pickle.dump(div_list, f)
        pickle.dump(fit_mean, f)

def associateDistro(drift_per_parent, retention, d_distro, density_distro, dd_list, g, mode='normal'):
    drift_per_parent = np.array(drift_per_parent)
    d_distro = np.array(d_distro).astype(np.int32)
    retention = np.array(retention).flatten()
    if mode != 'dnc':
        g -= 1
    if len(retention) < 1:
        return density_distro, dd_list
    
    #dnc returns an array of n/2 identical arrays, total shape is (n/2, #actually used in crossover, genome_length)
    #if mode == 'dnc': 
    #    summed_distro = d_distro[0, :, :]  # summed_distro has shape (len(retention), 49)
    #    print(summed_distro)
    for i, p_index in enumerate(retention):
        try:
            # If mode is 'dnc', use summed_distro, otherwise use original d_distro
            d_sum = d_distro[i, :] if mode == 'dnc' else d_distro[p_index, :]
            #d_sum = d_distro[p_index, :]
            if drift_per_parent[i] == 0:
                density_distro['d'] += d_sum
                dd_list['d'][g, :] = d_sum
            elif drift_per_parent[i] == 2:
                density_distro['b'] += d_sum
                dd_list['b'][g, :] = d_sum
            else:
                density_distro['n'] += d_sum
                dd_list['n'][g, :] = d_sum
        except (ValueError, IndexError) as e:
            print('---\nhelper.py::associateDistro')
            print(e)
            print(f'generation {g}')
            print(f'p_index {p_index}')
            print(f'retention {retention}')
            print(f'drift_per_parent {drift_per_parent}')
            print(f'd_distro {d_distro}')
            print(f'd_distro.shape {d_distro.shape}')
            print(f'density_distro {density_distro}')
            print(f"density_distro['n'].shape {density_distro['n'].shape}")
            print(f"dd_list['n'].shape {dd_list['n'].shape}")
            print(d_distro[p_index, :].shape)
            print(density_distro['d'].shape)
            print(dd_list['d'][g, :])

            exit(1)
    return density_distro, dd_list
def semantic_diversity(fitnesses):
    return np.nanstd(fitnesses)

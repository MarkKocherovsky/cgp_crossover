from helper import getNoise
from lgp_fitness import *
from similarity import *


def getNeighborMap(preds, sharp_out_manager, train_y):
    neighborhood = sharp_out_manager.perturb(preds)
    return [corr(neighbor, train_y) for neighbor in neighborhood]


def get_fitness_evaluators(x, b, y, pr, func, bank, n_inp, max_d, fit, arity):
    return [Fitness(x[i], b[i], y[i], [p], func, bank, n_inp, max_d, fit, arity) for p, i in zip(pr, range(len(pr)))]


def get_sam_in(noisy_x, b, noisy_y, pop, func, bank, n_inp, max_d, fit, arity):
    sharpness_evaluators = get_fitness_evaluators(noisy_x, b, noisy_y, pop, func, bank, n_inp, max_d, fit, arity)
    sharpness = np.array([evaluator()[0] for evaluator in sharpness_evaluators]).flatten()
    return sharpness


def processSharpnessLGP(train_x, train_x_bias, alignment, max_p, max_c, inputs, func, bank, n_inp, fit, arity, max_d,
                        sharp_in_manager, fitness_eval, pop, sharp_in_list, sharp_in_std, sharp_out_manager,
                        sharp_out_list, sharp_out_std, train_y):
    # Get noisy data
    noisy_x, noisy_y = getNoise(train_x_bias.shape, max_p, max_c, inputs, func, sharp_in_manager, opt=1)
    # Calculate sharpness for SAM-IN
    sharpness = get_sam_in(noisy_x[:, :, :inputs], noisy_x[:, :, inputs:], noisy_y, pop, func, bank, n_inp, max_d, fit,
                           arity)
    sharp_in_list.append(np.mean(sharpness))
    sharp_in_std.append(np.std(sharpness))

    # Calculate predictions
    preds = [fitness_eval.predict(p, A, B, inputs, train_x) for p, A, B in zip(pop, alignment[:, 0], alignment[:, 1])]

    neighbor_map = np.array(
        [getNeighborMap(pred, sharp_out_manager, train_y) for i, pred in zip(range(0, max_p), preds)])
    out_sharpness = np.std(neighbor_map, axis=1) ** 2
    sharp_out_list.append(np.mean(out_sharpness))
    sharp_out_std.append(np.std(out_sharpness))

    return sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std

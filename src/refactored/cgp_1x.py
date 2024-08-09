import numpy as np
from numpy import random
from pathlib import Path
from copy import deepcopy
from scipy.stats import skew, kurtosis
import warnings
import pickle
from cgp_utils import *

warnings.filterwarnings('ignore')

print("started")

t, max_g, max_n, max_p, max_c, p_mut, p_xov = setup_environment(
    t=int(argv[1]), 
    max_g=int(argv[2]), 
    max_n=int(argv[3]), 
    max_p=int(argv[4]), 
    max_c=int(argv[5]), 
    p_mut=float(argv[8]), 
    p_xov=float(argv[9]), 
    seed=t+100
)

run_name = 'cgp_1x'

# Define bank and functions
bank = (add, sub, mul, div)
bank_string = ("+", "-", "*", "/")
func_bank = Collection()
func = func_bank.func_list[int(argv[6])]
func_name = func_bank.name_list[int(argv[6])]
train_x = func.x_dom
train_y = func.y_test

f = int(argv[7])
fits = FitCollection()
fit = fits.fit_list[f]
fit_name  = fits.name_list[f]

alignment = np.zeros((max_p+max_c, 2))
alignment[:, 0] = 1.0

biases = np.arange(0, 10, 1).astype(np.int32)
bias = biases.shape[0]
train_x_bias = np.zeros((train_x.shape[0], biases.shape[0]+1))
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = biases

parents, fitness_objects, fitnesses = initialize_population(max_p, max_n, bank, outputs=1, arity=2)
density_distro = np.zeros(max_n*(outputs+arity), dtype=np.int32)
sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()

# Initial fitness evaluation
fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, parent) for i, parent in zip(range(0, max_p), parents)])
fitnesses[:max_p] = fit_temp[:, 0].copy().flatten()
alignment[:max_p, 0] = fit_temp[:, 1].copy() #a
alignment[:max_p, 1] = fit_temp[:, 2].copy() #b

# SAM-IN
noisy_x, noisy_y = get_noise(train_x_bias.shape, sharp_in_manager, inputs=1, func=func, pop_size=max_p+max_c)
sharpness, sharp_in_list, sharp_in_std = calculate_sharpness(noisy_x, noisy_y, parents, fitness_objects, max_p, max_c)

# SAM-OUT
preds = [fitness_objects[i](train_x_bias, train_y, parent, opt=1)[0] for i, parent in zip(range(0, max_p), parents)]
neighbor_map = calculate_neighbor_map(preds, sharp_out_manager, fitness_objects, train_y)
sharp_out_list = [np.mean(np.std(neighbor_map, axis=1)**2)]
sharp_out_std = [np.std(np.std(neighbor_map, axis=1))]

fit_track = []
ret_avg_list, ret_std_list = [], []
avg_change_list, avg_hist_list, std_change_list = [], [], []
best_i = np.argmin(fitnesses[:max_p])
p_size = [cgp_active_nodes(parents[best_i][0], parents[best_i][1], opt=2)]

mut_impact = MutationImpact(neutral_limit=0.1)
num_elites = 7

for g in range(1, max_g+1):
    children, retention, density_distro = xover(deepcopy(parents), density_distro, method='OnePoint', max_n=max_n)
    children = mutate(deepcopy(children))
    fitnesses, alignment, pop = update_population(fitnesses, parents, children, max_p, max_c, fitness_objects, train_x_bias, train_y)
    
    if any(np.isnan(fitnesses)): # Replace nans with positive infinity to screen them out
        nans = np.isnan(fitnesses)
        fitnesses[nans] = np.PINF 
    
    arg_sorted = np.argsort(fitnesses)
    parents = [pop[model] for model in arg_sorted[:max_p]]
    
    best_i = np.argmin(fitnesses[:max_p])
    fit_track.append(deepcopy(np.min(fitnesses[:max_p])))
    
    ret_avg_list.append(np.mean(retention[:max_p]))
    ret_std_list.append(np.std(retention[:max_p]))
    
    sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std, sharpness, out_sharpness = perturb_and_calculate_sharpness(
        sharp_in_manager, parents, fitness_objects, train_x_bias, train_y, max_p, max_c
    )
    
    pop = parents + children
    p_size.append(cgp_active_nodes(parents[best_i][0], parents[best_i][1], opt=2))

log_and_save_results(output_dir=Path.cwd(), trial=t, results=[fit_track, ret_avg_list, ret_std_list, avg_change_list, avg_hist_list, std_change_list, p_size])

plt.plot(fit_track)
plt.ylabel("Fitness")
plt.xlabel("Generations")
plt.show()

print("Script complete")


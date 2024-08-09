import numpy as np
from copy import deepcopy

def create_input_vector(x, biases, inputs):
    """Create input vector with biases."""
    vec = np.zeros((x.shape[0], biases.shape[0] + 1))
    vec[:, :inputs] = x
    vec[:, inputs:] = biases
    return vec

def initialize_population(max_p, max_n, bank, arity):
    """Generate initial parent population."""
    return generate_parents(max_p, max_n, bank, first_body_node=11, outputs=1, arity=arity)

def get_noise_data(shape, inputs, func, sharp_in_manager, opt=0):
    """Generate noisy input and output data."""
    x = []
    y = []
    fixed_inputs = sharp_in_manager.perturb_data()[:, :inputs] if opt == 1 else None
    for _ in range(shape[0]):
        noisy_x = np.zeros(shape)
        noisy_x[:, :inputs] = fixed_inputs if opt == 1 else sharp_in_manager.perturb_data()[:, :inputs]
        noisy_x[:, inputs:] = sharp_in_manager.perturb_constants()[:, inputs:]
        noisy_y = np.fromiter(map(func.func, noisy_x[:, :inputs].flatten()), dtype=np.float32)
        x.append(noisy_x)
        y.append(noisy_y)
    return np.array(x), np.array(y)

def evaluate_neighbors(preds, sharp_out_manager, fitness, train_y):
    """Evaluate fitness of neighbors."""
    neighborhood = sharp_out_manager.perturb(preds)
    return [fitness.fit(neighbor, train_y) for neighbor in neighborhood]

def process_generation(g, max_g, parents, train_x_bias, train_y, fitness_objects, sharp_in_manager, sharp_out_manager, max_c, func, arity):
    """Process a single generation of the evolutionary algorithm."""
    mutate = mutate_1_plus_4
    select = tournament_elitism
    
    # Generate offspring
    children = [mutate(deepcopy(parent)) for parent in parents for _ in range(max_c)]
    pop = parents + children
    fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in enumerate(pop)])
    fitnesses = fit_temp[:, 0].flatten()
    alignment = fit_temp[:, 1:3]
    
    if np.any(np.isnan(fitnesses)):
        fitnesses[np.isnan(fitnesses)] = np.PINF
    
    mut_impact = MutationImpact(neutral_limit=0.1)
    mut_impact(fitnesses, len(parents), option='OneParent', children=4)
    
    ret_avg_list, std_change_list = [], []
    avg_change_list, avg_hist_list = [], []
    
    for i in range(len(parents)):
        best_p = parents[i]
        best_p_fit = fitnesses[i]
        c_fits = fitnesses[len(parents) + i*max_c: len(parents) + (i+1)*max_c]
        cs = [children[j] for j in range(i*max_c, (i+1)*max_c)]
        
        full_change_list = [percent_change(c_fit, best_p_fit) for c_fit in c_fits]
        ret = [find_similarity(c[0], best_p[0], c[1], best_p[1]) for c in cs]
        
        best_c_idx = np.argmin(c_fits)
        best_c_fit = c_fits[best_c_idx]
        best_c = cs[best_c_idx]
        
        avg_change_list.append(np.nanmean(full_change_list))
        std_change_list.append(np.nanstd(full_change_list))
        ret_avg_list.append(np.nanmean(ret))
        
        if len(full_change_list) > 0:
            cl_std = np.nanstd(full_change_list)
            avg_hist_list.append((g, np.histogram(full_change_list, bins=10, range=(cl_std*-2, cl_std*2))))
    
    noisy_x, noisy_y = get_noise_data(train_x_bias.shape, inputs, func, sharp_in_manager, opt=1)
    sharpness = np.array([fitness_objects[i](noisy_x[i], noisy_y[i], individual)[0] for i, individual in enumerate(pop)])
    
    sharp_in_list = [np.mean(sharpness)]
    sharp_in_std = [np.std(sharpness)]
    
    preds = [fitness_objects[i](train_x_bias, train_y, individual, opt=1)[0] for i, individual in enumerate(pop)]
    neighbor_map = np.array([evaluate_neighbors(pred, sharp_out_manager, fitness_objects[i], train_y) for i, pred in enumerate(preds)])
    out_sharpness = np.std(neighbor_map, axis=1)**2
    sharp_out_list = [np.mean(out_sharpness)]
    sharp_out_std = [np.std(out_sharpness)]
    
    best_i = np.argmin(fitnesses)
    best_fit = fitnesses[best_i]
    p_size = [cgp_active_nodes(pop[best_i][0], pop[best_i][1], opt=2)]
    
    return pop, parents, fit_temp, fitnesses, best_i, best_fit, sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std, avg_change_list, std_change_list, ret_avg_list, avg_hist_list, p_size


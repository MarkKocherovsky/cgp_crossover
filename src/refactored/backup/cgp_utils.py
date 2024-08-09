import numpy as np

def get_noise(shape, pop_size, inputs, func, opt=0):
    """Generate noisy input and output data."""
    x, y = [], []
    fixed_inputs = None
    if opt == 1:
        fixed_inputs = SAM_IN().perturb_data()[:, :inputs]
    
    for _ in range(pop_size):
        noisy_x = np.zeros(shape)
        if opt == 1:
            noisy_x[:, :inputs] = fixed_inputs
        else:
            noisy_x[:, :inputs] = SAM_IN().perturb_data()[:, :inputs]
        noisy_x[:, inputs:] = SAM_IN().perturb_constants()[:, inputs:]
        noisy_y = np.fromiter(map(func.func, noisy_x[:, :inputs].flatten()), dtype=np.float32)
        x.append(noisy_x)
        y.append(noisy_y)
    
    return np.array(x), np.array(y)

def get_neighbor_map(preds, sharp_out_manager, fitness, train_y):
    """Evaluate fitness of neighbors."""
    neighborhood = sharp_out_manager.perturb(preds)
    return [fitness.fit(neighbor, train_y) for neighbor in neighborhood]

def process_generation_data(pop, fitnesses, max_p):
    """Process generation data for statistics."""
    change_list, full_change_list, ret_list = [], [], []
    retention = []  # Placeholder, implement logic as needed
    
    for p in retention:
        ps = [pop[p], pop[p + 1]]
        p_fits = np.array([fitnesses[p], fitnesses[p + 1]])
        cs = [pop[p + max_p], pop[p + max_p + 1]]
        c_fits = np.array([fitnesses[p + max_p], fitnesses[p + max_p + 1]])
        
        best_p = np.argmin(p_fits)
        best_c = np.argmin(c_fits)
        
        change_list.append(percent_change(c_fits[best_c], p_fits[best_p]))
        ret_list.append(find_similarity(cs[best_c][0], ps[best_p][0], cs[best_c][1], ps[best_p][1], 'cgp'))
        
        full_change_list.append([percent_change(c, best_p) for c in c_fits])
    
    full_change_list = np.array(full_change_list).flatten()
    full_change_list = full_change_list[np.isfinite(full_change_list)]
    
    return change_list, full_change_list, ret_list

def save_results(func_name, run_name, trial, best_pop, preds, best_fit, p_size, fit_track,
                 avg_change_list, ret_avg_list, bin_centers, hist_gens, avg_hist_list,
                 drift_list, drift_cum, sharp_in_list, sharp_out_list, sharp_in_std,
                 sharp_out_std, density_distro):
    """Save results to a pickle file."""
    Path(f"../output/{run_name}/{func_name}/log/").mkdir(parents=True, exist_ok=True)
    with open(f"../output/{run_name}/{func_name}/log/output_{trial}.pkl", "wb") as f:
        pickle.dump(best_pop[0], f)
        pickle.dump(best_pop[1], f)
        pickle.dump(preds, f)
        pickle.dump(best_fit, f)
        pickle.dump(p_size, f)
        pickle.dump(fit_track, f)
        pickle.dump(avg_change_list, f)
        pickle.dump(ret_avg_list, f)
        pickle.dump(bin_centers, f)
        pickle.dump(hist_gens, f)
        pickle.dump(avg_hist_list, f)
        pickle.dump(drift_list, f)
        pickle.dump(drift_cum, f)
        pickle.dump(sharp_in_list, f)
        pickle.dump(sharp_out_list, f)
        pickle.dump(sharp_in_std, f)
        pickle.dump(sharp_out_std, f)
        pickle.dump(density_distro, f)


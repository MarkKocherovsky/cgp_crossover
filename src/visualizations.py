from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cgp_plots

methods = ["lexicase-0.1", "ranked",
           "roulette", "tourn-1", "tourn-4", "tourn-8"]
settings = ["cgp_1x", "cgp_2x"]
test_name = "Koza 3"
setting_name = "cgp_1x"

# # ALL > Get best trial
# method_best = {}
# for m in methods:
#     setting_best = {}
#     for s in settings:
#         trial_best = []
#         for trial in range(1, 51):
#             path = f"selection_output\\{m}\\{s}\\{test_name}\\log\\output_{trial}.pkl"
#             try:
#                 with open(path, "rb") as f:
#                     results = []
#                     for i in range(6):
#                         results.append(pickle.load(f))
#                 trial_best.append(results[4])
#             except:
#                 trial_best.append(99)
#         setting_best[s] = np.argmin(trial_best) + 1
#     method_best[m] = setting_best

# # CGPNOX > Get best trial
# trial_best = []
# for t in range(1, 51):
#     path = f"selection_output\\cgp_nox\\{test_name}\\log\\output_{trial}.pkl"
#     with open(path, "rb") as f:
#         results = []
#         for i in range(6):
#             results.append(pickle.load(f))
#         trial_best.append(results[4])
# method_best["cgp_nox"] = np.argmin(trial_best) + 1

# # ALL > Get best fitness history
# fitness_tracks = {}
# for m in methods:
#     setting_fitness = {}
#     for s in settings:
#         path = f"selection_output\\{m}\\{s}\\{test_name}\\log\\output_{method_best[m][s]}.pkl"
#         try:
#             with open(path, "rb") as f:
#                 results = []
#                 for i in range(7):
#                     results.append(pickle.load(f))
#             setting_fitness[s] = results[6]
#         except:
#             setting_fitness[s] = [1 for x in range(1000)]
#     fitness_tracks[m] = setting_fitness

# # CGPNOX > Get best fitness history
# path = f"selection_output\\cgp_nox\\{test_name}\\log\\output_{method_best['cgp_nox']}.pkl"
# with open(path, "rb") as f:
#     results = []
#     for i in range(7):
#         results.append(pickle.load(f))
#     fitness_tracks["cgp_nox"] = results[6]

# # ALL > Get average fitness history
# fitness_tracks = {}
# for m in methods:
#     setting_fitness = {}
#     for s in settings:
#         setting_fitness[s] = [0 for x in range(10000)]
#         num_trials = 1
#         for trial in range(1, 51):
#             path = f"selection_output\\{m}\\{s}\\{test_name}\\log\\output_{trial}.pkl"
#             try:
#                 with open(path, "rb") as f:
#                     results = []
#                     for i in range(7):
#                         results.append(pickle.load(f))
#                 setting_fitness[s] = np.add(setting_fitness[s], results[6])
#                 num_trials += 1
#             except:
#                 pass
#         setting_fitness[s] = [x / num_trials for x in setting_fitness[s]]
#     fitness_tracks[m] = setting_fitness

# # CGPNOX > Get average fitness history
# fitness_tracks["cgp_nox"] = [0 for x in range(10000)]
# for trial in range(1, 51):
#     path = f"selection_output\\cgp_nox\\{test_name}\\log\\output_{trial}.pkl"
#     with open(path, "rb") as f:
#         results = []
#         for i in range(7):
#             results.append(pickle.load(f))
#     fitness_tracks["cgp_nox"] = np.add(fitness_tracks["cgp_nox"], results[6])
# fitness_tracks["cgp_nox"] = [x / 50 for x in fitness_tracks["cgp_nox"]]

# ALL > Calculate standard deviation (only works for cgp1x and 2x)
method_mean = {}
method_std = {}
for m in methods:
    setting_mean = {}
    setting_std = {}
    for s in settings:
        trial_record = [[1 for x in range(10000)]]
        for trial in range(1, 51):
            path = f"selection_output\\{m}\\{s}\\{test_name}\\log\\output_{trial}.pkl"
            try:
                with open(path, "rb") as f:
                    results = []
                    for i in range(7):
                        results.append(pickle.load(f))
                trial_record.append(results[6])
            except:
                pass
        setting_mean[s] = np.mean(trial_record, axis=0)
        setting_std[s] = np.std(trial_record, axis=0)
    method_mean[m] = setting_mean
    method_std[m] = setting_std

# CGPNOX > Calculate standard deviation (only works for cgp1x and 2x)
trial_record = []
for trial in range(1, 51):
    path = f"selection_output\\cgp_nox\\{test_name}\\log\\output_{trial}.pkl"
    try:
        with open(path, "rb") as f:
            results = []
            for i in range(7):
                results.append(pickle.load(f))
        trial_record.append(results[6])
    except:
        pass
method_mean["cgp_nox"] = np.mean(trial_record, axis=0)
method_std["cgp_nox"] = np.std(trial_record, axis=0)


# Draw graphs
fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# Line graph
y1 = np.subtract(method_mean["cgp_nox"], [x/10 for x in method_std["cgp_nox"]])
y2 = np.add(method_mean["cgp_nox"], [x/10  for x in method_std["cgp_nox"]])
ax.fill_between(list(range(10000)), y1, y2, alpha=0.4)
ax.plot(method_mean["cgp_nox"], label="cgp_nox")

for m in methods:
    y1 = np.subtract(method_mean[m][setting_name], [x/10 for x in method_std[m][setting_name]])
    y2 = np.add(method_mean[m][setting_name], [x/10 for x in method_std[m][setting_name]])
    ax.fill_between(list(range(10000)), y1, y2, alpha=0.4)    
    ax.plot(method_mean[m][setting_name], label=f"{m}{setting_name[-3:]}")
ax.set_yscale('log')
ax.set_title(f'{test_name} Avg 50 Trials')
ax.set_ylabel("1-R^2")
ax.set_xlabel("Generations")
ax.legend()

cgp1x_plots = [method_mean["cgp_nox"]]
for m in methods:
    cgp1x_plots.append(method_mean[m][setting_name])

# Box plot
ax2.boxplot(cgp1x_plots, vert=True, labels=["cgp_nox"] + methods)
ax2.set_yscale('log')
ax.set_ylabel("1-R^2")
ax2.set_title(f'{test_name} Avg 50 Trials')
plt.show()

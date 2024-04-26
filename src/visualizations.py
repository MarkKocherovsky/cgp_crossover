from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
  

methods = ["lexicase-0.1", "roulette",
           "etourn-1", "etourn-4", "etourn-8"]
settings = ["cgp_1x", "cgp_2x"]
test_name = "Koza 2"
setting_name = "cgp_2x"

# ALL > Calculate standard deviation (only works for cgp1x and 2x)
method_mean = {}
method_std = {}
method_best = {}
for m in methods:
    setting_mean = {}
    setting_std = {}
    setting_best = {}
    for s in settings:
        trial_record = []
        best = []
        for trial in range(1, 51):
            path = f"selection_output\\{m}\\{s}\\{test_name}\\log\\output_{trial}.pkl"
            try:
                with open(path, "rb") as f:
                    results = []
                    for i in range(7):
                        results.append(pickle.load(f))
                trial_record.append(results[6])
                best.append(results[4])
            except:
                pass
        setting_mean[s] = np.mean(trial_record, axis=0)
        setting_std[s] = np.std(trial_record, axis=0)
        setting_best[s] = best
    method_mean[m] = setting_mean
    method_std[m] = setting_std
    method_best[m] = setting_best

# CGPNOX > Calculate standard deviation (only works for cgp1x and 2x)
trial_record = []
best = []
for trial in range(1, 51):
    path = f"selection_output\\cgp_nox\\{test_name}\\log\\output_{trial}.pkl"
    try:
        with open(path, "rb") as f:
            results = []
            for i in range(7):
                results.append(pickle.load(f))
        trial_record.append(results[6])
        best.append(results[4])
    except:
        pass
method_mean["cgp_nox"] = np.mean(trial_record, axis=0)
method_std["cgp_nox"] = np.std(trial_record, axis=0)
method_best["cgp_nox"] = best

# Draw graphs
fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 7))

# Line graph
y1 = np.subtract(method_mean["cgp_nox"], [x/5 for x in method_std["cgp_nox"]])
y2 = np.add(method_mean["cgp_nox"], [x/5 for x in method_std["cgp_nox"]])
ax.fill_between(list(range(10000)), y1, y2, alpha=0.25)
ax.plot(method_mean["cgp_nox"], label="cgp_nox")

for m in methods:
    y1 = np.subtract(method_mean[m][setting_name], [
                     x/5 for x in method_std[m][setting_name]])
    y2 = np.add(method_mean[m][setting_name], [
                x/5 for x in method_std[m][setting_name]])
    ax.fill_between(list(range(10000)), y1, y2, alpha=0.25)
    ax.plot(method_mean[m][setting_name], label=f"{m}{setting_name[-3:]}")
ax.set_yscale('log')
ax.set_title(f'NoX vs CGP{setting_name[-3:]}, {test_name} Avg 50 Trials')
ax.set_ylabel("1-R^2")
ax.set_xlabel("Generations")
ax.legend()

cgp1x_plots = [method_best["cgp_nox"]]
for m in methods:
    # print(len(method_best[m][setting_name]))
    cgp1x_plots.append(method_best[m][setting_name])

# Box plot
ax2.boxplot(cgp1x_plots, vert=True, labels=["cgp_nox"] + methods)
ax2.set_yscale('log')
ax.set_ylabel("1-R^2")
ax2.set_title(f'NoX vs CGP{setting_name[-3:]}, {test_name} Best 50 Trials')
plt.show()

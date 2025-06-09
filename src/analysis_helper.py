import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

from dataclasses import dataclass

from matplotlib import pyplot as plt


@dataclass(frozen=True)
class Method:
    """Represents a crossover method with metadata for visualization."""
    code_name: str
    short_name: str
    long_name: str
    color: str
    linestyle: str

@dataclass(frozen=True)
class Metric:
    code_name: str
    full_name: str
    short_name: str
    log: bool

# Path structure:
# Output
#   Problem
#       Crossover
#           Selection
#               Trial 0...n
#                   best_model.csv
#                   statistics.csv
#                   xover_density_beneficial.csv
#                   xover_density_deleterious.csv
#                   xover_density_neutral.csv

class AnalysisToolkit:
    def __init__(self, crossover_methods: dict, selection_methods: dict, problem_list: dict, metrics: list, trials: int,
                 max_generations: int):
        self.crossover_methods = crossover_methods
        self.selection_methods = selection_methods
        self.problems = problem_list
        self.base_path = Path('../output')
        self.trials = trials
        self.max_generations = max_generations
        self.metrics = metrics

    def _load_trial_data(self, problem: str, xover: str, selection_key: str) -> list:
        """Load statistics data for all trials."""
        trial_data = []
        for trial in range(self.trials):
            path = self.base_path / problem / xover / selection_key / f'trial_{trial}/statistics.csv'
            #print(path)
            trial_data.append(np.loadtxt(path, delimiter=','))
            if len(trial_data[trial][-1]) < self.max_generations:
                print(path, len(trial_data[-1]))
            if np.array(trial_data)[trial, -1, 0] > 0.99:
                print(f'Recommend re-running {path}, best fitness = {np.array(trial_data)[trial, -1, 0]}')
        return trial_data

    def compile_averages(self, metric_list):
        """
        metric_list: positions of metrics we want
        """
        output_dir = Path('../output/intermediate_results')
        output_dir.mkdir(parents=True, exist_ok=True)

        for problem in self.problems:
            for xover_method in self.crossover_methods:
                xover = self.crossover_methods[xover_method].code_name
                if xover == "None":
                    selection_list = ["elite"]  # Ensure "none" crossover only uses "elite"
                else:
                    selection_list = list(self.selection_methods.keys())
                for selection in selection_list:
                    table_list = self._load_trial_data(problem, xover, selection)

                    for m, metric in enumerate(self.metrics):
                        try:
                            metric_idx = metric_list[m]
                            data = np.array([table[0:self.max_generations, metric_idx] for table in table_list])
                            quartiles = np.quantile(data, [0, 0.25, 0.5, 0.75, 1], axis=0)
                            q_table = pd.DataFrame(quartiles.T,
                                columns=['Minimum', 'First Quartile', 'Median', 'Third Quartile',
                                         'Maximum'])
                            q_table.to_csv(output_dir / f'{problem}_{xover}_{selection}_{self.metrics[metric].code_name}.csv', index=False)
                        except ValueError as e:
                            print(f'analysis_helper.py::AnalysisToolkit::compile_averages ValueError {e}')
                            print(f'{problem} {xover_method} {selection} {metric}\n#########')
                            print(f'{problem} {xover_method} {selection} {metric.full_name}\n#########')
    def make_path(self, problem: str, xover: str, selection: str, metric: str):
        return Path(f'../output/intermediate_results/{problem}_{xover}_{selection}_{metric}.csv')

    def plot_line_graph(self, selection_method: str, metric: str, graph_filename:str, title: str, x_label: str, y_label: str, log:bool = False):
        if isinstance(metric, str):
            metric = next((m for m in self.metrics if m.code_name == metric), None)
            if metric is None:
                raise ValueError(f"Metric '{metric}' not found in self.metrics")

        n_problems = len(self.problems)
        fig, axs = plt.subplots(int(np.round(n_problems/2)), 2, figsize=(15, 20))
        for i, problem in enumerate(self.problems):
            for xover_method in self.crossover_methods:
                xover = self.crossover_methods[xover_method].code_name
                if xover == 'None':
                    sel_key = 'elite'
                else:
                    sel_key = selection_method
                file_name = self.make_path(problem, xover, sel_key, metric.code_name)
                if file_name.exists():
                    data = pd.read_csv(file_name)
                    median = data['Median']
                    quartile_1 = data['First Quartile']
                    quartile_3 = data['Third Quartile']
                else:
                    raise FileNotFoundError(f"File {file_name} not found.")
                if log:
                    axs[i//2, i%2].set_yscale('log')
                axs[i//2, i%2].plot(range(self.max_generations), median, label=self.crossover_methods[xover_method].short_name, c=self.crossover_methods[xover_method].color, linestyle=self.crossover_methods[xover_method].linestyle)
                axs[i//2, i%2].fill_between(range(self.max_generations), quartile_1, quartile_3, alpha=0.2)
                axs[i//2, i%2].set_title(self.problems[problem])
                axs[i//2, i%2].set_xlabel(x_label)
                axs[i//2, i%2].set_ylabel(y_label)
                axs[i//2, i%2].legend()
        fig.suptitle(f'{title}\n{self.selection_methods[selection_method]}\n{metric.full_name}')
        fig.tight_layout()
        file_path = f"../output/graphs_raw/{graph_filename}.pkl"  # Path to save the binary file
        with open(file_path, "wb") as file:
            pickle.dump(plt.gcf(), file)
        output_dir = "../output/graphs/"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"../output/graphs/{graph_filename}.png")
        print(f'{graph_filename} saved')

    def plot_box_plots(self, selection_method: str, metric: Metric, graph_filename: str, title: str,
                       x_label: str, y_label: str, log:bool = False, violin:bool = False, jitter:bool = False):
        if isinstance(metric, str):
            metric = next((m for m in self.metrics if m.code_name == metric), None)
            if metric is None:
                raise ValueError(f"Metric '{metric}' not found in self.metrics")


        n_problems = len(self.problems)
        fig, axs = plt.subplots(int(np.round(n_problems / 2)), 2, figsize=(15, 20))
        axs = axs.flatten() if n_problems > 1 else [axs]

        for i, (problem_key, problem_name) in enumerate(self.problems.items()):
            ax = axs[i]
            box_data = []
            labels = []
            colors = []

            for xover_key, xover_obj in self.crossover_methods.items():
                xover = xover_obj.code_name
                sel_key = 'elite' if xover == 'None' else selection_method

                try:
                    trials = self._load_trial_data(problem_key, xover, sel_key)
                except Exception as e:
                    print(f"Error loading data for {problem_key} {xover}: {e}")
                    continue

                # Extract metric at final generation for each trial
                try:
                    metric_idx = next(
                        i for i, m in enumerate(self.metrics.values()) if m.code_name == metric.code_name
                    )

                    final_values = [trial[self.max_generations - 1, metric_idx] for trial in trials]
                except Exception as e:
                    print(f"Error extracting final values for {problem_key} {xover}: {e}")
                    continue

                box_data.append(final_values)
                labels.append(xover_obj.short_name)
                colors.append(xover_obj.color)

            # Draw box plot
            if violin:
                violins = ax.violinplot(box_data, widths=0.9, showmeans=False, showmedians=False)
                for pc,color in zip(violins['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_edgecolor('none')
                    pc.set_alpha(0.5)  # ✅ semi-transparent so boxplot is visible
            bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, notch=False, widths = 0.75)
            positions = np.arange(1, len(box_data) + 1)  # x-axis positions of boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            if jitter:
                for i, data in enumerate(box_data):
                    x = np.random.normal(loc=positions[i], scale=0.06, size=len(data))
                    ax.scatter(x, data, alpha=0.5, s=8, color='black', zorder=3)  # s=10 for point size

            ax.set_title(problem_name)
            if log:
                ax.set_yscale('log')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45)

        fig.suptitle(f'{title}\nSelection: {self.selection_methods[selection_method]}\nMetric: {metric.full_name}')
        fig.tight_layout()

        # Save to file
        Path("../output/graphs_raw").mkdir(exist_ok=True)
        Path("../output/graphs").mkdir(exist_ok=True)
        with open(f"../output/graphs_raw/{graph_filename}.pkl", "wb") as f:
            pickle.dump(fig, f)
        plt.savefig(f"../output/graphs/{graph_filename}.png")
        print(f'{graph_filename} saved')
        plt.close(fig)



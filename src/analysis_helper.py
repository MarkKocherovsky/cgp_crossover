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
            print(path)
            trial_data.append(np.loadtxt(path, delimiter=','))
            if len(trial_data[-1]) < self.max_generations:
                print(path, len(trial_data[-1]))
        return trial_data

    def compile_averages(self):
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

                    for metric in self.metrics:
                        try:
                            metric_idx = self.metrics.index(metric)  # assumes self.metrics is a list of metric names in fixed column order
                            data = np.array([table[0:self.max_generations, metric_idx] for table in table_list])
                            quartiles = np.quantile(data, [0, 0.25, 0.5, 0.75, 1], axis=0)
                            q_table = pd.DataFrame(quartiles.T,
                                columns=['Minimum', 'First Quartile', 'Median', 'Third Quartile',
                                         'Maximum'])
                            q_table.to_csv(output_dir / f'{problem}_{xover}_{selection}_{metric}.csv', index=False)
                        except ValueError as e:
                            print(f'analysis_helper.py::AnalysisToolkit::compile_averages ValueError {e}')
                            print(f'{problem} {xover_method} {selection} {metric}\n#########')
    def make_path(self, problem: str, xover: str, selection: str, metric: str):
        return Path(f'../output/intermediate_results/{problem}_{xover}_{selection}_{metric}.csv')

    def plot_line_graph(self, selection_method: str, metric: str, graph_filename:str, title: str, x_label: str, y_label: str, log:bool = True):
        n_problems = len(self.problems)
        fig, axs = plt.subplots(int(np.round(n_problems/2)), 2, figsize=(15, 15))
        for i, problem in enumerate(self.problems):
            for xover_method in self.crossover_methods:
                xover = self.crossover_methods[xover_method].code_name
                if xover == 'None':
                    sel_key = 'elite'
                else:
                    sel_key = selection_method
                file_name = self.make_path(problem, xover, sel_key, metric)
                if file_name.exists():
                    data = pd.read_csv(file_name)
                    median = data['Median']
                    quartile_1 = data['First Quartile']
                    quartile_3 = data['Third Quartile']
                else:
                    raise FileNotFoundError(f"File {file_name} not found.")
                if log:
                    axs[i//2, i%2].set_yscale('log')
                axs[i//2, i%2].plot(range(self.max_generations), median, label=self.crossover_methods[xover_method].short_name, c=self.crossover_methods[xover_method].color)
                axs[i//2, i%2].fill_between(range(self.max_generations), quartile_1, quartile_3, alpha=0.2)
                axs[i//2, i%2].set_title(self.problems[problem])
                axs[i//2, i%2].set_xlabel(x_label)
                axs[i//2, i%2].set_ylabel(y_label)
                axs[i//2, i%2].legend()
        fig.suptitle(f'{title}\n{self.selection_methods[selection_method]}\n{metric}')
        fig.tight_layout()
        file_path = f"../output/graphs_raw/{graph_filename}.pkl"  # Path to save the binary file
        with open(file_path, "wb") as file:
            pickle.dump(plt.gcf(), file)
        output_dir = "../output/graphs/"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"../output/graphs/{graph_filename}.png")
        print(f'{graph_filename} saved')


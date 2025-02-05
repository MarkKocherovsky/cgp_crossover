import pandas as pd
import numpy as np
from pathlib import Path

from dataclasses import dataclass


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
#               Trial 0..n
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
            trial_data.append(pd.read_csv(path))
        return trial_data

    def compile_averages(self):
        canonical_flag = set()
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
                        data = np.array([table[metric][0:self.max_generations].values for table in table_list])
                        quartiles = np.quantile(data, [0, 0.25, 0.5, 0.75, 1], axis=0)
                        q_table = pd.DataFrame(quartiles.T,
                                               columns=['Minimum', 'First Quartile', 'Median', 'Third Quartile',
                                                        'Maximum'])
                        q_table.to_csv(output_dir / f'{problem}_{xover}_{selection}_{metric}.csv', index=False)

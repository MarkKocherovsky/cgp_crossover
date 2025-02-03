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

    @staticmethod
    def _get_selection_key(xover: str, selection: str, canonical_flag: set) -> str | None:
        """Determine the correct selection key based on crossover type."""
        if xover == 'none':
            if 'elite' not in canonical_flag:
                canonical_flag.add('elite')
                return 'elite'
            return None  # Skip duplicate processing
        return selection

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
            for xover in self.crossover_methods:
                for selection in self.selection_methods:
                    selection_key = self._get_selection_key(xover, selection, canonical_flag)
                    if selection_key is None:
                        continue  # Skip redundant processing

                    table_list = self._load_trial_data(problem, xover, selection_key)

                    for metric in self.metrics:
                        data = np.array([table[metric].values for table in table_list])
                        quartiles = np.quantile(data, [0, 0.25, 0.5, 0.75, 1], axis=0)
                        q_table = pd.DataFrame(quartiles.T,
                                               columns=['Minimum', 'First Quartile', 'Median', 'Third Quartile',
                                                        'Maximum'])
                        q_table.to_csv(output_dir / f'{problem}_{xover}_{selection_key}_{metric}.csv', index=False)

import pandas as pd
import numpy as np
import pickle
import datetime
import os
import subprocess
from pathlib import Path
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import mannwhitneyu
from dataclasses import dataclass

from matplotlib.colors import ListedColormap, BoundaryNorm
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
    def __init__(self, crossover_methods: dict, selection_methods: dict, path: str, problem_list: dict, metrics: list,
                 trials: int,
                 max_generations: int, output_format: str = '.png'):
        self.crossover_methods = crossover_methods
        self.selection_methods = selection_methods
        self.problems = problem_list
        self.base_path = Path(path)
        self.trials = trials
        self.max_generations = max_generations
        self.metrics = metrics
        self.output_format = output_format

    def _load_trial_data(self, problem: str, xover: str, selection_key: str, restart: bool = False,
                         mutation: str = None) -> list:
        """Load statistics data for all trials with proper validation."""
        trial_data = []
        mutation = '' if mutation is None else mutation
        for trial in range(self.trials):
            path = self.base_path / problem / xover / mutation / selection_key / f'trial_{trial}/statistics.csv'
            try:
                mod_time = datetime.datetime.fromtimestamp(path.stat().st_mtime)
                if mod_time.month < 9:
                    print(f"{path} — Last modified before September: {mod_time}")
                    slurm_problem = problem.replace("_1d", "")
                    slurm_script = f"../output/slurm_files/kocherov_{slurm_problem}_{xover}_{trial}.slurm"
                    if restart:
                        try:
                            subprocess.run(['sbatch', str(slurm_script)], check=True)
                            print(f"[Resubmitted] {slurm_script}")
                        except subprocess.CalledProcessError as e:
                            print(f"[SLURM Error] Failed to resubmit {slurm_script}: {e}")

                # print(f"{path} — Last modified: {mod_time}")
                data = np.loadtxt(path, delimiter=',')
            except FileNotFoundError:
                print(f"[Missing] {path} not found. Skipping.")
                continue
            except ValueError as e:
                print(f"[ValueError] {path}: {e}. Skipping.")
                continue

            # Check for expected number of generations
            if data.shape[0] != self.max_generations and restart is True:
                print(f"[Mismatch] {path}: Expected {self.max_generations}, got {data.shape[0]}. Resubmitting job...")
                # Auto-submit SLURM job
                slurm_problem = problem.replace("_1d", "")
                slurm_script = f"../output/slurm_files/kocherov_{slurm_problem}_{xover}_{trial}.slurm"
                try:
                    # subprocess.run(['sbatch', str(slurm_script)], check=True)
                    print(f"[Resubmitted] {slurm_script}")
                except subprocess.CalledProcessError as e:
                    print(f"[SLURM Error] Failed to resubmit {slurm_script}: {e}")
            # Warn if final fitness is suspiciously high
            elif data.shape[0] != self.max_generations and restart is not True:
                print(f"[Mismatch] {path}: Expected {self.max_generations}, got {data.shape[0]}.")
                continue
            if data[-1, 0] > 0.99:
                print(f"[Warning] High final fitness in {path}: {data[-1, 0]}. Consider re-running.")
                continue

            trial_data.append(data[:self.max_generations])

        return trial_data

    def compile_averages(self, metric_list, restart=False, mutation=None):
        """
        metric_list: positions of metrics we want
        """
        mutation = '' if mutation is None else mutation
        output_dir = Path('../output/intermediate_results') / mutation
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f'max generations: {self.max_generations}')
        for problem in self.problems:
            print(problem)
            for xover_method in self.crossover_methods:
                xover = self.crossover_methods[xover_method].code_name
                if "None" in xover:
                    selection_list = ["elite"]  # Ensure "none" crossover only uses "elite"
                else:
                    selection_list = list(self.selection_methods.keys())
                for selection in selection_list:
                    table_list = self._load_trial_data(problem, xover, selection, restart=False, mutation=mutation)
                    best_fitness = np.array([table[-1, 0] for table in table_list])
                    best_test_fitness = np.array([table[-1, 5] for table in table_list])
                    best_sizes = np.array([table[-1, 10] for table in table_list])
                    for m, metric in enumerate(self.metrics):
                        try:
                            metric_idx = metric_list[m]
                            valid_tables = [table for table in table_list if table.shape[0] >= self.max_generations]
                            data = np.array([table[0:self.max_generations, metric_idx] for table in valid_tables])

                            quartiles = np.quantile(data, [0, 0.25, 0.5, 0.75, 1], axis=0)
                            q_table = pd.DataFrame(quartiles.T,
                                                   columns=['Minimum', 'First Quartile', 'Median', 'Third Quartile',
                                                            'Maximum'])
                            if '/full' in xover:
                                xover = xover.replace('/full', '_full')
                            print(f'{problem}_{xover}_{selection}_{self.metrics[metric].code_name}.csv')
                            q_table.to_csv(
                                output_dir / f'{problem}_{xover}_{selection}_{self.metrics[metric].code_name}.csv',
                                index=False)
                        except ValueError as e:
                            print(f'analysis_helper.py::AnalysisToolkit::compile_averages ValueError {e}')
                            print(f'{problem} {xover_method} {selection} {metric}\n#########')
                            print(f'{problem} {xover_method} {selection} {metric.full_name}\n#########')
                        except IndexError as e:
                            continue
                    np.savetxt(output_dir / f'{problem}_{xover}_{selection}_min_fitnesses.csv', best_fitness)
                    np.savetxt(output_dir / f'{problem}_{xover}_{selection}_min_test_fitnesses.csv', best_test_fitness)
                    np.savetxt(output_dir / f'{problem}_{xover}_{selection}_best_sizes.csv', best_sizes)

    def make_path(self, problem: str, xover: str, selection: str, metric: str):
        return Path(f'../output/intermediate_results/{problem}_{xover}_{selection}_{metric}.csv')

    def plot_line_graph(self, selection_method: str, metric: str, graph_filename: str, title: str, x_label: str,
                        y_label: str, log: bool = False):
        if isinstance(metric, str):
            metric = next((m for m in self.metrics if m.code_name == metric), None)
            if metric is None:
                raise ValueError(f"Metric '{metric}' not found in self.metrics")

        n_problems = len(self.problems)
        fig, axs = plt.subplots(int(np.round(n_problems / 2)), 2, figsize=(8, 10))
        axs = axs.flatten()
        legend_dict = {}
        for i, problem in enumerate(self.problems):
            for xover_method in self.crossover_methods:
                xover = self.crossover_methods[xover_method].code_name
                sel_key = 'elite' if 'None' in xover else selection_method
                if xover == 'None/full':
                    xover = 'None_full'
                file_name = self.make_path(problem, xover, sel_key, metric.code_name)
                if file_name.exists():
                    data = pd.read_csv(file_name)
                    median = data['Median']
                    quartile_1 = data['First Quartile']
                    quartile_3 = data['Third Quartile']
                else:
                    raise FileNotFoundError(f"File {file_name} not found.")
                if log:
                    axs[i].set_yscale('log')
                try:
                    line = axs[i].plot(
                        range(self.max_generations),
                        median,
                        label=self.crossover_methods[xover_method].short_name,
                        c=self.crossover_methods[xover_method].color,
                        linestyle=self.crossover_methods[xover_method].linestyle
                    )[0]
                except ValueError as e:
                    print(e)
                    print(f'{sel_key} {xover} {problem}')
                    exit()
                # axs[i//2, i%2].plot(range(self.max_generations), median, label=self.crossover_methods[xover_method].short_name, c=self.crossover_methods[xover_method].color, linestyle=self.crossover_methods[xover_method].linestyle)
                axs[i].fill_between(range(self.max_generations), quartile_1, quartile_3, alpha=0.1)
                axs[i].set_title(self.problems[problem], fontsize=11)
                axs[i].set_xlabel('', fontsize=8)
                if i % 2 == 0:
                    axs[i].set_ylabel(y_label, fontsize=8)
                else:
                    axs[i].set_ylabel('')
                # axs[i//2, i%2].legend()
                label = self.crossover_methods[xover_method].short_name
                legend_dict[label] = line  # overwrites duplicates automatically
        for ax in axs[-2:]:
            ax.set_xlabel(x_label, fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.88])  # Before legend + title
        fig.legend(list(legend_dict.values()), list(legend_dict.keys()), loc='upper center', ncol=3,
                   bbox_to_anchor=(0.5, 0.96), fontsize=10)
        # fig.suptitle(f'{title}\n{self.selection_methods[selection_method]}\n{metric.full_name}', fontsize=14)
        fig.suptitle(f'{title}', fontsize=14)
        file_path = f"../output/graphs_raw/{graph_filename}.pkl"  # Path to save the binary file
        with open(file_path, "wb") as file:
            pickle.dump(plt.gcf(), file)
        output_dir = "../output/graphs/"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"../output/graphs/{graph_filename}{self.output_format}")
        print(f'{graph_filename} saved')

    def plot_box_plots(self, selection_method: str, metric: Metric, graph_filename: str, title: str,
                       x_label: str, y_label: str, log: bool = False, violin: bool = False, jitter: bool = False):
        if isinstance(metric, str):
            metric = next((m for m in self.metrics if m.code_name == metric), None)
            if metric is None:
                raise ValueError(f"Metric '{metric}' not found in self.metrics")

        print(metric)
        n_problems = len(self.problems)
        fig, axs = plt.subplots(int(np.round(n_problems / 2)), 2, figsize=(8, 10))
        axs = axs.flatten() if n_problems > 1 else [axs]

        for i, (problem_key, problem_name) in enumerate(self.problems.items()):
            ax = axs[i]
            box_data = []
            labels = []
            colors = []

            for xover_key, xover_obj in self.crossover_methods.items():
                xover = xover_obj.code_name
                sel_key = 'elite' if 'None' in xover else selection_method
                if xover == 'None/full':
                    xover = 'None_full'
                try:
                    file_name = self.make_path(problem_key, xover, sel_key, metric.code_name)
                    final_values = np.loadtxt(file_name)
                except Exception as e:
                    print(f"Error loading data for {problem_key} {xover}: {e}")
                    continue
                box_data.append(final_values)
                labels.append(xover_obj.short_name)
                colors.append(xover_obj.color)

            # Draw box plot
            if violin:
                violins = ax.violinplot(box_data, widths=0.9, showmeans=False, showmedians=False)
                for pc, color in zip(violins['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_edgecolor('none')
                    pc.set_alpha(0.5)  # ✅ semi-transparent so boxplot is visible
            bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, notch=False, widths=0.75)
            positions = np.arange(1, len(box_data) + 1)  # x-axis positions of boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            if jitter:
                for i, data in enumerate(box_data):
                    x = np.random.normal(loc=positions[i], scale=0.06, size=len(data))
                    ax.scatter(x, data, alpha=0.5, s=8, color='black', zorder=3)  # s=10 for point size

            ax.set_title(problem_name, fontsize=10)
            if log:
                ax.set_yscale('log')
                # Extract box y-data
                box_bottoms = [box.get_path().vertices[:, 1].min() for box in bp['boxes']]  # Q1
                box_tops = [box.get_path().vertices[:, 1].max() for box in bp['boxes']]  # Q3

                ymin = min(box_bottoms)
                ymax = max(box_tops)
                # Calculate orders of magnitude
                lower_exp = np.floor(np.log10(ymin)) if ymin > 0 else -1
                upper_exp = np.ceil(np.log10(ymax)) if ymax > 0 else 1
                upper_exp = np.ceil(np.log10(ymax)) if ymax > 0 else 1

                # Set y-axis limits
                ax.set_ylim(10 ** lower_exp, 10 ** upper_exp)
            ax.set_xlabel('')
            if i % 2 == 0:
                ax.set_ylabel(y_label, fontsize=8)
            else:
                ax.set_ylabel('')
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, fontsize=7)
            # Then adjust alignment of the labels
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        for ax in axs.flatten()[-2:]:
            ax.set_xlabel(x_label, fontsize=8)
        fig.suptitle(f'{title}', fontsize='14')
        # fig.suptitle(f'{title}\nSelection: {self.selection_methods[selection_method]}\nMetric: {metric.full_name}', fontsize='14')
        fig.tight_layout()

        # Save to file
        Path("../output/graphs_raw").mkdir(exist_ok=True)
        Path("../output/graphs").mkdir(exist_ok=True)
        with open(f"../output/graphs_raw/{graph_filename}.pkl", "wb") as f:
            pickle.dump(fig, f)
        plt.savefig(f"../output/graphs/{graph_filename}{self.output_format}")
        print(f'{graph_filename} saved')
        plt.close(fig)

    def get_median_end_values(self, selection_method: str = 'elite_tournament',
                              save_path: str = 'median_end_fitnesses.csv',
                              metric: str = 'min_fitnesses',
                              plackett_dir: str = '../output/plackett_input'):
        results = {}
        full_results = {}

        os.makedirs(plackett_dir, exist_ok=True)

        for problem_key, problem_name in self.problems.items():
            row = {}
            full_row = {}
            for xover_key, xover_obj in self.crossover_methods.items():
                xover = xover_obj.code_name
                sel_key = 'elite' if 'None' in xover else selection_method
                if xover == 'None/full':
                    xover = 'None_full'
                try:
                    file_name = self.make_path(problem_key, xover, sel_key, metric)
                    final_values = np.loadtxt(file_name)
                    final_values = final_values.flatten()  # ensure 1D
                    median_value = np.median(final_values)
                    row[xover] = median_value
                    full_row[xover] = final_values
                except Exception as e:
                    print(f"Error loading data for {problem_key} {xover}: {e}")
                    row[xover] = np.nan
                    full_row[xover] = np.full(50, np.nan)  # pad to allow ranking

            full_results[problem_name] = full_row
            results[problem_name] = row

            # ---- Plackett-Luce ranking prep ----
            methods = list(full_row.keys())
            # transpose: rows = replicates, columns = methods
            try:
                lengths = {m: len(full_row[m]) for m in methods}
                print(f"Replicate lengths for {problem_name}: {lengths}")
                matrix = np.array([full_row[m] for m in methods])
                if matrix.ndim != 2:
                    raise ValueError(f"Matrix not 2D for {problem_name}: shape {matrix.shape}")
                matrix = matrix.T  # shape (n_replicates, n_methods)
                rankings = np.argsort(np.argsort(matrix, axis=1), axis=1) + 1
                df_out = pd.DataFrame(rankings, columns=methods)
                df_out.to_csv(os.path.join(plackett_dir, f'plackett_input_{problem_name}_{metric}.csv'), index=False)
            except Exception as e:
                print(f"Could not generate PL input for {problem_name}: {e}")

        # Save median summary if needed
        df_median = pd.DataFrame.from_dict(results, orient='index')
        df_median.to_csv(save_path)

        # Also return full data and medians if you want to reuse them
        return df_median, pd.DataFrame.from_dict(full_results, orient='index')

    def rankings_to_pairwise(self, rankings):
        pairwise = []
        for ranking in rankings:
            for i in range(len(ranking)):
                for j in range(i + 1, len(ranking)):
                    pairwise.append((ranking[i], ranking[j]))
        return pairwise

    def plackett_luce(self, rankings_path, save_path):
        """
        assumes that rankings are a df with problem names as row indices and xover names as column names
        """
        rankings = pd.read_csv(rankings_path)
        problem_list = rankings.index.tolist()
        xover_list = rankings.columns.tolist()
        alg_index = {name: i for i, name in enumerate(xover_list)}
        pl_probs = {}
        for problem_name, row in rankings.iterrows():
            # Sort algorithms from best (lowest rank) to worst
            row = pd.to_numeric(row, errors='coerce')
            sorted_algs = row.sort_values().index.tolist()
            ranking = [alg_index[alg] for alg in sorted_algs]

            # choix requires multiple rankings, so we can duplicate this one if needed
            # or you can collect many replicates — but for now, treat each row as a single ranking
            pairwise = self.rankings_to_pairwise([ranking])
            print(pairwise)
            theta = choix.ilsr_pairwise(n_items=len(xover_list), data=pairwise)
            probs = theta / theta.sum()

            pl_probs[problem_name] = dict(zip(xover_list, probs))

        # Create a DataFrame for output
        pl_df = pd.DataFrame(pl_probs).T  # transpose: problems as rows
        print(pl_df.round(3))
        pl_df.to_csv(save_path)

    def get_significance_tables(self, selection_method='elite_tournament', output_dir='../output/graphs/significance/'):
        os.makedirs(output_dir, exist_ok=True)

        for problem_key, problem_name in self.problems.items():
            data = {}
            method_names = []

            # Collect data for each crossover method
            for xover_key, xover_obj in self.crossover_methods.items():
                xover = xover_obj.code_name
                sel_key = 'elite' if 'None' in xover else selection_method
                if xover == 'None/full':
                    xover = 'None_full'

                try:
                    file_name = self.make_path(problem_key, xover, sel_key, 'min_fitnesses')
                    final_values = np.loadtxt(file_name)
                    data[xover] = final_values
                    method_names.append(xover)
                except Exception as e:
                    print(f"Error loading data for {problem_key} {xover}: {e}")
                    continue

            # Initialize p-value matrix
            p_matrix = pd.DataFrame(index=method_names, columns=method_names, dtype=float)

            for i, method_i in enumerate(method_names):
                for j, method_j in enumerate(method_names):
                    if i == j:
                        p_matrix.loc[method_i, method_j] = 1.0
                    else:
                        try:
                            stat, p = mannwhitneyu(data[method_i], data[method_j], alternative='two-sided')
                            p_matrix.loc[method_i, method_j] = p
                        except Exception as e:
                            print(f"Error comparing {method_i} vs {method_j} on {problem_key}: {e}")
                            p_matrix.loc[method_i, method_j] = np.nan

            # Save CSV
            csv_path = os.path.join(output_dir, f"{problem_name}_significance.csv")
            p_matrix.to_csv(csv_path)
            print(f"✅ Saved: {csv_path}")
            # Skip heatmap generation if table is empty or all NaN
            if p_matrix.dropna(how='all').empty or p_matrix.dropna(axis=1, how='all').empty:
                print(f"⚠️ Skipping heatmap for {problem_name}: no valid data.")
                continue

            # Plot heatmap

            # 1. Create custom colormap: green for p < 0.05, then maroons
            colors = ['green', '#f08080', '#cd5c5c', '#b22222', '#a52a2a', '#800000',
                      '#000000']  # green + maroon shades
            bounds = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.99, 1.0]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(bounds, ncolors=len(colors))
            scale = 1.2
            # 2. Plot heatmap with custom coloring
            plt.figure(figsize=(11 * scale, 9 * scale))
            ax = sns.heatmap(
                p_matrix.astype(float),
                annot=True,
                fmt=".3f",
                cmap=cmap,
                norm=norm,
                cbar_kws={'label': 'p-value'}
            )
            cbar = ax.collections[0].colorbar
            cbar.ax.set_ylabel('p-value', fontsize=14)  # change label and font size
            cbar.ax.tick_params(labelsize=12)
            plt.title(f"Mann–Whitney U Test p-values\n{problem_name}", fontsize=20)
            labels = [xover.short_name for xover in self.crossover_methods.values()]
            ax.set_xticklabels(labels, rotation=45, ha='left', fontsize=12)
            ax.xaxis.set_ticks_position('top')  # Tick marks on top
            ax.xaxis.set_label_position('top')  # Axis label (if any) on top
            ax.set_yticklabels(labels, rotation=45, va='top', fontsize=12)
            plt.tight_layout()

            img_path = os.path.join(output_dir, f"{problem_name}_heatmap{self.output_format}")
            plt.savefig(img_path)
            plt.close()
            print(f"📊 Saved heatmap: {img_path}")

    def plot_box_plots_compare_old_new(self, old_methods, old_problems, selection_method: str, metric: Metric,
                                       graph_filename: str, title: str,
                                       x_label: str, y_label: str, log: bool = False, jitter: bool = False):
        if isinstance(metric, str):
            metric = next((m for m in self.metrics if m.code_name == metric), None)
            if metric is None:
                raise ValueError(f"Metric '{metric}' not found in self.metrics")

        # External mappings — provide these in your main script or in class if preferred
        n_problems = len(self.problems)
        fig, axs = plt.subplots(int(np.round(n_problems / 2)), 2, figsize=(8, 8))
        axs = axs.flatten() if n_problems > 1 else [axs]

        for i, (problem_key, problem_name) in enumerate(self.problems.items()):
            ax = axs[i]
            box_data = []
            labels = []
            colors = []

            for xover_key, xover_obj in self.crossover_methods.items():
                xover = xover_obj.code_name
                sel_key = 'elite' if 'None' in xover else selection_method
                if xover == 'None/full':
                    xover = 'None_full'

                # Load OLD data using external mappings
                old_problem = old_problems.get(problem_key)
                old_method = old_methods.get(xover)
                if not old_problem or not old_method:
                    print(f"⚠️ Old mapping not found for: {problem_key}, {xover}")
                    continue
                try:
                    if metric.code_name == 'best_sizes':
                        old_path = Path(
                            f"../output/intermediate_results_old/intermediate_results_{old_problem}_{old_method}_size_old.csv")
                    else:
                        old_path = Path(
                            f"../output/intermediate_results_old/intermediate_results_{old_problem}_{old_method}_old.csv")
                    old_vals = np.loadtxt(old_path, delimiter=',')
                    box_data.append(old_vals)
                    labels.append(f'{xover_obj.short_name} (old)')
                    colors.append(xover_obj.color)
                except Exception as e:
                    print(f"⚠️ Old data missing: {old_path}: {e}")
                    continue
                # Load NEW data
                try:
                    temp_key = old_problems.get(problem_key, None)
                    if temp_key is None:
                        continue
                    new_file = self.make_path(problem_key, xover, sel_key, metric.code_name)
                    new_vals = np.loadtxt(new_file)
                    box_data.append(new_vals)
                    labels.append(f'{xover_obj.short_name} (new)')
                    colors.append(xover_obj.color)
                except Exception as e:
                    print(f"⚠️ New data missing: {problem_key} {xover}: {e}")
                    continue

            # Plotting
            bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, notch=False, widths=0.75)
            positions = np.arange(1, len(box_data) + 1)
            for patch, color in zip(bp['boxes'], colors):  # colors already interleaved
                patch.set_facecolor(color)

            if jitter:
                for j, data in enumerate(box_data):
                    x = np.random.normal(loc=positions[j], scale=0.06, size=len(data))
                    ax.scatter(x, data, alpha=0.5, s=8, color='black', zorder=3)

            ax.set_title(problem_name, fontsize=12)
            if log:
                ax.set_yscale('log')
                box_bottoms = [box.get_path().vertices[:, 1].min() for box in bp['boxes']]
                box_tops = [box.get_path().vertices[:, 1].max() for box in bp['boxes']]
                ymin = min(box_bottoms)
                ymax = max(box_tops)
                lower_exp = np.floor(np.log10(ymin)) if ymin > 0 else -1
                upper_exp = np.ceil(np.log10(ymax)) if ymax > 0 else 1
                ax.set_ylim(10 ** lower_exp, 10 ** upper_exp)

            ax.set_xlabel(x_label, fontsize=9)
            ax.set_ylabel(y_label, fontsize=9)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, fontsize=8)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')

        fig.suptitle(f'{title}\nSelection: {self.selection_methods[selection_method]}\nMetric: {metric.full_name}',
                     fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 1])
        Path("../output/graphs_raw").mkdir(exist_ok=True)
        Path("../output/graphs").mkdir(exist_ok=True)
        with open(f"../output/graphs_raw/{graph_filename}.pkl", "wb") as f:
            pickle.dump(fig, f)
        plt.savefig(f"../output/graphs/{graph_filename}{self.output_format}")
        print(f'{graph_filename} saved')
        plt.close(fig)

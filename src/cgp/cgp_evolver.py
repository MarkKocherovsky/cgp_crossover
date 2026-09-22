from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
import pickle
import time
import uuid

import numpy as np
from Bio.Align import PairwiseAligner, Seq

from .cgp_model import CGP
from .fitness_functions import correlation, corr_comp_fitness
from .helper import (
    _get_quartiles,
    pairwise_minkowski_distance,
    get_score,
    get_ssd,
    get_weights,
    clean_values,
    _get_semantic_alignment,
)
from .llm_helper import (
    choose_crossover_point_with_ollama,
    summarize_population_for_llm,
)
from .cgp_generator import node_to_int


# OPTIMIZATION: node types and category names are invariant.  Resolving them
# once avoids repeated list construction/string lookup in hot crossover paths.
INPUT_NODE = node_to_int("Input")
CONSTANT_NODE = node_to_int("Constant")
FUNCTION_NODE = node_to_int("Function")
OUTPUT_NODE = node_to_int("Output")
XOVER_CATEGORIES = ("deleterious", "neutral", "beneficial")


class CartesianGP:

    def _missing_key_error(self, key):
        raise KeyError(f"'{key}' is required but was not provided.")

    def __init__(self, parents=1, children=4, max_generations=100, mutation='Point', selection='Elite',
                 xover=None, fixed_length=True, fitness_function='correlation_complexity', model_parameters=None,
                 function_bank=None, solution_threshold=0.005, checkpoint_filename='checkpoint.pkl', seed=42,
                 **kwargs):
        # hyperparameter tuning
        # Basic attributes
        self.model_keys = None
        np.random.seed(seed)
        self.model_key_map = None
        self.child_id_counter = 0
        self.population = np.empty(parents + children, dtype=object)
        self.fitnesses = np.full(parents + children, np.inf, dtype=np.float64)
        self.corrs = np.full(parents + children, np.inf, dtype=np.float64)
        self.fitnesses_test = np.full(parents + children, np.inf, dtype=np.float64)
        self.corr_test = np.full(parents + children, np.inf, dtype=np.float64)
        self.best_model = None
        self.function_bank = function_bank
        self.x = None
        self.x_test = None
        self.y = None
        self.y_test = None
        self.solution_threshold = solution_threshold
        self.max_p = parents
        self.max_c = children
        self.max_g = max_generations
        self.original_max_g = max_generations
        self.fixed_length = fixed_length
        self.model_kwargs = model_parameters or {}
        self.ckpt_filename = checkpoint_filename
        self.current_generation = 0
        self.kwargs = kwargs
        self.first_submission = True
        self.dnc_hyperparams = None
        self.dnc = None
        # Config values with defaults
        self.mutation_type = mutation.lower()
        self.selection_type = selection.lower()
        self.xover_type = xover.lower() if xover else None
        self.ff_string = fitness_function
        self.semantic = 'semantic' in self.xover_type if xover else False
        self.aligned = 'aligned' in self.xover_type if xover else False
        self.homologous = 'homologous' in self.xover_type if xover else False

        self.elapsed = 0

        self.search_evaluations = 0 #Fitness evaluations that actually participate in evolution.
        self.diagnostic_evaluations = 0 #Extra evaluations used to inspect crossover/mutation effects
        self.test_evaluations = 0 #Test-set evaluations used only for reporting/generalization.
        self.total_fitness_calls = 0

        self.pareto_layers = []
        self.pareto_elite = []

        # Safe defaults
        self.tournament_size = int(kwargs.get('tournament_size', 2))
        self.n_elites = int(kwargs.get('n_elites', 0))
        self.n_points = int(kwargs.get('n_points', 1))
        self.tournament_diversity = kwargs.get('tournament_diversity', True)
        self.one_d = kwargs.get('one_dimensional_xover', False)
        if self.one_d:
            raise RuntimeError(f'One-Dimensional Crossover has been deprecated since 11 June 2026.')
        self.llm_model = kwargs.get('llm_model', None)
        if self.llm_model is not None:
            self.llm_model = f'crossover-{self.llm_model}'
            self.llm_window_size=int(kwargs.get('llm_window', 5))
            self.llm_window = deque(maxlen=self.llm_window_size)
        else:
            self.llm_window = None

        # Mutation fallback
        self.mutation_can_make_children = self.max_p < 2 or kwargs.get('asexual_reproduction', False)

        # Fitness function
        self.fitness_function = {'correlation': correlation,
                                 'correlation_complexity': corr_comp_fitness,
                                 }.get(fitness_function.lower())
        if not self.fitness_function:
            raise ValueError(f"Invalid fitness function: {fitness_function}")

        # Selection methods
        self.selection_methods = {
            'elite': self.elite_selection,
            'paretoelite': self.pareto_elite_selection,
            'tournament': self.tournament_selection,
            'paretotournament': self.pareto_tournament_selection,
            'elite_tournament': self.elite_tournament_selection,
            'competent_tournament': self.competent_tournament_selection
        }
        if self.selection_type not in self.selection_methods:
            raise ValueError(f"Invalid selection type: {self.selection_type}")
        self.selection = self.selection_methods[self.selection_type]

        # Crossover setup
        self.xover_methods = {
            'none': None,
            'n_point': self._n_point_xover,
            'uniform': self._uniform_xover,
            'semantic_uniform': self._uniform_xover,
            'aligned_semantic_uniform': self._uniform_xover,
            'aligned_homologous_semantic_uniform': self._uniform_xover,
            'homologous_semantic_uniform': self._uniform_xover,
            'semantic_n_point': self._n_point_xover,
            'homologous_semantic_n_point': self._n_point_xover,
            'aligned_homologous_semantic_n_point': self._n_point_xover,
            'aligned_semantic_n_point': self._n_point_xover,
            'subgraph': self._subgraph_xover,
            'llm_n_point': self._n_point_xover,
        }
        if self.xover_type:
            if self.xover_type not in self.xover_methods:
                raise ValueError(f"Invalid crossover type: {self.xover_type}")
            elif 'dnc' in self.xover_type:
                raise ValueError(f"Deep Neural Crossover has been deprecated since 11 June 2026.")
            self.xover = self.xover_methods[self.xover_type]
        else:
            self.xover = None

        # one dimensional xover is incompatible with semantic, subgraph, and llm methods:
        if self.one_d and ('semantic' in self.xover_type or 'subgraph' in self.xover_type or 'llm' in self.xover_type):
            raise ValueError(f'{self.xover_type} crossover is incompatible with one-dimensional crossover.')

        # llm crossover requires a specified model
        if self.llm_model is None and self.xover_type == 'llm_n_point':
            raise ValueError(f'LLM Crossover selected, but no model specified. Accepted inputs are:\n'
                             f'\tgemma')

        # Validate crossover points
        if self.xover_type and 'n_point' in self.xover_type:
            max_size = self.model_kwargs.get('max_size', 10)
            if self.n_points < 1 or self.n_points > max_size // 2:
                raise ValueError(f"Invalid n_points: {self.n_points}")

        # Metrics and tracking
        self.metrics = np.zeros((self.max_g + 1, 37), dtype=np.float64)
        self.xover_index = {
            category: np.zeros((self.max_g, self.max_p))
            for category in XOVER_CATEGORIES
        }
        self.mut_index = np.zeros((self.max_g, self.max_p))

        # self.stn = STN()

    @staticmethod
    def hash_model(m):
        """Return the MD5 digest without allocating an intermediate ``bytes`` copy."""
        # OPTIMIZATION: hashlib can consume a contiguous buffer directly.  Only
        # non-contiguous views require a compact temporary array.
        contiguous = np.ascontiguousarray(m)
        return hashlib.md5(memoryview(contiguous).cast("B")).hexdigest()

    def save_checkpoint(self, filename="cgp_checkpoint.pkl", generation=None):
        """Atomically save the evolver state."""
        checkpoint = Path(filename)
        temp_file = checkpoint.with_suffix(".tmp")

        # OPTIMIZATION: bound method tables and the optional alignment cache are
        # reconstructed on load, so excluding them reduces checkpoint size and
        # avoids serializing redundant references.
        state = self.__dict__.copy()
        for transient_key in (
            "selection_methods",
            "xover_methods",
            "selection",
            "xover",
            "_similarity_aligner",
        ):
            state.pop(transient_key, None)

        # HIGHEST_PROTOCOL is faster and more compact for NumPy-heavy state.
        with temp_file.open("wb") as file:
            pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)

        os.replace(temp_file, checkpoint)
        print(
            f"Checkpoint saved at generation "
            f"{generation or self.current_generation} in {filename}"
        )

    @classmethod
    def load_checkpoint(cls, filename="cgp_checkpoint.pkl"):
        """Load a checkpoint, falling back to its atomic temporary file."""

        def try_load(path):
            with Path(path).open("rb") as file:
                return pickle.load(file)

        checkpoint = Path(filename)
        try:
            data = try_load(checkpoint)
        except EOFError:
            print(
                f"⚠️ Warning: Failed to load checkpoint '{filename}' "
                "(EOFError). Trying backup..."
            )
            temp_file = checkpoint.with_suffix(".tmp")
            if not temp_file.exists():
                raise RuntimeError(
                    f"Checkpoint corrupted and no backup found: {filename}"
                )

            try:
                data = try_load(temp_file)
                print("✅ Loaded from backup:", temp_file)
            except Exception as error:
                raise RuntimeError(
                    f"Failed to load from both '{filename}' and backup: {error}"
                ) from error

        obj = cls.__new__(cls)
        obj.__dict__.update(data)

        # Restore bound callables, which should never be trusted from an old pickle.
        obj.selection_methods = {
            "elite": obj.elite_selection,
            "paretoelite": obj.pareto_elite_selection,
            "tournament": obj.tournament_selection,
            "paretotournament": obj.pareto_tournament_selection,
            "elite_tournament": obj.elite_tournament_selection,
            "competent_tournament": obj.competent_tournament_selection,
        }
        obj.xover_methods = {
            "none": None,
            "n_point": obj._n_point_xover,
            "uniform": obj._uniform_xover,
            "semantic_uniform": obj._uniform_xover,
            "aligned_semantic_uniform": obj._uniform_xover,
            "aligned_homologous_semantic_uniform": obj._uniform_xover,
            "homologous_semantic_uniform": obj._uniform_xover,
            "semantic_n_point": obj._n_point_xover,
            "homologous_semantic_n_point": obj._n_point_xover,
            "aligned_homologous_semantic_n_point": obj._n_point_xover,
            "aligned_semantic_n_point": obj._n_point_xover,
            "subgraph": obj._subgraph_xover,
            "llm_n_point": obj._n_point_xover,
        }
        obj.selection = obj.selection_methods.get(obj.selection_type)
        obj.xover = obj.xover_methods.get(obj.xover_type)
        obj.first_submission = False

        # Older checkpoints predate this lazy cache.
        obj.__dict__.pop("_similarity_aligner", None)

        print(f"Checkpoint loaded at generation {obj.current_generation}")
        return obj

    def initialize_population(self):
        """Initialize the parent population."""
        creation_kwargs = {
            "fixed_length": self.fixed_length,
            "fitness_function": self.ff_string,
            "mutation_type": self.mutation_type,
            **self.model_kwargs,
        }
        # CORRECTNESS: passing function_bank=None suppresses CGP's default bank.
        if self.function_bank is not None:
            creation_kwargs["function_bank"] = self.function_bank

        # OPTIMIZATION: the common (1 + λ) configuration avoids constructing a
        # thread pool whose startup costs exceed a single model generation.
        if self.max_p == 1:
            individuals = [CGP(**creation_kwargs)]
        else:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(CGP, **creation_kwargs)
                    for _ in range(self.max_p)
                ]
                individuals = [future.result() for future in futures]

        for index, individual in enumerate(individuals):
            if self.one_d:
                n_zeros = (
                    (individual.arity + 1) * individual.max_size
                    + individual.outputs
                )
                individual.xover_index = np.zeros(n_zeros)
                print(
                    f"📦 Initialized xover_index for individual {index}: "
                    f"{individual.xover_index.shape}"
                )
            else:
                individual.xover_index = np.zeros(
                    individual.max_size + individual.outputs
                )
            self.population[index] = individual

        # model_keys is a flat str->int dictionary; a shallow copy gives the
        # same independence as deepcopy without recursively walking scalars.
        self.model_keys = self.population[0].model_keys.copy()

    def elite_selection(self, models=None, n_elites=None, indices=False):
        """Select top-n elite individuals based on fitness."""
        n_elites = n_elites or self.max_p
        source = self.population if models is None else models

        # OPTIMIZATION: sort the shuffled list in place instead of creating a
        # second sorted list plus two full tuples.
        valid_pairs = [(i, model) for i, model in enumerate(source) if model is not None]
        np.random.shuffle(valid_pairs)
        valid_pairs.sort(key=lambda pair: pair[1].fitness)

        selected = valid_pairs[:n_elites]
        selected_models = [deepcopy(model) for _, model in selected]

        if indices:
            # Preserve the original tuple return type for selected indices.
            return selected_models, tuple(index for index, _ in selected)
        return selected_models

    def get_pareto_front(self, points: np.ndarray) -> np.ndarray:
        """
        Return indices of Pareto-optimal points for two minimized objectives.
        """
        if points.size == 0:
            return np.empty(0, dtype=np.intp)

        order = np.argsort(points[:, 0])
        best_complexity = np.inf

        # OPTIMIZATION: preallocate the largest possible result instead of growing
        # a Python list and converting it after the scan.
        front = np.empty(order.size, dtype=np.intp)
        count = 0
        for index in order:
            complexity = points[index, 1]
            if complexity < best_complexity:
                best_complexity = complexity
                front[count] = index
                count += 1

        return front[:count].copy()

    def pareto_elite_selection(self, models=None, n_elites=None, return_indices=False):
        """
        Select elites using correlation and complexity.

        Correlation is primary; complexity is used within a small correlation
        bucket.  The method always returns up to ``n_elites`` valid individuals.
        """
        source = self.population if models is None else models

        # OPTIMIZATION: build the original-index vector directly and avoid
        # constructing/unpacking a list of (index, model) tuples.
        original_indices = np.fromiter(
            (i for i, model in enumerate(source) if model is not None),
            dtype=np.intp,
        )
        if original_indices.size == 0:
            return np.empty(0, dtype=np.intp) if return_indices else []

        models_valid = [source[i] for i in original_indices]
        n_elites = n_elites or self.n_elites
        n_elites = min(n_elites, len(models_valid))

        # Build the dense objective matrix with one allocation.
        points = np.empty((len(models_valid), 2), dtype=np.float64)
        for i, model in enumerate(models_valid):
            points[i, 0] = model.correlation
            points[i, 1] = model.complexity

        front_indices = self.get_pareto_front(points)
        front_points = points[front_indices]

        correlation = front_points[:, 0]
        complexity = front_points[:, 1]
        best_correlation = np.min(correlation)
        correlation_bucket = np.floor(
            (correlation - best_correlation) / 1e-3
        )

        front_order = np.lexsort((complexity, correlation_bucket))
        chosen = front_indices[front_order[:n_elites]].tolist()

        if len(chosen) < n_elites:
            remaining_mask = np.ones(len(models_valid), dtype=bool)
            remaining_mask[chosen] = False
            remaining_indices = np.flatnonzero(remaining_mask)
            remaining_points = points[remaining_indices]
            remaining_order = np.lexsort(
                (remaining_points[:, 1], remaining_points[:, 0])
            )
            needed = n_elites - len(chosen)
            chosen.extend(remaining_indices[remaining_order[:needed]].tolist())

        chosen_indices = np.asarray(chosen, dtype=np.intp)
        if return_indices:
            return original_indices[chosen_indices]

        return [deepcopy(models_valid[i]) for i in chosen_indices]

    def tournament_selection(self, n_to_select=None):
        """Perform fitness tournament selection."""
        n_to_select = n_to_select or self.max_p
        new_population = np.empty(n_to_select, dtype=object)

        # CORRECTNESS + OPTIMIZATION: retain original population indices while
        # filtering None entries; the old compressed index range could select the
        # wrong model when gaps were present.
        available_indices = np.fromiter(
            (
                index
                for index, individual in enumerate(self.population)
                if individual is not None
            ),
            dtype=np.intp,
        )
        return self.t_select(
            available_indices,
            new_population,
            len(new_population),
            pareto=False,
        )

    def pareto_tournament_selection(self, n_to_select=None):
        """Perform Pareto tournament selection."""
        n_to_select = n_to_select or self.max_p
        new_population = np.empty(n_to_select, dtype=object)
        available_indices = np.fromiter(
            (
                index
                for index, individual in enumerate(self.population)
                if individual is not None
            ),
            dtype=np.intp,
        )
        return self.t_select(
            available_indices,
            new_population,
            len(new_population),
            pareto=True,
        )

    def elite_tournament_selection(self):
        """Combine elite selection with tournament selection."""
        elite_population, elite_indices = self.elite_selection(
            n_elites=self.n_elites,
            indices=True,
        )
        remaining_slots = self.max_p - len(elite_population)

        # OPTIMIZATION: set membership is O(1), unlike repeatedly scanning the
        # tuple of elite indices for every population member.
        elite_index_set = set(elite_indices)
        available_indices = np.fromiter(
            (
                i
                for i, individual in enumerate(self.population)
                if individual is not None and i not in elite_index_set
            ),
            dtype=np.intp,
        )

        tournament_population = np.empty(remaining_slots, dtype=object)
        tournament_population = self.t_select(
            available_indices,
            tournament_population,
            remaining_slots,
            pareto=False,
        )
        return np.concatenate((elite_population, tournament_population))

    def t_select(self, available_indices, new_population, remaining_slots, pareto=False):
        """Fill ``new_population`` from tournament winners."""
        # OPTIMIZATION: use a NumPy index pool consistently.  This also avoids the
        # list-vs-array comparison bug in diversity removal.
        available_indices = np.asarray(available_indices, dtype=np.intp)
        population = self.population
        tournament_size = self.tournament_size
        enforce_diversity = self.tournament_diversity

        for output_index in range(remaining_slots):
            if enforce_diversity and available_indices.size < tournament_size:
                contestants = available_indices
            else:
                contestants = np.random.choice(
                    available_indices,
                    size=tournament_size,
                    replace=False,
                )

            if not pareto:
                best_index = min(
                    contestants,
                    key=lambda index: population[index].fitness,
                )
            else:
                selected_models = [population[index] for index in contestants]
                relative_best = self.pareto_elite_selection(
                    models=selected_models,
                    return_indices=True,
                    n_elites=1,
                )
                best_index = contestants[relative_best[0]]

            new_population[output_index] = deepcopy(population[best_index])

            if enforce_diversity:
                available_indices = available_indices[
                    available_indices != best_index
                ]

        return new_population

    def _compute_semantics(self):
        """Compute one flattened semantic row for every valid individual."""
        valid_individuals = [
            individual for individual in self.population if individual is not None
        ]
        if not valid_individuals:
            return np.empty((0, len(self.x)))

        # OPTIMIZATION: np.stack knows the final row count up front and avoids some
        # of vstack's input normalization overhead.
        return np.stack(
            [individual(self.x).ravel() for individual in valid_individuals],
            axis=0,
        )

    def competent_tournament_selection(self, n_to_select=None):
        """
        Perform semantic-distance-based competent tournament selection.
        """
        n_to_select = n_to_select or self.max_p

        valid_population_indices = np.fromiter(
            (i for i, model in enumerate(self.population) if model is not None),
            dtype=np.intp,
        )
        parent_semantics = self._compute_semantics()
        if parent_semantics.size == 0:
            return np.empty(0, dtype=object)

        # OPTIMIZATION: normalize every semantic row in two vectorized passes,
        # replacing np.apply_along_axis and its Python callback overhead.
        row_means = np.mean(parent_semantics, axis=1, keepdims=True)
        row_stds = np.std(parent_semantics, axis=1, keepdims=True)
        parent_semantics = (
            parent_semantics - row_means
        ) / (row_stds + 1e-8)

        target = np.ravel(self.y)
        target = (target - np.mean(target)) / (np.std(target) + 1e-8)
        target_distances = np.linalg.norm(
            parent_semantics - target,
            axis=1,
        )

        selected_indices = []
        remaining = list(range(len(valid_population_indices)))
        tournament_size = min(self.tournament_size, len(remaining))
        enforce_diversity = self.tournament_diversity

        while (
            len(selected_indices) < n_to_select
            and len(remaining) >= tournament_size
        ):
            contestants = np.random.choice(
                remaining,
                size=tournament_size,
                replace=False,
            )

            first_index = contestants[
                np.argmin(target_distances[contestants])
            ]
            first_semantics = parent_semantics[first_index]
            first_target_distance = target_distances[first_index]

            # OPTIMIZATION: avoid the unused semantic-distance array and avoid a
            # temporary dictionary.  The strict '<' preserves first-item tie
            # behavior from min(dict, key=dict.get).
            second_index = contestants[0]
            second_score = get_score(
                first_target_distance,
                pairwise_minkowski_distance(
                    first_semantics,
                    parent_semantics[second_index],
                    p=2,
                ),
                target_distances[second_index],
            )
            for candidate in contestants[1:]:
                candidate_score = get_score(
                    first_target_distance,
                    pairwise_minkowski_distance(
                        first_semantics,
                        parent_semantics[candidate],
                        p=2,
                    ),
                    target_distances[candidate],
                )
                if candidate_score < second_score:
                    second_score = candidate_score
                    second_index = candidate

            selected_indices.extend((first_index, second_index))

            if enforce_diversity:
                remaining = [
                    index
                    for index in remaining
                    if index != first_index and index != second_index
                ]

        selected_original_indices = valid_population_indices[
            np.asarray(selected_indices[:n_to_select], dtype=np.intp)
        ]
        # self.population becomes a list inside fit(); gather explicitly so
        # this path supports both list and object-array populations.
        return np.asarray(
            [self.population[index] for index in selected_original_indices],
            dtype=object,
        )

    def crossover(self, parents, xover_rate, gen):
        """Perform crossover until ``max_c`` children have been produced."""
        children = []
        max_children = self.max_c
        child_counter = self.child_id_counter
        model_key_map = getattr(self, "model_key_map", None)

        # OPTIMIZATION: iterate parent indices directly instead of materializing a
        # list of parent-pair tuples on every generation.
        while len(children) < max_children:
            for parent_offset in range(0, len(parents), 2):
                pair_index = parent_offset // 2
                parent1 = parents[parent_offset]
                parent2 = parents[parent_offset + 1]

                if np.random.rand() < xover_rate:
                    if self.xover_type == "subgraph":
                        child1 = self.xover(parent1, parent2, gen)
                        child2 = self.xover(parent2, parent1, gen)
                    else:
                        child1, child2 = self.xover(parent1, parent2, gen)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)

                parent1_key = getattr(
                    parent1,
                    "child_keys",
                    f"Model_{2 * pair_index:03d}_g{gen - 1}",
                )
                parent2_key = getattr(
                    parent2,
                    "child_keys",
                    f"Model_{2 * pair_index + 1:03d}_g{gen - 1}",
                )
                parent_keys = [parent1_key, parent2_key]

                for child in (child1, child2):
                    child_key = f"Child_{child_counter:03d}_g{gen}"
                    child.set_parent_key(parent_keys.copy())
                    child.set_child_key(child_key)

                    if model_key_map is not None:
                        model_key_map[child_key] = child

                    child_counter += 1
                    children.append(child)
                    if len(children) >= max_children:
                        break

                if len(children) >= max_children:
                    break

        self.child_id_counter = child_counter
        return np.asarray(children, dtype=object)

    def flatten_parent(self, parent):
        """
        Flatten function nodes as [operator, operand0, ..., operandN].
        """
        node_type_column = self.model_keys["NodeType"]
        function_mask = parent.model[:, node_type_column] == FUNCTION_NODE
        nodes = parent.model[function_mask]

        # OPTIMIZATION: gather all required columns once and flatten in C order,
        # replacing the nested Python loops.
        columns = [self.model_keys["Operator"]]
        columns.extend(
            self.model_keys[f"Operand{i}"] for i in range(parent.arity)
        )
        flattened = nodes[:, columns].astype(np.int64, copy=False).ravel()
        return flattened, nodes[:, node_type_column]

    def unflatten_model(self, flattened_model, node_types, arity):
        """
        Reconstruct model rows from [operator, operand0, ..., operandN] records.
        """
        num_nodes = len(node_types)
        stride = arity + 1
        records = np.asarray(flattened_model).reshape(num_nodes, stride)
        model = np.zeros(
            (num_nodes, len(self.model_keys)),
            dtype=np.int64,
        )

        # OPTIMIZATION: assign complete columns at once.  The +1 offset is also a
        # correctness fix: operand0 follows the operator in the flattened record.
        model[:, self.model_keys["NodeType"]] = node_types
        model[:, self.model_keys["Operator"]] = records[:, 0]
        operand_columns = [
            self.model_keys[f"Operand{i}"] for i in range(arity)
        ]
        model[:, operand_columns] = records[:, 1:]
        return model

    def _n_point_xover(self, p1, p2, gen, **kwargs):
        """Perform n-point crossover, including semantic variants."""

        def get_crossover_points(length, offset=0, weights=None):
            indices = np.arange(offset, length)
            if weights is not None and weights.sum() > 0:
                if len(indices) > len(weights):
                    indices = indices[:len(weights)]
                return np.sort(
                    np.random.choice(
                        indices,
                        size=self.n_points,
                        replace=False,
                        p=weights,
                    )
                )
            return np.sort(
                np.random.choice(
                    indices,
                    size=self.n_points,
                    replace=False,
                )
            )

        first_body_node = min(p1.first_body_node, p2.first_body_node)
        weights = None

        if "llm" in self.xover_type:
            self.llm_window.appendleft(self.population)
            population_context = summarize_population_for_llm(self.llm_window)
            crossover_point = choose_crossover_point_with_ollama(
                p1.model,
                p2.model,
                p1.fitness,
                p2.fitness,
                p1.complexity,
                p2.complexity,
                self.llm_model,
                population_context,
            )
            crossover_points = np.asarray([crossover_point], dtype=np.intp)
        else:
            if self.semantic:
                if self.aligned:
                    weights = _get_semantic_alignment(p1, p2, self.x)
                else:
                    values1 = clean_values(p1, self.x)
                    values2 = clean_values(p2, self.x)
                    weights = get_weights(get_ssd(values1, values2))

                if self.homologous and not np.all(weights == weights[0]):
                    maximum = weights.max()
                    minimum = weights.min()
                    weights = (
                        maximum - weights
                    ) / (maximum - minimum + 1e-8)
                    weights /= weights.sum()

            crossover_points = get_crossover_points(
                len(p1.model),
                p1.first_body_node,
                weights,
            )

        tracking_indices = crossover_points - first_body_node
        p1.xover_index[tracking_indices] += 1
        p2.xover_index[tracking_indices] += 1

        # OPTIMIZATION: copy alternating slices directly into the final arrays.
        # This avoids np.split's view lists and np.concatenate's additional input
        # bookkeeping while retaining support for unequal parent lengths.
        points = crossover_points.tolist()
        segment_starts = [0, *points]
        segment_ends = [*points, None]

        child1_length = (
            len(p1.model)
            if len(points) % 2 == 0
            else len(p2.model)
        )
        child2_length = (
            len(p2.model)
            if len(points) % 2 == 0
            else len(p1.model)
        )
        child1 = np.empty(
            (child1_length, p1.model.shape[1]),
            dtype=np.result_type(p1.model.dtype, p2.model.dtype),
        )
        child2 = np.empty(
            (child2_length, p1.model.shape[1]),
            dtype=np.result_type(p1.model.dtype, p2.model.dtype),
        )

        child1_position = 0
        child2_position = 0
        for segment_index, (start, end) in enumerate(
            zip(segment_starts, segment_ends)
        ):
            if segment_index % 2 == 0:
                source1, source2 = p1.model, p2.model
            else:
                source1, source2 = p2.model, p1.model

            source1_slice = source1[start:end]
            source2_slice = source2[start:end]
            next1 = child1_position + len(source1_slice)
            next2 = child2_position + len(source2_slice)
            child1[child1_position:next1] = source1_slice
            child2[child2_position:next2] = source2_slice
            child1_position = next1
            child2_position = next2

        child_kwargs = {
            "model_keys": self.model_keys,
            "fixed_length": self.fixed_length,
            "fitness_function": self.ff_string,
            "mutation_type": self.mutation_type,
        }
        if self.function_bank is not None:
            child_kwargs["function_bank"] = self.function_bank
        return (
            CGP(model=child1, **child_kwargs),
            CGP(model=child2, **child_kwargs),
        )

    def _uniform_xover(self, p1, p2, gen, weights: np.ndarray | list = None, **kwargs):
        """Perform uniform crossover, including semantic variants."""
        n_outputs = 0

        if self.semantic:
            if self.aligned:
                weights = _get_semantic_alignment(p1, p2, self.x)
            else:
                values1 = clean_values(p1, self.x)
                values2 = clean_values(p2, self.x)
                weights = get_weights(
                    get_ssd(values1, values2),
                    epsilon=0.001,
                )

            if self.homologous and not np.all(weights == weights[0]):
                maximum = weights.max()
                minimum = weights.min()
                weights = (
                    maximum - weights
                ) / (maximum - minimum + 1e-8)
                weights /= weights.sum()

            n_outputs = p1.outputs

        first_body_node = min(p1.first_body_node, p2.first_body_node)
        assert len(p1.model) == len(p2.model), (
            "Parents in Uniform Xover must have the same length."
        )
        possible_indices = np.arange(
            first_body_node,
            len(p1.model) - n_outputs,
        )

        if weights is not None:
            positive_mask = weights > 1e-8
            filtered_indices = possible_indices[positive_mask]
            filtered_weights = weights[positive_mask]
            n_swap = min(
                len(filtered_indices),
                len(possible_indices) // 2,
            )
            if n_swap:
                swapped_indices = np.random.choice(
                    filtered_indices,
                    size=n_swap,
                    replace=False,
                    p=filtered_weights / filtered_weights.sum(),
                )
            else:
                swapped_indices = np.empty(0, dtype=np.intp)
        else:
            swapped_indices = np.random.choice(
                possible_indices,
                size=len(possible_indices) // 2,
                replace=False,
            )

        child1_model = p1.model.copy()
        child2_model = p2.model.copy()

        # OPTIMIZATION: skip advanced-index temporary assignments when no genes
        # were selected; this matters for sparse semantic weights.
        if swapped_indices.size:
            child1_model[swapped_indices] = p2.model[swapped_indices]
            child2_model[swapped_indices] = p1.model[swapped_indices]

        tracking_indices = swapped_indices - first_body_node
        p1.xover_index[tracking_indices] += 1
        p2.xover_index[tracking_indices] += 1

        child_kwargs = {
            "model_keys": self.model_keys,
            "fixed_length": self.fixed_length,
            "fitness_function": self.ff_string,
            "mutation_type": self.mutation_type,
        }
        if self.function_bank is not None:
            child_kwargs["function_bank"] = self.function_bank
        return (
            CGP(model=child1_model, **child_kwargs),
            CGP(model=child2_model, **child_kwargs),
        )

    def _subgraph_xover(self, p1: CGP, p2: CGP, gen: int, **kwargs):
        """Perform subgraph crossover using NumPy model arrays."""

        def random_node_number(n_inputs, input_nodes=None, active_nodes=None, maximum=None):
            """Choose from valid active/input connection candidates."""
            candidates = []

            if active_nodes is not None:
                if maximum is None:
                    candidates.append(
                        active_nodes[np.random.randint(0, len(active_nodes))]
                    )
                else:
                    eligible = active_nodes[active_nodes <= maximum]
                    if eligible.size:
                        candidates.append(
                            eligible[np.random.randint(0, len(eligible))]
                        )
                    else:
                        candidates.append(np.random.randint(0, n_inputs))

            if input_nodes is not None and len(input_nodes):
                candidates.append(
                    input_nodes[np.random.randint(0, len(input_nodes))]
                )

            if not candidates:
                raise ValueError("No valid node candidates were supplied.")

            # CORRECTNESS: randint's upper bound is exclusive.  Using len-1 made a
            # one-element candidate list fail and made the final candidate unreachable.
            return candidates[np.random.randint(0, len(candidates))]

        def determine_crossover_point(active1, active2):
            minimum1, maximum1 = min(active1), max(active1)
            minimum2, maximum2 = min(active2), max(active2)
            if minimum1 >= maximum1:
                maximum1 += 1
            if minimum2 >= maximum2:
                maximum2 += 1
            point1 = np.random.randint(minimum1, maximum1)
            point2 = np.random.randint(minimum2, maximum2)
            return min(point1, point2)

        def reconnect_active_nodes(n_inputs, active_nodes, crossover_point, model):
            """Reconnect operands that no longer target an active node."""
            node_type_column = self.model_keys["NodeType"]
            node_types = model[:, node_type_column]
            input_nodes = np.flatnonzero(
                (node_types == CONSTANT_NODE) | (node_types == INPUT_NODE)
            )
            operand_columns = np.fromiter(
                (
                    self.model_keys[f"Operand{i}"]
                    for i in range(p1.arity)
                ),
                dtype=np.intp,
                count=p1.arity,
            )
            active_set = set(np.asarray(active_nodes, dtype=np.intp).tolist())

            for node_index in active_nodes:
                if node_index <= crossover_point:
                    continue
                for operand_column in operand_columns:
                    if int(model[node_index, operand_column]) not in active_set:
                        model[node_index, operand_column] = random_node_number(
                            n_inputs,
                            input_nodes=input_nodes,
                            active_nodes=active_nodes,
                            maximum=crossover_point,
                        )

            output_nodes = np.flatnonzero(node_types == OUTPUT_NODE)
            operand0_column = self.model_keys["Operand0"]
            for output_index in output_nodes:
                if int(model[output_index, operand0_column]) not in active_set:
                    model[output_index, operand0_column] = random_node_number(
                        n_inputs,
                        input_nodes=input_nodes,
                        active_nodes=active_nodes,
                    )

        def ensure_active_nodes(parent):
            """Return a model copy and at least one active node."""
            active_nodes = np.fromiter(
                parent.get_active_nodes(),
                dtype=np.intp,
            )
            if active_nodes.size:
                return parent.model.copy(), active_nodes

            # OPTIMIZATION: allocate generic evaluation arrays only on the rare
            # fallback path, and reuse them across retries.
            generic_x = np.zeros((1, parent.inputs))
            generic_y = np.zeros((1, parent.outputs))
            while active_nodes.size == 0:
                parent.mutate_output()
                parent.fit(generic_x, generic_y)
                active_nodes = np.fromiter(
                    parent.get_active_nodes(),
                    dtype=np.intp,
                )
            return parent.model.copy(), active_nodes

        model1, active1 = ensure_active_nodes(p1)
        model2, active2 = ensure_active_nodes(p2)
        crossover_point = determine_crossover_point(active1, active2)

        child_kwargs = {
            "model_keys": self.model_keys,
            "fixed_length": self.fixed_length,
            "fitness_function": self.ff_string,
            "mutation_type": self.mutation_type,
        }
        if self.function_bank is not None:
            child_kwargs["function_bank"] = self.function_bank

        if crossover_point <= 0:
            return CGP(model=model1, **child_kwargs)

        # OPTIMIZATION: for equal-length parents, copy parent1 once and overwrite
        # the tail in place instead of concatenating two temporary slices.
        if model1.shape == model2.shape:
            child_model = model1
            child_model[crossover_point:] = model2[crossover_point:]
        else:
            child_model = np.concatenate(
                (model1[:crossover_point], model2[crossover_point:]),
                axis=0,
            )

        first_body_node = min(p1.first_body_node, p2.first_body_node)
        active_before = active1[active1 <= crossover_point]
        active_after = active2[active2 > crossover_point]

        if active_before.size and active_after.size:
            child_model[
                active_after[0],
                self.model_keys["Operand0"],
            ] = active_before[-1]

        active_nodes = np.concatenate((active_before, active_after))
        if active_nodes.size:
            reconnect_active_nodes(
                p1.inputs + len(p1.constants),
                active_nodes,
                crossover_point,
                child_model,
            )

        child = CGP(model=child_model, **child_kwargs)
        child.xover_index[crossover_point - first_body_node] += 1
        return child

    def _mutate(self, models, gen, mutation_rate, verbose=False):
        """Mutate models either into exactly ``max_c`` children or in place."""
        model_key_map = getattr(self, "model_key_map", None)
        child_counter = self.child_id_counter

        if self.mutation_can_make_children:
            children = []
            num_models = len(models)
            if num_models == 0:
                return np.empty(0, dtype=object)

            # CORRECTNESS + COMPATIBILITY: distribute children in parent-major
            # order.  This matches the old ordering when max_c was divisible by
            # the parent count, while handling remainders and max_c < parents.
            base_children, remainder = divmod(self.max_c, num_models)

            for model_index, model in enumerate(models):
                attempts = base_children + (model_index < remainder)
                for _ in range(attempts):
                    should_mutate = (
                        np.random.rand() < mutation_rate
                        or self.max_p == 1
                    )
                    if should_mutate:
                        child = model.mutate(verbose)
                    else:
                        # An unmutated reproduction is still a distinct child.
                        child = deepcopy(model)
                        child.id = uuid.uuid4()

                    assert child.id != model.id, (
                        "Reproduced child has same ID as parent"
                    )
                    assert child is not model, (
                        "Child is not a distinct instance"
                    )
                    assert id(child.model) != id(model.model), (
                        "Structured array not deeply copied"
                    )

                    child.fit(self.x, self.y, mutable=False)
                    child_key = f"Child_{child_counter:03d}_g{gen}"
                    parent_key = getattr(
                        model,
                        "child_keys",
                        f"Model_{model_index:03d}_g{gen - 1}",
                    )
                    child.set_parent_key([parent_key])
                    child.set_child_key(child_key)

                    if model_key_map is not None:
                        model_key_map[child_key] = child

                    child_counter += 1
                    children.append(child)

            self.child_id_counter = child_counter
            return np.asarray(children, dtype=object)

        for model_index, model in enumerate(models):
            if np.random.rand() >= mutation_rate:
                continue

            mutated = model.mutate(verbose)
            mutated.fit(self.x, self.y)

            child_key = f"Child_{child_counter:03d}_g{gen}"
            parent_key = getattr(
                model,
                "child_keys",
                f"Model_{model_index:03d}_g{gen - 1}",
            )
            mutated.set_parent_key([parent_key])
            mutated.set_child_key(child_key)

            if model_key_map is not None:
                model_key_map[child_key] = mutated

            child_counter += 1
            models[model_index] = mutated

        self.child_id_counter = child_counter
        return models

    def _get_fitnesses(
        self,
        mode="train",
        pop_list=None,
        eval_purpose="search",
        mutable=True,
        count_only_unchached=True,
    ):
        """Evaluate a population and optionally store its fitness arrays."""
        self.total_fitness_calls += 1

        if eval_purpose == "search":
            self.search_evaluations += 1
        elif eval_purpose == "diagnostic":
            self.diagnostic_evaluations += 1
        elif eval_purpose == "test":
            self.test_evaluations += 1
        else:
            raise ValueError(
                "cgp_evolver.py::_get_fitnesses: eval_purpose must be "
                '"search" or "diagnostic" or "test"'
            )

        if mode == "train":
            x, y = self.x, self.y
        elif mode == "test":
            x, y = self.x_test, self.y_test
        else:
            raise ValueError(f"Unknown mode: {mode}")

        store_results = pop_list is None
        population = self.population if store_results else pop_list
        population_size = len(population)

        fitnesses = np.full(population_size, np.inf, dtype=np.float64)
        correlations = np.full(population_size, np.inf, dtype=np.float64)

        # OPTIMIZATION: cache the array setters and avoid tuple unpacking into an
        # unused complexity slot on every iteration.
        for index, model in enumerate(population):
            if model is None:
                continue
            result = model.fit(x, y, mutable=mutable)
            correlations[index] = result[0]
            fitnesses[index] = result[2]

        if store_results:
            if mode == "train":
                self.fitnesses = fitnesses
                self.corrs = correlations
            else:
                self.fitnesses_test = fitnesses
                self.corr_test = correlations

        return fitnesses, correlations

    def _get_similarity_score(self, model1_obj, model2_obj):
        """Compute structural similarity using global sequence alignment."""
        model1 = model1_obj.model
        model2 = model2_obj.model

        if not isinstance(model1, np.ndarray) or model1.ndim != 2:
            print(f"Model1 not 2D: shape={getattr(model1, 'shape', None)}")
            return 0.0
        if not isinstance(model2, np.ndarray) or model2.ndim != 2:
            print(f"Model2 not 2D: shape={getattr(model2, 'shape', None)}")
            return 0.0

        try:
            # OPTIMIZATION: only columns 1: are consumed below, so copy that slice
            # instead of deepcopying both complete model matrices.
            sequence_matrix1 = np.array(model1[:, 1:], copy=True)
            sequence_matrix2 = np.array(model2[:, 1:], copy=True)

            # OPTIMIZATION: np.unique(return_inverse=True) performs the function
            # remapping in compiled code and replaces np.vectorize(dict.get).
            functions = np.concatenate(
                (sequence_matrix1[:, 0], sequence_matrix2[:, 0])
            )
            _, inverse = np.unique(functions, return_inverse=True)
            split = len(sequence_matrix1)
            sequence_matrix1[:, 0] = inverse[:split]
            sequence_matrix2[:, 0] = inverse[split:]

            sequence1 = sequence_matrix1.ravel().astype(str)
            sequence2 = sequence_matrix2.ravel().astype(str)
        except Exception as error:
            print("Error during sequence mapping or flattening:", error)
            return 0.0

        # OPTIMIZATION: configure the immutable scoring object once per evolver.
        aligner = getattr(self, "_similarity_aligner", None)
        if aligner is None:
            aligner = PairwiseAligner()
            aligner.mode = "global"
            aligner.match_score = 2
            aligner.mismatch_score = -1
            aligner.open_gap_score = -2
            aligner.extend_gap_score = -2
            self._similarity_aligner = aligner

        try:
            score = aligner.score(
                Seq("".join(sequence1)),
                Seq("".join(sequence2)),
            )
            return score if np.isfinite(score) else 0.0
        except Exception as error:
            print("Alignment error:", error)
            return 0.0

    def _analyze_similarity(self):
        """
        Computes the similarity between parents and their best offspring.
        Uses structural similarity to compare genetic representations.
        """
        parent_child_groups = self._group_parents_and_children()
        similarity_scores = []

        for parent_pair, (parents, children) in parent_child_groups.items():
            # Filter out None or invalid entries
            if any("g-1" in p for p in parent_pair):
                continue
            parents = [p for p in parents if p is not None]
            children = [c for c in children if c is not None]
            if not parents or not children:
                continue

            best_parent = min(parents, key=lambda ind: getattr(ind, 'fitness', np.inf))
            best_child = min(children, key=lambda ind: getattr(ind, 'fitness', np.inf))

            score = self._get_similarity_score(best_parent, best_child)
            similarity_scores.append((parent_pair, score))

        return similarity_scores

    def _record_metrics(self, gen: int, elapsed: float = 0):
        """Record all generation-level statistics."""
        # OPTIMIZATION: compute the valid population positions once, then reuse the
        # same compact views for every metric family.
        valid_indices = np.fromiter(
            (
                index
                for index, individual in enumerate(self.population)
                if individual is not None
            ),
            dtype=np.intp,
        )
        valid_population = [self.population[i] for i in valid_indices]

        fitness = self.fitnesses[valid_indices]
        test_fitness = self.fitnesses_test[valid_indices]
        test_correlation = self.corr_test[valid_indices]

        # CORRECTNESS + OPTIMIZATION: model.correlation is overwritten by the
        # most recent test evaluation.  The dedicated training array is
        # authoritative and already contiguous.
        correlation_values = self.corrs[valid_indices]
        complexity_values = np.fromiter(
            (model.complexity for model in valid_population),
            dtype=np.float64,
            count=len(valid_population),
        )
        active_nodes = np.fromiter(
            (model.count_active_nodes() for model in valid_population),
            dtype=np.float64,
            count=len(valid_population),
        )

        best_relative_index = int(np.argmin(fitness))
        self.best_model = valid_population[best_relative_index]

        correlation_statistics = _get_quartiles(correlation_values)
        test_correlation_statistics = _get_quartiles(test_correlation)
        complexity_statistics = _get_quartiles(complexity_values)
        fitness_statistics = _get_quartiles(fitness)
        test_fitness_statistics = _get_quartiles(test_fitness)
        active_node_statistics = _get_quartiles(active_nodes)
        semantic_diversity = np.nanstd(fitness)

        self.metrics[gen] = (
            *correlation_statistics,
            *test_correlation_statistics,
            *complexity_statistics,
            *fitness_statistics,
            *test_fitness_statistics,
            self.best_model.count_active_nodes(),
            *active_node_statistics,
            semantic_diversity,
            self.search_evaluations,
            self.diagnostic_evaluations,
            self.test_evaluations,
            self.total_fitness_calls,
            elapsed,
        )

    def save_metrics(self, path=None):
        path = path if path is not None else '.'
        print(f'{path}/statistics.csv')
        # Save the metrics DataFrame only once
        np.savetxt(f'{path}/statistics.csv', self.metrics, delimiter=',')

        # Save the xover_index categories efficiently
        for cat in ['deleterious', 'neutral', 'beneficial']:
            np.savetxt(f'{path}/xover_density_{cat}.csv', self.xover_index[cat].astype(np.int32), delimiter=",")

    def save_stn(self, path=None):
        path = path if path is not None else '.'
        print(f'{path}/stn.json')
        # Save the metrics DataFrame only once
        with open(f'{path}/stn.json', 'w') as f:
            json.dump(self.stn.to_dict(), f, indent = 4)

    def _report_generation(self, g: int):
        """Print the best stored training result for a generation."""
        best_index = int(np.argmin(self.fitnesses))
        best_individual = self.population[best_index]
        print(f"Generation {g}")
        print(
            f"Best Fitness: {self.fitnesses[best_index]}:\t"
            f"Correlation: {self.corrs[best_index]}\t"
            f"Complexity: {best_individual.complexity}\t"
            f"Elapsed Time: {self.elapsed}"
        )
        print("################")

    def _compare_child_parents(self):
        parent_child_groups = self._group_parents_and_children()

        for parent_pair, (parents, children) in parent_child_groups.items():
            if not parents or not children:
                continue

            # Extract fitness values from parents for quicker access
            parent_fitness = [parent.fitness for parent in parents]

            for child in children:
                if child.better_than_parents is None:
                    # Check if the child's fitness is better or worse than any parent
                    child_fitness = child.fitness
                    if any(child_fitness > f for f in parent_fitness):
                        child.better_than_parents = 'deleterious'
                    elif any(child_fitness < f for f in parent_fitness):
                        child.better_than_parents = 'beneficial'

    def _box_distribution(self, gen):
        """Accumulate crossover-density statistics and reset individual counters."""
        xover_index = self.xover_index
        expected_length = xover_index["beneficial"].shape[1]
        row = gen - 1

        # OPTIMIZATION: process the population in one pass instead of first
        # allocating an intermediate filtered-individual list.
        for individual in self.population:
            if (
                individual is None
                or individual.parent_keys is None
                or individual.better_than_parents is None
            ):
                continue

            if individual.xover_index.shape[0] != expected_length:
                raise ValueError(
                    f"xover_index length mismatch at generation {gen}: "
                    f"expected {expected_length}, "
                    f"got {individual.xover_index.shape[0]}"
                )

            category = individual.better_than_parents
            if category not in XOVER_CATEGORIES:
                category = "neutral"
            xover_index[category][row] += individual.xover_index
            individual.xover_index.fill(0)

    def set_max_gens(self, gens):
        self.max_g = gens

    def expand_generations_if_needed(self, new_max_g: int):
        """Grow generation-indexed arrays exactly once to ``new_max_g``."""
        if new_max_g <= self.original_max_g:
            return

        def grow_rows(array, target_rows):
            if array.shape[0] >= target_rows:
                return array

            # OPTIMIZATION: direct allocation/copy avoids np.pad's generalized
            # argument processing and prevents cumulative over-padding on resume.
            grown = np.zeros(
                (target_rows, *array.shape[1:]),
                dtype=array.dtype,
            )
            grown[: array.shape[0]] = array
            return grown

        self.metrics = grow_rows(self.metrics, new_max_g + 1)
        self.mut_index = grow_rows(self.mut_index, new_max_g)
        for category in self.xover_index:
            self.xover_index[category] = grow_rows(
                self.xover_index[category],
                new_max_g,
            )

        # CORRECTNESS: advance the recorded capacity so a later resume extends
        # from the current size instead of padding by the total historical delta.
        self.original_max_g = new_max_g

    def initialize_xover_index(self):
        """Initialize generation-by-gene crossover-density arrays."""
        xover_length = len(self.population[0].xover_index)
        self.xover_index = {
            category: np.zeros(
                (self.max_g, xover_length),
                dtype=np.float64,
            )
            for category in XOVER_CATEGORIES
        }

    def _reinsert_elites(self, protected_parents):
        """Reinsert the top elites after verifying their stored fitness."""
        if self.n_elites <= 0:
            return

        parent_fitnesses = np.fromiter(
            (parent.fitness for parent in protected_parents),
            dtype=np.float64,
            count=len(protected_parents),
        )
        elite_indices = np.argsort(parent_fitnesses)[: self.n_elites]
        corrected_elites = []

        for elite_index in elite_indices:
            elite = protected_parents[elite_index]
            original_fitness = elite.fitness

            # OPTIMIZATION: one deepcopy is sufficient; the previous code copied
            # every selected elite twice before evaluation.
            elite_copy = deepcopy(elite)
            result = elite_copy.fit(self.x, self.y, mutable=False)
            recomputed_fitness = result[2]
            difference = abs(original_fitness - recomputed_fitness)

            # CORRECTNESS: test the larger threshold first; the previous elif was
            # unreachable for differences above 1e-2.
            if difference > 1e-2:
                raise RuntimeError(
                    "⚠️ Mismatch in elite fitness — overwriting stored fitness: "
                    f"{original_fitness} → {recomputed_fitness}"
                )
            if difference > 1e-5:
                print(f"⚠️ Minor mismatch ({difference:.2e}) — tolerating.")

            elite_copy.fitness = recomputed_fitness
            corrected_elites.append(elite_copy)

        population_fitnesses = np.fromiter(
            (individual.fitness for individual in self.population),
            dtype=np.float64,
            count=len(self.population),
        )
        worst_indices = np.argsort(population_fitnesses)[-self.n_elites :]
        for index, elite in zip(worst_indices, corrected_elites):
            self.population[index] = elite
            self.fitnesses[index] = elite.fitness

    def fit(
        self,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
        step_size: int = None,
        xover_rate: float = 0.5,
        mutation_rate: float = 0.5,
        budget_hrs = 1
    ):
        """
        Train the Cartesian Genetic Programming population.
        """
        self.x, self.x_test = train_x, test_x
        self.y, self.y_test = train_y, test_y

        if len(self.x) < 1:
            raise ValueError("Must have at least one input value.")
        if len(self.y) != len(self.x):
            raise ValueError(
                "Must have a 1:1 mapping for input set to output values."
            )
        if step_size is not None and not isinstance(step_size, int):
            raise TypeError(
                "Step size must be either of type `int` or `None`."
            )

        if self.first_submission:
            print("First Time Setup")
            print(f"1D Xover {self.one_d}")
            self.initialize_population()
            self.initialize_xover_index()
            self.model_key_map = {}

            self._get_fitnesses(
                mode="train",
                eval_purpose="search",
            )
            self._get_fitnesses(
                mode="test",
                eval_purpose="test",
            )

            self.metrics = np.zeros(
                (self.max_g + 1, 37),
                dtype=np.float64,
            )
            # Keep the gene-indexed crossover arrays produced by
            # initialize_xover_index(); replacing them with max_p columns made
            # per-gene statistics incompatible with individual counters.
            self.mut_index = np.zeros(
                (self.max_g, self.max_p),
                dtype=np.float64,
            )

            self._record_metrics(0, self.elapsed)
            self._report_generation(0)
            # CORRECTNESS: subsequent fit() calls on the same instance should
            # resume rather than silently replacing the evolved population.
            self.first_submission = False

            genes_per_instruction = self.population[0].arity + 1
            # Preserve the existing mutation-density layout while avoiding repeated
            # len/dictionary lookups.
            model_size = len(self.population[0].xover_index)
            if self.one_d:
                self.mut_index = np.zeros(
                    (self.max_g, model_size),
                    dtype=np.float64,
                )
            else:
                self.mut_index = np.zeros(
                    (
                        self.max_g,
                        model_size * genes_per_instruction,
                    ),
                    dtype=np.float64,
                )
        else:
            self.expand_generations_if_needed(self.max_g)
            print(
                f"Resuming generation {self.current_generation}, "
                f"elapsed={self.elapsed:.2f} seconds"
            )

        # Preserve the original pre-loop evaluations and their accounting effects.
        self._get_fitnesses(
            mode="test",
            mutable=False,
            eval_purpose="test",
        )
        self._get_fitnesses(
            mode="train",
            mutable=True,
            eval_purpose="search",
        )

        n_elites = getattr(self, "n_elites", 1)
        elite_previous = self.elite_selection(n_elites=n_elites)
        assert all(
            isinstance(elite, CGP) for elite in elite_previous
        ), "Elite_selection returned non-CGPs"
        assert all(
            hasattr(elite, "fitness") and elite.fitness is not None
            for elite in elite_previous
        ), "Elite has no fitness"

        elapsed_before_segment = float(getattr(self, "elapsed", 0.0))
        segment_start = time.perf_counter()

        # OPTIMIZATION: cache hot bound methods outside the generation loop.
        get_fitnesses = self._get_fitnesses
        mutate = self._mutate
        record_metrics = self._record_metrics
        report_generation = self._report_generation
        save_checkpoint = self.save_checkpoint

        for generation in range(
            self.current_generation + 1,
            self.max_g + 1,
        ):
            self.current_generation = generation

            selected = self.selection()

            # OPTIMIZATION: all built-in selection methods except competent
            # tournament already return independent copies.  Avoid copying those
            # complete model arrays a second time.
            if self.selection_type == "competent_tournament":
                selected_parents = [
                    deepcopy(parent) for parent in selected
                ]
            else:
                selected_parents = list(selected)

            if self.xover:
                children = self.crossover(
                    selected_parents,
                    xover_rate,
                    generation,
                )
                # Keep this diagnostic evaluation: it updates child fitness and
                # experiment evaluation counters even though its arrays are unused.
                get_fitnesses(
                    pop_list=children,
                    mutable=False,
                    mode="train",
                    eval_purpose="diagnostic",
                )
            else:
                if self.mutation_can_make_children:
                    # OPTIMIZATION: child-making mutation never modifies these
                    # sources, so another full-model deepcopy is unnecessary.
                    children = selected_parents
                else:
                    # CORRECTNESS: no-crossover reproduction must still create
                    # exactly max_c candidates, not one child per parent.
                    parent_count = len(selected_parents)
                    children = np.asarray(
                        [
                            deepcopy(
                                selected_parents[index % parent_count]
                            )
                            for index in range(self.max_c)
                        ],
                        dtype=object,
                    )

            # OPTIMIZATION: the old cloned_parents intermediate deep-copied every
            # parent and was immediately deep-copied again.  One protected copy is
            # sufficient to guarantee model-memory independence.
            protected_parents = [
                deepcopy(parent) for parent in selected_parents
            ]

            for index, (original, protected) in enumerate(
                zip(selected_parents, protected_parents)
            ):
                assert not np.shares_memory(
                    original.model,
                    protected.model,
                ), f"Memory shared at index {index}"

            mutated_children = mutate(
                children,
                generation,
                mutation_rate,
            )

            if self.mutation_can_make_children:
                # OPTIMIZATION: set membership replaces the previous O(P*C)
                # all-pairs ID comparison.
                parent_ids = {parent.id for parent in selected_parents}
                assert all(
                    child.id not in parent_ids
                    for child in mutated_children
                ), "Mutation may be in-place!"

            if isinstance(mutated_children, CGP):
                mutated_children = [mutated_children]
            elif isinstance(mutated_children, np.ndarray):
                mutated_children = mutated_children.tolist()
            else:
                mutated_children = list(mutated_children)

            assert all(
                isinstance(parent, CGP)
                for parent in protected_parents
            ), "Non-CGP in protected_parents"
            assert all(
                isinstance(child, CGP)
                for child in mutated_children
            ), "Non-CGP in mutated_children"
            assert all(
                parent is not None for parent in protected_parents
            ), "protected_parents contains None"
            assert all(
                child is not None for child in mutated_children
            ), "mutated_children contains None"

            self.population = protected_parents + mutated_children

            expected_population_size = self.max_p + self.max_c
            actual_population_size = len(self.population)
            if actual_population_size != expected_population_size:
                raise RuntimeError(
                    "Population size mismatch after reproduction: "
                    f"expected {expected_population_size}, "
                    f"got {actual_population_size}. "
                    f"len(protected_parents)={len(protected_parents)}, "
                    f"len(mutated_children)={len(mutated_children)}"
                )

            assert actual_population_size == len(self.fitnesses)
            assert actual_population_size == len(self.corrs)
            assert actual_population_size == len(self.fitnesses_test)
            assert actual_population_size == len(self.corr_test)

            get_fitnesses(
                mutable=False,
                mode="test",
                eval_purpose="test",
            )
            get_fitnesses(
                mutable=True,
                mode="train",
                eval_purpose="search",
            )

            # The former best_fitness_train/test lists were local, never returned,
            # and never read.  Omitting them prevents unbounded per-run list growth.
            self.elapsed = (
                elapsed_before_segment
                + time.perf_counter()
                - segment_start
            )
            record_metrics(generation, self.elapsed)

            if step_size and generation % step_size == 0:
                report_generation(generation)
                save_checkpoint(
                    filename=self.ckpt_filename,
                    generation=generation,
                )
            if self.elapsed >= (budget_hrs*3600):
                print(f'Timed out in generation {generation}. Elapsed Time: {self.elapsed/3600}/{budget_hrs} Hours.')
                break

        return (
            self.population[np.argmin(self.fitnesses)],
            self.population[np.argmin(self.fitnesses_test)],
        )

    def return_stn(self):
        return self.stn
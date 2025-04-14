import numpy as np
import pickle
import os
import heapq
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy, copy

from cgp_model import CGP
from fitness_functions import correlation
from helper import _validate_int_param, _get_quartiles, pairwise_minkowski_distance, get_score, get_ssd, get_weights, \
    clean_values
from scipy.spatial.distance import cdist
from Bio.Align import PairwiseAligner, Seq


class CartesianGP:
    def _missing_key_error(self, key):
        raise KeyError(f"'{key}' is required but was not provided.")

    def __init__(self, parents=1, children=4, max_generations=100, mutation='Point', selection='Elite',
                 xover=None, fixed_length=True, fitness_function='Correlation', model_parameters=None,
                 function_bank=None, solution_threshold=0.005, checkpoint_filename='checkpoint.pkl', **kwargs):

        # Basic attributes
        self.model_key_map = None
        self.child_id_counter = 0
        self.population = np.empty(parents + children, dtype=object)
        self.fitnesses = np.full(parents + children, np.inf, dtype=np.float64)
        self.best_model = None
        self.function_bank = function_bank
        self.x = None
        self.y = None
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

        # Config values with defaults
        self.mutation_type = mutation.lower()
        self.selection_type = selection.lower()
        self.xover_type = xover.lower() if xover else None
        self.ff_string = fitness_function
        self.semantic = 'semantic' in self.xover_type if xover else False
        self.homologous = 'homologous' in self.xover_type if xover else False

        # Safe defaults
        self.tournament_size = int(kwargs.get('tournament_size', 2))
        self.n_elites = int(kwargs.get('n_elites', 0))
        self.n_points = int(kwargs.get('n_points', 1))
        self.tournament_diversity = kwargs.get('tournament_diversity', True)

        # Mutation fallback
        self.mutation_can_make_children = self.max_p < 2 or kwargs.get('asexual_reproduction', False)

        # Fitness function
        self.fitness_function = {'correlation': correlation}.get(fitness_function.lower())
        if not self.fitness_function:
            raise ValueError(f"Invalid fitness function: {fitness_function}")

        # Selection methods
        self.selection_methods = {
            'elite': self.elite_selection,
            'tournament': self.tournament_selection,
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
            'homologous_semantic_uniform': self._uniform_xover,
            'semantic_n_point': self._n_point_xover,
            'homologous_semantic_n_point': self._n_point_xover,
            'subgraph': self._subgraph_xover
        }
        if self.xover_type:
            if self.xover_type not in self.xover_methods:
                raise ValueError(f"Invalid crossover type: {self.xover_type}")
            self.xover = self.xover_methods[self.xover_type]
        else:
            self.xover = None

        # Validate crossover points
        if self.xover_type and 'n_point' in self.xover_type:
            max_size = self.model_kwargs.get('max_size', 10)
            if self.n_points < 1 or self.n_points > max_size // 2:
                raise ValueError(f"Invalid n_points: {self.n_points}")

        # Metrics and tracking
        self.metrics = np.zeros((self.max_g + 1, 12), dtype=np.float64)
        self.xover_index = {cat: np.zeros((self.max_g, self.max_p)) for cat in ['deleterious', 'neutral', 'beneficial']}
        self.mut_index = np.zeros((self.max_g, self.max_p))

    def save_checkpoint(self, filename="cgp_checkpoint.pkl", generation=None):
        temp_file = filename + ".tmp"
        with open(temp_file, "wb") as f:
            pickle.dump(self.__dict__, f)
        os.replace(temp_file, filename)  # atomic rename to avoid partial writes
        print(f"Checkpoint saved at generation {generation or self.current_generation} in {filename}")

    @classmethod
    def load_checkpoint(cls, filename="cgp_checkpoint.pkl"):
        def _try_load(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        try:
            data = _try_load(filename)
        except EOFError:
            print(f"⚠️ Warning: Failed to load checkpoint '{filename}' (EOFError). Trying backup...")
            tmp_file = filename + ".tmp"
            if os.path.exists(tmp_file):
                try:
                    data = _try_load(tmp_file)
                    print("✅ Loaded from backup:", tmp_file)
                except Exception as e:
                    raise RuntimeError(f"❌ Failed to load from both '{filename}' and backup: {e}")
            else:
                raise RuntimeError(f"❌ Checkpoint corrupted and no backup found: {filename}")

        obj = cls.__new__(cls)  # Don't call __init__!
        obj.__dict__.update(data)

        # Restore method bindings after unpickling
        obj.selection_methods = {
            'elite': obj.elite_selection,
            'tournament': obj.tournament_selection,
            'elite_tournament': obj.elite_tournament_selection,
            'competent_tournament': obj.competent_tournament_selection
        }

        obj.xover_methods = {
            'none': None,
            'n_point': obj._n_point_xover,
            'uniform': obj._uniform_xover,
            'semantic_uniform': obj._uniform_xover,
            'homologous_semantic_uniform': obj._uniform_xover,
            'semantic_n_point': obj._n_point_xover,
            'homologous_semantic_n_point': obj._n_point_xover,
            'subgraph': obj._subgraph_xover
        }

        obj.selection = obj.selection_methods.get(obj.selection_type)
        obj.xover = obj.xover_methods.get(obj.xover_type)

        obj.first_submission = False

        print(f"Checkpoint loaded at generation {obj.current_generation}")
        return obj

    def initialize_population(self):
        """Initialize population in parallel."""
        with ThreadPoolExecutor() as executor:
            future_models = {
                i: executor.submit(CGP, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                                   mutation_type=self.mutation_type, function_bank=self.function_bank,
                                   **self.model_kwargs)
                for i in range(self.max_p)
            }

        for i, future in future_models.items():
            self.population[i] = future.result()

    def elite_selection(self, n_elites=None):
        """Select the top `n_elites` individuals."""
        n_elites = n_elites or self.max_p
        elite_indices = np.argsort(self.fitnesses)[:n_elites]
        return self.population[elite_indices]

    def tournament_selection(self, n_to_select=None):
        """Performs tournament selection with optional diversity enforcement."""
        n_to_select = n_to_select or self.max_p
        new_population = np.empty(n_to_select, dtype=object)

        # Filter out None values from population
        t_pop = self.population[self.population != None]

        if self.tournament_diversity:
            # Use a list to track available individuals to prevent reselection
            available_indices = list(range(len(t_pop)))

        for i in range(n_to_select):
            if self.tournament_diversity and len(available_indices) < self.tournament_size:
                # If fewer individuals remain than tournament size, use all available ones
                contestants_indices = available_indices
            else:
                # Randomly select contestants from available indices
                contestants_indices = np.random.choice(available_indices, size=self.tournament_size, replace=False)

            contestants = t_pop[contestants_indices]

            # Select the best individual based on fitness
            best_index = min(contestants_indices, key=lambda idx: t_pop[idx].fitness)
            new_population[i] = t_pop[best_index]

            # Remove selected individual from available pool if enforcing diversity
            if self.tournament_diversity:
                available_indices.remove(best_index)

        return new_population

    def elite_tournament_selection(self):
        """Combines elite selection with tournament selection, ensuring diversity enforcement."""

        # Step 1: Select elite individuals
        elite_population = self.elite_selection(self.n_elites)
        elite_indices = np.array([np.where(self.population == elite)[0][0] for elite in elite_population])

        # Step 2: Prepare for tournament selection
        remaining_slots = self.max_p - len(elite_population)
        t_pop = self.population[self.population != None]  # Remove None values

        # If enforcing diversity, track available individuals
        if self.tournament_diversity:
            available_indices = list(range(len(t_pop)))
            # Remove elite indices from available pool
            available_indices = [idx for idx in available_indices if idx not in elite_indices]

        new_population = np.empty(remaining_slots, dtype=object)

        # Step 3: Perform tournament selection
        for i in range(remaining_slots):
            if self.tournament_diversity and len(available_indices) < self.tournament_size:
                # Use all remaining individuals if fewer than tournament size
                contestants_indices = available_indices
            else:
                # Randomly select contestants
                contestants_indices = np.random.choice(available_indices, size=self.tournament_size, replace=False)

            contestants = t_pop[contestants_indices]

            # Select the best individual based on fitness
            best_index = min(contestants_indices, key=lambda idx: t_pop[idx].fitness)
            new_population[i] = t_pop[best_index]

            # Remove selected individual to enforce diversity
            if self.tournament_diversity:
                available_indices.remove(best_index)

        # Step 4: Combine elite and tournament-selected individuals
        final_population = np.concatenate((elite_population, new_population))
        return final_population

    def _compute_semantics(self):
        """
        Compute semantics for all individuals in the CGP population at once.
        Uses vectorized operations for better efficiency.

        Returns:
            np.ndarray: A matrix where each row is an individual's output.
        """
        valid_individuals = [ind for ind in self.population if ind is not None]

        if not valid_individuals:  # Handle empty population case
            return np.empty((0, len(self.x)))

        # Compute outputs for all individuals in one go
        semantics = np.vstack([ind(self.x).flatten() for ind in valid_individuals])

        return semantics

    def competent_tournament_selection(self, n_to_select=None):
        """
        Perform competent tournament selection with semantic distance-based scoring,
        using optimized vectorized operations.

        Returns:
            np.ndarray: Array of selected parent CGP models.
        """
        n_to_select = n_to_select or self.max_p

        # Filter out None individuals and get their indices
        t_pop = np.array([ind for ind in self.population if ind is not None])
        parent_indices = np.arange(len(t_pop))

        # Compute all parent semantics at once
        parent_semantics = self._compute_semantics()

        # Compute target vector (ground truth)
        target = np.ravel(self.y)

        # Compute distances to target using vectorized Minkowski distance (p=2)
        target_distances = np.linalg.norm(parent_semantics - target, axis=1)

        # Prepare selected indices and remaining indices
        selected_indices = []
        remaining = list(parent_indices)

        t_size = min(self.tournament_size, len(remaining))
        enforce_diversity = self.tournament_diversity

        while len(selected_indices) < n_to_select and len(remaining) >= t_size:

            # Ensure we have enough candidates to sample
            if len(remaining) < t_size:
                break

            # Sample contestants
            contestants = np.random.choice(remaining, size=t_size, replace=False)

            # Select first parent (lowest target distance)
            first_index = contestants[np.argmin(target_distances[contestants])]

            first_sem = parent_semantics[first_index]
            first_target_dist = target_distances[first_index]

            # Compute semantic distances between first parent and all contestants
            sem_distances = np.linalg.norm(parent_semantics[contestants] - first_sem, axis=1)

            # Compute composite scores
            scores = {
                idx: get_score(
                    first_target_dist,
                    pairwise_minkowski_distance(first_sem, parent_semantics[idx], p=2),
                    target_distances[idx]
                )
                for idx in contestants
            }

            # Select second parent (minimum composite score)
            second_index = min(scores, key=scores.get)

            # Add both parents to selected list
            selected_indices.extend([first_index, second_index])

            # Enforce diversity
            if enforce_diversity:
                remaining = [i for i in remaining if i not in (first_index, second_index)]

        return self.population[np.array(selected_indices[:n_to_select])]

    def crossover(self, parents, xover_rate, gen):
        """Perform crossover to generate children."""
        children = []
        parent_pairs = [(parents[i], parents[i + 1]) for i in range(0, len(parents), 2)]

        while len(children) < self.max_c:
            for i, (p1, p2) in enumerate(parent_pairs):
                if np.random.rand() < xover_rate:
                    if self.xover_type == 'subgraph':
                        c1 = self.xover(p1, p2, gen)
                        c2 = self.xover(p2, p1, gen)
                    else:
                        c1, c2 = self.xover(p1, p2, gen)
                else:
                    c1, c2 = deepcopy(p1), deepcopy(p2)

                # Get parent keys
                p1_key = getattr(p1, 'child_keys', f'Model_{2 * i:03d}_g{gen - 1}')
                p2_key = getattr(p2, 'child_keys', f'Model_{2 * i + 1:03d}_g{gen - 1}')

                # Assign keys to both children
                for child in [c1, c2]:
                    child_key = f'Child_{self.child_id_counter:03d}_g{gen}'
                    child.set_parent_key([p1_key, p2_key])
                    child.set_child_key(child_key)

                    if hasattr(self, 'model_key_map'):
                        self.model_key_map[child_key] = child

                    self.child_id_counter += 1
                    children.append(child)

                    if len(children) >= self.max_c:
                        break
                if len(children) >= self.max_c:
                    break

        return np.array(children, dtype=object)

    def _n_point_xover(self, p1, p2, gen, **kwargs):
        """Performs n-point crossover, supporting semantic and homologous crossover."""

        def get_crossover_points(parent, first_index, xover_weights=None, include_output=False):
            """Generate sorted crossover points for a parent."""
            n_outputs = 0 if include_output else parent.outputs
            possible_indices = np.arange(first_index, len(parent.model) - n_outputs)

            if xover_weights is not None and np.sum(xover_weights) > 0:
                return np.sort(np.random.choice(possible_indices, size=self.n_points, replace=False, p=xover_weights))

            return np.sort(np.random.choice(possible_indices, size=self.n_points, replace=False))

        fb_node = min(p1.first_body_node, p2.first_body_node)
        first_index_p1 = p1.first_body_node
        first_index_p2 = p2.first_body_node

        # Compute weights for semantic and homologous crossover
        weights = None
        if self.semantic:
            vmat_1, vmat_2 = clean_values(p1, self.x), clean_values(p2, self.x)
            weights = get_weights(get_ssd(vmat_1, vmat_2))  # Compute weight distribution

            if self.homologous and not np.all(weights == weights[0]):  # Normalize if needed
                max_weight, min_weight = weights.max(), weights.min()
                weights = (max_weight - weights) / (max_weight - min_weight + 1e-8)

                if not np.allclose(weights.sum(), 1.0):
                    weights /= weights.sum()  # Re-normalize if sum deviates from 1.0

        # Determine crossover points
        if self.fixed_length:
            xover_points = get_crossover_points(p1, first_index_p1, weights)
            xover_points_p1 = xover_points_p2 = xover_points
        else:
            xover_points_p1 = get_crossover_points(p1, first_index_p1, weights)
            xover_points_p2 = get_crossover_points(p2, first_index_p2, weights)

        # Track crossover points for statistical analysis
        p1.xover_index[xover_points_p1 - fb_node] += 1
        p2.xover_index[xover_points_p2 - fb_node] += 1

        # Perform crossover by interleaving segments
        p1_parts = np.split(p1.model, xover_points_p1)
        p2_parts = np.split(p2.model, xover_points_p2)

        child1 = np.concatenate(p1_parts[::2] + p2_parts[1::2])
        child2 = np.concatenate(p2_parts[::2] + p1_parts[1::2])

        # Create new CGP models for children
        c1 = CGP(model=child1, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        c2 = CGP(model=child2, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        return c1, c2

    def _uniform_xover(self, p1, p2, gen, weights=None, **kwargs):
        """Performs uniform crossover, supporting semantic and homologous crossover."""

        n_outputs = 0
        if self.semantic:
            vmat_1, vmat_2 = clean_values(p1, self.x), clean_values(p2, self.x)
            weights = get_weights(get_ssd(vmat_1, vmat_2), epsilon=0.001)  # Compute semantic weights

            if self.homologous and not np.all(weights == weights[0]):  # Normalize if needed
                max_weight, min_weight = weights.max(), weights.min()
                weights = (max_weight - weights) / (max_weight - min_weight + 1e-8)

                if not np.allclose(weights.sum(), 1.0):
                    weights /= weights.sum()  # Re-normalize if sum deviates from 1.0

            n_outputs = p1.outputs  # Adjust for outputs

        fb_node = min(p1.first_body_node, p2.first_body_node)

        assert len(p1.model) == len(p2.model), "Parents in Uniform Xover must have the same length."
        possible_indices = np.arange(fb_node, len(p1.model) - n_outputs)

        if weights is not None:
            #weights = weights[fb_node:len(p1.model) - n_outputs]  # Align weights with crossover region
            mask = weights > 1e-8
            filtered_indices = possible_indices[mask]
            filtered_weights = weights[mask]
            n_swap = min(len(filtered_indices), len(possible_indices) // 2)

            if n_swap > 0:
                swapped_indices = np.random.choice(
                    filtered_indices, size=n_swap, replace=False,
                    p=filtered_weights / filtered_weights.sum())
            else:
                swapped_indices = np.array([], dtype=int)
        else:
            swapped_indices = np.random.choice(
                possible_indices, size=len(possible_indices) // 2, replace=False)

        # Perform crossover with minimal copying
        c1_model, c2_model = p1.model.copy(), p2.model.copy()
        c1_model[swapped_indices] = p2.model[swapped_indices]
        c2_model[swapped_indices] = p1.model[swapped_indices]

        # Track crossover points for statistical analysis
        p1.xover_index[swapped_indices - fb_node] += 1
        p2.xover_index[swapped_indices - fb_node] += 1

        # Create new CGP models for children
        c1 = CGP(model=c1_model, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        c2 = CGP(model=c2_model, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        return c1, c2

    def _subgraph_xover(self, p1: CGP, p2: CGP, gen: int, **kwargs):
        """Performs subgraph crossover using NumPy arrays."""

        def random_node_number(n_i, I=None, n_f=None, m=None):
            """Selects a random node number with constraints."""
            n_r = []  # List of valid random choices

            if n_f is not None:
                if m is not None:
                    n_m = n_f[n_f <= m]
                    if len(n_m) == 0:
                        n_r.append(np.random.randint(0, n_i))
                    else:
                        n_r.append(np.random.choice(n_m))
                else:
                    n_r.append(np.random.choice(n_f))

            if I is not None:
                n_r.append(np.random.choice(I))

            return np.random.choice(n_r)

        def determine_crossover_point(m1, m2):
            """Determines a crossover point for two models."""
            a, b, c, d = min(m1), max(m1), min(m2), max(m2)
            if a >= b:
                b += 1  # Ensure valid range
            if c >= d:
                d += 1
            cp1, cp2 = np.random.randint(a, b), np.random.randint(c, d)
            return min(cp1, cp2)

        def neighborhood_connect(nf, nb, model):
            """Ensures neighborhood connectivity in the new model."""
            model[nb]['Operand0'] = nf
            return model

        def random_active_connect(n_i, n_a, c_p, model):
            """Reconnects inactive nodes in the new model."""
            operand_indices = [f'Operand{i}' for i in range(p1.arity)]
            input_nodes = np.where((model['NodeType'] == 'Constant') | (model['NodeType'] == 'Input'))[0]

            for n in n_a:
                if n > c_p:
                    for operand in operand_indices:
                        if model[n][operand] not in n_a:
                            model[n][operand] = random_node_number(n_i, I=input_nodes, n_f=n_a, m=c_p)

            output_nodes = np.where(model['NodeType'] == 'Output')[0]
            for idx in output_nodes:
                if model[idx]['Operand0'] not in n_a:
                    model[idx]['Operand0'] = random_node_number(n_i, I=input_nodes, n_f=n_a)

            return model

        def ensure_active_nodes(parent):
            """Ensures a model has active nodes before crossover."""
            active_nodes = np.where(parent.get_active_nodes())[0]
            while len(active_nodes) < 1:  # If there are no active nodes, mutate until one is found
                parent.mutate()
                generic = np.zeros((1, parent.inputs))
                parent.fit(generic, generic)
                active_nodes = np.where(parent.get_active_nodes())[0]
            return parent.model.copy(), active_nodes

        # Ensure active nodes for both parents
        g1, m1 = ensure_active_nodes(p1)
        g2, m2 = ensure_active_nodes(p2)

        # Determine crossover point
        xover_point = determine_crossover_point(m1, m2)
        if xover_point <= 0:
            return CGP(model=g1, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                       mutation_type=self.mutation_type)

        # Create new model by swapping sections
        g0 = np.concatenate((g1[:xover_point], g2[xover_point:]))  # NumPy-based slicing and concatenation

        # Determine first function node for reference
        fb_node = min(p1.first_body_node, p2.first_body_node)

        # Extract active nodes surrounding the crossover point
        n_a1 = m1[m1 <= xover_point]
        n_a2 = m2[m2 > xover_point]

        if len(n_a1) > 0 and len(n_a2) > 0:  # Ensure both lists contain active function nodes
            n_f, n_b = n_a1[-1], n_a2[0]  # Identify boundary nodes
            g0 = neighborhood_connect(n_f, n_b, g0)  # Ensure neighborhood connectivity

        # Merge active nodes and reconnect
        n_a = np.concatenate((n_a1, n_a2))
        if len(n_a) > 0:
            g0 = random_active_connect((p1.inputs + len(p1.constants)), n_a, xover_point, g0)

        # Create and return new CGP instance
        g0 = CGP(model=g0, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        g0.xover_index[xover_point - fb_node] += 1
        return g0

    def _mutate(self, models, gen, mutation_rate, verbose=False):
        children = []

        if self.mutation_can_make_children:
            while len(children) < self.max_c:
                for m, model in enumerate(models):
                    if np.random.rand() < mutation_rate or self.max_p == 1:
                        ref_model = deepcopy(model)
                        ref_model.mutate(verbose)

                        child_key = f'Child_{self.child_id_counter:03d}_g{gen}'
                        parent_key = getattr(model, 'child_keys', f'Model_{m:03d}_g{gen - 1}')

                        ref_model.set_parent_key([parent_key])
                        ref_model.set_child_key(child_key)

                        if hasattr(self, 'model_key_map'):
                            self.model_key_map[child_key] = ref_model

                        self.child_id_counter += 1
                        children.append(ref_model)

                    if len(children) >= self.max_c:
                        break

            return np.concatenate((models, np.array(children, dtype=object)))
        else:  # In-place mutation
            for m, model in enumerate(models):
                if np.random.rand() < mutation_rate:
                    model.mutate(verbose)

                    # Assign keys
                    child_key = f'Child_{self.child_id_counter:03d}_g{gen}'
                    parent_key = getattr(model, 'child_keys', f'Model_{m:03d}_g{gen - 1}')

                    model.set_parent_key([parent_key])
                    model.set_child_key(child_key)
                    self.child_id_counter += 1

            return models

    def _get_fitnesses(self):
        """Compute fitnesses for all models."""
        for i in range(len(self.population)):
            if self.population[i] is not None:
                self.fitnesses[i] = self.population[i].fit(self.x, self.y)

    def _group_parents_and_children(self):
        individuals_with_parents = [
            model for model in self.population
            if model is not None and hasattr(model, 'parent_keys') and model.parent_keys is not None
        ]

        parent_child_map = {}

        for child in individuals_with_parents:
            parent_pair = tuple(child.parent_keys)
            parent_child_map.setdefault(parent_pair, []).append(child)

        parent_child_groups = {
            parent_pair: (
                [self.model_key_map[p_key] for p_key in parent_pair if p_key in self.model_key_map],
                children
            )
            for parent_pair, children in parent_child_map.items()
        }

        return parent_child_groups

    def _get_similarity_score(self, model1_obj, model2_obj):
        """
        Computes a similarity score between two models based on structural alignment.

        Args:
            model1_obj: First CGP model object (must have .model as 2D array).
            model2_obj: Second CGP model object (must have .model as 2D array).

        Returns:
            float: A similarity score (higher = more similar).
        """
        m1 = deepcopy(model1_obj.model)
        print(m1)
        m2 = deepcopy(model2_obj.model)

        # Validate dimensionality
        if not (isinstance(m1, np.ndarray) and m1.ndim == 2):
            print(f"Model1 not 2D: shape={getattr(m1, 'shape', None)}")
            return 0.0
        if not (isinstance(m2, np.ndarray) and m2.ndim == 2):
            print(f"Model2 not 2D: shape={getattr(m2, 'shape', None)}")
            return 0.0

        def _map_functions(mo1, mo2):
            unique_functions = np.unique(np.concatenate((mo1[:, 1], mo2[:, 1])))
            function_map = {func: i for i, func in enumerate(unique_functions)}
            mo1[:, 1] = np.vectorize(function_map.get)(mo1[:, 1])
            mo2[:, 1] = np.vectorize(function_map.get)(mo2[:, 1])
            return mo1, mo2

        try:
            m1, m2 = _map_functions(m1, m2)
            sequence1 = m1[:, 1:].flatten().astype(str)
            sequence2 = m2[:, 1:].flatten().astype(str)
        except Exception as e:
            print("Error during sequence mapping or flattening:", e)
            return 0.0

        aligner = PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -2
        aligner.extend_gap_score = -2

        try:
            seq1 = Seq("".join(sequence1))
            seq2 = Seq("".join(sequence2))
            score = aligner.score(seq1, seq2)
            return score if np.isfinite(score) else 0.0
        except Exception as e:
            print("Alignment error:", e)
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

    def _record_metrics(self, gen: int):
        """
        Records key performance metrics for each generation, including fitness statistics
        and similarity measurements.
        """
        # Extract fitness values & active node counts
        fit_list = np.array(list(self.fitnesses))
        active_nodes_list = []
        for p in range(len(self.population)):
            if self.population[p] is not None:
                active_nodes_list.append(self.population[p].count_active_nodes())
        active_nodes_list = np.atleast_1d(active_nodes_list)

        # Compute similarity scores
        #similarity_scores = self._analyze_similarity()
        #similarity_values = np.array([score for _, score in similarity_scores])

        # Find the best model by fitness (lower is better)
        best_model_index = np.argmin(fit_list)
        self.best_model = self.population[best_model_index]

        # Compute quartile statistics efficiently
        fit_statistics = _get_quartiles(fit_list)
        active_nodes_statistics = _get_quartiles(active_nodes_list)
        semantic_diversity = np.nanstd(fit_list)
        """
        # Compute similarity quartiles (handle empty case)
        if len(similarity_values) > 0:
            similarity_quartiles = _get_quartiles(similarity_values)
        else:
            similarity_quartiles = [np.nan] * 5  # Default to NaN if no similarity data
        """
        # Store metrics in the NumPy structured array
        self.metrics[gen] = (
            fit_statistics[0], fit_statistics[1], fit_statistics[2], fit_statistics[3], fit_statistics[4],
            # Fitness stats
            self.best_model.count_active_nodes(),
            active_nodes_statistics[0], active_nodes_statistics[1], active_nodes_statistics[2],
            active_nodes_statistics[3], active_nodes_statistics[4],  # Model size stats
            semantic_diversity  # Semantic diversity
            #similarity_quartiles[0], similarity_quartiles[1], similarity_quartiles[2],
            #similarity_quartiles[3], similarity_quartiles[4]  # Similarity stats
        )

    def save_metrics(self, path=None):
        path = path if path is not None else '.'
        print(f'{path}/statistics.csv')
        # Save the metrics DataFrame only once
        np.savetxt(f'{path}/statistics.csv', self.metrics, delimiter=',')

        # Save the xover_index categories efficiently
        for cat in ['deleterious', 'neutral', 'beneficial']:
            np.savetxt(f'{path}/xover_density_{cat}.csv', self.xover_index[cat].astype(np.int32), delimiter=",")

    def _report_generation(self, g: int):
        # Efficient logging instead of multiple print calls
        print(f'Generation {g}')
        print(f'Fitness statistics:')
        print(self.metrics[g])
        print('################')

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
        # Access the population once and filter individuals with parent keys
        individuals_with_parents = [
            ind for ind in self.population if
            ind is not None and ind.parent_keys is not None and ind.better_than_parents is not None
        ]

        # Create a reference to xover_index for quicker access
        xover_index = self.xover_index

        # Set indices based on 'better_than_parents' value and reset xover_index in one pass
        for ind in individuals_with_parents:
            if ind.better_than_parents == 'beneficial':
                xover_index['beneficial'][gen - 1, :] += ind.xover_index
            elif ind.better_than_parents == 'deleterious':
                xover_index['deleterious'][gen - 1, :] += ind.xover_index
            else:
                xover_index['neutral'][gen - 1, :] += ind.xover_index

            # Reset the xover_index after assignment
            ind.xover_index.fill(0)  # More efficient than np.zeros() if we're resetting the entire array

    def set_max_gens(self, gens):
        self.max_g = gens

    def expand_generations_if_needed(self, new_max_g: int):
        if new_max_g <= self.original_max_g:
            return
        pad = new_max_g - self.original_max_g
        self.metrics = np.pad(self.metrics, ((0, pad), (0, 0)), mode='constant')
        self.mut_index = np.pad(self.mut_index, ((0, pad), (0, 0)), mode='constant')
        for k in self.xover_index:
            self.xover_index[k] = np.pad(self.xover_index[k], ((0, pad), (0, 0)), mode='constant')

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, step_size: int = None,
            xover_rate: float = 0.5, mutation_rate: float = 0.5):
        """
        Trains the Cartesian Genetic Programming model using evolutionary techniques.

        Args:
            train_x (np.ndarray): Training input data.
            train_y (np.ndarray): Training target data.
            step_size (int, optional): Frequency of reporting progress.
            xover_rate (float): Probability of performing crossover.
            mutation_rate (float): Probability of performing mutation.

        Returns:
            CGP: The best evolved model.
        """
        self.x, self.y = train_x, train_y

        # Sanity checks
        if len(self.x) < 1:
            raise ValueError("Must have at least one input value.")
        if len(self.y) != len(self.x):
            raise ValueError("Must have a 1:1 mapping for input set to output values.")
        if step_size is not None and not isinstance(step_size, int):
            raise TypeError("Step size must be either of type `int` or `None`.")

        if self.first_submission: #first time setup
            self.initialize_population()
            self.model_key_map = {}  # Add this at the beginning of fit()

            # After initializing population (gen = 0)
            #for i, model in enumerate(self.population):
            #    if model is not None:
            #        model.set_child_key(f'Model_{i:03d}_g0')
            #        self.model_key_map[model.child_keys] = model

            # Compute fitnesses for the initial population
            self._get_fitnesses()
            self._record_metrics(0)
            self._report_generation(0)

            # Setup mutation and crossover tracking
            model_size = self.model_kwargs.get('max_size', -1)
            if model_size < 0:
                model_size = self.population[0].max_size
            model_size += self.population[0].outputs
            self.xover_index = {key: np.zeros((self.max_g, model_size)) for key in self.xover_index}
            genes_per_instruction = self.population[0].arity + 1  # for the operator
            self.mut_index = np.zeros((self.max_g, model_size * genes_per_instruction))

        else:
            self.expand_generations_if_needed(self.max_g)

        # ✅ **Evolutionary Process**
        for gen in range(self.current_generation, self.max_g + 1):
            self.current_generation = gen
            # **Parent Selection**
            selected_parents = self.selection()

            # **Crossover to Generate Children**
            if self.xover:
                children = self.crossover(selected_parents, xover_rate, gen)
                #for model in children:
                #    self.model_key_map[model.child_keys] = model
            else:
                children = selected_parents  # No crossover, pass parents as is

            # **Compare children to parents & track distributions**
            self._get_fitnesses()
            self._compare_child_parents()
            self._box_distribution(gen)
            # **Mutate Children**
            mutated_children = self._mutate(children, gen, mutation_rate, verbose=False)
            #for model in mutated_children:
            #    self.model_key_map[model.child_keys] = model

            # **Update Population with New Children**
            if self.xover:
                selected_parents = np.concatenate((selected_parents, mutated_children))
                self.population = selected_parents
            else:
                self.population = mutated_children

            # **Compute New Fitnesses**
            self._get_fitnesses()
            # self._clear_fitnesses()
            self._record_metrics(gen)

            # **Step-wise Reporting**
            if step_size and gen % step_size == 0:
                self._report_generation(gen)
                self.save_checkpoint(filename=self.ckpt_filename, generation=gen)

        # ✅ **Return the Best Model**
        return self.population[np.argmin(self.fitnesses)]

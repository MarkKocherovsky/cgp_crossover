import numpy as np
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
                 function_bank=None, solution_threshold=0.005, **kwargs):

        self.population = np.empty(parents + children, dtype=object)  # Array for storing models
        self.fitnesses = np.full(parents + children, np.inf, dtype=np.float64)  # Fitness values
        self.best_model = None
        self.function_bank = function_bank
        self.x = None
        self.y = None
        self.solution_threshold = solution_threshold
        self.max_p = parents
        self.max_c = children
        self.max_g = max_generations
        self.fixed_length = fixed_length
        self.mutation_type = mutation.lower()
        self.selection_type = selection.lower()
        self.xover_type = xover.lower() if xover else None
        self.model_kwargs = model_parameters or {}
        self.semantic = 'semantic' in self.xover_type if xover else ''
        self.homologous = 'homologous' in self.xover_type if xover else ''
        self.ff_string = fitness_function
        self.tournament_size = 2

        if self.max_p < 2 or kwargs.get('asexual_reproduction', False):
            self.mutation_can_make_children = True
        else:
            self.mutation_can_make_children = False

        # Validate fitness function
        self.fitness_function = {'correlation': correlation}.get(fitness_function.lower())
        if not self.fitness_function:
            raise ValueError(f"Invalid fitness function: {fitness_function}")

        # Validate selection method
        self.selection_methods = {
            'elite': self.elite_selection,
            'tournament': self.tournament_selection,
            'elite_tournament': self.elite_tournament_selection,
            'competent_tournament': self.competent_tournament_selection
        }

        self.tournament_diversity = kwargs.get('tournament_diversity', True)  # Default: enforce diversity
        if 'tournament' in self.selection_type:
            self.tournament_size = int(kwargs.get('tournament_size')) or self._missing_key_error('tournament_size')
            if 'elite' in self.selection_type:
                self.n_elites = int(kwargs.get('n_elites')) or self._missing_key_error('n_elites')
        if xover:
            if 'n_point' in self.xover_type or 'semantic' in self.xover_type:
                self.n_points = int(kwargs.get('n_points')) or self._missing_key_error('n_points')
                if self.n_points < 1 or self.n_points > self.model_kwargs['max_size'] // 2:
                    raise ValueError(
                        f"Invalid n_points parameter: {self.n_points}, must be between 1 and {self.model_kwargs['max_size'] // 2}")

        if self.selection_type not in self.selection_methods:
            raise ValueError(f"Invalid selection type: {self.selection_type}")

        self.selection = self.selection_methods[self.selection_type]

        # Setup crossover
        self.xover_methods = {
            'n_point': self._n_point_xover,
            'uniform': self._uniform_xover,
            'semantic_uniform': self._uniform_xover,
            'homologous_semantic_uniform': self._uniform_xover,
            'semantic_n_point': self._n_point_xover,
            'homologous_semantic_n_point': self._n_point_xover,
            'subgraph': self._subgraph_xover
        }
        self.semantic = False
        self.homologous = False

        # Allow modifiers like 'semantic_n_point' or 'homologous_uniform'
        if self.xover_type:
            if self.xover_type not in self.xover_methods:
                raise ValueError(f"Invalid crossover type: {self.xover_type}")

            # Extract modifiers
            if 'semantic' in self.xover_type:
                self.semantic = True
            if 'homologous' in self.xover_type:
                self.homologous = True

            # Set crossover function
            self.xover = self.xover_methods[self.xover_type]
        else:
            self.xover = None  # Allow for no crossover

        # Setup mutation
        if self.mutation_type not in ['point']:
            raise ValueError(f"Invalid mutation type: {self.mutation_type}")

        # Initialize tracking structures
        self.metrics = np.zeros((self.max_g + 1, 17), dtype=np.float64)  # Store min, max, median, etc.
        self.xover_index = {cat: np.zeros((self.max_g, self.max_p)) for cat in ['deleterious', 'neutral', 'beneficial']}
        self.mut_index = np.zeros((self.max_g, self.max_p))

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

                children.append(c1)
                children.append(c2)

        return np.array(children, dtype=object)[:self.max_c]

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
        swapped_indices = np.random.choice(
            possible_indices, size=len(possible_indices) // 2, replace=False, p=weights)

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

    def _mutate(self, models, gen, mutation_rate):
        if self.mutation_can_make_children:  # Reproduction through mutation
            children = []
            while len(children) < self.max_c:
                for m, model in enumerate(models):
                    if np.random.rand() < mutation_rate or self.max_p == 1:
                        # Directly mutate the model (avoid deepcopy if not needed)
                        ref_model = copy(model)  # Assuming a 'copy' method is available for models
                        ref_model.mutate()

                        # Use NumPy-style enumerated key formatting for consistency
                        child_key = f'Child_{m:03d}_g{gen}'

                        # Ensure unique child keys and proper parent key tracking
                        p_key = f'Model_{m:03d}_g{gen - 1}' if child_key not in models else model.parent_keys

                        # Set parent key and store the child
                        ref_model.set_parent_key([p_key])
                        children.append(ref_model)

            models = np.concatenate(
                (models, np.array(children, dtype=object)))  # Efficiently update models with the new children
            return models  # No need to deepcopy models, just return the updated dictionary

        else:  # Mutate models in-place
            for m, model in enumerate(models):
                if np.random.rand() < mutation_rate:
                    model.mutate()
            return models  # No need for deepcopy in this case

    def _get_fitnesses(self):
        """Compute fitnesses for all models."""
        for i in range(len(self.population)):
            if self.population[i] is not None:
                self.fitnesses[i] = self.population[i].fit(self.x, self.y)

    def _group_parents_and_children(self):
        """
        Groups parents with their corresponding children using their parent_keys.
        """
        # Find all individuals with parent keys
        individuals_with_parents = [(p, self.population[p]) for p in range(len(self.population)) if
                                    self.population[p] is not None and self.population[p].parent_keys is not None]

        parent_child_map = {}

        for key, ind in individuals_with_parents:
            parent_pair = tuple(ind.parent_keys)  # Convert parent keys to a tuple
            if parent_pair not in parent_child_map:
                parent_child_map[parent_pair] = []
            parent_child_map[parent_pair].append(ind)  # Store child under parent pair

        # Create a dictionary mapping parent tuples to their children
        parent_child_groups = {
            parent_pair: (
                [self.population.get(p) for p in parent_pair if p in self.population],  # Retrieve parents
                children  # Associated children
            )
            for parent_pair, children in parent_child_map.items()
        }

        return parent_child_groups

    def _get_similarity_score(self, model1, model2):
        """
        Computes a similarity score between two models based on structural alignment.

        Args:
            model1 (np.ndarray): First CGP model.
            model2 (np.ndarray): Second CGP model.

        Returns:
            float: A similarity score.
        """
        m1, m2 = deepcopy(model1), deepcopy(model2)  # Ensure copies to avoid modifying originals

        def _map_functions(mo1, mo2):
            """
            Maps function nodes in two models to unique integer labels for alignment.
            """
            unique_functions = np.unique(np.concatenate((mo1[:, 1], mo2[:, 1])))  # Extract unique operators
            function_map = {func: i for i, func in enumerate(unique_functions)}  # Assign unique indices

            # Apply function mapping to operator columns
            mo1[:, 1] = np.vectorize(function_map.get)(mo1[:, 1])
            mo2[:, 1] = np.vectorize(function_map.get)(mo2[:, 1])

            return mo1, mo2

        m1, m2 = _map_functions(m1, m2)

        # Convert function nodes to sequences
        try:
            sequence1 = m1[:, 1:].flatten().astype(str)
            sequence2 = m2[:, 1:].flatten().astype(str)
        except IndexError:
            return 0  # If insufficient rows, return 0 similarity

        # Compute alignment score using Biopython's PairwiseAligner
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -2
        aligner.extend_gap_score = -2

        seq1, seq2 = Seq("".join(sequence1)), Seq("".join(sequence2))
        score = aligner.score(seq1, seq2)

        return score if np.isfinite(score) else 0.0  # Ensure valid score

    def _analyze_similarity(self):
        """
        Computes the similarity between parents and their best offspring.
        Uses semantic similarity to compare genetic structures.
        """

        parent_child_groups = self._group_parents_and_children()
        similarity_scores = []

        for parent_pair, (parents, children) in parent_child_groups.items():
            if not parents or not children:
                continue  # Skip if no parents or children exist

            # Find the best parent and best child based on fitness
            best_parent = min(parents, key=lambda ind: ind.fitness)
            best_child = min(children, key=lambda ind: ind.fitness)

            # Compute similarity between best parent and best child
            score = self._get_similarity_score(best_parent.model, best_child.model)
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
        similarity_scores = self._analyze_similarity()
        similarity_values = np.array([score for _, score in similarity_scores])

        # Find the best model by fitness (lower is better)
        best_model_index = np.argmin(fit_list)
        self.best_model = self.population[best_model_index]

        # Compute quartile statistics efficiently
        fit_statistics = _get_quartiles(fit_list)
        active_nodes_statistics = _get_quartiles(active_nodes_list)
        semantic_diversity = np.nanstd(fit_list)

        # Compute similarity quartiles (handle empty case)
        if len(similarity_values) > 0:
            similarity_quartiles = _get_quartiles(similarity_values)
        else:
            similarity_quartiles = [np.nan] * 5  # Default to NaN if no similarity data

        # Store metrics in the NumPy structured array
        self.metrics[gen] = (
            fit_statistics[0], fit_statistics[1], fit_statistics[2], fit_statistics[3], fit_statistics[4],
            # Fitness stats
            self.best_model.count_active_nodes(),
            active_nodes_statistics[0], active_nodes_statistics[1], active_nodes_statistics[2],
            active_nodes_statistics[3], active_nodes_statistics[4],  # Model size stats
            semantic_diversity,  # Semantic diversity
            similarity_quartiles[0], similarity_quartiles[1], similarity_quartiles[2],
            similarity_quartiles[3], similarity_quartiles[4]  # Similarity stats
        )

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
        self.initialize_population()

        # Sanity checks
        if len(self.x) < 1:
            raise ValueError("Must have at least one input value.")
        if len(self.y) != len(self.x):
            raise ValueError("Must have a 1:1 mapping for input set to output values.")
        if step_size is not None and not isinstance(step_size, int):
            raise TypeError("Step size must be either of type `int` or `None`.")

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

        # ✅ **Evolutionary Process**
        for gen in range(1, self.max_g + 1):
            # **Parent Selection**
            selected_parents = self.selection()

            # **Crossover to Generate Children**
            if self.xover:
                children = self.crossover(selected_parents, xover_rate, gen)
            else:
                children = selected_parents  # No crossover, pass parents as is

            # **Compare children to parents & track distributions**
            self._get_fitnesses()
            self._compare_child_parents()
            self._box_distribution(gen)
            # **Mutate Children**
            mutated_children = self._mutate(children, gen, mutation_rate)

            # **Update Population with New Children**
            selected_parents = np.concatenate((selected_parents, mutated_children))
            self.population = selected_parents

            # **Compute New Fitnesses**
            self._get_fitnesses()
            # self._clear_fitnesses()
            self._record_metrics(gen)

            # **Step-wise Reporting**
            if step_size and gen % step_size == 0:
                self._report_generation(gen)

        # ✅ **Return the Best Model**
        return self.population[np.argmin(self.fitnesses)]


test = CartesianGP

import numpy as np
import heapq
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from cgp_model import CGP
from fitness_functions import correlation
from helper import _validate_int_param, _get_quartiles, pairwise_minkowski_distance, get_score, get_ssd, get_weights, clean_values
from Bio.Align import PairwiseAligner, Seq

class CartesianGP:
    def __init__(self, parents=1, children=4, max_generations=100, mutation='Point', selection='Elite',
                 xover=None, fixed_length=True, fitness_function='Correlation', model_parameters=None,
                 function_bank=None, solution_threshold=0.005, **kwargs):

        self.population = np.empty(parents + children, dtype=object)  # Array for storing models
        self.fitnesses = np.full(parents + children, np.inf, dtype=np.float64)  # Fitness values
        self.best_model = None
        self.function_bank = function_bank
        self.x = None
        self.y = None
        self.mutation_can_make_children = False
        self.solution_threshold = solution_threshold
        self.max_p = parents
        self.max_c = children
        self.max_g = max_generations
        self.fixed_length = fixed_length
        self.mutation_type = mutation.lower()
        self.selection_type = selection.lower()
        self.xover_type = xover.lower() if xover else None
        self.model_kwargs = model_parameters or {}
        self.semantic = 'semantic' in self.xover_type
        self.homologous = 'homologous' in self.xover_type
        self.ff_string = fitness_function

        # Validate fitness function
        self.fitness_function = {'correlation': correlation}.get(fitness_function.lower())
        if not self.fitness_function:
            raise ValueError(f"Invalid fitness function: {fitness_function}")

        # Validate selection method
        self.selection_methods = {
            'elite': self.elite_selection,
            'tournament': self.tournament_selection,
            'competent tournament': self.competent_tournament_selection
        }
        if self.selection_type not in self.selection_methods:
            raise ValueError(f"Invalid selection type: {self.selection_type}")

        self.selection = self.selection_methods[self.selection_type]

        # Setup crossover
        self.xover_methods = {
            'n_point': self._n_point_xover,
            'uniform': self._uniform_xover
        }
        self.semantic = False
        self.homologous = False

        # Allow modifiers like 'semantic_n_point' or 'homologous_uniform'
        if self.xover_type:
            xover_parts = self.xover_type.split('_')

            # Extract primary crossover type
            xover_name = xover_parts[-1]  # Last part should be the actual xover method

            if xover_name not in self.xover_methods:
                raise ValueError(f"Invalid crossover type: {self.xover_type}")

            # Extract modifiers
            if 'semantic' in xover_parts:
                self.semantic = True
            if 'homologous' in xover_parts:
                self.homologous = True

            # Set crossover function
            self.xover = self.xover_methods[xover_name]
        else:
            self.xover = None  # Allow for no crossover

        # Setup mutation
        if self.mutation_type not in ['point']:
            raise ValueError(f"Invalid mutation type: {self.mutation_type}")

        # Initialize tracking structures
        self.metrics = np.zeros((self.max_g + 1, 5), dtype=np.float64)  # Store min, max, median, etc.
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

        self._get_fitnesses()
        self._record_metrics(0)

    def elite_selection(self, n_elites=None):
        """Select the top `n_elites` individuals."""
        n_elites = n_elites or self.max_p
        elite_indices = np.argsort(self.fitnesses)[:n_elites]
        return self.population[elite_indices]

    def tournament_selection(self, n_to_select=None):
        """Performs tournament selection."""
        n_to_select = n_to_select or self.max_p
        new_population = np.empty(n_to_select, dtype=object)

        for i in range(n_to_select):
            contestants = np.random.choice(self.population[:self.max_p], size=3, replace=False)
            new_population[i] = min(contestants, key=lambda ind: ind.fitness)

        return new_population

    def competent_tournament_selection(self, n_to_select=None):
        """
        Perform competent tournament selection with semantic distance-based scoring,
        using NumPy arrays for efficiency.

        Returns:
            np.ndarray: Array of selected parent CGP models.
        """
        # Default number to select is the number of parents
        n_to_select = n_to_select or self.max_p

        # Assume population is stored in a NumPy array (first max_p individuals are parents)
        parent_indices = np.arange(self.max_p)

        # Compute semantic vectors (assumed to be 1D NumPy arrays) for all parents.
        # For instance, each CGP model could expose a method _compute_semantics()
        parent_semantics = np.array([parent._compute_semantics(self.x) for parent in self.population[parent_indices]])

        # Compute target vector (ground truth) as 1D NumPy array
        target = np.ravel(self.y)

        # Compute each parent’s distance to the target using a pairwise Minkowski function.
        # (pairwise_minkowski_distance is assumed to accept two 1D arrays and return a scalar distance)
        target_distances = np.array([pairwise_minkowski_distance(sem, target, p=2)
                                     for sem in parent_semantics])

        # Prepare an array to hold the indices of selected individuals
        selected_indices = []

        # For efficient lookup, create a mutable list of remaining parent indices
        remaining = list(parent_indices)

        # Tournament parameters: we require tournament_size and diversity flag to be set
        t_size = self.tournament_size  # must have been set by _setup_tournament_and_elite()
        enforce_diversity = self.tournament_diversity

        while len(selected_indices) < n_to_select and len(remaining) >= t_size:
            # Randomly sample tournament contestants from remaining indices
            contestants = np.random.choice(remaining, size=t_size, replace=False)
            # Select the contestant with the best (lowest) fitness
            first_index = contestants[np.argmin(self.fitnesses[contestants])]

            first_sem = parent_semantics[first_index]
            first_target_dist = target_distances[first_index]

            # For each contestant, compute the semantic distance to the first parent’s semantics,
            # and compute a composite score using a helper function get_score().
            # get_score() is assumed to combine the parent's target distance,
            # the distance between semantics, and the contestant's target distance.
            scores = {}
            for idx in contestants:
                sem_dist = pairwise_minkowski_distance(first_sem, parent_semantics[idx], p=2)
                # get_score is assumed to return a scalar (the lower the score, the better)
                scores[idx] = get_score(first_target_dist, sem_dist, target_distances[idx])

            # Select the contestant with the minimum score as the second parent
            second_index = min(scores, key=scores.get)

            # Add both indices to the selected list
            selected_indices.extend([first_index, second_index])

            # Optionally enforce diversity by removing selected indices from remaining pool
            if enforce_diversity:
                remaining = [i for i in remaining if i not in (first_index, second_index)]

        # If we selected more than needed, take only the first n_to_select.
        selected_indices = np.array(selected_indices[:n_to_select])
        return self.population[selected_indices]

    def elite_tournament_selection(self, n_elites=None):
        """Combines elite selection with tournament selection for diversity."""

        # Step 1: Select elite individuals
        elite_population = self.elite_selection(n_elites)
        elite_indices = np.array([np.where(self.population == elite)[0][0] for elite in elite_population])

        # Step 2: Perform tournament selection for the remaining slots
        remaining_slots = self.max_p - len(elite_population)
        remaining_population = self.tournament_selection(remaining_slots)

        # Step 3: Combine results
        final_population = np.concatenate((elite_population, remaining_population))
        return final_population

    def crossover(self, parents, xover_rate, gen):
        """Perform crossover to generate children."""
        children = np.empty(self.max_c, dtype=object)
        parent_pairs = [(parents[i], parents[i + 1]) for i in range(0, len(parents), 2)]

        for i, (p1, p2) in enumerate(parent_pairs):
            if np.random.rand() < xover_rate:
                c1, c2 = self.xover(p1, p2, gen)
            else:
                c1, c2 = deepcopy(p1), deepcopy(p2)

            children[i * 2], children[i * 2 + 1] = c1, c2

        return children


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

            if self.homologous:
                max_weight = weights.max()
                min_weight = weights.min()
                weights = (max_weight - weights) / (max_weight - min_weight + 1e-8)  # Normalize

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
            operand_indices = [f'Operand{i}' for i in range(self.arity)]
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
                parent.fit([0, 0, 0], [0, 0, 0])
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

    def _get_fitnesses(self):
        """Compute fitnesses for all models."""
        for i in range(self.max_p + self.max_c):
            self.fitnesses[i] = self.population[i].fit(self.x, self.y)

    def _group_parents_and_children(self):
        """
        Groups parents with their corresponding children using their parent_keys.
        """
        # Find all individuals with parent keys
        individuals_with_parents = [(key, ind) for key, ind in self.population.items() if ind.parent_keys is not None]

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
        fit_list = np.array(list(self.fitnesses.values()))
        active_nodes_list = np.array([self.population[p].count_active_nodes() for p in self.population])

        # Compute similarity scores
        similarity_scores = self._analyze_similarity()
        similarity_values = np.array([score for _, score in similarity_scores])

        # Find the best model by fitness (lower is better)
        best_model_index = np.argmin(fit_list)
        self.best_model = list(self.population.values())[best_model_index]

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

        # ✅ **Parallelized Population Initialization**
        with ThreadPoolExecutor() as executor:
            self.population = {
                f'Model_{p}': executor.submit(CGP, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                                              mutation_type=self.mutation_type, function_bank=self.function_bank,
                                              **self.model_kwargs)
                for p in range(self.max_p)
            }

        # Wait for all model initializations to complete
        self.population = {key: model.result() for key, model in self.population.items()}

        # Compute fitnesses for the initial population
        self._get_fitnesses()
        self._record_metrics(0)
        self._report_generation(0)

        # Setup mutation and crossover tracking
        model_size = self.model_kwargs.get('max_size', -1)
        if model_size < 0:
            model_size = self.population['Model_0'].max_size
        model_size += self.population['Model_0'].outputs
        self.xover_index = {key: np.zeros((self.max_g, model_size)) for key in self.xover_index}
        genes_per_instruction = self.population['Model_0'].arity + 1  # for the operator
        self.mut_index = np.zeros((self.max_g, model_size * genes_per_instruction))

        # ✅ **Evolutionary Process**
        for gen in range(1, self.max_g + 1):
            # **Parent Selection**
            selected_parents = self.selection()

            # **Crossover to Generate Children**
            if self.xover_type:
                children = self.crossover(selected_parents, xover_rate, gen)
            else:
                children = selected_parents  # No crossover, pass parents as is

            # **Compare children to parents & track distributions**
            self._compare_child_parents()
            self._box_distribution(gen)

            # **Mutate Children**
            mutated_children = self._mutate(children, gen, mutation_rate)

            # **Update Population with New Children**
            selected_parents.update(mutated_children)
            self.population = selected_parents

            # **Compute New Fitnesses**
            self._get_fitnesses()
            self._clear_fitnesses()
            self._record_metrics(gen)

            # **Step-wise Reporting**
            if step_size and gen % step_size == 0:
                self._report_generation(gen)

        # ✅ **Return the Best Model**
        return self.population[np.argmin(self.fitnesses)]


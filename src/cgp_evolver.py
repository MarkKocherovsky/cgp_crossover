from copy import copy
from copy import deepcopy

import numpy as np
import pandas as pd
import heapq
from Bio.Align import PairwiseAligner, Seq
from concurrent.futures import ThreadPoolExecutor


from cgp_model import CGP
from fitness_functions import correlation
# from helper import *
from helper import _validate_int_param, _get_quartiles, _split, pairwise_minkowski_distance, clean_values, get_score, \
    get_ssd, get_weights


class CartesianGP:
    def _missing_key_error(self, key):
        raise KeyError(f"'{key}' is required but was not provided.")

    def _setup_n_point_xover(self, kwargs):
        if 'n_point' in self.xover_type or 'semantic' in self.xover_type:
            self.n_points = _validate_int_param(
                'n_points',
                kwargs.get('n_points') or self._missing_key_error('n_points'),
                min_val=1,
                max_val=self.model_kwargs['max_size'] // 2 if self.model_kwargs is not None else 2
            )

    def _setup_tournament_and_elite(self, kwargs):
        if 'tournament' in self.selection_type:
            self.tournament_size = _validate_int_param(
                'tournament_size',
                kwargs.get('tournament_size') or self._missing_key_error('tournament_size'),
                min_val=1,
                max_val=self.max_p,
            )

            self.tournament_diversity = kwargs.get('tournament_diversity', True)  # Default: enforce diversity

            if 'elite' in self.selection_type:
                self.n_elites = _validate_int_param(
                    'n_elites',
                    kwargs.get('n_elites') or self._missing_key_error('n_elites'),
                    min_val=1,
                    max_val=self.max_p,
                )

    def __init__(self, parents: int = 1, children: int = 4,
                 max_generations: int = 100, mutation: str = 'Basic', selection: str = 'Elite',
                 xover: str = None, fixed_length: bool = True, fitness_function: str = 'Correlation',
                 model_parameters: dict = None, function_bank=None, solution_threshold=0.005, **kwargs):
        # extract from header
        self.population = {}
        self.fitnesses = {}
        self.y = None
        self.x = None
        self.mutation_can_make_children = None
        self.function_bank = function_bank
        self.best_model = None
        self.weights = None
        self.semantic = False
        self.xover_index = {'deleterious': None, 'neutral': None, 'beneficial': None}
        self.mut_index = None
        self.max_p = parents
        self.max_c = children
        self.max_g = max_generations
        self.mutation_type = mutation.lower()
        self.xover_type = xover.lower()
        self.model_kwargs = model_parameters
        self.fixed_length = fixed_length
        self.selection_type = selection.lower()
        self.solution_threshold = solution_threshold
        self.ff_string = fitness_function

        metrics = ['Min Fitness', 'Fitness 1st Quartile', 'Median Fitness', 'Fitness 3rd Quartile',
                   'Max Fitness',
                   'Best Model Size', 'Min Model Size', 'Model Size 1st Quartile', 'Median Model Size',
                   'Model Size 3rd Quartile', 'Max Model Size', 'Semantic Diversity', 'Min Similarity',
                   'Similarity 1st Quartile',
                   'Median Similarity', 'Similarity 3rd Quartile', 'Max Similarity']

        metrics_initialized = np.zeros((self.max_g + 1, len(metrics)))
        self.metrics = pd.DataFrame(metrics_initialized, columns=metrics)

        # Sanitize inputs
        if self.max_p < 1:
            raise ValueError("Must have at least one parent.")
        if self.max_c < 1:
            raise ValueError("Must have at least one child.")

        # Get fitness, mutation, xover, and selection
        possible_fitness_functions = {
            'correlation': correlation
        }
        possible_selection_functions = {
            'elite': self.elite_selection,
            'tournament': self.tournament_selection,
            'elite_tournament': self.elite_tournament_selection,
            'competent tournament': self.competent_tournament_selection
        }

        possible_xover_functions = {
            'n_point': self._n_point_xover,
            'uniform': self._uniform_xover,
            'subgraph': self._subgraph_xover,
            'semantic_n_point': self._n_point_xover,
            'semantic_uniform': self._uniform_xover,
            # 'semantic uniform': self._semantic_uniform_xover
        }
        if self.mutation_type not in ['point']:
            raise ValueError(f"Invalid mutation type: {self.mutation_type}")
        if self.selection_type not in possible_selection_functions:
            raise ValueError(f"Invalid selection type: {self.selection_type}")
        if self.xover_type.lower() not in possible_xover_functions:
            raise ValueError(f"Invalid crossover type: {self.xover_type}")
        if 'semantic' in self.xover_type:
            self.semantic = True
        if self.model_kwargs is not None and not isinstance(self.model_kwargs, dict):
            raise TypeError("Model parameters must be a dictionary.")
        self.fitness_function = possible_fitness_functions.get(fitness_function.lower())
        self.xover = possible_xover_functions.get(self.xover_type.lower()) if self.xover_type is not None else None
        self.selection = possible_selection_functions.get(self.selection_type.lower())
        self._setup_tournament_and_elite(kwargs)
        self._setup_n_point_xover(kwargs) if ('n_point' in self.xover_type) else None
        if self.fitness_function is None:
            raise KeyError(f"{fitness_function} is an invalid fitness function.")
        if self.selection is None:
            raise KeyError(f"{self.selection_type} is an invalid selection operator.")

        if kwargs.get('mutation_breeding', False) or (self.max_p == 1 and self.max_c > 1) or self.xover_type is None:
            self.mutation_can_make_children = True
        else:
            self.mutation_can_make_children = False

    """
    Selection Algorithms
    """

    def elite_selection(self, n_elites=None):
        """
        Selects the top `n_elites` individuals based on fitness.
        """
        n_elites = n_elites or self.max_p  # Default to max_p if n_elites is not specified
        # Use `heapq.nsmallest` for efficient top-k selection
        top_keys = heapq.nsmallest(n_elites, self.fitnesses, key=self.fitnesses.get)
        # Use a dictionary comprehension to construct the elite population
        return {key: self.population[key] for key in top_keys}

    def tournament_selection(self, n_to_select=None, population_keys=None):
        """
        Performs tournament selection to pick `n_to_select` individuals.
        """
        n_to_select = n_to_select or self.max_p  # Default to max_p if n_to_select is not specified
        population_keys = population_keys or list(self.population.keys())  # Default to all population keys
        new_population = {}
        fitness_getter = self.fitnesses.get  # Cache the getter for performance

        # Precompute the size of population keys for reuse
        keys_size = len(population_keys)

        while len(new_population) < n_to_select:
            # Select contestants for the tournament
            if keys_size > self.tournament_size:
                contestants = np.random.choice(population_keys, size=self.tournament_size, replace=False)
            else:
                contestants = population_keys  # Use all available keys if not enough

            # Find the best individual directly without creating an intermediate dictionary
            best_individual = min(contestants, key=fitness_getter)

            # Add the best individual to the new population

            new_population[best_individual] = self.population[best_individual]

            # Remove the selected individual for diversity, if needed
            if self.tournament_diversity:
                population_keys.remove(best_individual)
                keys_size -= 1  # Update the size of population_keys

        return new_population

    """
        Tomasz P Pawlak and Krzysztof Krawiec. Competent Geometric Semantic
        Genetic Programming for Symbolic Regression and Boolean Function Synthesis.
        Evolutionary Computation, 26(2):177–212, 2018
    """

    def competent_tournament_selection(self, n_to_select=None, population_keys=None):
        """
        Perform competent tournament selection with semantic distance-based scoring.
        """
        # Set default values
        n_to_select = n_to_select or self.max_p
        population_keys = population_keys or list(self.population.keys())
        target = self.y.flatten()  # Ground truth

        # Precompute parent semantics and target distances
        parent_semantic_list = self._determine_parent_semantics()
        target_distances = {key: pairwise_minkowski_distance(predictions, target, p=2)
                            for key, predictions in parent_semantic_list.items()}

        new_population = {}
        fitness_getter = self.fitnesses.get  # Cache fitness getter for performance

        while len(new_population) < n_to_select:
            # Select the first parent via regular tournament selection
            first_parent_key = min(
                np.random.choice(population_keys, size=self.tournament_size, replace=False),
                key=fitness_getter
            )
            first_parent = self.population[first_parent_key]
            parent_semantics = parent_semantic_list[first_parent_key]
            parent_distance_to_target = target_distances[first_parent_key]

            # Sample tournament contestants
            if len(population_keys) > self.tournament_size:
                contestants = np.random.choice(population_keys, size=self.tournament_size, replace=False)
            else:
                contestants = population_keys  # Use all available keys if not enough

            # Compute scores for all contestants in a vectorized manner
            contestant_distances = {key: pairwise_minkowski_distance(parent_semantics, parent_semantic_list[key], p=2)
                                    for key in contestants}
            scores = {
                key: get_score(parent_distance_to_target, contestant_distances[key], target_distances[key])
                for key in contestants
            }

            # Select the contestant with the best score
            second_parent_key = min(scores, key=scores.get)
            new_population[first_parent_key] = first_parent
            new_population[second_parent_key] = self.population[second_parent_key]

            # Remove selected contestant from future tournaments
            population_keys.remove(second_parent_key)

            # Remove selected contestants from population_keys if diversity is required
            if self.tournament_diversity:
                population_keys.remove(first_parent_key)
                population_keys.remove(second_parent_key)
        return new_population

    def _determine_parent_semantics(self):
        """
        Precompute unique parent semantics by evaluating predictions for each parent.
        """
        # Precompute and flatten predictions, avoiding redundant computations
        parent_predictions = {}
        seen_predictions = set()  # Use a set to efficiently check for duplicates

        for p, parent in self.population.items():
            predictions = parent(self.x).flatten()
            predictions_tuple = tuple(predictions)  # Convert to tuple for hashing
            if predictions_tuple not in seen_predictions:
                parent_predictions[p] = predictions
                seen_predictions.add(predictions_tuple)

        return parent_predictions

    def elite_tournament_selection(self, n_elites=None):
        """
        Combines elite selection and tournament selection to create the new population.

        Args:
            n_elites (int): Number of elites to select.
        Returns:
            dict: A new population combining elites and tournament-selected individuals.
        """
        # Determine the number of elites (default: one)
        n_elites = n_elites or 1

        # Step 1: Select elites
        elite_population = self.elite_selection(n_elites)
        elite_keys = set(elite_population.keys())

        # Step 2: Perform tournament selection for the remaining slots
        remaining_slots = self.max_p - n_elites
        population_keys = [key for key in self.population.keys() if key not in elite_keys]

        if self.tournament_diversity:
            # Remove selected elites from the tournament pool for diversity
            remaining_population = self.tournament_selection(
                n_to_select=remaining_slots,
                population_keys=population_keys
            )
        else:
            # Tournament selection without removing elites from the pool
            remaining_population = self.tournament_selection(
                n_to_select=remaining_slots,
                population_keys=list(self.population.keys())
            )

        # Step 3: Combine the results and update the population
        elite_population.update(remaining_population)
        return elite_population

    """
    Crossover Operators
    """

    def crossover(self, selected_parents: dict, xover_rate: float, gen: int):
        """
        Perform crossover to generate a new population of children.

        Args:
            selected_parents (dict): Dictionary of selected parent individuals.
            xover_rate (float): Probability of performing crossover.
            gen (int): Current generation.

        Returns:
            dict: A dictionary of children generated through crossover.
        """
        # Flatten parent data into lists for efficient processing
        parent_list = list(selected_parents.values())
        name_list = list(selected_parents.keys())

        # Ensure the number of parents is even for pairing
        if len(parent_list) % 2 != 0:
            parent_list.pop()
            name_list.pop()

        children = {}
        n_c = 0

        # Perform crossover in pairs
        for p in range(0, len(parent_list), 2):
            if len(children) >= self.max_c:
                break  # Stop when the maximum number of children is reached

            p1, p2 = parent_list[p], parent_list[p + 1]
            if np.random.rand() < xover_rate:
                # Perform crossover
                if self.xover_type.lower() == 'subgraph':
                    c1 = self.xover(p1, p2, gen)
                    c2 = self.xover(p2, p1, gen)
                else:
                    c1, c2 = self.xover(p1, p2, gen=gen, weights=self.weights)
            else:
                # Copy parents without crossover
                c1, c2 = copy(p1), copy(p2)

            # Assign children to the dictionary
            children[f'Child_{n_c}_g{gen}'] = c1
            children[f'Child_{n_c + 1}_g{gen}'] = c2

            # Set parent keys for lineage tracking
            c1.set_parent_key([name_list[p], name_list[p + 1]])
            c2.set_parent_key([name_list[p], name_list[p + 1]])

            n_c += 2

        return children

    def _n_point_xover(self, p1, p2, **kwargs):
        """
        Performs n-point crossover.
        """

        def get_crossover_points(parent, first_index, xover_weights=None, include_output=False):
            """Generate sorted crossover points for a parent."""
            n_outputs = 0 if include_output else parent.outputs
            possible_indices = np.arange(first_index, len(parent.model) - n_outputs)
            if xover_weights is not None and np.sum(xover_weights) > 0:
                return np.sort(np.random.choice(possible_indices, size=self.n_points, replace=False, p=xover_weights))
            return np.sort(np.random.choice(possible_indices, size=self.n_points, replace=False))

        def interleave_parts(p1_sections, p2_sections):
            """Efficiently interleave parts from two parents."""
            child1 = pd.concat(p1_sections[::2] + p2_sections[1::2], ignore_index=False)
            child2 = pd.concat(p2_sections[::2] + p1_sections[1::2], ignore_index=False)
            return child1, child2

        # Generate crossover points
        fb_node = min(p1.first_body_node, p2.first_body_node)
        first_index_p1 = p1.model['NodeType'].eq('Function').idxmax()
        first_index_p2 = p2.model['NodeType'].eq('Function').idxmax()

        weights = None
        if self.semantic:
            vmat_1, vmat_2 = clean_values(p1, self.x), clean_values(p2, self.x)
            weights = get_weights(get_ssd(vmat_1, vmat_2))

        if self.fixed_length:
            xover_points = get_crossover_points(p1, first_index_p1, weights)
            xover_points_p1 = xover_points_p2 = xover_points
        else:
            xover_points_p1 = get_crossover_points(p1, first_index_p1, weights)
            xover_points_p2 = get_crossover_points(p2, first_index_p2, weights)

        p1.xover_index[xover_points_p1 - fb_node] += 1
        p2.xover_index[xover_points_p2 - fb_node] += 1

        # Split and interleave parts
        p1_parts = _split(p1.model, xover_points_p1)
        p2_parts = _split(p2.model, xover_points_p2)
        child1, child2 = interleave_parts(p1_parts, p2_parts)

        # Create children
        c1 = CGP(model=child1, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        c2 = CGP(model=child2, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        return c1, c2

    def _uniform_xover(self, p1, p2, gen, weights=None, **kwargs):
        """
        Performs uniform crossover.
        """
        n_outputs = 0
        if self.semantic:
            vmat_1, vmat_2 = clean_values(p1, self.x), clean_values(p2, self.x)
            weights = get_weights(get_ssd(vmat_1, vmat_2), epsilon=0.001)
            n_outputs = p1.outputs

        fb_node = min(p1.first_body_node, p2.first_body_node)
        assert len(p1.model) == len(p2.model), "Parents in Uniform Xover must have the same length."

        possible_indices = np.arange(fb_node, len(p1.model) - n_outputs)
        swapped_indices = np.random.choice(
            possible_indices, size=len(possible_indices) // 2, replace=False, p=weights)

        # Perform crossover with minimal deep copying
        c1_model, c2_model = p1.model.copy(), p2.model.copy()
        c1_model.loc[swapped_indices] = p2.model.loc[swapped_indices]
        c2_model.loc[swapped_indices] = p1.model.loc[swapped_indices]

        # Update crossover indices
        p1.xover_index[swapped_indices - fb_node] += 1
        p2.xover_index[swapped_indices - fb_node] += 1

        # Create children
        c1 = CGP(model=c1_model, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        c2 = CGP(model=c2_model, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        return c1, c2

    # Kalkreuth, R. (2021). Reconsideration and extension of Cartesian genetic programming
    # (Doctoral dissertation, Technical University of Dortmund, Germany).
    def _subgraph_xover(self, p1: CGP, p2: CGP, gen: int, **kwargs):
        def random_node_number(n_i, I=None, n_f=None, m=None):
            n_r = []  # random list and function node numbers
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
            a, b, c, d = min(m1), max(m1), min(m2), max(m2)
            if a >= b:
                b += 1  # should always return a
            if c >= d:
                d += 1
            cp1, cp2 = np.random.randint(a, b), np.random.randint(c, d)
            return min(cp1, cp2)

        def neighborhood_connect(nf, nb, model):
            model.loc[nb, 'Operand0'] = nf
            return model

        def random_active_connect(n_i, n_a, c_p, model):
            operand_list = model.filter(regex='Operand').columns
            input_nodes = model[
                (model['NodeType'] == 'Constant') | (model['NodeType'] == 'Input')
                ].index.to_list()

            for n in n_a:
                if n > c_p:
                    for operand in operand_list:
                        if model.loc[n, operand] not in n_a:
                            model.loc[n, operand] = random_node_number(n_i, I=input_nodes, n_f=n_a, m=c_p)
            for index, row in model[model['NodeType'] == 'Output'].iterrows():
                if row['Operand0'] not in n_a:
                    model.at[index, 'Operand0'] = random_node_number(input_nodes, n_a)

            return model

        def ensure_active_nodes(parent):
            active_nodes = np.array(parent.get_active_nodes().index.to_list())
            while len(active_nodes) < 1:  # If there are no active nodes, mutate until you get one
                parent.mutate()
                parent.fit([0, 0, 0], [0, 0, 0])
                active_nodes = np.array(parent.get_active_nodes().index.to_list())
            return parent.model.copy(), active_nodes

        # Ensure active nodes for both parents
        g1, m1 = ensure_active_nodes(p1)
        g2, m2 = ensure_active_nodes(p2)

        xover_point = determine_crossover_point(m1, m2)  # determine the crossover point
        if xover_point <= 0:
            xover_point = 0
        g0 = pd.concat([g1.loc[0:xover_point - 1], g2.loc[xover_point:]], ignore_index=True)  # Create new model
        fb_node = min(p1.first_body_node, p2.first_body_node)

        n_a1 = m1[m1 <= xover_point]
        n_a2 = m2[m2 > xover_point]

        if len(n_a1) > 0 and len(n_a2) > 0:  # check if both lists contain active function node numbers
            n_f, n_b = n_a1[-1], n_a2[0]  # determine the nodes immediately surrounding the crossover point
            g0 = neighborhood_connect(n_f, n_b, g0)  # perform neighborhood connect

        n_a = np.concat((n_a1, n_a2))
        if len(n_a) > 0:  # check if any function nodes are active
            g0 = random_active_connect((p1.inputs + len(p1.constants)), n_a, xover_point, g0)  # random active connect

        g0 = CGP(model=g0, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        g0.xover_index[xover_point - fb_node] += 1
        return g0

    def _get_fitnesses(self):
        for p in self.population:
            self.fitnesses[p] = self.population[p].fit(self.x, self.y)

    def _clear_fitnesses(self):
        """Remove fitnesses not associated with the current population."""
        population_set = set(self.population)  # Convert to set for O(1) lookups
        self.fitnesses = {key: value for key, value in self.fitnesses.items() if key in population_set}

    """
    Data Recording/Reporting
    """

    def _group_parents_and_children(self):
        """Group parents and their children."""
        # Collect individuals with valid parent keys
        individuals_with_parents = [
            (key, ind) for key, ind in self.population.items() if ind.parent_keys is not None
        ]

        # Group children by their parent pairs
        parent_child_map = {}
        for key, ind in individuals_with_parents:
            parent_pair = tuple(ind.parent_keys)
            if parent_pair not in parent_child_map:
                parent_child_map[parent_pair] = []
            parent_child_map[parent_pair].append(ind)

        # Construct the parent-child groups
        parent_child_groups = {
            parent_pair: (
                [self.population.get(p) for p in parent_pair if p in self.population],  # Retrieve parent individuals
                children  # Associated children
            )
            for parent_pair, children in parent_child_map.items()
        }

        return parent_child_groups

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
            ind for ind in self.population.values() if
            ind.parent_keys is not None and ind.better_than_parents is not None
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

    def _analyze_similarity(self):
        parent_child_groups = self._group_parents_and_children()

        # Preallocate similarity_scores list
        similarity_scores = []

        # Access population values once and minimize repeated function calls
        for parent_pair, (parents, children) in parent_child_groups.items():
            if not parents or not children:
                continue  # Skip groups without parents or children

            # Find the best parent by fitness (assume higher fitness is better)
            best_parent = min(parents, key=lambda ind: ind.fitness)

            # Find the best child by minimum fitness
            best_child = min(children, key=lambda ind: ind.fitness)

            # Calculate similarity score between the best parent and best child
            score = self._get_similarity_score(best_parent.model, best_child.model)

            # Append the result
            similarity_scores.append((parent_pair, score))

        return similarity_scores

    def _record_metrics(self, gen: int):
        # Get lists of fitness values and active node counts in a single loop
        fit_list = np.array(list(self.fitnesses.values()))
        len_active_nodes_list = np.array([self.population[p].count_active_nodes() for p in self.population])

        # Analyze similarity scores (already optimized)
        similarity_scores = self._analyze_similarity()
        scores = np.array([score for _, score in similarity_scores])

        # Get the best model index (optimized by using np.argmin directly)
        best_model_index = np.argmin(fit_list)
        self.best_model = list(self.population.values())[best_model_index]

        # Do stats (precompute quartiles and standard deviation in one go)
        fit_statistics = _get_quartiles(fit_list)
        active_nodes_statistics = _get_quartiles(len_active_nodes_list)
        semantic_diversity = np.nanstd(fit_list)
        similarity = _get_quartiles(scores) if gen > 0 else [np.nan] * 5

        # Prepare fitness statistics labels in a tuple format to avoid re-creation in the loop
        fitness_labels = (
            ('Min Fitness', fit_statistics[0]),
            ('Fitness 1st Quartile', fit_statistics[1]),
            ('Median Fitness', fit_statistics[2]),
            ('Fitness 3rd Quartile', fit_statistics[3]),
            ('Max Fitness', fit_statistics[4]),
            ('Best Model Size', self.best_model.count_active_nodes()),
            ('Min Model Size', active_nodes_statistics[0]),
            ('Model Size 1st Quartile', active_nodes_statistics[1]),
            ('Median Model Size', active_nodes_statistics[2]),
            ('Model Size 3rd Quartile', active_nodes_statistics[3]),
            ('Max Model Size', active_nodes_statistics[4]),
            ('Semantic Diversity', semantic_diversity),
            ('Min Similarity', similarity[0]),
            ('Similarity 1st Quartile', similarity[1]),
            ('Median Similarity', similarity[2]),
            ('Similarity 3rd Quartile', similarity[3]),
            ('Max Similarity', similarity[4])
        )

        # Update current generation with fitness statistics in one step
        current_gen = self.metrics.loc[gen]
        pd.set_option('display.float_format', '{:.2e}'.format)  # Set global display format for floats

        # Update the statistics in a single loop to minimize iteration
        for label, value in fitness_labels:
            current_gen[label] = value

    def _report_generation(self, g: int):
        # Efficient logging instead of multiple print calls
        print(f'Generation {g}')
        print(self.metrics.iloc[g].to_string(index=True))  # Convert to string without index for more concise output
        print('################')

    def save_metrics(self, path=None):
        path = path if path is not None else '.'

        # Save the metrics DataFrame only once
        self.metrics.to_csv(f'{path}/statistics.csv')

        # Save the xover_index categories efficiently
        for cat in ['deleterious', 'neutral', 'beneficial']:
            np.savetxt(f'{path}/xover_density_{cat}.csv', self.xover_index[cat].astype(np.int32), delimiter=",")

    def _mutate(self, models, gen, mutation_rate):
        if self.mutation_can_make_children:  # Reproduction through mutation
            children = {}
            n_c = 0
            while len(children) < self.max_c:
                for model_key, model in models.items():
                    if np.random.rand() < mutation_rate or self.max_p == 1:
                        # Directly mutate the model (avoid deepcopy if not needed)
                        ref_model = model.copy()  # Assuming a 'copy' method is available for models
                        ref_model.mutate()

                        # Ensure unique child names and check if model already exists in the dictionary
                        child_key = f'Child_{n_c}_g{gen}'
                        if child_key not in models:
                            p_key = model_key
                        else:
                            p_key = model.parent_keys

                        # Set parent key and store the child
                        ref_model.set_parent_key([p_key])
                        children[child_key] = ref_model
                        n_c += 1

            models.update(children)  # Efficiently update models with the new children
            return models  # No need to deepcopy models, just return the updated dictionary

        else:  # Mutate models in-place
            for model_key, model in models.items():
                if np.random.rand() < mutation_rate:
                    model.mutate()
            return models  # No need for deepcopy in this case

    @staticmethod
    def _get_similarity_score(model1, model2):
        m1 = deepcopy(model1)
        m2 = deepcopy(model2)

        def _map_functions(mo1, mo2):
            # Combine unique operators from both dataframes
            functions = pd.unique(pd.concat([mo1['Operator'], mo2['Operator']]))

            # Create a mapping dictionary
            # You can replace this logic with a more sophisticated mapping rule
            function_map = {func: i for i, func in enumerate(functions)}

            # Apply the mapping to both dataframes
            mo1['Operator'] = mo1['Operator'].map(function_map)
            mo2['Operator'] = mo2['Operator'].map(function_map)

            mo1 = mo1.query("NodeType != 'Input' and NodeType != 'Constant'")
            mo1 = mo1.drop(['Value', 'NodeType', 'Active'], axis=1)

            mo2 = mo2.query("NodeType != 'Input' and NodeType != 'Constant'")
            mo2 = mo2.drop(['Value', 'NodeType', 'Active'], axis=1)

            return mo1, mo2

        m1, m2 = _map_functions(m1, m2)

        # Ensure rows are comparable and of the same length
        try:
            sequence1 = m1.values[:, 0:].tolist()  # First row
            sequence2 = m2.values[:, 0:].tolist()  # Second row
        except IndexError:
            # If there are not enough rows, return 0
            return 0

        # Convert sequences to a format that the aligner can process
        # For numerical data, Biopython aligners work well with raw lists
        aligner = PairwiseAligner()
        aligner.mode = 'global'

        # Configure scoring
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -2
        aligner.extend_gap_score = -2

        # Perform the alignment and calculate the score
        # Convert sequences to strings and handle non-string values
        sequence1 = [str(int(y)) if pd.notna(y) else "" for x in sequence1 for y in x]
        sequence2 = [str(int(y)) if pd.notna(y) else "" for x in sequence2 for y in x]

        # Create Biopython sequences
        seq1 = Seq("".join(sequence1))
        seq2 = Seq("".join(sequence2))

        score = aligner.score(seq1, seq2)
        if not np.isfinite(score):
            score = 0.0

        # Return the alignment score
        return score

    import numpy as np
    from copy import copy
    from concurrent.futures import ThreadPoolExecutor

    def fit(self, train_x: list | np.ndarray, train_y: list | np.ndarray, step_size: int = None,
            xover_rate: float = 0.5, mutation_rate: float = 0.5):
        self.x = train_x
        self.y = train_y
        if len(self.x) < 1:
            raise ValueError("Must have at least one input value.")
        if len(self.y) != len(self.x):
            raise ValueError("Must have a 1:1 mapping for input set to output values.")
        if step_size is not None and not isinstance(step_size, int):
            raise TypeError("Step size must be either of type `int` or `None`.")

        # initialize population in parallel
        with ThreadPoolExecutor() as executor:
            self.population = {
                f'Model_{p}': executor.submit(CGP, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                                              mutation_type=self.mutation_type, function_bank=self.function_bank,
                                              **self.model_kwargs)
                for p in range(self.max_p)
            }

        # Wait for all model initializations to complete
        self.population = {key: model.result() for key, model in self.population.items()}

        self._get_fitnesses()
        self._record_metrics(0)
        self._report_generation(0)

        model_size = self.model_kwargs.get('max_size', -1)
        if model_size < 0:
            model_size = self.population['Model_0'].max_size
        model_size += self.population['Model_0'].outputs
        self.xover_index = {key: np.zeros((self.max_g, model_size)) for key in self.xover_index}
        genes_per_instruction = self.population['Model_0'].arity + 1  # for the operator
        self.mut_index = np.zeros((self.max_g, model_size * genes_per_instruction))

        # Get parents for generation
        for g in range(1, self.max_g + 1):
            if g > 1 or (g <= 1 and self.xover_type == 'semantic'):
                selected_parents = self.selection()
            else:
                selected_parents = self.population

            if not self.mutation_can_make_children:
                children = self.crossover(selected_parents, xover_rate, g) if self.xover_type else selected_parents
                selected_parents.update(children)
                self.population = selected_parents
                self._get_fitnesses()

                # Compare child and parent
                self._compare_child_parents()
                self._box_distribution(g)

                # Mutate the children (don't mutate the parents)
                mutated_children = self._mutate(children, g, mutation_rate)
                self._get_fitnesses()

                # Update selected_parents with mutated children
                selected_parents.update(mutated_children)
                self.population = selected_parents
            else:
                self.population = self._mutate(selected_parents, g, mutation_rate)
                self._compare_child_parents()
                self._box_distribution(g)

            # Get fitnesses and clear them only when necessary
            self._get_fitnesses()
            self._clear_fitnesses()
            self._record_metrics(g)

            if g % step_size == 0:
                self._report_generation(g)

        return self.best_model

from copy import copy

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from Bio.Align import PairwiseAligner
from cgp_model import CGP
from copy import deepcopy
from fitness_functions import correlation


def _get_quartiles(data):
    data = np.where(data == np.inf, 1, data)
    return np.quantile(data, [0, 0.25, 0.5, 0.75, 1])


def _validate_int_param(param_name, value, min_val, max_val):
    """
    Validates and converts a parameter to an integer if necessary.
    Raises errors or issues warnings as needed.
    """
    if not isinstance(value, int):
        print(f"Warning: {param_name} {value} is not an integer. Rounding to {int(value)}.")
        value = int(value)
    assert min_val <= value <= max_val, (
        f"{param_name} must be between {min_val} and {max_val}. Current value: {value}."
    )
    return value


def _split(m, points: [int]):
    previous_point = 0
    parts = []
    points = np.append(points, len(m))
    for point in points:
        parts.append(m[previous_point:point])
        previous_point = point

    return parts


class CartesianGP:
    def _missing_key_error(self, key):
        raise KeyError(f"'{key}' is required but was not provided.")

    def _setup_n_point_xover(self, kwargs):
        if 'n_point' in self.xover_type:
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
                 model_parameters: dict = None, solution_threshold=0.005, **kwargs):
        # extract from header
        self.population = {}
        self.fitnesses = {}
        self.y = None
        self.x = None
        self.mutation_can_make_children = None
        self.best_model = None
        self.weights = None
        self.max_p = parents
        self.max_c = children
        self.max_g = max_generations
        self.mutation_type = mutation
        self.xover_type = xover
        self.model_kwargs = model_parameters
        self.fixed_length = fixed_length
        self.selection_type = selection
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
        if self.mutation_type not in ['point']:
            raise ValueError(f"Invalid mutation type: {self.mutation_type}")
        if self.selection_type not in ['elite', 'tournament', 'elite tournament']:
            raise ValueError(f"Invalid selection type: {self.selection_type}")
        if self.xover_type not in [None, 'n_point', 'uniform']:
            raise ValueError(f"Invalid crossover type: {self.xover_type}")
        if self.model_kwargs is not None and not isinstance(self.model_kwargs, dict):
            raise TypeError("Model parameters must be a dictionary.")

        # Get fitness, mutation, xover, and selection
        possible_fitness_functions = {
            'correlation': correlation
        }
        possible_selection_functions = {
            'elite': self.elite_selection,
            'tournament': self.tournament_selection,
            'elite tournament': self.elite_tournament_selection
        }

        possible_xover_functions = {
            'n_point': self._n_point_xover,
            'uniform': self._uniform_xover
        }
        self.fitness_function = possible_fitness_functions.get(fitness_function.lower())
        self.xover = possible_xover_functions.get(self.xover_type.lower()) if self.xover_type is not None else None
        self.selection = possible_selection_functions.get(self.selection_type.lower())
        self._setup_tournament_and_elite(kwargs)
        self._setup_n_point_xover(kwargs) if self.xover_type == 'n_point' else None
        if self.fitness_function is None:
            raise KeyError(f"{fitness_function} is an invalid fitness function.")
        if self.selection is None:
            raise KeyError(f"{self.selection_type} is an invalid selection operator.")

        if kwargs.get('mutation_breeding', False) or (self.max_p == 1 and self.max_c > 1):
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
        n_elites = n_elites if n_elites is not None else self.max_p
        # Sort the population keys by fitness
        sorted_fitnesses = sorted(self.fitnesses, key=self.fitnesses.get)[:n_elites]
        # Return the selected elite individuals as a dictionary
        return {key: self.population[key] for key in sorted_fitnesses}

    def tournament_selection(self, n_to_select=None, population_keys=None):
        """
        Performs tournament selection to pick `n_to_select` individuals.
        """
        n_to_select = n_to_select if n_to_select is not None else self.max_p
        # Use the entire population if no specific keys are provided
        population_keys = population_keys if population_keys is not None else list(self.population.keys())
        new_population = {}

        while len(new_population) < n_to_select:
            # Randomly select contestants for the tournament
            contestants = np.random.choice(
                population_keys,
                size=self.tournament_size,
                replace=False
            )

            # Get the fitness values for the contestants
            contestant_fitnesses = {key: self.fitnesses[key] for key in contestants}

            # Select the best individual
            best_individual = min(contestant_fitnesses, key=contestant_fitnesses.get)

            # Add the best individual to the new population
            new_population[best_individual] = self.population[best_individual]

            # Remove the selected individual if diversity is required
            if self.tournament_diversity:
                population_keys.remove(best_individual)

        return new_population

    def elite_tournament_selection(self, n_elites=None):
        """
        Combines elite selection and tournament selection to create the new population.
        """
        # Determine the number of elites (default: one)
        n_elites = n_elites if n_elites is not None else 1

        # Step 1: Select elites
        elite_population = self.elite_selection(n_elites)

        # Step 2: Perform tournament selection for the remaining slots
        remaining_slots = self.max_p - len(elite_population)
        remaining_population = self.tournament_selection(
            n_to_select=remaining_slots,
            population_keys=[key for key in self.population.keys() if key not in elite_population]
        )

        # Step 3: Combine the results and update the population
        elite_population.update(remaining_population)
        return elite_population

    """
    Crossover Algorithms
    """

    def crossover(self, selected_parents: dict, xover_rate: float, gen: int):  # we're assuming two parents for now
        parent_list = list(selected_parents.values())
        name_list = list(selected_parents.keys())
        children = {}
        n_c = 0
        while len(children) < self.max_c:
            for p in range(0, len(parent_list), 2):
                p1 = parent_list[p]
                p2 = parent_list[p + 1]
                if np.random.rand() < xover_rate:
                    c1, c2 = self.xover(p1, p2, self.weights)
                else:
                    c1, c2 = copy(p1), copy(p2)
                children[f'Child_{n_c}_g{gen}'], children[f'Child_{n_c + 1}_g{gen}'] = c1, c2
                c1.set_parent_key([name_list[p], name_list[p + 1]])
                c2.set_parent_key([name_list[p], name_list[p + 1]])
                n_c += 2
        return children

    def _n_point_xover(self, p1, p2, **kwargs):
        """
        Performs n-point crossover
        @param p1: first parent
        @param p2: second parent
        @param kwargs: useless but used to soak up extra args
        @return: children
        """

        def get_crossover_points(parent, first_index):
            """Generate sorted crossover points for a parent."""
            possible_indices = range(first_index, len(parent.model))
            return np.sort(np.random.choice(possible_indices, size=self.n_points, replace=False))

        def interleave(p1_sections, p2_sections, columns):
            """Interleave parts from two parents, alternating between p1 and p2 sections."""
            child1_parts = [
                p1_sections[i] if i % 2 == 0 else p2_sections[i]
                for i in range(min(len(p1_sections), len(p2_sections)))
            ]
            child2_parts = [
                p2_sections[i] if i % 2 == 0 else p1_sections[i]
                for i in range(min(len(p1_sections), len(p2_sections)))
            ]
            return (
                pd.concat(child1_parts, ignore_index=False),
                pd.concat(child2_parts, ignore_index=False)
            )

        # Determine the first crossover index for each parent
        first_index_p1 = p1.model.loc[p1.model['NodeType'] == 'Function'].index[0]
        try:
            first_index_p2 = p2.model.loc[p2.model['NodeType'] == 'Function'].index[0]
        except:
            print(p2)
            print(p2.model)
            raise AttributeError

        columns = list(p1.model.columns)

        # Generate crossover points
        if self.fixed_length:
            xover_points_p1 = xover_points_p2 = get_crossover_points(p1, first_index_p1)
        else:
            xover_points_p1 = get_crossover_points(p1, first_index_p1)
            xover_points_p2 = get_crossover_points(p2, first_index_p2)

        # Split parents into parts
        p1_parts = _split(p1.model, xover_points_p1)
        p2_parts = _split(p2.model, xover_points_p2)

        # Interleave and return children
        child1, child2 = interleave(p1_parts, p2_parts, columns)
        c1 = CGP(model=child1, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        c2 = CGP(model=child2, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        return c1, c2

    def _uniform_xover(self, p1, p2, weights=None, **kwargs):
        """
        performs uniform crossover
        @param p1: First Parent
        @param p2: Second Parent
        @param weights: weights for choosing if an instruction is swapped, should be ((n_operations, [p, 1-p]))
        @return: children
        """
        weights = weights if weights is not None else np.full((len(p1.model), 2), 0.5)
        c1 = deepcopy(p1)
        c2 = deepcopy(p2)
        fb_node = min(c1.first_body_node, c2.first_body_node)
        m_len = min(len(c1.model), len(c2.model))
        swap = [True, False]
        for i in range(fb_node, m_len):
            if np.random.choice(swap, p=weights[i, :]):
                c1.model.loc[i] = deepcopy(p2.model.loc[i])
                c2.model.loc[i] = deepcopy(p1.model.loc[i])

        return c1, c2

    def _get_fitnesses(self):
        for p in self.population:
            self.fitnesses[p] = self.population[p].fit(self.x, self.y)

    def _clear_fitnesses(self):  # get rid of fitnesses no longer in population
        self.fitnesses = {key: self.fitnesses[key] for key in self.fitnesses if key in self.population}

    """
    Data Recording/Reporting
    """

    def _analyze_similarity(self):
        # Collect individuals where `parent_key` is not None
        individuals_with_parents = [
            ind for ind in self.population.values() if ind.parent_keys is not None
        ]

        # Find all unique parents (as tuples for consistency in sets)
        unique_parent_pairs = {
            tuple(ind.parent_keys) for ind in individuals_with_parents
        }

        # Group each parent pair with their children
        parent_child_groups = {}
        for parent_pair in unique_parent_pairs:
            # Retrieve the parent individuals
            parent_individuals = [self.population.get(p) for p in parent_pair if p in self.population]

            # Get the children for the current parent pair
            children = [
                ind for ind in individuals_with_parents if ind.parent_keys == list(parent_pair)
            ]

            # Combine parents and children into the group
            parent_child_groups[parent_pair] = (parent_individuals, children)

        # Iterate through parent-child groups to calculate similarity scores
        similarity_scores = []
        for parent_pair, (parents, children) in parent_child_groups.items():
            if not parents or not children:
                continue  # Skip groups without parents or children

            # Find the best parent by fitness (assume higher fitness is better)
            best_parent = min(parents, key=lambda ind: ind.fitness)

            # Find the best child by minimum fitness
            best_child = min(children, key=lambda ind: ind.fitness)

            # Calculate similarity score between the best parent and best child
            score = self._get_similarity_score(best_parent.model, best_child.model)
            similarity_scores.append((parent_pair, score))

        # Return or process similarity scores
        return similarity_scores

    def _record_metrics(self, gen: int):
        # get lists
        fit_list = np.array(list(self.fitnesses.values()))
        # len active nodes list
        len_active_nodes_list = [self.population[p].count_active_nodes() for p in self.population]

        similarity_scores = self._analyze_similarity()
        scores = np.array([score for _, score in similarity_scores])

        best_model_index = np.argmin(fit_list)
        self.best_model = list(self.population.values())[best_model_index]

        # do stats
        fit_statistics = _get_quartiles(fit_list)
        active_nodes_statistics = _get_quartiles(len_active_nodes_list)
        semantic_diversity = np.nanstd(fit_list)
        similarity = _get_quartiles(scores) if gen > 0 else [np.nan] * 5

        """
        metrics = ['Min Fitness', 'Fitness 1st Quartile', 'Median Fitness', 'Fitness 3rd Quartile',
                   'Max Fitness',
                   'Best Model Size', 'Min Model Size', 'Model Size 1st Quartile', 'Median Model Size',
                   'Model Size 3rd Quartile', 'Max Model Size']
        """

        # Define the statistic labels for fitness
        fitness_labels = [
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
        ]

        # Update current generation with fitness statistics
        current_gen = self.metrics.loc[gen]
        for label, value in fitness_labels:
            current_gen[label] = value

    def _report_generation(self, g: int):
        print(f'Generation {g}')
        print(self.metrics.iloc[[g]])
        print('################')

    def save_metrics(self, filename=None):
        filename = filename if filename is not None else 'statistics.csv'
        self.metrics.to_csv(filename)

    def _mutate(self, models, gen, mutation_rate):
        if self.mutation_can_make_children:  # Reproduction through mutation
            children = {}
            n_c = 0
            while len(children) < self.max_c:
                for model in models:
                    if np.random.rand() < mutation_rate or self.max_p == 1:
                        ref_model = deepcopy(models[model])
                        ref_model.mutate()
                        children[f'Child_{n_c}_g{gen}'] = deepcopy(ref_model)
                        del ref_model
                        n_c += 1
            models.update(children)
            return models
        else:  # Mutate models in-place
            for model_key, model in models.items():
                if np.random.rand() < mutation_rate:
                    model.mutate()
            return deepcopy(models)

    def _get_similarity_score(self, m1, m2):
        try:
            # Concatenate the two DataFrames along the first axis
            all_active = np.concatenate((m1.values, m2.values), axis=0)
        except ValueError:
            # If concatenation fails, return a score of 0
            return 0

        # Ensure rows are comparable and of the same length
        try:
            sequence1 = list(all_active[0])  # First row
            sequence2 = list(all_active[1])  # Second row
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
        # TODO: Fix this
        score = aligner.score(sequence1, sequence2)

        # Return the alignment score
        return score

    def fit(self, x: list | np.ndarray, y: list | np.ndarray, step_size: int = None,
            xover_rate: float = 0.5, mutation_rate: float = 0.5):
        self.x = x
        self.y = y
        if len(self.x) < 1:
            raise ValueError("Must have at least one input value.")
        if len(self.y) != len(self.x):
            raise ValueError("Must have a 1:1 mapping for input set to output values.")
        if step_size is not None and not isinstance(step_size, int):
            raise TypeError("Step size must be either of type `int` or `None`.")

        # initialize population
        for p in range(self.max_p):
            self.population[f'Model_{p}'] = CGP(fixed_length=self.fixed_length, fitness_function=self.ff_string,
                                                mutation_type=self.mutation_type,
                                                **self.model_kwargs)
        self._get_fitnesses()
        self._record_metrics(0)
        self._report_generation(0)

        # get parents for generation
        for g in range(1, self.max_g + 1):
            if g > 1:
                selected_parents = self.selection()
            else:
                selected_parents = self.population

            children = {}  # just so pycharm stops bitching
            if not self.mutation_can_make_children:
                children = (
                    self.crossover(selected_parents, xover_rate, g)
                    if self.xover_type is not None
                    else selected_parents
                )
                children = self._mutate(children, g, mutation_rate)
                selected_parents.update(children)
                self.population = selected_parents  # Merge the dictionaries
            else:
                self.population = self._mutate(selected_parents, g, mutation_rate)

            # get fitnesses
            self._get_fitnesses()
            self._clear_fitnesses()
            self._record_metrics(g)
            if g % step_size == 0:
                self._report_generation(g)
        return self.best_model


np.random.seed(1)
evolver = CartesianGP(parents=8, children=16, max_generations=50, mutation='point', selection='elite tournament',
                      xover='uniform', fixed_length=True, fitness_function='Correlation',
                      model_parameters={'max_size': 24},
                      solution_threshold=0.05, n_points=1, tournament_size=8, n_elites=2, mutation_breeding=False)
x = np.array([0, 1, 2, 3, 4, 5, 6])
y = (x ** 2).reshape(-1, 1)  # fix this
best_model = evolver.fit(x, y, 10, 1.0, 1.0)
evolver.save_metrics()

best_model.print_model()
best_model.to_csv()

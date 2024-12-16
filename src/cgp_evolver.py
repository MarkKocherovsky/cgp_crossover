from copy import copy
from copy import deepcopy

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, Seq

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
            # Sample tournament contestants
            if len(population_keys) > self.tournament_size:
                contestants = np.random.choice(population_keys, size=self.tournament_size, replace=False)
            else:
                contestants = population_keys  # Use all available keys if not enough

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

    """
        Tomasz P Pawlak and Krzysztof Krawiec. Competent Geometric Semantic
        Genetic Programming for Symbolic Regression and Boolean Function Synthesis.
        Evolutionary Computation, 26(2):177–212, 2018
    """

    def competent_tournament_selection(self, n_to_select=None, population_keys=None):
        # Set default values
        n_to_select = n_to_select or self.max_p
        population_keys = population_keys or list(self.population.keys())
        target = self.y.flatten()  # Ground truth from Pawlak and Krawiec, 2018

        # Precompute distances from the target for all individuals
        parent_semantic_list = self._determine_parent_semantics()
        target_distances = {key: pairwise_minkowski_distance(predictions, target, p=2) for key, predictions in
                            parent_semantic_list.items()}

        new_population = {}

        while len(new_population) < n_to_select:
            # Select the first parent via regular tournament selection
            first_parent_key, first_parent = next(
                iter(self.tournament_selection(n_to_select=1, population_keys=population_keys).items()))

            parent_semantics = parent_semantic_list[first_parent_key]
            parent_distance_to_target = target_distances[first_parent_key]

            # Sample tournament contestants
            if len(population_keys) > self.tournament_size:
                contestants = np.random.choice(population_keys, size=self.tournament_size, replace=False)
            else:
                contestants = population_keys  # Use all available keys if not enough

            scores = {}

            for key in contestants:
                contestant_semantics = parent_semantic_list[key]
                distance_to_parent = pairwise_minkowski_distance(parent_semantics, contestant_semantics, p=2)
                distance_to_target = target_distances[key]
                scores[key] = get_score(parent_distance_to_target, distance_to_parent, distance_to_target)

            # Select the contestant with the best score
            second_parent_key = min(scores, key=scores.get)
            new_population[first_parent_key] = first_parent
            new_population[second_parent_key] = self.population[second_parent_key]

            # Remove selected contestant from future tournaments
            population_keys.remove(second_parent_key)

        return new_population

    def _determine_parent_semantics(self):
        # essentially what I want to do is organize by distance of node values
        parent_predictions = {}
        # loop over each existing parent and get the predictions output
        for p in self.population:
            predictions = self.population[p](self.x)
            # Check if predictions is not already in parent_predictions values
            if not any(np.array_equal(predictions, v) for v in parent_predictions.values()):
                parent_predictions[p] = predictions.flatten()

        return parent_predictions

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
    Crossover Operators
    """

    def crossover(self, selected_parents: dict, xover_rate: float, gen: int):  # we're assuming two parents for now
        new_parents = {k: [v] for k, v in selected_parents.items()}  # Wrap in lists for uniformity

        # Flatten the new_parents structure for further use
        parent_list = [item for sublist in new_parents.values() for item in sublist]
        name_list = [name for name, sublist in new_parents.items() for _ in sublist]

        children = {}
        n_c = 0
        while len(children) < self.max_c:
            for p in range(0, len(parent_list), 2):
                p1 = parent_list[p]
                p2 = parent_list[p + 1]
                if np.random.rand() < xover_rate:
                    if self.xover_type.lower() == 'subgraph':
                        c1 = self.xover(p1, p2, gen)
                        c2 = self.xover(p2, p1, gen)
                    else:
                        c1, c2 = self.xover(p1, p2, gen=gen, weights=self.weights)
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

        def get_crossover_points(parent, first_index, xover_weights=None, include_output=False):
            """Generate sorted crossover points for a parent."""
            if include_output:
                n_outputs = 0
            else:
                n_outputs = parent.outputs
            possible_indices = range(first_index, len(parent.model) - n_outputs)
            if np.sum(xover_weights) == 0:
                xover_weights = None  # `None` makes np.random.choice default to uniform probabilities
            return np.sort(np.random.choice(possible_indices, size=self.n_points, replace=False, p=xover_weights))

        def interleave(p1_sections, p2_sections):
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

        # Generate crossover points
        fb_node = min(p1.first_body_node, p2.first_body_node)
        weights = None
        if self.semantic:
            vmat_1 = clean_values(p1, self.x)
            vmat_2 = clean_values(p2, self.x)
            ssd = get_ssd(vmat_1, vmat_2)
            weights = get_weights(ssd)

        if self.fixed_length:
            xover_points_p1 = xover_points_p2 = get_crossover_points(p1, first_index_p1, weights)
        else:
            xover_points_p1 = get_crossover_points(p1, first_index_p1, weights)
            xover_points_p2 = get_crossover_points(p2, first_index_p2, weights)
        p1.xover_index[xover_points_p1 - fb_node] += 1
        p2.xover_index[xover_points_p2 - fb_node] += 1

        # Split parents into parts
        p1_parts = _split(p1.model, xover_points_p1)
        p2_parts = _split(p2.model, xover_points_p2)

        # Interleave and return children
        child1, child2 = interleave(p1_parts, p2_parts)
        c1 = CGP(model=child1, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        c2 = CGP(model=child2, fixed_length=self.fixed_length, fitness_function=self.ff_string,
                 mutation_type=self.mutation_type)
        return c1, c2

    def _uniform_xover(self, p1, p2, gen, weights=None, **kwargs):
        """
        performs uniform crossover
        @param p1: First Parent
        @param p2: Second Parent
        @param weights: weights for choosing if an instruction is swapped, should be ((n_operations, [p, 1-p]))
        @return: children
        """
        weights = None
        n_outputs = 0  # outputs allowed to xover
        if self.semantic:
            vmat_1 = clean_values(p1, self.x)
            vmat_2 = clean_values(p2, self.x)
            ssd = get_ssd(vmat_1, vmat_2)
            weights = get_weights(ssd, epsilon=0.001)
            n_outputs = p1.outputs
        c1 = deepcopy(p1)
        c2 = deepcopy(p2)
        fb_node = min(c1.first_body_node, c2.first_body_node)
        assert len(c1.model) == len(c2.model), 'Parents in Uniform Xover have to be the same length.'
        possible_indices = list(range(fb_node, len(c1.model) - n_outputs))
        swapped_indices = np.sort(
            np.random.choice(possible_indices, (len(possible_indices) // 2,), p=weights, replace=False))
        for i in swapped_indices:
            c1.model.loc[i] = deepcopy(p2.model.loc[i])
            c2.model.loc[i] = deepcopy(p1.model.loc[i])
            c1.xover_index[i - fb_node] += 1
            c2.xover_index[i - fb_node] += 1
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

    def _clear_fitnesses(self):  # get rid of fitnesses no longer in population
        self.fitnesses = {key: self.fitnesses[key] for key in self.fitnesses if key in self.population}

    """
    Data Recording/Reporting
    """

    def _group_parents_and_children(self):
        """Group parents and their children."""
        # Collect individuals where `parent_key` is not None
        individuals_with_parents = [
            ind for ind in self.population.values() if ind.parent_keys is not None
        ]

        # Find all unique parent pairs
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

        return parent_child_groups

    def _compare_child_parents(self):
        parent_child_groups = self._group_parents_and_children()
        for parent_pair, (parents, children) in parent_child_groups.items():
            if not parents or not children:
                continue
            for child in children:
                if child.better_than_parents is None:
                    if any(child.fitness > parent.fitness for parent in parents):
                        child.better_than_parents = 'deleterious'
                    elif any(child.fitness < parent.fitness for parent in parents):
                        child.better_than_parents = 'beneficial'

    def _box_distribution(self, gen):
        individuals_with_parents = [
            ind for ind in self.population.values() if ind.parent_keys is not None
        ]
        for ind in individuals_with_parents:
            if ind.better_than_parents == 'beneficial':
                self.xover_index['beneficial'][gen - 1, :] = ind.xover_index
            elif ind.better_than_parents == 'deleterious':
                self.xover_index['deleterious'][gen - 1, :] = ind.xover_index
            else:
                self.xover_index['neutral'][gen - 1, :] = ind.xover_index
            ind.xover_index = np.zeros(ind.xover_index.shape)  # prepare for next generation

    def _analyze_similarity(self):
        parent_child_groups = self._group_parents_and_children()

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
        pd.set_option('display.float_format', '{:.2e}'.format)  # Set global display format for floats
        for label, value in fitness_labels:
            current_gen[label] = value

    def _report_generation(self, g: int):
        print(f'Generation {g}')
        print(self.metrics.iloc[[g]])
        print('################')

    def save_metrics(self, path=None):
        path = path if path is not None else 'statistics'
        self.metrics.to_csv(f'{path}/statistics.csv')
        [np.savetxt(f'{path}/xover_density_{cat}.csv', self.xover_index[cat].astype(np.int32), delimiter=",") for
         cat in ['deleterious', 'neutral', 'beneficial']]

    def _mutate(self, models, gen, mutation_rate):
        if self.mutation_can_make_children:  # Reproduction through mutation
            children = {}
            n_c = 0
            while len(children) < self.max_c:
                for model in models:
                    if np.random.rand() < mutation_rate or self.max_p == 1:
                        ref_model = deepcopy(models[model])
                        ref_model.mutate()
                        if f'Child_{n_c}_g{gen}' not in models:
                            p_key = model
                        else:
                            p_key = models[model].parent_keys
                        children[f'Child_{n_c}_g{gen}'] = deepcopy(ref_model)
                        children[f'Child_{n_c}_g{gen}'].set_parent_key([p_key])
                        del ref_model
                        n_c += 1
            models.update(children)
            return deepcopy(models)
        else:  # Mutate models in-place
            for model_key, model in models.items():
                if np.random.rand() < mutation_rate:
                    model.mutate()
            return deepcopy(models)

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

        # initialize population
        for p in range(self.max_p):
            self.population[f'Model_{p}'] = CGP(fixed_length=self.fixed_length, fitness_function=self.ff_string,
                                                mutation_type=self.mutation_type, function_bank=self.function_bank,
                                                **self.model_kwargs)
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
        # get parents for generation
        for g in range(1, self.max_g + 1):
            if g > 1 or (g <= 1 and self.xover_type == 'semantic'):  # parents in semantic xover need to be ordered
                selected_parents = self.selection()
            else:
                selected_parents = self.population

            if not self.mutation_can_make_children:
                children = (
                    self.crossover(selected_parents, xover_rate, g)
                    if self.xover_type is not None
                    else selected_parents
                )
                # Add the original children to selected_parents
                selected_parents.update(children)
                self.population = copy(selected_parents)
                self._get_fitnesses()
                # Compare child and parent (assuming this method is handling comparisons for other purposes)
                self._compare_child_parents()
                self._box_distribution(g)

                # Mutate the children (don't mutate the parents)
                mutated_children = self._mutate(children, g, mutation_rate)
                self._get_fitnesses()
                # Remove original children before updating with mutated children
                # Assuming children are stored with the same keys
                # selected_parents = {k: v for k, v in selected_parents.items() if k not in mutated_children}

                # Update selected_parents with mutated children only
                selected_parents.update(mutated_children)
                self.population = copy(selected_parents)

            else:
                self.population = self._mutate(selected_parents, g, mutation_rate)
                self._compare_child_parents()
                self._box_distribution(g)
            # get fitnesses
            self._get_fitnesses()
            self._clear_fitnesses()
            self._record_metrics(g)
            if g % step_size == 0:
                self._report_generation(g)
        return self.best_model

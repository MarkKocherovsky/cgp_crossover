from copy import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cgp_model import CGP
from fitness_functions import correlation


# from old.cgp_vlen import max_g


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
                   'Model Size 3rd Quartile', 'Max Model Size']

        metrics_initialized = np.zeros((self.max_g + 1, len(metrics)))
        self.metrics = pd.DataFrame(metrics_initialized, columns=metrics)

        # Sanitize inputs
        if self.max_p < 1:
            raise ValueError("Must have at least one parent.")
        if self.max_c < 1:
            raise ValueError("Must have at least one child.")
        if self.mutation_type not in ['point']:
            raise ValueError(f"Invalid mutation type: {self.mutation_type}")
        if self.selection_type not in ['elite', 'tournament', 'tournament_elite']:
            raise ValueError(f"Invalid selection type: {self.selection_type}")
        if self.xover_type not in [None, 'n_point']:
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
            'n_point': self._n_point_xover
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
        children = {}
        n_c = 0
        while len(children) < self.max_c:
            for p in range(0, len(parent_list), 2):
                p1 = parent_list[p]
                p2 = parent_list[p + 1]
                if np.random.rand() < xover_rate:
                    c1, c2 = self.xover(p1, p2)
                else:
                    c1, c2 = copy(p1), copy(p2)
                children[f'Child_{n_c}_g{gen}'], children[f'Child_{n_c + 1}_g{gen}'] = c1, c2
                n_c += 2
        return children

    def _n_point_xover(self, p1, p2):
        def get_crossover_points(parent, first_index):
            """Generate sorted crossover points for a parent."""
            possible_indices = range(first_index, len(parent.model))
            return np.sort(np.random.choice(possible_indices, size=self.n_points, replace=False))

        def interleave(p1_sections, p2_sections, columns):
            """Interleave parts from two parents."""
            child1_parts = p1_sections[::2] + p2_sections[1::2]
            child2_parts = p2_sections[::2] + p1_sections[1::2]
            return (
                pd.concat(child1_parts, ignore_index=True),
                pd.concat(child2_parts, ignore_index=True)
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

    def _get_fitnesses(self):
        for p in self.population:
            self.fitnesses[p] = self.population[p].fit(self.x, self.y)

    def _clear_fitnesses(self):  # get rid of fitnesses no longer in population
        self.fitnesses = {key: self.fitnesses[key] for key in self.fitnesses if key in self.population}

    """
    Data Recording/Reporting
    """

    def _record_metrics(self, gen: int):
        # get lists
        fit_list = np.array(list(self.fitnesses.values()))
        # len active nodes list
        len_active_nodes_list = np.array([len(self.population[p].get_active_nodes()) for p in self.population])

        best_model_index = np.argmin(fit_list)
        best_model = list(self.population.values())[best_model_index]
        # do stats
        fit_statistics = _get_quartiles(fit_list)
        active_nodes_statistics = _get_quartiles(len_active_nodes_list)

        # Define the statistic labels for fitness
        fitness_labels = [
            ('Min Fitness', fit_statistics[0]),
            ('Fitness 1st Quartile', fit_statistics[1]),
            ('Median Fitness', fit_statistics[2]),
            ('Fitness 3rd Quartile', fit_statistics[3]),
            ('Max Fitness', fit_statistics[4]),
            ('Best Model Size', len(best_model.get_active_nodes())),
            ('Min Model Size', active_nodes_statistics[0]),
            ('Model Size 1st Quartile', active_nodes_statistics[1]),
            ('Median Model', active_nodes_statistics[2]),
            ('Model Size 3rd Quartile', active_nodes_statistics[3]),
            ('Max Model Size', active_nodes_statistics[4])
        ]

        # Update current generation with fitness statistics
        current_gen = self.metrics.loc[gen]
        for label, value in fitness_labels:
            current_gen[label] = value

    def _report_generation(self, g: int):
        print(f'Generation {g}')
        print(self.metrics.iloc[[g]])
        print('################')

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

        # get parents for generation
        for g in range(1, self.max_g + 1):
            if g > 1:
                selected_parents = self.selection()
            else:
                selected_parents = self.population

            children = self.crossover(selected_parents, xover_rate,
                                      g) if self.xover_type is not None else selected_parents

            # perform mutation
            # TODO: mutation that can create new children
            for child_key in children:
                if np.random.rand() < mutation_rate:
                    children[child_key].mutate()

            # get fitnesses
            selected_parents.update(children)  # Merge the dictionaries
            self.population = selected_parents.copy()  # Create a copy of the updated selected_parents
            print([m.print_model() for m in self.population.values()])
            exit()
            self._get_fitnesses()
            self._clear_fitnesses()
            self._record_metrics(g)
            if g % step_size == 0:
                self._report_generation(g)
        return self.metrics


np.random.seed(1)
evolver = CartesianGP(parents=1, children=4, max_generations=1000, mutation='point', selection='elite',
                      xover=None, fixed_length=True, fitness_function='Correlation',
                      model_parameters={'max_size': 16},
                      solution_threshold=0.05, n_points=1, tournament_size=4)
x = np.array([0, 1, 2, 3, 4, 5, 6])
y = (x ** 2).reshape(-1, 1)  # fix this
metrics = evolver.fit(x, y, 50, 0.5, 1.0)

fig, ax = plt.subplots()
ax.plot(metrics['Min Fitness'], label='Best Fitness')
ax.plot(metrics['Median Fitness'], label='Median Fitness')
ax.set_yscale('log')
plt.savefig('plt.png')

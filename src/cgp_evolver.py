import numpy as np

from cgp_model import CGP
from src.backup.test_align import fitness
from src.fitness_functions import correlation


class CartesianGP:
    def __init__(self, parents: int = 1, children: int = 4,
                 max_generations: int = 100, mutation: str = 'Basic', selection: str = 'Elite',
                 xover: str = None, fixed_length: bool = True, fitness_function: str = 'Correlation',
                 model_parameters: dict = None, solution_threshold = 0.005):
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

        # Sanitize inputs
        if self.max_p < 1:
            raise ValueError("Must have at least one parent.")
        if self.max_c < 1:
            raise ValueError("Must have at least one child.")
        if self.mutation_type not in ['Basic']:
            raise ValueError(f"Invalid mutation type: {self.mutation_type}")
        if self.selection_type not in ['Elite']:
            raise ValueError(f"Invalid selection type: {self.selection_type}")
        if self.xover_type not in [None, 'n_point']:
            raise ValueError(f"Invalid crossover type: {self.xover_type}")
        if self.model_kwargs is not None and not isinstance(self.model_kwargs, dict):
            raise TypeError("Model parameters must be a dictionary.")

        # Get fitness, mutation, xover, and selection
        possible_fitness_functions = {
            'Correlation': correlation
        }
        possible_selection_functions = {
            'Elite': elite_selection
            'Tournament': tournament_selection
            'Elite Tournament': elite_tournament_selection
        }
        possible_mutation_functions = {
            'Basic': basic_mutation
        }
        possible_xover_functions = {
            'NPoint': n_point
        }
        self.fitness_function = possible_fitness_functions.get(fitness_function)
        self.mutation = possible_fitness_functions.get(self.mutation_type)
        self.xover = possible_fitness_functions.get(self.xover_type)
        self.selection = possible_fitness_functions.get(self.selection_type)

        if self.fitness_function is None:
            raise KeyError(f"{fitness_function} is an invalid fitness function.")
        if self.fitness_function is None:
            raise KeyError(f"{fitness_function} is an invalid mutation operator.")
        if self.selection is None:
            raise KeyError(f"{selection} is an invalid selection operator.")

    def _get_fitnesses(self):
        for p in range(0, len(self.population)):
            self.fitnesses[p] = self.population[p].fit(self.x, self.y)

    def fit(self, x: list | np.ndarray, y: list | np.ndarray, step_size: int = None):
        self.x = x
        self.y = y
        if len(self.x) < 1:
            raise ValueError("Must have at least one input value.")
        if len(self.y) != len(self.x):
            raise ValueError("Must have a 1:1 mapping for input set to output values.")
        if step_size is not None and not isinstance(step_size, int):
            raise TypeError("Step size must be either of type `int` or `None`.")

        for p in range(self.max_p):
            self.population[f'Model_{p}'] = CGP(self.fixed_length, self.fitness_function, self.model_kwargs)

        self._get_fitnesses()

        print()
        if self.xover_type is not None:
            selected_parents =

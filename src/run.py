# run.py
# a single run of CGP for the experiment
# to actually queue up jobs run job_scheduler.py
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from cgp_evolver import CartesianGP
from cgp_operators import add, sub, mul, div
from test_problems import Collection

# get arguments, until the end of the print statements this is a chatgpt innovation
parser = argparse.ArgumentParser(description="Run a genetic algorithm with configurable parameters.")

# Required arguments
parser.add_argument("trial_number", type=int, help="Trial number for the experiment.")
parser.add_argument("max_generations", type=int, help="Maximum number of generations.")
parser.add_argument("model_size", type=int, help="Maximum Size of Models.")
parser.add_argument("max_parents", type=int, help="Maximum number of parents.")
parser.add_argument("max_children", type=int, help="Maximum number of children.")
parser.add_argument("xover_type", type=str, help="Crossover type.")
parser.add_argument("xover_rate", type=float, help="Crossover rate.")
parser.add_argument("mutation_type", type=str, help="Mutation type.")
parser.add_argument("mutation_rate", type=float, help="Mutation rate.")
parser.add_argument("selection_type", type=str, help="Selection type.")
parser.add_argument("test_problem_key", type=str, help="Key of the test problem.")

# Optional arguments with defaults
parser.add_argument("--fitness_function", type=str, default='correlation', help="Fitness function.")
parser.add_argument("--n_points", type=int, default=1, help="Number of points (default: 1).")
parser.add_argument("--tournament_size", type=int, default=5, help="Size of the tournament (default: 5).")
parser.add_argument("--n_elites", type=int, default=1, help="Number of elites (default: 1).")
parser.add_argument("--problem_dimensions", type=int, default=1, help="Controls number of dimensions for test data.")
parser.add_argument("--step_size", type=int, default=100, help="Prints out generation data every N generations.")

args = parser.parse_args()

# Access arguments
trial_number = args.trial_number
max_generations = args.max_generations
model_size = args.model_size
xover_type = args.xover_type
xover_rate = args.xover_rate
max_parents = args.max_parents
max_children = args.max_children
mutation_type = args.mutation_type
mutation_rate = args.mutation_rate
selection_type = args.selection_type
fitness_function = args.fitness_function
test_problem_key = args.test_problem_key
problem_dimensions = args.problem_dimensions
n_points = args.n_points
tournament_size = args.tournament_size
n_elites = args.n_elites
step_size = args.step_size

np.random.seed(trial_number)

print(f"Trial Number: {trial_number}")
print(f"Max Generations: {max_generations}")
print(f"Number of Parents: {max_parents}")
print(f"Number of Children: {max_children}")
print(f"Crossover Type: {xover_type}")
print(f"Crossover Rate: {xover_rate}")
print(f"Mutation Type: {mutation_type}")
print(f"Mutation Rate: {mutation_rate}")
print(f"Selection Type: {selection_type}")
print(f"Fitness Function: {fitness_function}")
print(f"Test Problem: {test_problem_key}")
print(f"Number of Points: {n_points}")
print(f"Tournament Size: {tournament_size}")
print(f"Number of Elites: {n_elites}")

# data sanitizing is done in the CartesianGP class, see cgp_evolver.py

# initialize problem data

problem_list = Collection()
test_function = problem_list(test_problem_key, n_dims=problem_dimensions)
train_x, test_x, train_y, test_y = test_function.return_points()

# establish output path
run_path = f'../output/{test_problem_key}_{problem_dimensions}d/{xover_type}/{selection_type}/trial_{trial_number}'
Path(run_path).mkdir(parents=True, exist_ok=True)

# model parameters
model_parameters = {
    'max_size': model_size,
    'inputs': test_function.dimensions,
    'outputs': 1,
    'arity': 2,
    'constants': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
}
function_bank = (add, sub, mul, div)

if max_parents < max_children:
    mutation_breeding = True
else:
    mutation_breeding = False
evolution_module = CartesianGP(parents=max_parents, children=max_children,
                               max_generations=max_generations, mutation=mutation_type, selection=selection_type,
                               xover=xover_type, fixed_length=True, fitness_function=fitness_function,
                               model_parameters=model_parameters, n_points=n_points, n_elites=n_elites,
                               tournament_size=tournament_size, function_bank=function_bank,
                               mutation_breeding=mutation_breeding)

start = datetime.now()
best_model = evolution_module.fit(train_x, train_y, step_size=step_size)
end = datetime.now()

duration = end - start
print(f'Duration: {duration}')
print(f'Best Model Training Fitness: {best_model.fitness}')
if test_x is not None:
    test_fitness = best_model.fit(test_x, test_y)
    print(f'Best Model Testing Fitness: {test_fitness}')

evolution_module.save_metrics(run_path)
best_model.print_model()
best_model.to_csv(f'{run_path}/best_model.csv')

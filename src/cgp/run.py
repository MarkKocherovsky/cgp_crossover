# run.py
# a single run of CGP for the experiment
# to actually queue up jobs run job_scheduler.py
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cgp_evolver import CartesianGP
from cgp_operators import add, sub, mul, div
from test_problems import Collection
from fitness_functions import *

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from ConfigSpace.conditions import InCondition

from smac import HyperparameterOptimizationFacade, Scenario


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def configspace(seed=0) -> ConfigurationSpace:
    # Build Configuration Space which defines all parameters and their ranges
    # https://automl.github.io/SMAC3/latest/examples/1%20Basics/2_svm_cv/#__tabbed_1_1
    cs = ConfigurationSpace(seed=seed)
    max_size = Categorical("max_size", [4, 8, 16, 32, 64, 128, 256], default=64)
    # version without crossover
    # n_parents = Categorical("n_parents", [1, 2, 4, 8], default=1)
    # version with crossover
    n_parents = Categorical("n_parents", [2, 4, 8, 16, 32, 64], default=32)
    # n_children = Categorical("n_children", [4, 8, 16], default=4)
    n_children = Categorical("n_children", [2, 4, 8, 16, 32, 64], default=32)
    m_type = Categorical("m_type", ["point", "full"], default="point")
    x_rate = Categorical("x_rate", [0.05, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0], default=0.5)
    m_rate = Categorical("m_rate", [0.05, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0], default=0.2)
    # t_size = Categorical("t_size", [4, 6, 8, 10], default=4)

    # (mu+lambda)
    #cs.add([max_size, n_children])
    # (1+lambda)
    cs.add([max_size, n_children, n_parents, x_rate, m_rate])

    return cs


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
parser.add_argument("--asexual_reproduction", type=str2bool, default=False,
                    help="Controls whether or not asexual reproduction can occur.")
parser.add_argument("--one_dimensional_xover", type=str2bool, default=False,
                    help="If True, parents will be flattened before crossover. Not compatible with subgraph or semantic methods.")
parser.add_argument("--tuning", type=str2bool, default=False,
                    help="If True, hyperparameters will be tuned. Try to give a smaller generation size for testing.")
parser.add_argument("--cfg_name", type=str, help="Name of Configuration")
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
asex = args.asexual_reproduction
one_d = args.one_dimensional_xover
tuning = args.tuning
cfg_name = args.cfg_name
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
print(f"One Dimensional Crossover: {one_d}")
print(f"Configuration Name: {cfg_name}")
# data sanitizing is done in the CartesianGP class, see cgp_evolver.py

# initialize problem data

problem_list = Collection()
test_function = problem_list(test_problem_key, n_dims=problem_dimensions)
train_x, test_x, train_y, test_y = test_function.return_points()

if mutation_type != 'point':
    mut_type = mutation_type
else:
    mut_type = ''
# establish output path
CHECKPOINT_PATH = os.path.join(os.environ.get("SCRATCH", "/tmp"), "ckpt")
if not one_d:
    run_path = f'/mnt/gs21/scratch/kocherov/Documents/cgp/output/{test_problem_key}_{problem_dimensions}d/{xover_type}/{mut_type}/{selection_type}/{cfg_name}/trial_{trial_number}'
    CHECKPOINT_FILE = f"{CHECKPOINT_PATH}/{test_problem_key}_{problem_dimensions}d_{xover_type}{mut_type}_{selection_type}_{cfg_name}_trial_{trial_number}_ckpt.pkl"
else:
    run_path = f'/mnt/gs21/scratch/kocherov/Documents/cgp/output/{test_problem_key}_{problem_dimensions}d/{xover_type}_1d/{mut_type}/{selection_type}/{cfg_name}/trial_{trial_number}'
    CHECKPOINT_FILE = f"{CHECKPOINT_PATH}/{test_problem_key}_{problem_dimensions}d_{xover_type}{mut_type}_1d_{selection_type}_trial_{trial_number}_ckpt.pkl"
Path(run_path).mkdir(parents=True, exist_ok=True)
print(run_path)
# model parameters
model_parameters = {
    'max_size': model_size,
    'inputs': test_function.dimensions,
    'outputs': 1,
    'arity': 2,
    'constants': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
}
function_bank = {'add': add, 'sub': sub, 'mul': mul, 'div': div}

if asex or max_parents < max_children:
    mutation_breeding = True
else:
    mutation_breeding = False

# if it's DNC semantic, then the learning should be on the semantics, not the actual nodes
sequence_length = model_size if 'semantic' not in xover_type else train_x.shape[0]
# a gene is NodeType, Value, Operator, *Operands, Active
input_dim = (1 + 1 + model_parameters['arity'] + 1 + 1) if 'semantic' not in xover_type else train_x.shape[0]
dnc_hyperparameters = {
    'embedding_dim': 64,
    'sequence_length': sequence_length,
    'input_dim': input_dim,
    'get_fitness_function': correlation,
    'batch_size': 820,
    'epsilon_greedy': 0.2,
    'learning_rate': 0.0001,
    'running_mean_decay': 0.0001,
    'adam_decay': 0.0001
}
if 'dnc' in xover_type:
    print(dnc_hyperparameters)
# CHECKPOINT_PATH = '../output/ckpt'
print(CHECKPOINT_PATH)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

seconds = int(time.time())
if tuning:
    cs = configspace(seed=seconds)
    scenario = Scenario(
        cs,
        n_trials=100,
        name=f'trial_{trial_number}',
        output_directory=Path(f'../output/{test_problem_key}_{problem_dimensions}d/{xover_type}/SMAC'),
        objectives = ['correlation', 'complexity']
        # n_workers = 8
    )
    d_path = Path(f'../output/{test_problem_key}_{problem_dimensions}d/{xover_type}/SMAC')
    d_path.mkdir(parents=True, exist_ok=True)
    #with open(f'../output/{test_problem_key}_{problem_dimensions}d/{xover_type}/SMAC/seeds.txt', "w+") as f:
    #    f.write('')


    def train(config: Configuration, seed: int = 0):

        config_model_parameters = {'max_size': config.get('max_size', model_size), 'inputs': test_function.dimensions,
                                   'outputs': 1, 'arity': 2, 'constants': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

        config_max_children = config.get('n_children', max_children)
        config_max_parents = config.get('n_parents', max_parents)
        config_mutation_type = config.get('m_type', mutation_type)
        config_mutation_rate = config.get('m_rate', mutation_rate)
        config_xover_rate = config.get('x_rate', xover_rate)

        #with open(f'../output/{test_problem_key}_{problem_dimensions}d/{xover_type}/SMAC/seeds.txt', "a") as f:
        #    f.write(f"Seed: {config_seed}\tConfig: {config}\n#####\n")

        tuning_evolution_module = CartesianGP(
            parents=config_max_parents,
            children=config_max_children,
            max_generations=max_generations,
            mutation=config_mutation_type,
            selection=selection_type,
            xover=xover_type,
            fixed_length=True,
            fitness_function=fitness_function,
            model_parameters=config_model_parameters,
            n_points=n_points,
            n_elites=n_elites,
            tournament_size=tournament_size,
            function_bank=function_bank,
            mutation_breeding=mutation_breeding,
            checkpoint_filename=CHECKPOINT_FILE,
            one_dimensional_xover=one_d,
            seed=seconds,
            dnc_hp=dnc_hyperparameters,
            tuning=tuning
        )

        _, best_test_model = tuning_evolution_module.fit(train_x, test_x, train_y, test_y, step_size=step_size,
                                                         xover_rate=config_xover_rate,
                                                         mutation_rate=config_mutation_rate)
        return {'correlation': best_test_model.fitness, 'complexity': best_test_model.complexity}


    # We want to run the facade's default initial design, but we want to change the number
    # of initial configs to 5.
    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=0)
    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario,
        max_config_calls=5,
    )

    # Now we use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(
        scenario,
        train,
        initial_design=initial_design,
        intensifier=intensifier,
        multi_objective_algorithm=HyperparameterOptimizationFacade.get_multi_objective_algorithm(
            scenario,
            objective_weights=[5, 1],
        ),
        overwrite=True,  # If the run exists, we overwrite it; alternatively, we can continue from last state
    )
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(cs.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
    print(incumbent)

else:
    if os.path.exists(CHECKPOINT_FILE):
        evolution_module = CartesianGP.load_checkpoint(filename=CHECKPOINT_FILE)
        evolution_module.set_max_gens(max_generations)
        print(f"Resuming from generation {evolution_module.current_generation}")
    else:
        evolution_module = CartesianGP(
            parents=max_parents,
            children=max_children,
            max_generations=max_generations,
            mutation=mutation_type,
            selection=selection_type,
            xover=xover_type,
            fixed_length=True,
            fitness_function=fitness_function,
            model_parameters=model_parameters,
            n_points=n_points,
            n_elites=n_elites,
            tournament_size=tournament_size,
            function_bank=function_bank,
            mutation_breeding=mutation_breeding,
            checkpoint_filename=CHECKPOINT_FILE,
            one_dimensional_xover=one_d,
            seed=trial_number,
            dnc_hp=dnc_hyperparameters,
            tuning=tuning
        )

    start = datetime.now()

    best_model, best_test_model = evolution_module.fit(train_x, test_x, train_y, test_y, step_size=step_size)
    end = datetime.now()

    duration = end - start
    print(f'Duration: {duration}')
    # print(f'Best Model Training Fitness: {best_model.fitness}')
    # print(f'Best Model Testing Fitness: {best_model.fitness}')
    # if test_x is not None:
    #    test_fitness = best_model.fit(test_x, test_y)
    #    print(f'Best Model Testing Fitness: {test_fitness}')

    evolution_module.save_metrics(run_path)
    best_model.print_model()
    print('---')
    best_test_model.print_model()
    df = pd.DataFrame(best_test_model.model)
    df.to_csv(f'{run_path}/best_model.csv', index=True)
    print(f'Complexity: {best_test_model.count_active_nodes()}')

    # Clean up checkpoint and temporary file if run completes successfully
    try:
        os.remove(CHECKPOINT_FILE)
        tmp_file = CHECKPOINT_FILE + ".tmp"
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        print("✅ Checkpoint and temporary file removed after successful run.")
    except Exception as e:
        print(f"⚠️ Could not remove checkpoint files: {e}")

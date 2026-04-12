import numpy as np
import os
from .cgp_evolver import CartesianGP
from .cgp_operators import add, sub, mul, div
from .test_problems import Collection
def set_up_cgp(x, y, seed):
    trial_number = 0
    max_generations = 30
    model_size = 32
    xover_type = None
    xover_rate = 0.0
    max_parents = 1
    max_children = 4
    mutation_type = "full"
    mutation_rate = 1.0
    selection_type = "paretoelite"
    fitness_function = "correlation"
    test_problem_key = "poet"
    n_points = 1
    tournament_size = 4
    n_elites = 1
    step_size = 20
    asex = True
    print(x.shape[-1])
    print(y.shape[-1])
    model_parameters = {
        'max_size': model_size,
        'inputs': x.shape[-1],
        'outputs': y.shape[-1],
        'arity': 2,
        'constants': np.array([])
    }
    function_bank = {'add': add, 'sub': sub, 'mul': mul, 'div': div}

    if asex or max_parents < max_children:
        mutation_breeding = True
    else:
        mutation_breeding = False
    CHECKPOINT_PATH = os.path.join(os.environ.get("SCRATCH", "/tmp"), "ckpt")
    CHECKPOINT_FILE = f"{CHECKPOINT_PATH}/test_{test_problem_key}_trial_{trial_number}_ckpt.pkl"

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
            one_dimensional_xover=False,
            seed=seed,
            tuning=False
        )
    return evolution_module

problems = Collection()
test_function = problems("EnergyEfficiency")
train_x, test_x, train_y, test_y = test_function.return_points()
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

evolution_module = set_up_cgp(train_x, train_y, 0)
best_model, _ = evolution_module.fit(train_x, test_x, train_y, test_y)

print(best_model([0, 0, 0, 0, 0, 1.2, 1, 2]))
print(best_model.slope)
print(best_model.intercept)

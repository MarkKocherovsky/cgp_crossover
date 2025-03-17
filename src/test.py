from cgp_evolver import CartesianGP
from cgp_operators import add, sub, mul, div
import matplotlib.pyplot as plt
import numpy as np
# Mock test for CartesianGP initialization
np.random.seed(0)
def test_cartesian_gp_init():
    # Define some mock parameters
    model_params = {"max_size": 64}
    function_bank = [add, sub, mul, div]

    # Initialize the CartesianGP instance
    cgp = CartesianGP(
        parents=2,
        children=4,
        max_generations=200,
        mutation="point",
        selection="elite",
        xover=None,
        fixed_length=True,
        fitness_function="correlation",
        model_parameters=model_params,
        function_bank=function_bank,
        solution_threshold=0.001,
        tournament_size= 5,
        n_points= 1,
        n_elites= 40
    )
    x = np.atleast_2d(np.arange(-5, 5.01, 0.5)).T
    y = x
    best_model = cgp.fit(x,y)

    preds = best_model(x)
    print(best_model.fitness)
    print(best_model.model)
    print("Test passed: CartesianGP initialized correctly!")
    print(preds)
    print(cgp.metrics)

# Run the test
test_cartesian_gp_init()

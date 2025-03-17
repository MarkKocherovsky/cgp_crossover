import numpy as np
from cgp_operators import add, sub, mul, div
from cgp_generator import generate_model
from fitness_functions import correlation
from cgp_model import CGP  # Ensure this points to your optimized CGP class file

# Set seed for reproducibility
np.random.seed(42)

# Define test parameters
inputs = 3
outputs = 2
arity = 2
max_size = 10
constants = [0.1, 0.5, 1.0]
function_bank = (add, sub, mul, div)
fixed_length = True

# Generate a test model
test_model = generate_model(
    max_size, inputs, constants, arity, outputs, len(function_bank), function_bank, fixed_length
)

# Initialize CGP instance with generated model
cgp_instance = CGP(model=test_model, fitness_function='Correlation')

# Print initial model details
print("Initial Model Parameters:")
cgp_instance.print_parameters()

# Generate test input data
test_input = np.random.rand(5, inputs)  # 5 test cases, 3 input features each
print("\nTest Input Data:")
print(test_input)

# Evaluate model output
output = cgp_instance(test_input)
print("\nModel Output:")
print(output)

# Test active node tracking
active_nodes = cgp_instance.get_active_nodes()
print("\nActive Nodes:")
print(active_nodes)

# Count active nodes
num_active_nodes = cgp_instance.count_active_nodes()
print(f"\nNumber of Active Nodes: {num_active_nodes}")

# Perform mutation
print("\nPerforming Mutation...")
cgp_instance.mutate()

# Validate mutation
mutated_output = cgp_instance(test_input)
print("\nMutated Model Output:")
print(mutated_output)

# Test fitness function with dummy ground truth
ground_truth = np.random.rand(5, outputs)
fitness = cgp_instance.fit(test_input, ground_truth)
print(f"\nModel Fitness: {fitness}")

# Ensure fitness computation is valid
assert isinstance(fitness, (float, np.float64)), "Fitness function should return a float value!"

print("\n✅ All tests passed successfully!")

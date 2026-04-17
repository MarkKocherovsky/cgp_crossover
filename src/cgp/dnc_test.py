from dnc.multiparent_wrapper import NeuralCrossoverWrapper
from dnc.multiparent_dnc import NeuralCrossover
from fitness_functions import correlation
model_size = 64
arity = 2
sequence_length = model_size*(1+arity+1+1)
dnc_hyperparameters = {
    'embedding_dim': 32,
    'sequence_length': sequence_length,
    'input_dim': 6,
    'get_fitness_function': correlation
}

dnc = NeuralCrossoverWrapper(**dnc_hyperparameters, crossover_type='uniform', n_points = 1)
def print_hyperparams(model: NeuralCrossover):
    keys = [
        'input_size', 'hidden_size', 'input_dim', 'ind_length',
        'num_layers', 'dropout', 'n_parents', 'device', 'epsilon_greedy'
    ]
    for key in keys:
        val = getattr(model, key, '[missing]')
        print(f"{key}: {val}")

print_hyperparams(dnc.neural_crossover)
print(f'batch_size: {dnc.batch_size}')

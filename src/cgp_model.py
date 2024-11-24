import numpy as np
import pandas as pd
from numpy import random

from cgp_generator import generate_model
from cgp_operators import add, sub, mul, div
from fitness_functions import correlation, align


def _validate_model(model):
    """Ensure the provided model is valid."""
    required_columns = {'NodeType', 'Operator', 'Operand0', 'Value'}
    assert isinstance(model, pd.DataFrame), "Pre-built model must be a Pandas DataFrame."
    assert required_columns.issubset(model.columns), f"Model must contain the required columns: {required_columns}"


class CGP:
    def __init__(self, model=None, fixed_length=True, fitness_function='Correlation', mutation_type='Point', **kwargs):
        self.mutation = None
        self.slope = None
        self.intercept = None
        self.fitness = None
        self.fixed_length = fixed_length
        if fitness_function == 'Correlation':
            self.fitness_function = correlation
        else:
            raise ValueError(f'Invalid Fitness Function: {fitness_function}')

        # Set the function bank and number of operations
        self.function_bank = kwargs.get('function_bank', (add, sub, mul, div))
        self.n_operations = len(self.function_bank)
        assert self.n_operations >= 1, 'There has to be at least one operator.'

        if model is not None:
            _validate_model(model)
            self.model = model
            self._initialize_from_model()
        else:
            self._initialize_from_kwargs(kwargs)
        self.first_body_node = self.inputs + len(self.constants)
        self.last_body_node = self.inputs + len(self.constants) + self.max_size - 1

        self._choose_mutation(mutation_type)

    def _initialize_from_model(self):
        """Initialize attributes from a pre-built model."""
        self.constants = self.model['NodeType'][self.model['NodeType'] == 'Constants']
        self.inputs = len(self.model['NodeType'][self.model['NodeType'] == 'Input'])
        self.outputs = len(self.model['NodeType'][self.model['NodeType'] == 'Output'])
        self.max_size = len(self.model) - self.inputs - len(self.constants)
        self.n_operations = self.model['Operator'].nunique()
        self.arity = len(self.model.filter(regex='Operand').columns)

    def _initialize_from_kwargs(self, kwargs):
        """Initialize attributes from keyword arguments."""
        self.constants = kwargs.get('constants', np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

        self.inputs = kwargs.get('inputs', 1)
        assert self.inputs >= 1, 'There has to be at least one input feature.'

        self.outputs = kwargs.get('outputs', 1)
        assert self.outputs >= 1, 'There has to be at least one output.'

        self.arity = kwargs.get('arity', 2)
        assert self.arity >= 1, 'Operators have to take at least one input.'

        self.max_size = kwargs.get('max_size', 16)
        assert self.max_size >= 1, 'There has to be at least one instruction.'

        # Generate a new model based on the provided or default parameters
        self.model = generate_model(
            self.max_size, self.inputs, self.constants, self.arity, self.outputs, self.n_operations,
            self.function_bank, self.fixed_length,
        )

    def print_model(self):
        print(self.model)

    def print_parameters(self):
        print(f'Constants\t {self.constants}'
              f'\nInputs\t {self.inputs}'
              f'\nOutputs\t {self.outputs}'
              f'\nArity\t {self.arity}'
              f'\nMaximum Number of Instructions\t {self.max_size}'
              f'\nFunctions\t {self.function_bank}'
              f'\nNumber of Functions\t {self.n_operations}')

    def to_csv(self, filename='model.csv'):
        self.model.to_csv(filename)

    def __call__(self, data: float | list | np.ndarray):
        data = np.atleast_2d(data)  # Ensure data is at least 2D
        if self.inputs == 1:
            data = data.flatten()  # Flatten if single input
            outputs = np.array([self._compute_single_input(datum) for datum in data])
        else:
            outputs = np.array([self._compute_multiple_inputs(datum) for datum in data])
        return outputs

    def _get_node_value(self, operand):
        new_node = self.model.loc[operand]
        node_type = new_node['NodeType']

        if node_type in {'Input', 'Constant'}:
            return new_node['Value']

        if node_type == 'Function':
            # Retrieve operand values recursively
            new_operands = [self._get_node_value(val) for val in new_node.filter(regex='Operand')]
            try:
                return new_node['Operator'](*new_operands)
            except TypeError as e:
                print(e)
                print(self.model)
                print(*new_operands)
                print(new_node['Operator'])
                exit()

        # Handle invalid node type
        print(self.model)
        raise ValueError(f"Invalid node type: {node_type}")

    def _run(self):
        # Get the rows where NodeType is 'Output'
        output_nodes = self.model[self.model['NodeType'] == 'Output']

        # Update the 'Value' column with computed values
        new_values = [self._get_node_value(o) for o in output_nodes['Operand0']]
        self.model.loc[output_nodes.index, 'Value'] = new_values
        return new_values

    def _compute_single_input(self, datum):
        """Handle computation for single input."""
        self.model.loc[self.model['NodeType'] == 'Input', 'Value'] = datum
        return self._run()

    def _compute_multiple_inputs(self, datum):
        """Handle computation for multiple inputs."""
        input_nodes = self.model.loc[self.model['NodeType'] == 'Input']
        input_nodes['Value'] = datum
        return self._run()

    def fit(self, data, ground_truth):
        predictions = self.__call__(data)
        self.fitness = self.fitness_function(predictions, ground_truth)
        if self.fitness_function == correlation:
            self.slope, self.intercept = align(predictions, ground_truth)
        return self.fitness

    def get_active_nodes(self):
        # Start with output nodes
        output_nodes = self.model[self.model['NodeType'] == 'Output']
        active_nodes = set()  # Use a set to avoid duplicates

        # Recursively find all active nodes
        def trace_dependencies(current_node_index):
            node = self.model.loc[current_node_index]
            if node['NodeType'] == 'Function':
                # Add the function node to active nodes
                active_nodes.add(current_node_index)
                # Trace its operands
                operands = node.filter(regex='Operand').dropna()
                for operand in operands:
                    operand_index = int(operand)
                    if operand_index not in active_nodes:
                        trace_dependencies(operand_index)

        # Trace all output nodes
        for node_index in output_nodes['Operand0']:
            trace_dependencies(node_index)

        return list(active_nodes)

    def _choose_mutation(self, mutation_type):
        """Select and assign the mutation function based on the mutation type."""
        mutation_type = mutation_type.lower()
        possible_mutation_functions = {
            'point': self._point_mutation
        }
        if mutation_type not in possible_mutation_functions:
            raise AttributeError(f'{mutation_type} is an invalid mutation operator.')
        self.mutation = possible_mutation_functions[mutation_type]

    def _point_mutation(self):
        """Simple point mutation on function and output nodes."""
        # Mutate only function and output nodes
        mutation_index = random.randint(self.first_body_node, self.last_body_node)

        # Access the row and determine the mutation
        if self.model.at[mutation_index, 'NodeType'] == 'Function':
            mutation_column = random.choice(['Operator'] + [f'Operand{n}' for n in range(self.arity)])

            if mutation_column == 'Operator':
                self.model.at[mutation_index, mutation_column] = random.choice(self.function_bank)
            elif mutation_column.startswith('Operand'):
                self.model.at[mutation_index, mutation_column] = random.randint(0, mutation_index)
            else:
                raise AttributeError(f'Column {mutation_column} not recognized for Point Mutation')

        elif self.model.at[mutation_index, 'NodeType'] == 'Output':
            mutation_column = random.choice([f'Operand{n}' for n in range(self.outputs)])
            self.model.at[mutation_index, mutation_column] = random.randint(0, self.last_body_node)

    def mutate(self):
        """Perform mutation using the assigned mutation function."""
        if self.mutation is None:
            raise ValueError("Mutation function is not selected. Call _choose_mutation() first.")
        self.mutation()

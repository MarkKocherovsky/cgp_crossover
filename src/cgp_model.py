import numpy as np
import pandas as pd

from cgp_generator import generate_model
from cgp_operators import add, sub, mul, div
from fitness_functions import correlation, align


def _validate_model(model):
    """Ensure the provided model is valid."""
    required_columns = {'NodeType', 'Operator', 'Operand', 'Value'}
    assert isinstance(model, pd.DataFrame), "Pre-built model must be a Pandas DataFrame."
    assert required_columns.issubset(model.columns), f"Model must contain the required columns: {required_columns}"


class CGP:
    def __init__(self, model=None, fixed_length=True, fitness_function='Correlation', **kwargs):
        self.slope = None
        self.intercept = None
        self.fitness = None
        self.fixed_length = fixed_length
        if fitness_function == 'Correlation':
            self.fitness_function = correlation
        else:
            raise ValueError(f'Invalid Fitness Function: {fitness_function}')

        if model is not None:
            _validate_model(model)
            self.model = model
            self._initialize_from_model()
        else:
            self._initialize_from_kwargs(kwargs)

    def _initialize_from_model(self):
        """Initialize attributes from a pre-built model."""
        self.constants = self.model['NodeType'][self.model['NodeType'] == 'Constants']
        self.inputs = len(self.model['NodeType'][self.model['NodeType'] == 'Input'])
        self.outputs = len(self.model['NodeType'][self.model['NodeType'] == 'Output'])
        self.max_size = len(self.model) - self.inputs - len(self.constants)
        self.n_operations = self.model['Operator'].nunique()
        self.function_bank = tuple(self.model['Operator'].unique())
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

        # Set the function bank and number of operations
        self.function_bank = kwargs.get('function_bank', (add, sub, mul, div))
        self.n_operations = len(self.function_bank)
        assert self.n_operations >= 1, 'There has to be at least one operator.'

        # Generate a new model based on the provided or default parameters
        self.model = generate_model(
            self.max_size, self.inputs, self.constants, self.arity, self.outputs,
            self.n_operations, self.function_bank, self.fixed_length
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
            return new_node['Operator'](*new_operands)

        # Handle invalid node types
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

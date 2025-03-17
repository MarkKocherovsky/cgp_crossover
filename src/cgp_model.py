import numpy as np
from numpy.random import randint, choice
from cgp_generator import generate_model
from cgp_operators import add, sub, mul, div
from fitness_functions import correlation, align


class CGP:
    def __init__(self, model=None, fixed_length=True, fitness_function='Correlation', mutation_type='Point',
                 parent_keys=None, **kwargs):

        self.mutation = None
        self.slope = None
        self.intercept = None
        self.fitness = None
        self.fixed_length = fixed_length
        self.parent_keys = parent_keys
        self.better_than_parents = None

        # Assign fitness function
        if fitness_function.lower() == 'correlation':
            self.fitness_function = correlation
        else:
            raise ValueError(f'Invalid Fitness Function: {fitness_function}')

        # Function bank setup
        self.function_bank = kwargs.get('function_bank', (add, sub, mul, div))
        self.n_operations = len(self.function_bank)
        assert self.n_operations >= 1, 'At least one operator is required.'

        # Initialize model
        if model is not None:
            self.model = model
            self._initialize_from_model()
        else:
            self._initialize_from_kwargs(kwargs)

        self.first_body_node = self.inputs + len(self.model[self.model['NodeType'] == 'Constant'])
        self.last_body_node = self.first_body_node + self.max_size - 1
        self.xover_index = np.zeros(self.max_size + self.outputs)

        self._choose_mutation(mutation_type)

    def _initialize_from_model(self):
        """Initialize attributes from a pre-built model."""
        self.constants = self.model['Value'][self.model['NodeType'] == 'Constant']
        self.inputs = np.sum(self.model['NodeType'] == 'Input')
        self.outputs = np.sum(self.model['NodeType'] == 'Output')
        self.max_size = np.sum(self.model['NodeType'] == 'Function')
        self.n_operations = len(set(self.model['Operator']))
        self.arity = sum(name.startswith('Operand') for name in self.model.dtype.names)

    def _initialize_from_kwargs(self, kwargs):
        """Initialize attributes from keyword arguments."""
        #self.constants = np.array(kwargs.get('constants', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        self.constants = np.atleast_1d(kwargs.get('constants', [1]))
        self.inputs = kwargs.get('inputs', 1)
        self.outputs = kwargs.get('outputs', 1)
        self.arity = kwargs.get('arity', 2)
        self.max_size = kwargs.get('max_size', 16)

        assert self.inputs >= 1, 'There must be at least one input feature.'
        assert self.outputs >= 1, 'There must be at least one output.'
        assert self.arity >= 1, 'Operators must take at least one input.'
        assert self.max_size >= 1, 'There must be at least one instruction.'

        # Generate the model using NumPy instead of Pandas
        self.model = generate_model(
            self.max_size, self.inputs, self.constants, self.arity, self.outputs,
            self.n_operations, self.function_bank, self.fixed_length
        )

    def __call__(self, data):
        data = np.atleast_2d(data)  # Ensure data is at least 2D
        outputs = np.array([self._compute_single_input(datum) for datum in data])

        # Apply scaling if needed
        if self.slope is not None and self.intercept is not None:
            outputs = outputs * self.slope + self.intercept
        return outputs

    def _get_node_value(self, operand):
        """Compute the value of a given node using NumPy array indexing."""
        node = self.model[operand]
        node_type = node['NodeType']

        if node_type in {'Input', 'Constant'}:
            return node['Value']

        if node_type == 'Function':
            # Get operand values recursively
            operand_values = np.array([self._get_node_value(node[f'Operand{i}']) for i in range(self.arity)])
            self.model[operand]['Active'] = 1  # Mark as active node
            operator = node['Operator']
            result = operator(*operand_values)

            self.model[operand]['Value'] = result
            return result

        raise ValueError(f"Invalid node type: {node_type}")

    def _run(self):
        """Run the model and compute output values."""
        self.model['Active'] = 0  # Reset active nodes
        self.model['Value'][self.model['NodeType'] == 'Function'] = 0  # Reset function nodes

        output_indices = np.where(self.model['NodeType'] == 'Output')[0]
        output_values = np.array([self._get_node_value(self.model[i]['Operand0']) for i in output_indices])

        self.model['Value'][output_indices] = output_values
        return output_values

    def _compute_single_input(self, datum):
        """Compute output for a single input."""
        datum = np.asarray(datum, dtype=np.float64).item()
        input_indices = np.where(self.model['NodeType'] == 'Input')[0]
        input_indices = np.atleast_1d(input_indices)  # Ensure it's an array
        self.model['Value'][input_indices] = np.float64(datum)


        return self._run()

    def fit(self, data, ground_truth):
        predictions = self.__call__(data)
        self.fitness = self.fitness_function(predictions, ground_truth)

        if self.fitness_function == correlation:
            self.slope, self.intercept = align(predictions, ground_truth)

        return self.fitness

    def get_active_nodes(self):
        return self.model[self.model['Active'] == 1]

    def count_active_nodes(self):
        return np.sum(self.model['Active'] == 1)

    def _choose_mutation(self, mutation_type):
        """Assign mutation function."""
        mutation_type = mutation_type.lower()
        mutations = {'point': self._point_mutation}
        if mutation_type not in mutations:
            raise ValueError(f'{mutation_type} is an invalid mutation operator.')
        self.mutation = mutations[mutation_type]

    def _point_mutation(self):
        """Efficient point mutation ensuring changes affect active nodes."""
        #active_indices = np.where((self.model['NodeType'] == 'Function') & (self.model['Active'] == 1))[0]
        active_indices = np.where((self.model['NodeType'] == 'Function') | (self.model['NodeType'] == 'Output'))[0]

        if len(active_indices) == 0:
            # Fallback: Mutate any function node if no active ones exist
            active_indices = np.where((self.model['NodeType'] == 'Function') | (self.model['NodeType'] == 'Output'))[0]

        # Select a random active function node
        mutation_index = np.random.choice(active_indices)
        if self.model[mutation_index]['NodeType'] == 'Function':
            mutation_column = np.random.choice(['Operator'] + [f'Operand{i}' for i in range(self.arity)])

            if mutation_column == 'Operator':
                # Ensure new operator is different
                new_operator = np.random.choice(self.function_bank)
                while new_operator == self.model[mutation_index]['Operator']:
                    new_operator = np.random.choice(self.function_bank)
                self.model[mutation_index]['Operator'] = new_operator
            else:
                # Mutate operand, ensuring a different value
                new_operand = np.random.randint(0, mutation_index)
                while new_operand == self.model[mutation_index][mutation_column]:
                    new_operand = np.random.randint(0, mutation_index)
                self.model[mutation_index][mutation_column] = new_operand
        else: #mutate output node
            new_operand = np.random.randint(0, mutation_index)
            self.model[mutation_index]['Operand0'] = new_operand

    def mutate(self):
        """Perform mutation using the assigned function."""
        if self.mutation is None:
            raise ValueError("Mutation function not set. Call _choose_mutation() first.")
        self.mutation()

    def print_parameters(self):
        """Print key parameters of the CGP model."""
        print(f"Constants: {self.constants}")
        print(f"Inputs: {self.inputs}")
        print(f"Outputs: {self.outputs}")
        print(f"Arity: {self.arity}")
        print(f"Max Instructions: {self.max_size}")
        print(f"Function Bank: {[func.__name__ for func in self.function_bank]}")
        print(f"Number of Functions: {self.n_operations}")
    def set_parent_key(self, key):
        self.parent_keys = key

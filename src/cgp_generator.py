import numpy as np
from cgp_operators import *


def generate_model(max_size: int, inputs: int, constants: list | np.ndarray, arity: int, outputs: int,
                   n_operations: int, function_bank: tuple, fixed_length: bool = True):
    """
    Generates a CGP model using NumPy arrays instead of pandas.

    Args:
        max_size (int): Maximum number of body nodes.
        inputs (int): Number of variable inputs.
        constants (list | np.ndarray): List of constants.
        arity (int): Number of arguments per function node.
        outputs (int): Number of outputs.
        n_operations (int): Number of operations in the function bank.
        function_bank (tuple): Tuple of available operations.
        fixed_length (bool): If False, the model size can be <= max_size.

    Returns:
        np.ndarray: Structured NumPy array representing the model.
    """

    constants = np.array(constants) if isinstance(constants, list) else constants

    # Define structured NumPy dtype for the model
    dtype = [
        ('NodeType', 'U8'),  # 'Input', 'Constant', 'Function', or 'Output'
        ('Value', 'f8'),  # Value for constants (or 0 for others)
        ('Operator', 'U10'),  # Function reference (only for function nodes)
        *[(f'Operand{i}', 'i4') for i in range(arity)],  # Operands (for function nodes)
        ('Active', 'i4')  # Active status (0 or 1)
    ]

    # Input and Constant Nodes
    num_constants = len(constants)
    num_inputs = inputs + num_constants
    input_nodes = np.zeros(num_inputs, dtype=dtype)
    input_nodes['NodeType'][:inputs] = 'Input'
    input_nodes['NodeType'][inputs:] = 'Constant'
    input_nodes['Value'][inputs:] = constants  # Assign constant values

    # Determine the actual number of function nodes
    model_size = max_size if fixed_length else np.random.randint(1, max_size)
    first_body_node = num_inputs
    last_body_node = first_body_node + model_size

    # Function Nodes
    body_nodes = np.zeros(model_size, dtype=dtype)
    body_nodes['NodeType'] = 'Function'
    body_nodes['Operator'] = np.random.choice(list(function_bank), model_size)

    # Generate operands correctly
    for i in range(arity):
        body_nodes[f'Operand{i}'] = np.random.randint(0, first_body_node + np.arange(model_size), size=model_size)

    # Output Nodes
    output_nodes = np.zeros(outputs, dtype=dtype)
    output_nodes['NodeType'] = 'Output'
    output_nodes['Operand0'] = np.random.randint(0, last_body_node, outputs)

    # Combine all nodes into a single structured NumPy array
    model = np.concatenate((input_nodes, body_nodes, output_nodes))

    return model

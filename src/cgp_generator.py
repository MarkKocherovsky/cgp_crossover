import pandas as pd

from cgp_operators import *


def generate_model(max_size: int, inputs: int, constants: list | np.ndarray, arity: int, outputs: int,
                   n_operations: int, function_bank: tuple, fixed_length: bool = True):
    """
    @param function_bank:
    @param n_operations: number of operations
    @param max_size: maximum number of body node
    @param inputs: number of variable inputs
    @param constants: list of constants
    @param arity: number of arguments
    @param outputs: number of outputs
    @param fixed_length: whether the model has to be exactly max_size. If false, the model can be <= max_size
    """

    # Ensure constants is a list, even if tuple was provided
    constants = list(constants) if not isinstance(constants, list) else constants

    # Initialize input nodes with 'Input' and 'Value' columns
    input_values = [('Input', 0)] * inputs + [('Constant', const) for const in constants]
    input_dataframe = pd.DataFrame(data=input_values, columns=['NodeType', 'Value'])

    # Generate function body nodes with random operations and operands
    first_body_node = inputs + len(constants)
    last_body_node = max_size + first_body_node

    # Randomly populate the 'Operator' and 'Operand' columns
    model_size = max_size if fixed_length else np.random.randint(1, max_size)

    # Generate body nodes
    body = np.hstack([
        np.random.randint(0, n_operations, (model_size, 1)),  # Random operators
        np.array([np.random.randint(0, row_idx + first_body_node, arity) for row_idx in range(model_size)])  # Random operands
    ])

    columns = ['Operator'] + [f'Operand{i}' for i in range(arity)]
    body_dataframe = pd.DataFrame(data=body, columns=columns)
    body_dataframe.insert(0, 'NodeType', 'Function')
    body_dataframe['Operator'] = [function_bank[op] for op in body_dataframe['Operator']]

    # Ensure Operand columns are integers
    body_dataframe[columns[1:]] = body_dataframe[columns[1:]].astype(int)

    # Define output nodes with random connections to body nodes
    output_values = [('Output', 0, node) for node in np.random.randint(0, last_body_node, outputs)]
    output_dataframe = pd.DataFrame(data=output_values, columns=['NodeType', 'Value', 'Operand0'])

    # Ensure Value column in output nodes is integer
    output_dataframe['Value'] = output_dataframe['Value'].astype(int)

    # Combine input, function body, and output dataframes into the final model
    model = pd.concat([input_dataframe, body_dataframe, output_dataframe], ignore_index=True)
    return model

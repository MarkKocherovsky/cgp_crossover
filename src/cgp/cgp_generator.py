import numpy as np
from functools import lru_cache as _lru_cache

# Preserve the module's existing re-export behavior for operator symbols.
from .cgp_operators import *


_NODE_TYPES = ('Input', 'Constant', 'Function', 'Output')
_NODE_TO_INT = {node: index for index, node in enumerate(_NODE_TYPES)}


@_lru_cache(maxsize=None)
def _model_key_template(arity: int) -> dict[str, int]:
    """Build the column map once per arity; callers receive a copy."""
    keys = ('NodeType', 'Value', 'Operator',
            *(f'Operand{i}' for i in range(arity)),
            'Active')
    return {key: index for index, key in enumerate(keys)}


def int_to_node(i):
    return _NODE_TYPES[i]


def node_to_int(node):
    try:
        return _NODE_TO_INT[node]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f'Tried to convert node {node} to an integer. The only valid types are '
            '"Input", "Constant", "Function", and "Output"'
        ) from exc
    except RecursionError as exc:
        raise RecursionError(
            f'Recursion Error in cgp_generator.py::node_to_int\tTried to call {node}'
        ) from exc


def generate_model(max_size: int, inputs: int, constants: list | np.ndarray, arity: int, outputs: int,
                   n_operations: int, function_bank: dict, fixed_length: bool = True):
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
        fixed_length (bool): If False, the model size can be < max_size.

    Returns:
        np.ndarray: NumPy array representing the model.
        dict: Mapping from model-column names to column indices.

    Structured NumPy arrays are intentionally not used because of copy issues.

    Notes:
        ``n_operations`` remains in the signature for API compatibility. As in
        the original implementation, operator IDs are sampled from the actual
        length of ``function_bank``.
    """
    constants = np.array(constants) if isinstance(constants, list) else constants

    # Return a fresh dict, preserving the original function's ownership semantics
    # while avoiding repeated string construction for common arities.
    model_keys = _model_key_template(arity).copy()
    num_keys = len(model_keys)

    num_constants = len(constants)
    num_inputs = inputs + num_constants

    # Preserve the original variable-length sampling behavior and RNG call order.
    randint = np.random.randint
    model_size = max_size if fixed_length else randint(1, max_size)
    first_body_node = num_inputs
    last_body_node = first_body_node + model_size

    # Allocate the returned matrix directly. This avoids three temporary matrices
    # plus the full-array copy performed by np.concatenate.
    model = np.zeros((last_body_node + outputs, num_keys))

    # These positions are invariant for every arity in the returned schema.
    node_type_col = 0
    value_col = 1
    operator_col = 2
    first_operand_col = 3

    # Input rows already have NodeType == 0 because the model is zero-initialized.
    if num_constants:
        constant_rows = slice(inputs, num_inputs)
        model[constant_rows, node_type_col] = 1
        model[constant_rows, value_col] = constants

    body_rows = slice(first_body_node, last_body_node)
    model[body_rows, node_type_col] = 2
    model[body_rows, operator_col] = randint(0, len(function_bank), model_size)

    # Every body node may reference inputs/constants or an earlier body node.
    # Reusing this array avoids rebuilding it once per operand column.
    operand_upper_bounds = first_body_node + np.arange(model_size)
    for operand_col in range(first_operand_col, first_operand_col + arity):
        model[body_rows, operand_col] = randint(
            0, operand_upper_bounds, size=model_size
        )

    output_rows = slice(last_body_node, last_body_node + outputs)
    model[output_rows, node_type_col] = 3
    model[output_rows, model_keys['Operand0']] = randint(
        0, last_body_node, outputs
    )

    return model, model_keys
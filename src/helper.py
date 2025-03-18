import numpy as np


def _get_quartiles(data):
    data = np.where(data == np.inf, 1, data)
    return np.quantile(data, [0, 0.25, 0.5, 0.75, 1])


def _validate_int_param(param_name, value, min_val, max_val):
    """
    Validates and converts a parameter to an integer if necessary.
    Raises errors or issues warnings as needed.
    """
    if not isinstance(value, int):
        print(f"Warning: {param_name} {value} is not an integer. Rounding to {int(value)}.")
        value = int(value)
    assert min_val <= value <= max_val, (
        f"{param_name} must be between {min_val} and {max_val}. Current value: {value}."
    )
    return value


def _split(m, points: [int]):
    previous_point = 0
    parts = []
    points = np.append(points, len(m))
    for point in points:
        parts.append(m[previous_point:point])
        previous_point = point

    return parts


def pairwise_minkowski_distance(xa, xb, p=2):
    """
    Computes pairwise Minkowski distance between two collections of 1D arrays.
    :param xa: First 2D array or 1D array
    :param xb: Second 2D array or 1D array
    :param p: Minkowski distance parameter
    :return: Pairwise Minkowski distances as a 2D array
    """
    xa = np.atleast_2d(xa)  # Ensure XA is 2D
    xb = np.atleast_2d(xb)  # Ensure XB is 2D
    distance = np.sum(np.abs(xa[:, None] - xb) ** p, axis=-1) ** (1 / p)
    return distance[0, 0]


def get_score(p_to_t, p_to_m, m_to_t):
    """

    @param p_to_t: distance from parent to target
    @param p_to_m: distance from potential mate to parent
    @param m_to_t: distance from potential mate to target
    """
    if p_to_m == 0:
        return np.inf
    return (m_to_t / p_to_m) * (1 + np.abs(m_to_t - p_to_t))


def clean_values(model, x_train, include_output=False):
    """
    Gets the values matrix for each input.

    @param model: CGP model
    @param x_train: NumPy array of x inputs
    @param include_output: Whether output values are included in the set
    @return: Matrix of values with rows as node numbers and columns as inputs
    """
    # Get boolean masks for function and output nodes
    function_mask = model.model['NodeType'] == 'Function'
    output_mask = model.model['NodeType'] == 'Output'

    # Compute matrix size
    model_length = np.sum(function_mask)
    if include_output:
        model_length += np.sum(output_mask)

    number_of_inputs = x_train.shape[0]
    values_matrix = np.zeros((model_length, number_of_inputs))

    # Iterate over inputs
    for i, input_value in enumerate(x_train):
        model(input_value)  # Run the model with the input

        # Extract function node values
        function_values = model.model['Value'][function_mask]

        if include_output:
            # Extract output node values
            output_values = model.model['Value'][output_mask]
            values = np.concatenate([function_values, output_values])
        else:
            values = function_values

        values_matrix[:, i] = values

    # Ensure all values are finite, replacing NaNs/Infs with 0
    values_matrix[~np.isfinite(values_matrix)] = 0

    return values_matrix


"""
Uy, N. Q., Hien, N. T., Hoai, N. X., & O’Neill, M. (2010). Improving the generalisation ability of genetic
 programming with semantic similarity based crossover. In Genetic Programming: 13th European Conference, 
 EuroGP 2010, Istanbul, Turkey, April 7-9, 2010. Proceedings 13 (pp. 184-195). Springer Berlin Heidelberg.
"""


# instead of picking a point at random, we get the SSD of all points and then later can pick N according to operator
# , alpha=1e-4, beta=0.4
def get_ssd(v_matrix_1, v_matrix_2):
    assert v_matrix_1.shape == v_matrix_2.shape, "Internal Semantic Matrices must be the same size"
    return np.sum(np.abs(v_matrix_1 - v_matrix_2), axis=1) / v_matrix_1.shape[1]


def get_weights(ssd_matrix: np.ndarray, alpha: float = 1e-4, beta: float = 0.4, epsilon: float = 0.0):
    # Set weights to zero where conditions are met
    weight_matrix = np.where((ssd_matrix < alpha) | (ssd_matrix >= beta), epsilon, ssd_matrix)
    total = np.sum(weight_matrix)

    # Normalize weights or use uniform distribution if the sum is zero
    return weight_matrix / total if total != 0 else np.full_like(ssd_matrix, 1 / ssd_matrix.size)



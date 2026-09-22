"""Cartesian Genetic Programming individual representation.

This module keeps the public API and model layout of the original implementation,
while reducing repeated allocations and Python-level work in evaluation, mutation,
copying, LLM export, and hashing.
"""

from copy import deepcopy
import hashlib
import uuid

import numpy as np

from .cgp_generator import generate_model, node_to_int
from .cgp_operators import add, div, mul, sub
from .fitness_functions import align, corr_comp_fitness, correlation


# Resolve these once rather than repeatedly constructing/searching node-name lists.
_INPUT_NODE = node_to_int("Input")
_CONSTANT_NODE = node_to_int("Constant")
_FUNCTION_NODE = node_to_int("Function")
_OUTPUT_NODE = node_to_int("Output")
_FAST_NUMERIC_OPERATORS = (add, sub, mul, div)


class CGP:
    def __init__(
        self,
        model=None,
        model_keys=None,
        fixed_length=True,
        fitness_function="Correlation",
        mutation_type="Point",
        parent_keys=None,
        xover_length=None,
        **kwargs,
    ):
        self.correlation = 1.0
        self.complexity = 1.0
        self.id = uuid.uuid4()
        self.mutation = None
        self.slope = None
        self.intercept = None
        self.fitness = None
        self.fixed_length = fixed_length
        self.parent_keys = parent_keys
        self.child_keys = None
        self.better_than_parents = None
        self.visited = set()

        fitness_name = fitness_function.lower()
        if fitness_name == "correlation":
            self.fitness_function = correlation
        elif fitness_name == "correlation_complexity":
            self.fitness_function = corr_comp_fitness
        else:
            raise ValueError(f"Invalid Fitness Function: {fitness_function}")

        supplied_bank = kwargs.get(
            "function_bank", {"add": add, "sub": sub, "mul": mul, "div": div}
        )
        # Preserve the original public representation: integer -> callable.
        self.function_bank = dict(enumerate(supplied_bank.values()))
        self.n_operations = len(self.function_bank)
        assert self.n_operations >= 1, "At least one operator is required."

        if model is not None and model_keys is not None:
            self.model = model
            self.model_keys = model_keys
            self._initialize_from_model()
        elif (model is None) != (model_keys is None):
            raise RuntimeError("Must specify both model and model_keys.")
        else:
            self._initialize_from_kwargs(kwargs)

        self._initialize_column_cache()

        # Inputs and constants are contiguous at the beginning of a CGP model.
        self.first_body_node = self.inputs + len(self.constants)
        self.last_body_node = self.first_body_node + self.max_size - 1

        xover_size = xover_length if xover_length is not None else self.max_size + self.outputs
        self.xover_index = np.zeros(xover_size)

        self._choose_mutation(mutation_type)

    def _initialize_column_cache(self):
        """Cache immutable column positions used in hot loops."""
        keys = self.model_keys
        self._node_type_col = keys["NodeType"]
        self._value_col = keys["Value"]
        self._operator_col = keys["Operator"]
        self._active_col = keys["Active"]
        self._operand_cols = tuple(keys[f"Operand{i}"] for i in range(self.arity))
        self._mutation_cols = (self._operator_col, *self._operand_cols)

    def _initialize_from_model(self):
        """Initialize attributes from a pre-built model."""
        node_type_col = self.model_keys["NodeType"]
        value_col = self.model_keys["Value"]
        operator_col = self.model_keys["Operator"]
        node_types = self.model[:, node_type_col]

        constant_mask = node_types == _CONSTANT_NODE
        self.constants = self.model[constant_mask, value_col]
        self.inputs = np.count_nonzero(node_types == _INPUT_NODE)
        self.outputs = np.count_nonzero(node_types == _OUTPUT_NODE)
        self.max_size = np.count_nonzero(node_types == _FUNCTION_NODE)

        # Preserve the original definition, including zero-valued operator fields
        # belonging to non-function rows.
        self.n_operations = len(set(self.model[:, operator_col]))
        self.arity = sum(name.startswith("Operand") for name in self.model_keys)

    def _initialize_from_kwargs(self, kwargs):
        """Initialize attributes from keyword arguments."""
        self.constants = np.atleast_1d(kwargs.get("constants", [1]))
        self.inputs = kwargs.get("inputs", 1)
        self.outputs = kwargs.get("outputs", 1)
        self.arity = kwargs.get("arity", 2)
        self.max_size = kwargs.get("max_size", 16)

        assert self.inputs >= 1, "There must be at least one input feature."
        assert self.outputs >= 1, "There must be at least one output."
        assert self.arity >= 1, "Operators must take at least one input."
        assert self.max_size >= 1, "There must be at least one instruction."

        self.model, self.model_keys = generate_model(
            self.max_size,
            self.inputs,
            self.constants,
            self.arity,
            self.outputs,
            self.n_operations,
            self.function_bank,
            self.fixed_length,
        )

    def _index_error(self, model, operand, error):
        """Retain the original diagnostic behavior for invalid node indices."""
        print(f"_get_node_value(): {error}")
        print(f"operand: {operand}")
        print(model)
        print(f"model size: {model.shape}")
        raise SystemExit from error

    def _compile_execution_plans(self, model):
        """Compile depth-first evaluation sequences for all outputs.

        A separate sequence is retained for every output, and shared subgraphs are
        intentionally repeated. Consequently, operator invocation order and count
        match the original recursive evaluator for valid acyclic models.
        """
        node_type_col = self._node_type_col
        operator_col = self._operator_col
        operand_cols = self._operand_cols
        function_bank = self.function_bank
        arity = self.arity

        output_start = len(model) - self.outputs
        output_operands = model[
            output_start:, self._operand_cols[0]
        ].astype(np.intp, copy=True)

        plans = []
        all_active = set()

        for root in output_operands:
            root = int(root)
            node_indices = []
            operand_indices = []
            operators = []
            active_path = set()
            stack = [(root, False)]

            while stack:
                node_index, exiting = stack.pop()
                try:
                    node = model[node_index]
                except IndexError as error:
                    self._index_error(model, node_index, error)

                node_type = node[node_type_col]

                if exiting:
                    active_path.remove(node_index)
                    node_indices.append(node_index)
                    operand_row = tuple(int(node[col]) for col in operand_cols)
                    operand_indices.append(operand_row)
                    operators.append(function_bank[node[operator_col]])
                    all_active.add(node_index)
                    continue

                if node_type == _INPUT_NODE or node_type == _CONSTANT_NODE:
                    continue

                if node_type != _FUNCTION_NODE:
                    print(node)
                    raise ValueError(f"Invalid node type: {node_type}")

                if node_index in active_path:
                    raise RuntimeError(
                        f"Cycle detected at node {node_index} — already visited"
                    )

                active_path.add(node_index)
                stack.append((node_index, True))

                # Reverse push order so operands execute from Operand0 upward,
                # exactly as in the original recursive list comprehension.
                for col in reversed(operand_cols):
                    child = int(node[col])
                    if child in active_path:
                        raise RuntimeError(
                            f"Cycle detected at node {child} — already visited"
                        )
                    stack.append((child, False))

            if operand_indices:
                operand_array = np.asarray(operand_indices, dtype=np.intp)
            else:
                operand_array = np.empty((0, arity), dtype=np.intp)

            plans.append(
                (
                    np.asarray(node_indices, dtype=np.intp),
                    operand_array,
                    tuple(operators),
                    root,
                )
            )

        return tuple(plans), all_active

    @staticmethod
    def _sanitize_result(result, operand_values):
        if np.isfinite(result):
            return result
        print(
            "Warning: result of operation on "
            f"{operand_values} is infinite or invalid. Returning 0.0"
        )
        return 0.0

    def _evaluate_plans(self, values, plans, output_values):
        """Evaluate precompiled plans against one node-value vector."""
        arity = self.arity

        if arity == 1:
            for output_i, (nodes, operands, operators, root) in enumerate(plans):
                for node_index, operand_row, operator in zip(nodes, operands, operators):
                    result = operator(values[operand_row[0]])
                    if not np.isfinite(result):
                        result = self._sanitize_result(result, values[operand_row].copy())
                    values[node_index] = result
                output_values[output_i] = values[root]
            return

        if arity == 2:
            for output_i, (nodes, operands, operators, root) in enumerate(plans):
                for node_index, operand_row, operator in zip(nodes, operands, operators):
                    result = operator(values[operand_row[0]], values[operand_row[1]])
                    if not np.isfinite(result):
                        result = self._sanitize_result(result, values[operand_row].copy())
                    values[node_index] = result
                output_values[output_i] = values[root]
            return

        for output_i, (nodes, operands, operators, root) in enumerate(plans):
            for node_index, operand_row, operator in zip(nodes, operands, operators):
                operand_values = values[operand_row]
                result = operator(*operand_values)
                if not np.isfinite(result):
                    result = self._sanitize_result(result, operand_values)
                values[node_index] = result
            output_values[output_i] = values[root]

    def _can_use_compiled_evaluator(self):
        """Return whether the allocation-light numeric evaluator is exact.

        Arbitrary user operators may return objects or higher-precision values
        that must not be cast through the float64 model buffer. Such banks use
        the compatibility evaluator instead.
        """
        if self.model.dtype != np.dtype(float):
            return False
        return all(
            any(operator is known for known in _FAST_NUMERIC_OPERATORS)
            for operator in self.function_bank.values()
        )

    def _call_recursive_compat(self, data, mutable):
        """Compatibility path matching recursive evaluation semantics."""
        self.visited = set()
        model = self.model if mutable else self.model.copy()
        outputs = np.array(
            [
                self._compute_single_input_recursive(datum, model, mutable)
                for datum in data
            ]
        )
        if self.slope is not None and self.intercept is not None:
            outputs = outputs * self.slope + self.intercept
        return outputs

    def __call__(self, data, mutable=True):
        data = np.atleast_2d(data)

        if not self._can_use_compiled_evaluator():
            return self._call_recursive_compat(data, mutable)
        n_samples = len(data)
        self.visited = set()

        # Match np.array([]) from the original list-comprehension path and avoid
        # changing model state when no rows are supplied.
        if n_samples == 0:
            outputs = np.array([])
            if self.slope is not None and self.intercept is not None:
                outputs = outputs * self.slope + self.intercept
            return outputs

        plans, active_nodes = self._compile_execution_plans(self.model)
        self.visited = active_nodes

        outputs = np.empty((n_samples, self.outputs), dtype=float)
        output_values = np.empty(self.outputs, dtype=float)

        if mutable:
            values = self.model[:, self._value_col]
            # The active graph is structural and therefore identical for every
            # datum. Reset and mark it once instead of once per row.
            self.model[:, self._active_col] = 0
            if active_nodes:
                self.model[np.fromiter(active_nodes, dtype=np.intp), self._active_col] = 1
        else:
            # Only values vary during immutable evaluation; copying the complete
            # model once per datum is unnecessary.
            values = self.model[:, self._value_col].copy()

        for row_index, datum in enumerate(data):
            try:
                values[: self.inputs] = datum
            except ValueError as error:
                print("model.py::_compute_single_input")
                print(error)
                print(f"model[input_indices]: {self.model[: self.inputs]}")
                raise SystemExit from error

            self._evaluate_plans(values, plans, output_values)
            outputs[row_index] = output_values

        if mutable:
            # Preserve the final output-node values left by the original loop.
            output_start = len(self.model) - self.outputs
            self.model[output_start:, self._value_col] = outputs[-1]

        if self.slope is not None and self.intercept is not None:
            outputs = outputs * self.slope + self.intercept
        return outputs

    def _get_node_value(self, model, operand, mutable, visited=None):
        """Compute one node recursively, retaining the public helper API."""
        if visited is None:
            visited = set()
        if not hasattr(self, "visited") or self.visited is None:
            self.visited = set()

        operand = int(operand)
        if operand in visited:
            raise RuntimeError(f"Cycle detected at node {operand} — already visited")

        try:
            node = model[operand]
        except IndexError as error:
            self._index_error(model, operand, error)

        node_type = node[self._node_type_col]
        if node_type == _INPUT_NODE or node_type == _CONSTANT_NODE:
            return node[self._value_col]

        if node_type != _FUNCTION_NODE:
            print(node)
            raise ValueError(f"Invalid node type: {node_type}")

        visited.add(operand)
        self.visited.add(operand)
        try:
            operand_values = np.array(
                [
                    self._get_node_value(model, node[col], mutable, visited)
                    for col in self._operand_cols
                ]
            )
        except RecursionError as error:
            print(f"Recursion error: {error}")
            print(model)
            print(node_type)
            print(node)
            raise SystemExit from error
        finally:
            visited.remove(operand)

        operator = self.function_bank[node[self._operator_col]]
        result = operator(*operand_values)
        if not np.isfinite(result):
            result = self._sanitize_result(result, operand_values)

        if mutable:
            try:
                model[operand, self._value_col] = result
            except OverflowError:
                print(
                    f"Cannot cast {result}\noperand: {operand}\n"
                    f"operand values: {operand_values}\noperator: {operator}\n"
                    "Returning np.inf"
                )
        model[operand, self._active_col] = 1
        return result

    def _run_recursive(self, model, mutable):
        """Original recursive execution semantics for compatibility cases."""
        model[:, self._active_col] = 0
        output_start = len(model) - self.outputs
        output_values = np.empty(self.outputs)

        for output_i, model_index in enumerate(range(output_start, len(model))):
            output_values[output_i] = self._get_node_value(
                model,
                model[model_index, self._operand_cols[0]],
                mutable,
                visited=set(),
            )

        if mutable:
            model[output_start:, self._value_col] = output_values
        return output_values

    def _compute_single_input_recursive(self, datum, model, mutable):
        if not mutable:
            model = model.copy()
        try:
            model[: self.inputs, self._value_col] = datum
        except ValueError as error:
            print("model.py::_compute_single_input")
            print(error)
            print(f"model[input_indices]: {model[: self.inputs]}")
            raise SystemExit from error
        return self._run_recursive(model, mutable)

    def _run(self, model, mutable):
        """Run a model whose input values have already been assigned."""
        if not self._can_use_compiled_evaluator() or model.dtype != np.dtype(float):
            return self._run_recursive(model, mutable)

        model[:, self._active_col] = 0
        plans, active_nodes = self._compile_execution_plans(model)
        self.visited.update(active_nodes)

        if mutable:
            values = model[:, self._value_col]
        else:
            values = model[:, self._value_col].copy()

        output_values = np.empty(self.outputs, dtype=float)
        self._evaluate_plans(values, plans, output_values)

        if active_nodes:
            model[np.fromiter(active_nodes, dtype=np.intp), self._active_col] = 1
        if mutable:
            output_start = len(model) - self.outputs
            model[output_start:, self._value_col] = output_values
        return output_values

    def _compute_single_input(self, datum, model, mutable):
        if not mutable:
            model = model.copy()
        try:
            model[: self.inputs, self._value_col] = datum
        except ValueError as error:
            print("model.py::_compute_single_input")
            print(error)
            print(f"model[input_indices]: {model[: self.inputs]}")
            raise SystemExit from error
        return self._run(model, mutable)

    def fit(self, data, ground_truth, mutable=True):
        predictions = self(data, mutable=mutable)
        n_active_nodes = self.count_active_nodes()

        self.correlation, self.complexity, self.fitness = self.fitness_function(
            predictions, ground_truth, n_active_nodes, float(self.max_size)
        )

        if self.fitness_function is correlation:
            self.slope, self.intercept = align(predictions, ground_truth)

        return self.correlation, self.complexity, self.fitness

    def get_active_nodes(self):
        return self.visited

    def count_active_nodes(self):
        return len(self.visited)

    def _choose_mutation(self, mutation_type):
        """Assign mutation function without allocating a temporary mapping."""
        mutation_name = mutation_type.lower()
        if mutation_name == "point":
            self.mutation = self._point_mutation
        elif mutation_name == "full":
            self.mutation = self._full_mutation
        else:
            raise ValueError(f"{mutation_name} is an invalid mutation operator.")

    def mutate_output(self):
        node_types = self.model[:, self._node_type_col]
        output_indices = np.flatnonzero(node_types == _OUTPUT_NODE)
        mutation_index = np.random.choice(output_indices)
        row = self.model[mutation_index]
        row.fill(0)
        row[self._node_type_col] = _OUTPUT_NODE
        row[self._operand_cols[0]] = np.random.randint(
            0, self.first_body_node + self.max_size
        )

    def _full_mutation(self, verbose=True):
        node_types = self.model[:, self._node_type_col]
        mutable_indices = np.flatnonzero(
            (node_types == _FUNCTION_NODE) | (node_types == _OUTPUT_NODE)
        )
        mutation_index = np.random.choice(mutable_indices)
        if verbose:
            print(f"Mutating at index {mutation_index}")

        old_row = self.model[mutation_index]
        old_node_type = old_row[self._node_type_col]

        if old_node_type == _FUNCTION_NODE:
            new_node = [
                _FUNCTION_NODE,
                0,
                np.random.choice(tuple(self.function_bank.keys())),
                *[
                    np.random.randint(0, mutation_index)
                    for _ in range(self.arity)
                ],
                1,
            ]
        else:
            new_node = [
                _OUTPUT_NODE,
                0,
                0,
                np.random.randint(0, self.first_body_node + self.max_size),
                *[0 for _ in range(self.arity - 1)],
                0,
            ]

        if verbose:
            print(f"Replacing {old_row.copy()} at index {mutation_index} to {new_node}")
        old_row[:] = new_node

    def _point_mutation(self, verbose=False):
        node_types = self.model[:, self._node_type_col]
        mutable_indices = np.flatnonzero(
            (node_types == _FUNCTION_NODE) | (node_types == _OUTPUT_NODE)
        )
        mutation_index = np.random.choice(mutable_indices)
        if verbose:
            print(f"Mutating at index {mutation_index}")

        if self.model[mutation_index, self._node_type_col] == _FUNCTION_NODE:
            mutation_column = np.random.choice(self._mutation_cols)

            if mutation_column == self._operator_col:
                current_op = self.model[mutation_index, self._operator_col]
                operators = tuple(self.function_bank.keys())

                # Retain rejection sampling (and therefore seeded behavior) in
                # the normal multi-operator case. Avoid the original infinite
                # loop when only one operator exists.
                if len(operators) > 1:
                    new_operator = np.random.choice(operators)
                    while new_operator == current_op:
                        new_operator = np.random.choice(operators)
                    if verbose:
                        print(f"Mutating column {mutation_column} to {new_operator}")
                    self.model[mutation_index, self._operator_col] = new_operator
                    return

                # The operator gene cannot change; mutate a viable operand gene.
                viable_columns = [
                    col
                    for col in self._operand_cols
                    if mutation_index > 1
                    or self.model[mutation_index, col] != 0
                ]
                if not viable_columns:
                    return
                mutation_column = viable_columns[0]

            current_operand = self.model[mutation_index, mutation_column]
            # A body node at index 1 can only reference node 0, so no distinct
            # operand exists. Prefer another mutable gene rather than hanging.
            if mutation_index <= 1:
                if len(self.function_bank) <= 1:
                    return
                current_op = self.model[mutation_index, self._operator_col]
                new_operator = np.random.choice(tuple(self.function_bank.keys()))
                while new_operator == current_op:
                    new_operator = np.random.choice(tuple(self.function_bank.keys()))
                self.model[mutation_index, self._operator_col] = new_operator
                return

            new_operand = np.random.randint(0, mutation_index)
            while new_operand == current_operand:
                new_operand = np.random.randint(0, mutation_index)
            if verbose:
                print(f"Mutating column {mutation_column} to {new_operand}")
            self.model[mutation_index, mutation_column] = new_operand
            return

        old_operand = self.model[mutation_index, self._operand_cols[0]]
        new_operand = old_operand
        while old_operand == new_operand:
            new_operand = np.random.randint(0, mutation_index)
        if verbose:
            print(f"Mutating output to {new_operand}")
        self.model[mutation_index, self._operand_cols[0]] = new_operand

    def mutate(self, verbose=False):
        """Return a mutated, fully independent copy of the individual."""
        clone = deepcopy(self)
        clone.id = uuid.uuid4()
        if clone.mutation is None:
            raise ValueError("Mutation function not set. Call _choose_mutation() first.")
        clone.mutation(verbose)
        return clone

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        mutation = self.__dict__.get("mutation")
        for key, value in self.__dict__.items():
            if key == "mutation":
                continue
            if isinstance(value, np.ndarray):
                copied = value.copy()
                memo[id(value)] = copied
                setattr(result, key, copied)
            elif key == "model_keys" or key == "function_bank":
                copied = value.copy()
                memo[id(value)] = copied
                setattr(result, key, copied)
            elif isinstance(value, set):
                copied = value.copy()
                memo[id(value)] = copied
                setattr(result, key, copied)
            else:
                setattr(result, key, deepcopy(value, memo))

        if mutation is None:
            result.mutation = None
        elif getattr(mutation, "__func__", None) is CGP._point_mutation:
            result.mutation = result._point_mutation
        elif getattr(mutation, "__func__", None) is CGP._full_mutation:
            result.mutation = result._full_mutation
        else:
            result.mutation = deepcopy(mutation, memo)

        return result

    def print_parameters(self):
        """Print key parameters of the CGP model."""
        print(f"Constants: {self.constants}")
        print(f"Inputs: {self.inputs}")
        print(f"Outputs: {self.outputs}")
        print(f"Arity: {self.arity}")
        print(f"Max Instructions: {self.max_size}")
        print(f"Function Bank: {[func.__name__ for func in self.function_bank.values()]}")
        print(f"Number of Functions: {self.n_operations}")

    def print_model(self):
        print(self.model)

    def set_parent_key(self, key):
        self.parent_keys = key

    def set_child_key(self, key):
        self.child_keys = key

    def model_for_llm(self, with_inputs=False):
        """Return an indexed NumPy copy of the model for LLM prompting."""
        if with_inputs:
            source = self.model
            indices = np.arange(len(source), dtype=source.dtype)
        else:
            node_types = self.model[:, self._node_type_col]
            row_indices = np.flatnonzero(
                (node_types == _FUNCTION_NODE) | (node_types == _OUTPUT_NODE)
            )
            source = self.model[row_indices]
            indices = row_indices.astype(self.model.dtype, copy=False)

        # Allocate the final matrix once instead of deepcopy + concatenate (+
        # boolean-filter copy in the filtered case).
        result = np.empty(
            (len(source), self.model.shape[1] + 1), dtype=self.model.dtype
        )
        result[:, 0] = indices
        result[:, 1:] = source
        return result

    @staticmethod
    def _hash_model(model: np.ndarray) -> str:
        """Return an MD5 hash without always allocating a full bytes copy."""
        contiguous = np.ascontiguousarray(model)
        try:
            raw_view = memoryview(contiguous).cast("B")
            return hashlib.md5(raw_view).hexdigest()
        except (TypeError, ValueError):
            # Fallback for uncommon dtypes whose buffer cannot be byte-cast.
            return hashlib.md5(contiguous.tobytes()).hexdigest()
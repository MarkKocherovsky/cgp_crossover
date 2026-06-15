import json
import os
import re
from typing import Tuple

import numpy as np
import ollama
import pandas as pd
from pandas.testing import assert_frame_equal

from .cgp_evolver import CartesianGP
from .cgp_model import CGP
from .cgp_operators import add, sub, mul, div, op_or, op_and, op_not
from .test_problems import Collection


OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "crossover-gemma")


def one_point_table_crossover(
    parent1: pd.DataFrame,
    parent2: pd.DataFrame,
    crossover_point: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform deterministic one-point row-wise crossover.

    Child 0:
        rows before crossover_point from parent1
        rows at/after crossover_point from parent2

    Child 1:
        rows before crossover_point from parent2
        rows at/after crossover_point from parent1
    """
    parent1 = parent1.reset_index(drop=True).copy()
    parent2 = parent2.reset_index(drop=True).copy()

    if list(parent1.columns) != list(parent2.columns):
        raise ValueError(
            "Parents must have identical columns.\n"
            f"Parent 1 columns: {parent1.columns.tolist()}\n"
            f"Parent 2 columns: {parent2.columns.tolist()}"
        )

    if len(parent1) != len(parent2):
        raise ValueError(
            f"Parents must have the same number of rows. "
            f"Got {len(parent1)} and {len(parent2)}."
        )

    n_rows = len(parent1)

    if not (0 <= crossover_point <= n_rows):
        raise ValueError(
            f"crossover_point must be in [0, {n_rows}], got {crossover_point}"
        )

    child0 = pd.concat(
        [
            parent1.iloc[:crossover_point],
            parent2.iloc[crossover_point:],
        ],
        ignore_index=True,
    )

    child1 = pd.concat(
        [
            parent2.iloc[:crossover_point],
            parent1.iloc[crossover_point:],
        ],
        ignore_index=True,
    )

    # Keep idx consistent with row position after crossover.
    child0["idx"] = np.arange(n_rows)
    child1["idx"] = np.arange(n_rows)

    return child0, child1


def validate_one_point_children(
    parent1: pd.DataFrame,
    parent2: pd.DataFrame,
    child0: pd.DataFrame,
    child1: pd.DataFrame,
    crossover_point: int,
) -> None:
    """
    Verify that the generated children exactly match the deterministic
    one-point crossover result.
    """
    expected_child0, expected_child1 = one_point_table_crossover(
        parent1,
        parent2,
        crossover_point,
    )

    assert_frame_equal(
        child0.reset_index(drop=True),
        expected_child0.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )

    assert_frame_equal(
        child1.reset_index(drop=True),
        expected_child1.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )


def parse_crossover_point(raw_text: str, min_point: int, max_point: int) -> int:
    """
    Extract crossover_point from either JSON or plain text.

    Accepts:
        {"crossover_point": 5}
        5
        crossover_point: 5
    """
    raw_text = raw_text.strip()

    if not raw_text:
        raise ValueError("Ollama returned an empty crossover-point response.")

    # Remove markdown fences if present
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)
    raw_text = raw_text.strip()

    # Prefer JSON if possible
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict) and "crossover_point" in obj:
            point = int(obj["crossover_point"])
        else:
            raise ValueError(f"JSON did not contain crossover_point: {obj}")
    except json.JSONDecodeError:
        matches = re.findall(r"-?\d+", raw_text)
        if not matches:
            raise ValueError(
                "Could not find an integer crossover point in Ollama response.\n"
                f"Raw response:\n{raw_text}"
            )
        point = int(matches[0])

    if not (min_point <= point <= max_point):
        raise ValueError(
            f"Ollama chose crossover point {point}, but expected a value in "
            f"[{min_point}, {max_point}].\n\nRaw response:\n{raw_text}"
        )

    return point


def choose_crossover_point_with_ollama(
    parent1_model: pd.DataFrame,
    parent2_model: pd.DataFrame,
    parent1_f: float,
    parent1_c: float,
    parent2_f: float,
    parent2_c: float,
) -> int:
    """
    Ask Ollama to infer structure directly from the full parent tables
    and choose a structurally-aware one-point crossover point.

    Python still performs the actual crossover afterward.
    """
    parent1_model = parent1_model.reset_index(drop=True).copy()
    parent2_model = parent2_model.reset_index(drop=True).copy()

    n_rows = len(parent1_model)

    nonterminal_rows = parent1_model[
        parent1_model["node_type"].astype(int).isin([2, 3])
    ]

    if nonterminal_rows.empty:
        raise ValueError("No function/output rows found for crossover.")

    first_nonterminal_row = int(nonterminal_rows.index.min())

    # Avoid crossover before the first function/output row if that would merely
    # swap all structure. +1 makes the first copied structural row nontrivial.
    min_point = min(first_nonterminal_row + 1, n_rows - 1)

    # Allow swapping the output row, but avoid point == n_rows, which copies whole parents.
    max_point = n_rows - 1

    fallback_point = (min_point + max_point) // 2

    prompt = f"""
You are choosing a structurally-aware one-point crossover point for two Cartesian Genetic Programming parent models.

Return only JSON in exactly this form:
{{"crossover_point": <int>}}

Valid crossover_point range:
{min_point} through {max_point}, inclusive.

Crossover semantics:
- Child 0 will copy rows before crossover_point from Parent 1.
- Child 0 will copy rows at or after crossover_point from Parent 2.
- Child 1 will copy rows before crossover_point from Parent 2.
- Child 1 will copy rows at or after crossover_point from Parent 1.
- Python will construct the children. You only choose the point.

Structural interpretation:
- idx is the node index.
- node_type 0 = input.
- node_type 1 = constant.
- node_type 2 = function.
- node_type 3 = output.
- Function nodes use operand0 and operand1.
- Output nodes use operand0.
- Operands are node-index references.
- Active nodes are recursively used by the output node.
- Prefer preserving useful active structure.
- Prefer a point that recombines meaningful substructure rather than merely copying a parent.
- Avoid points that are likely to disrupt the output's active dependency chain too severely.
- Lower fitness is better.
- Lower complexity is better, but fitness has priority.

Parent 1 Fitness: {parent1_f}
Parent 1 Complexity: {parent1_c}

Parent 1:
{parent1_model.to_csv(index=False)}

Parent 2 Fitness: {parent2_f}
Parent 2 Complexity: {parent2_c}

Parent 2:
{parent2_model.to_csv(index=False)}

Return only JSON. No explanation. No markdown. No code fences.
"""

    # Do not use format=schema for now. Some local Ollama/Gemma combinations
    # return empty content with schema-constrained output. Prompt-only JSON is
    # often easier to debug.
    for attempt in range(2):
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in structurally-aware crossover for "
                        "Cartesian Genetic Programming. Analyze the full parent "
                        "tables and return only JSON with a crossover_point."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0,
                "num_ctx": 16384,
                "num_predict": 128,
            }
        )

        print(f"FULL OLLAMA RESPONSE, attempt {attempt + 1}:")
        print(response)

        raw_text = response.get("message", {}).get("content", "")

        print("RAW CROSSOVER POINT RESPONSE:")
        print(repr(raw_text))
        print(raw_text)

        try:
            return parse_crossover_point(raw_text, min_point, max_point)
        except ValueError as e:
            print(f"Ollama attempt {attempt + 1} failed: {e}")

            # Simpler retry, but still includes the full tables.
            prompt = f"""
Analyze these two CGP parent tables and choose one structurally-aware crossover point.

Valid range: {min_point} to {max_point}.

Return only:
{{"crossover_point": <integer>}}

Parent 1 fitness={parent1_f}, complexity={parent1_c}
Parent 1:
{parent1_model.to_csv(index=False)}

Parent 2 fitness={parent2_f}, complexity={parent2_c}
Parent 2:
{parent2_model.to_csv(index=False)}
"""

    print(
        f"Ollama did not return a valid point. "
        f"Using fallback crossover point: {fallback_point}"
    )

    return fallback_point


def set_up_cgp(x, y, seed):
    trial_number = 999
    max_generations = 1
    model_size = 64
    xover_type = "subgraph"
    max_parents = 16
    max_children = 16
    mutation_type = "full"

    selection_type = "paretotournament"
    fitness_function = "correlation"
    test_problem_key = "poet"
    n_points = 1
    tournament_size = 4
    n_elites = 1
    asex = True

    print("x inputs:", x.shape[-1])
    print("y outputs:", 1 if y.ndim == 1 else y.shape[-1])

    model_parameters = {
        "max_size": model_size,
        "inputs": x.shape[-1],
        "outputs": 1 if y.ndim == 1 else y.shape[-1],
        "arity": 2,
        "constants": np.array([1]),
    }

    function_bank = {
        "add": add,
        "sub": sub,
        "mul": mul,
        "div": div,
    }

    function_bool = {
        "op_and": op_and,
        "op_or": op_or,
        "op_not": op_not,
    }

    mutation_breeding = asex or max_parents < max_children

    checkpoint_path = os.path.join(os.environ.get("SCRATCH", "/tmp"), "ckpt")
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_file = (
        f"{checkpoint_path}/test_{test_problem_key}_trial_{trial_number}_ckpt.pkl"
    )

    evolution_module = CartesianGP(
        parents=max_parents,
        children=max_children,
        max_generations=max_generations,
        mutation=mutation_type,
        selection=selection_type,
        xover=xover_type,
        fixed_length=True,
        fitness_function=fitness_function,
        model_parameters=model_parameters,
        n_points=n_points,
        n_elites=n_elites,
        tournament_size=tournament_size,

        # For Koza3 / symbolic regression, use arithmetic operators.
        # Use function_bool only for Boolean truth-table problems.
        function_bank=function_bank,

        mutation_breeding=mutation_breeding,
        checkpoint_filename=checkpoint_file,
        one_dimensional_xover=False,
        seed=seed,
        tuning=False,
    )

    return evolution_module

MODEL_DF_COLUMNS = [
    "idx", "node_type", "value", "operator", "operand0", "operand1", "active"
]

CGP_ARRAY_COLUMNS = [
    "node_type", "value", "operator", "operand0", "operand1", "active"
]


def validate_child_dataframe(child_df: pd.DataFrame, name: str) -> None:
    """
    Validate that a child DataFrame can be converted back into a CGP model.
    """
    missing = set(MODEL_DF_COLUMNS) - set(child_df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    child_df = child_df[MODEL_DF_COLUMNS].copy()

    expected_idx = np.arange(len(child_df))
    actual_idx = child_df["idx"].astype(int).to_numpy()

    if not np.array_equal(actual_idx, expected_idx):
        raise ValueError(
            f"{name} idx column must be 0..{len(child_df) - 1}.\n"
            f"Got: {actual_idx}"
        )

    for _, row in child_df.iterrows():
        idx = int(row["idx"])
        node_type = int(row["node_type"])

        if node_type not in {0, 1, 2, 3}:
            raise ValueError(f"{name} has invalid node_type {node_type} at idx {idx}")

        if node_type == 2:
            operand0 = int(row["operand0"])
            operand1 = int(row["operand1"])

            if operand0 >= idx or operand1 >= idx:
                raise ValueError(
                    f"{name} function node {idx} violates feed-forward constraint: "
                    f"operand0={operand0}, operand1={operand1}"
                )

        elif node_type == 3:
            operand0 = int(row["operand0"])

            if operand0 >= idx:
                raise ValueError(
                    f"{name} output node {idx} violates feed-forward constraint: "
                    f"operand0={operand0}"
                )


def child_dataframe_to_cgp(
    child_df: pd.DataFrame,
    template_parent,
    fitness_function: str = "correlation",
    mutation_type: str = "full",
):
    """
    Convert a child DataFrame back into a valid CGP object.

    template_parent is one of the existing CGP parents from the evolved population.
    It supplies:
    - the CGP class
    - model_keys
    - fixed_length setting
    - function_bank
    """
    child_df = child_df[MODEL_DF_COLUMNS].copy()
    validate_child_dataframe(child_df, name="child")

    # The CGP.model array does not include the idx column.
    model_array = child_df[CGP_ARRAY_COLUMNS].to_numpy(dtype=float)

    child_cgp = template_parent.__class__(
        model=model_array,
        model_keys=template_parent.model_keys.copy(),
        fixed_length=template_parent.fixed_length,
        fitness_function=fitness_function,
        mutation_type=mutation_type,
        function_bank=template_parent.function_bank,
    )

    return child_cgp


def fit_child_model(child_cgp, x, y, name: str, mutable: bool = True):
    """
    Run CGP.fit() on a child model and print the resulting metrics.
    """
    corr, complexity, fitness = child_cgp.fit(x, y, mutable=mutable)

    print(f"{name} metrics")
    print(f"  correlation: {corr}")
    print(f"  complexity:  {complexity}")
    print(f"  fitness:     {fitness}")

    return corr, complexity, fitness

def main():
    problems = Collection()
    test_function = problems("Koza3")

    train_x, test_x, train_y, test_y = test_function.return_points()

    evolution_module = set_up_cgp(train_x, train_y, seed=4)

    mutation_rate = 0.5
    xover_rate = 0.5

    best_model, _ = evolution_module.fit(
        train_x,
        test_x,
        train_y,
        test_y,
        xover_rate=xover_rate,
        mutation_rate=mutation_rate,
    )

    pop = evolution_module.population

    if len(pop) < 2:
        raise ValueError("Population must contain at least two parents.")

    parent1 = pop[0]
    parent2 = pop[1]

    parent1_model = parent1.model_for_llm(with_inputs=True).reset_index(drop=True)
    parent1_f = parent1.fitness
    parent1_c = parent1.complexity

    parent2_model = parent2.model_for_llm(with_inputs=True).reset_index(drop=True)
    parent2_f = parent2.fitness
    parent2_c = parent2.complexity

    print("parents")
    print("parent1")
    print(parent1_model)
    print("parent2")
    print(parent2_model)

    crossover_point = choose_crossover_point_with_ollama(
        parent1_model=parent1_model,
        parent2_model=parent2_model,
        parent1_f=parent1_f,
        parent1_c=parent1_c,
        parent2_f=parent2_f,
        parent2_c=parent2_c,
    )

    print(f"Chosen crossover point: {crossover_point}")

    child0, child1 = one_point_table_crossover(
        parent1=parent1_model,
        parent2=parent2_model,
        crossover_point=crossover_point,
    )

    validate_one_point_children(
        parent1=parent1_model,
        parent2=parent2_model,
        child0=child0,
        child1=child1,
        crossover_point=crossover_point,
    )

    print("child0")
    print(child0)

    print("child1")
    print(child1)

    # Convert child DataFrames into actual CGP model objects.
    child0_cgp = child_dataframe_to_cgp(
        child_df=child0,
        template_parent=parent1,
        fitness_function="correlation",
        mutation_type="full",
    )

    child1_cgp = child_dataframe_to_cgp(
        child_df=child1,
        template_parent=parent1,
        fitness_function="correlation",
        mutation_type="full",
    )

    print("child0 CGP model array")
    print(child0_cgp.model)

    print("child1 CGP model array")
    print(child1_cgp.model)

    # Run individual CGP.fit() on the children.
    fit_child_model(
        child_cgp=child0_cgp,
        x=train_x,
        y=train_y,
        name="child0 train",
        mutable=True,
    )

    fit_child_model(
        child_cgp=child1_cgp,
        x=train_x,
        y=train_y,
        name="child1 train",
        mutable=True,
    )

    print("child0 after fit")
    print(child0_cgp.model_for_llm(with_inputs=True))
    print(child0_cgp.fitness)

    print("child1 after fit")
    print(child1_cgp.model_for_llm(with_inputs=True))
    print(child1_cgp.fitness)


if __name__ == "__main__":
    main()
import csv
import json
import re
from io import StringIO

import numpy as np
import ollama

LLM_COLUMNS = [
    "idx", "node_type", "value", "operator", "operand0", "operand1", "active"
]

LLM_IDX_COL = 0
LLM_NODE_TYPE_COL = 1

def llm_model_to_csv(model_array: np.ndarray) -> str:
    """
    Convert the NumPy LLM-view model to CSV text only for prompting.

    The in-memory representation remains a NumPy array.
    """
    output = StringIO()
    writer = csv.writer(output)

    writer.writerow(LLM_COLUMNS)

    for row in model_array:
        writer.writerow(row.tolist())

    return output.getvalue()


def validate_llm_model_array(model_array: np.ndarray, name: str) -> None:
    """
    Ensure the LLM-view model is a full NumPy model view:
    idx,node_type,value,operator,operand0,operand1,active
    """
    if not isinstance(model_array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(model_array)}")

    if model_array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {model_array.shape}")

    if model_array.shape[1] != len(LLM_COLUMNS):
        raise ValueError(
            f"{name} must have {len(LLM_COLUMNS)} columns "
            f"{LLM_COLUMNS}, got shape {model_array.shape}"
        )

    expected_idx = np.arange(model_array.shape[0])
    actual_idx = model_array[:, LLM_IDX_COL].astype(int)

    if not np.array_equal(actual_idx, expected_idx):
        raise ValueError(
            f"{name} must be a full model view with idx 0..n-1. "
            f"Got idx values: {actual_idx}"
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

    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)
    raw_text = raw_text.strip()

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
    parent1_model: np.ndarray,
    parent2_model: np.ndarray,
    parent1_f: float,
    parent1_c: float,
    parent2_f: float,
    parent2_c: float,
    ollama_model: str,
    population_context: str
) -> int:
    """
    Ask Ollama to infer structure directly from full parent NumPy arrays
    and choose a structurally-aware one-point crossover point.

    parent1_model and parent2_model should come from:

        parent.model_for_llm(with_inputs=True)

    They remain NumPy arrays in memory.
    """
    LLM_COLUMNS = [
        "idx", "node_type", "value", "operator", "operand0", "operand1", "active"
    ]

    RAW_MODEL_COLUMNS = [
        "node_type", "value", "operator", "operand0", "operand1", "active"
    ]

    LLM_IDX_COL = 0
    LLM_NODE_TYPE_COL = 1

    def ensure_llm_model_array(model_array: np.ndarray, name: str) -> np.ndarray:
        """
        Accept either:

        1. Raw internal CGP model array:
           node_type,value,operator,operand0,operand1,active
           shape = (n_nodes, 6)

        2. LLM-view model array:
           idx,node_type,value,operator,operand0,operand1,active
           shape = (n_nodes, 7)

        Return the 7-column LLM-view NumPy array.
        """
        model_array = np.asarray(model_array).copy()

        if model_array.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {model_array.shape}")

        if model_array.shape[1] == len(RAW_MODEL_COLUMNS):
            idx_col = np.arange(model_array.shape[0], dtype=model_array.dtype).reshape(-1, 1)
            model_array = np.concatenate([idx_col, model_array], axis=1)

        elif model_array.shape[1] == len(LLM_COLUMNS):
            pass

        else:
            raise ValueError(
                f"{name} must have either 6 raw CGP columns {RAW_MODEL_COLUMNS} "
                f"or 7 LLM-view columns {LLM_COLUMNS}, got shape {model_array.shape}"
            )

        expected_idx = np.arange(model_array.shape[0])
        actual_idx = model_array[:, LLM_IDX_COL].astype(int)

        if not np.array_equal(actual_idx, expected_idx):
            raise ValueError(
                f"{name} idx column must be 0..{model_array.shape[0] - 1}.\n"
                f"Got: {actual_idx}"
            )

        return model_array

    parent1_model = ensure_llm_model_array(parent1_model, "parent1_model")
    parent2_model = ensure_llm_model_array(parent2_model, "parent2_model")

    # validate_llm_model_array(parent1_model, "parent1_model")
    # validate_llm_model_array(parent2_model, "parent2_model")

    if parent1_model.shape != parent2_model.shape:
        raise ValueError(
            f"Parents must have the same LLM-view shape. "
            f"Got {parent1_model.shape} and {parent2_model.shape}."
        )

    n_rows = parent1_model.shape[0]

    node_types = parent1_model[:, LLM_NODE_TYPE_COL].astype(int)
    nonterminal_positions = np.where(np.isin(node_types, [2, 3]))[0]

    if nonterminal_positions.size == 0:
        raise ValueError("No function/output rows found for crossover.")

    first_nonterminal_row = int(nonterminal_positions.min())

    # Avoid crossover before the first structural row.
    min_point = min(first_nonterminal_row + 1, n_rows - 1)

    # Avoid point == n_rows, which would copy whole parents.
    max_point = n_rows - 1

    fallback_point = (min_point + max_point) // 2

    parent1_csv = llm_model_to_csv(parent1_model)
    parent2_csv = llm_model_to_csv(parent2_model)

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

Use this population-level context as background:
{population_context}

Now analyze these two selected parents directly.

Parent 1 Fitness: {parent1_f}
Parent 1 Complexity: {parent1_c}

Parent 1:
{parent1_csv}

Parent 2 Fitness: {parent2_f}
Parent 2 Complexity: {parent2_c}

Parent 2:
{parent2_csv}

Return only JSON. No explanation. No markdown. No code fences.
"""

    response = ollama.chat(
        model=ollama_model,
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
            "num_predict": 512,
        },
        think=False
    )

    #print(f"FULL OLLAMA RESPONSE, attempt {attempt + 1}:")
    #print(response)

    raw_text = response.get("message", {}).get("content", "")
    return parse_crossover_point(raw_text, min_point, max_point)


def summarize_population_for_llm(population_window, max_elites=3):
    populations = [
        [ind for ind in population if ind is not None]
        for population in population_window
    ]

    fitnesses = [
        np.array([ind.fitness for ind in population], dtype=float)
        for population in populations
    ]

    complexities = [
        np.array([ind.complexity for ind in population], dtype=float)
        for population in populations
    ]

    sorted_pops = [
        sorted(
            population,
            key=lambda ind: (ind.fitness, ind.complexity)
        )
        for population in populations
    ]

    lines = []
    lines.append("Population context over recent generations:")
    lines.append(f"- generations_stored: {len(population_window)}")
    lines.append(f"- latest_population_size: {len(populations[-1])}")

    lines.append("")
    lines.append("Per-generation fitness summary:")

    for gen_idx, fit_arr in enumerate(fitnesses):
        if len(fit_arr) == 0:
            lines.append(f"- generation_window_index {gen_idx}: empty")
            continue

        lines.append(
            f"- generation_window_index {gen_idx}: "
            f"min={np.min(fit_arr):.6g}, "
            f"median={np.median(fit_arr):.6g}, "
            f"max={np.max(fit_arr):.6g}"
        )

    lines.append("")
    lines.append("Per-generation complexity summary:")

    for gen_idx, comp_arr in enumerate(complexities):
        if len(comp_arr) == 0:
            lines.append(f"- generation_window_index {gen_idx}: empty")
            continue

        lines.append(
            f"- generation_window_index {gen_idx}: "
            f"min={np.min(comp_arr):.6g}, "
            f"median={np.median(comp_arr):.6g}, "
            f"max={np.max(comp_arr):.6g}"
        )

    lines.append("")
    lines.append("Elite summaries by generation:")

    for gen_idx, sorted_pop in enumerate(sorted_pops):
        lines.append(f"- generation_window_index {gen_idx}:")

        for rank, ind in enumerate(sorted_pop[:max_elites]):
            active_nodes = np.array(list(ind.get_active_nodes()))
            try:
                active_min = np.min(active_nodes)
                active_max = np.max(active_nodes)
            except ValueError as e:
                active_min = np.nan
                active_max = np.nan
            try:
                lines.append(
                    f"  - elite_rank={rank}, "
                    f"fitness={ind.fitness:.6g}, "
                    f"complexity={ind.complexity:.6g} "
                    f"Active Nodes Range: {active_min} - {active_max} "
                )
            except ValueError as e:
                print(f'llm_helper.py::summarize_population_for_llm: {e}')
                print(f'active_nodes: {active_nodes}')
                exit()

    #print(lines)
    return "\n".join(lines)
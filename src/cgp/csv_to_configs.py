import json
import math
from pathlib import Path

import pandas as pd


CSV_PATH = "config_data.csv"

DEFAULT_CONFIG = {
    "mutation": "full",
    "selection": "paretotournament",
    "max_g": 6000,
    "max_p": 1,
    "max_c": 2,
    "max_n": 64,
    "x_rate": 1.0,
    "m_rate": 1.0,
    "n_points": 1,
    "n_elites": 1,
    "t_size": 4,
    "p_dim": 1,
    "step_size": 100,
    "asexual_reproduction": False,
    "job_count": 0,
    "one_d": False,
}


def is_missing(value):
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    return str(value).strip() in {"", "-"}


def parse_value(value, default):
    if is_missing(value):
        return default

    value = str(value).strip()

    if isinstance(default, bool):
        return value.lower() in {"true", "1", "yes", "y"}

    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(value))

    if isinstance(default, float):
        return float(value)

    return value


def normalize_xover_type(value):
    """
    In this CSV, blank xover_type values mean None.
    """
    if is_missing(value):
        return "None"

    return str(value).strip()


def make_config(row):
    config = DEFAULT_CONFIG.copy()

    for key, default in DEFAULT_CONFIG.items():
        if key in row:
            config[key] = parse_value(row[key], default)

    xover_type = normalize_xover_type(row["xover_type"])

    if xover_type == "None":
        config["selection"] = "paretoelite"
        config["asexual_reproduction"] = True
    else:
        config["selection"] = "paretotournament"
        config["asexual_reproduction"] = False

    return config


def generate_configs(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)

    required_columns = {"Problem", "xover_type"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    counters = {}

    for _, row in df.iterrows():
        problem = str(row["Problem"]).strip()
        xover_type = normalize_xover_type(row["xover_type"])

        key = (problem, xover_type)
        counters[key] = counters.get(key, -1)+1

        config = make_config(row)

        output_dir = Path("./configs") / problem / xover_type
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"config_{counters[key]}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Wrote {output_path}")


if __name__ == "__main__":
    generate_configs()

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_search_trajectory(
    json_path,
    output_path="search_trajectory_network.png",
    target_complexity=1e-16,
    target_fitness=1e-16,
    annotate_steps=False,
    show=True
):
    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    items = list(data.values())
    if not items:
        raise ValueError("The JSON file is empty.")

    # Extract values
    complexities = np.array([float(item["complexity"]) for item in items])
    fitnesses = np.array([float(item["fitness"]) for item in items])

    n = len(items)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Trajectory line
    ax.plot(complexities, fitnesses, color="black", linewidth=0.8, zorder=1)

    # Intermediate individuals
    if n > 2:
        ax.scatter(
            complexities[1:-1],
            fitnesses[1:-1],
            color="black",
            marker="o",
            s=30,
            label="Intermediate individuals",
            zorder=3
        )

    # First individual
    ax.scatter(
        complexities[0],
        fitnesses[0],
        color="green",
        marker="^",
        s=120,
        label="First individual",
        zorder=4
    )

    # Ending individual
    ax.scatter(
        complexities[-1],
        fitnesses[-1],
        color="red",
        marker="s",
        s=120,
        label="Ending individual",
        zorder=4
    )

    # Target
    ax.scatter(
        target_complexity,
        target_fitness,
        color="orange",
        marker="*",
        s=200,
        label="Target",
        zorder=5
    )

    # Optional step annotations
    if annotate_steps:
        for i, (x, y) in enumerate(zip(complexities, fitnesses)):
            ax.annotate(str(i), (x, y), fontsize=8, xytext=(4, 4),
                        textcoords="offset points")

    ax.set_xlabel("Complexity")
    ax.set_ylabel("Fitness")
    ax.set_xlim([1e-16, 1])
    ax.set_ylim([1e-16, 1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Search Trajectory Network (Fitness vs Complexity)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
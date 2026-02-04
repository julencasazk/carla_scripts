"""
Plots per-iteration mean values from a JSON log of PSO runs.

Outputs
  - Matplotlib plots for selected fields.

Run (example)
  python3 tuning/pso/json_plot.py --json metrics_scan.json
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import argparse


def load_data(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)


def compute_iteration_means(records):
    """
    records: list[dict] from the JSON file
    returns: dict[iter] -> dict[field] -> mean_value
    """
    # Collect numeric fields per iter
    per_iter = defaultdict(lambda: defaultdict(list))

    for rec in records:
        it = rec["iter"]
        for k, v in rec.items():
            if k == "iter":
                continue
            # Only average numeric values
            if isinstance(v, (int, float)):
                per_iter[it][k].append(v)

    # Compute means
    iter_means = {}
    for it, fields in per_iter.items():
        iter_means[it] = {
            field: float(np.mean(values))
            for field, values in fields.items()
        }

    return iter_means


def plot_fields(iter_means, fields_to_plot):
    # Sort iterations
    iters = sorted(iter_means.keys())

    for field in fields_to_plot:
        y = [iter_means[it][field] for it in iters]
        plt.figure()
        plt.plot(iters, y, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel(field)
        plt.title(f"Average {field} per iteration")
        plt.grid(True)
        plt.tight_layout()

    plt.show()


def main():
    default_fields = [
        "kp",
        "ki",
        "kd",
        "J1_SSE",
        "J2_OS",
        "J3_Ts_Tr",
        "J4_RVG",
        "J5_ODJ",
        "J6_CEC",
        "J1_weighted",
        "J2_weighted",
        "J3_weighted",
        "J4_weighted",
        "J5_weighted",
        "J6_weighted",
        "Total_J",
    ]

    parser = argparse.ArgumentParser(
        description="Plot PSO PID tuning results from JSON file"
    )
    parser.add_argument("file", type=str, nargs="?", default="pso_results.json")
    # Use nargs='+' (one or more) and let Python collect them into a list
    parser.add_argument(
        "-p",
        "--plot",
        metavar="FIELD",
        nargs="+",         # one or more values
        help="Variables to plot (space-separated list)",
    )

    args = parser.parse_args()

    json_file = Path(args.file)
    data = load_data(json_file)
    iter_means = compute_iteration_means(data)

    # If -p/--plot is not given, use defaults
    fields_to_plot = args.plot if args.plot is not None else default_fields

    # Filter to only fields that actually exist
    existing_fields = [
        f for f in fields_to_plot
        if f in next(iter(iter_means.values()))
    ]

    if not existing_fields:
        raise ValueError("None of the requested fields exist in the data")

    plot_fields(iter_means, existing_fields)


if __name__ == "__main__":
    main()

# filepath: /home/jcazk/carla_scripts/plot_csv.py
#!/usr/bin/env python3
# filepath: /home/jcazk/carla_scripts/plot_csv.py
import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot one or more CSV columns vs a reference column.\n"
            "Example: plot_csv.py data.csv --ref time_s -p throttle speed_m_s"
        )
    )
    p.add_argument("csv", type=Path, help="Input CSV file.")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Optional output image path (e.g. plot.png).",
    )
    p.add_argument(
        "--ref",
        type=str,
        default="time_s",
        help="X-axis column name (default: time_s)",
    )
    p.add_argument(
        "-p",
        "--plot",
        metavar="FIELD",
        nargs="+",  # one or more values
        required=True,
        help="Columns to plot on Y-axis (space-separated list)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional overall figure title",
    )
    return p.parse_args()


def validate_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found columns: {list(df.columns)}")


def main():
    args = parse_args()

    if not args.csv.is_file():
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = [args.ref] + list(args.plot)
    try:
        validate_columns(df, required_cols)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    df = df.dropna(subset=required_cols)
    x = df[args.ref]

    # color sequence: orange, blue, red, green, purple, brown, pink, gray, olive, cyan
    colors = ["orange", "blue", "red", "green", "purple", "brown", "pink", "gray", "olive", "cyan"]

    n_plots = len(args.plot)
    # one tall figure, height grows with number of plots
    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(10, 3 * n_plots),
        sharex=True,
    )

    # if only one subplot, axes is not a list; make it iterable
    if n_plots == 1:
        axes = [axes]

    for i, (col, ax) in enumerate(zip(args.plot, axes)):
        color = colors[i % len(colors)]
        ax.plot(x, df[col], color=color)
        ax.set_ylabel(col)

        ax.grid(True, linestyle="--", alpha=0.4)

    # common x-label on the last subplot
    axes[-1].set_xlabel(args.ref)

    if args.title:
        fig.suptitle(args.title, y=0.99)

    fig.tight_layout()

    if args.out:
        try:
            fig.savefig(args.out, dpi=150)
            print(f"Saved plot: {args.out}")
        except Exception as e:
            print(f"Error saving figure: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        plt.show()


if __name__ == "__main__":
    main()

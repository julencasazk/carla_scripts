import argparse
import sys
from pathlib import Path
import pandas as pd

#!/usr/bin/env python3

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot throttle and speed vs time from a CSV (columns: time_s, throttle, speed_m_s)."
    )
    p.add_argument("csv", type=Path, help="Input CSV file.")
    p.add_argument("-o", "--out", type=Path, help="Optional output image path (e.g. plot.png).")
    p.add_argument("--title", default="Throttle and Speed vs Time", help="Figure title.")
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

    try:
        validate_columns(df, ["time_s", "throttle", "speed_m_s"])
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Drop rows with NaNs in required columns
    df = df.dropna(subset=["time_s", "throttle", "speed_m_s"])

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(df["time_s"], df["throttle"], color="tab:blue", label="Throttle")
    ax2.plot(df["time_s"], df["speed_m_s"], color="tab:red", label="Speed")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Throttle (0-1)", color="tab:blue")
    ax2.set_ylabel("Speed (m/s)", color="tab:red")

    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle(args.title)

    # Combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

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

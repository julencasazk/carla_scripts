# filepath: /home/jcazk/carla_scripts/plot_csv.py
#!/usr/bin/env python3
# filepath: /home/jcazk/carla_scripts/plot_csv.py
import argparse
import csv
import sys
from pathlib import Path

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

def _coalesce_bracket_groups(tokens: list[str]) -> list[str]:
    """
    Allow '-p [col1, col2] [col3]' by joining tokens between '[' and ']'.
    Users may still need to quote to avoid shell globbing, but this makes the
    parser tolerant of spaces inside bracketed groups.
    """
    out: list[str] = []
    buf: list[str] = []
    for t in tokens:
        if not buf:
            if "[" in t and "]" not in t:
                buf.append(t)
            else:
                out.append(t)
            continue

        buf.append(t)
        if "]" in t:
            out.append(" ".join(buf))
            buf = []

    if buf:
        out.append(" ".join(buf))
    return out


def _parse_plot_groups(raw: list[str]) -> list[list[str]]:
    """
    Parse '-p' arguments into groups.

    Examples:
      -p col1 col2           -> [[col1],[col2]]
      -p col1,col2 col3      -> [[col1,col2],[col3]]
      -p "[col1, col2]" col3 -> [[col1,col2],[col3]]
    """
    raw = _coalesce_bracket_groups(raw)

    groups: list[list[str]] = []
    for token in raw:
        s = token.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        # split on commas if present; otherwise single column
        if "," in s:
            cols = [c.strip() for c in s.split(",") if c.strip()]
        else:
            cols = [s] if s else []
        if not cols:
            continue
        groups.append(cols)

    return groups


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot one or more CSV columns vs a reference column.\n"
            "Examples:\n"
            "  plot_csv.py data.csv                 # prints column names\n"
            "  plot_csv.py data.csv --ref time_s -p throttle speed_m_s\n"
            "  plot_csv.py data.csv --ref time_s -p throttle,brake speed_m_s\n"
            "  plot_csv.py data.csv --ref time_s -p \"[throttle, brake]\" \"[speed_m_s]\""
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
        default=None,
        help=(
            "Columns to plot on Y-axis.\n"
            "Default behavior: each token becomes its own subplot (stacked vertically).\n"
            "Bundling: use commas or [..] to plot multiple columns on the same subplot.\n"
            "Examples:\n"
            "  -p col1 col2           # 2 stacked plots\n"
            "  -p col1,col2 col3      # (col1+col2) on one plot, col3 on another\n"
            "  -p \"[col1, col2]\" col3 # same as above"
        ),
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


def read_csv_fallback(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        data: dict[str, list[str]] = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                v = row.get(c)
                data[c].append("" if v is None else str(v))
    return cols, data

def main():
    args = parse_args()

    if not args.csv.is_file():
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = None
    columns: list[str]
    data_raw: dict[str, list[str]] | None = None

    if pd is not None:
        try:
            df = pd.read_csv(args.csv)
        except Exception as e:
            print(f"Error reading CSV: {e}", file=sys.stderr)
            sys.exit(1)
        columns = list(df.columns)
    else:
        try:
            columns, data_raw = read_csv_fallback(args.csv)
        except Exception as e:
            print(f"Error reading CSV: {e}", file=sys.stderr)
            sys.exit(1)

    if not args.plot:
        cols = list(columns)
        if not cols:
            print("(no columns found)")
        else:
            print("Columns:")
            for i, c in enumerate(cols):
                print(f"  {i:02d}: {c}")
        return

    try:
        import numpy as np  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"Error: plotting requires numpy + matplotlib: {e}", file=sys.stderr)
        sys.exit(1)

    plot_groups = _parse_plot_groups(list(args.plot))
    if not plot_groups:
        print("Error: no valid plot columns parsed from -p/--plot.", file=sys.stderr)
        sys.exit(1)

    flat_cols: list[str] = []
    for g in plot_groups:
        flat_cols.extend(g)
    # keep order but remove duplicates
    seen: set[str] = set()
    flat_cols_unique: list[str] = []
    for c in flat_cols:
        if c not in seen:
            seen.add(c)
            flat_cols_unique.append(c)

    required_cols = [args.ref] + flat_cols_unique
    if df is not None:
        try:
            validate_columns(df, required_cols)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        df = df.dropna(subset=required_cols)
        x = np.asarray(df[args.ref], dtype=float)
        y_arrays = {c: np.asarray(df[c], dtype=float) for c in flat_cols_unique}
    else:
        assert data_raw is not None
        missing = [c for c in required_cols if c not in columns]
        if missing:
            print(f"Error: Missing columns: {missing}. Found columns: {columns}", file=sys.stderr)
            sys.exit(1)

        n = len(data_raw[required_cols[0]]) if required_cols else 0
        arrs: dict[str, np.ndarray] = {}
        for c in required_cols:
            vals = data_raw[c]
            if len(vals) != n:
                print(f"Error: column length mismatch for {c}", file=sys.stderr)
                sys.exit(1)
            out = np.empty(n, dtype=float)
            out.fill(np.nan)
            for i, s in enumerate(vals):
                try:
                    out[i] = float(s)
                except Exception:
                    out[i] = np.nan
            arrs[c] = out

        mask = np.ones(n, dtype=bool)
        for c in required_cols:
            mask &= np.isfinite(arrs[c])
        x = arrs[args.ref][mask]
        y_arrays = {c: arrs[c][mask] for c in flat_cols_unique}

    # color sequence: orange, blue, red, green, purple, brown, pink, gray, olive, cyan
    colors = ["orange", "blue", "red", "green", "purple", "brown", "pink", "gray", "olive", "cyan"]

    n_plots = len(plot_groups)
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

    for gi, (group, ax) in enumerate(zip(plot_groups, axes)):
        for li, col in enumerate(group):
            try:
                yv = y_arrays[col]
                print(f"Column {col}, avg: {np.average(yv)}, max: {np.max(yv)}, min: {np.min(yv)}")
            except Exception:
                pass

            color = colors[(gi + li) % len(colors)]
            ax.plot(x, y_arrays[col], color=color, label=col)

        if len(group) == 1:
            ax.set_ylabel(group[0])
        else:
            ax.set_ylabel("value")
            ax.legend(loc="best")

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

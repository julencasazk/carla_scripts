"""
CSV plotting utility with thesis-style formatting.

What it does
  - Reads CSV columns and renders stacked plots with custom styling.

Outputs
  - Matplotlib plots.

Run (example)
  python3 tools/plot_csv.py -f data.csv -x time_s -p speed throttle

# filepath: /home/jcazk/carla_scripts/plot_csv.py
#!/usr/bin/env python3
# filepath: /home/jcazk/carla_scripts/plot_csv.py
"""

import argparse
import csv
import sys
from pathlib import Path

# -----------------------
# Plot style (edit here)
# -----------------------
# Hardcoded style values so you can tweak thesis figures without adding CLI args.
FIG_WIDTH_IN = 10.0
FIG_HEIGHT_PER_PLOT_IN = 3.0
LINE_WIDTH = 2.0
GRID_ALPHA = 0.35
GRID_STYLE = "--"
DPI = 400

FONT_SIZE = 24
TITLE_FONT_SIZE = 30
AXIS_LABEL_FONT_SIZE = 24
TICK_LABEL_FONT_SIZE = 20
LEGEND_FONT_SIZE = 20

# Layout tuning (edit here)
# Goal: plots fill the window with minimal margins.
MARGIN_LEFT = 0.06
MARGIN_RIGHT = 0.99
MARGIN_BOTTOM = 0.07
MARGIN_TOP_NO_TITLE = 0.98
MARGIN_TOP_WITH_TITLE = 0.94
HSPACE = 0.18
TITLE_Y = 0.985

# High-contrast color cycle (avoid muted defaults)
COLOR_CYCLE = [
    "#ff0000",  # red
    "#0000ff",  # blue
    "#00cc00",  # green
    "#ff00ff",  # magenta
    "#00ffff",  # cyan
    "#ffa500",  # orange
    "#000000",  # black
    "#ffff00",  # yellow
    "#8a2be2",  # blueviolet
    "#00ff7f",  # springgreen
]

MATPLOTLIB_RCPARAMS = {
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_LABEL_FONT_SIZE,
    "ytick.labelsize": TICK_LABEL_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "lines.linewidth": LINE_WIDTH,
}

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
        "--hist",
        action="store_true",
        default=False,
        help="Plot histograms of the selected columns instead of time-series vs --ref.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Histogram bin count (default: 50). Only used with --hist.",
    )
    p.add_argument(
        "--density",
        action="store_true",
        default=False,
        help="Normalize histograms to probability density. Only used with --hist.",
    )
    p.add_argument(
        "--dropout-base",
        type=float,
        default=None,
        help=(
            "Base value for dropout calculation (same units as the plotted column, e.g. ms). "
            "If omitted, uses the column median as the base."
        ),
    )
    p.add_argument(
        "--dropout-mult",
        type=float,
        default=2.0,
        help="Dropout threshold multiplier (default: 2.0). A sample is a dropout if value > dropout_mult * base.",
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
    p.add_argument(
        "--rename",
        metavar="OLD=NEW",
        nargs="*",
        default=None,
        help=(
            "Temporarily rename columns for plot labeling (legend/ylabel only).\n"
            "Does not change which column is read from the CSV.\n"
            "Examples:\n"
            "  --rename speed_carla_mps=CARLA speed_model_mps=Model\n"
            "  --rename throttle_carla=throttle(CARLA) throttle_model=throttle(plant)\n"
        ),
    )
    p.add_argument(
        "--xlabel",
        type=str,
        default=None,
        help="Override x-axis label (default: --ref column name, optionally renamed via --rename).",
    )
    p.add_argument(
        "--ylabel",
        type=str,
        default=None,
        help="Override y-axis label for all subplots (default: column name for singletons, or 'value' for bundled).",
    )
    p.add_argument(
        "--ylabels",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional per-subplot y-axis labels (one per subplot, in order).\n"
            "Overrides default ylabel behavior, but is still overridden by --ylabel."
        ),
    )
    p.add_argument(
        "--legend",
        action="store_true",
        default=True,
        help="Show a legend on each subplot (default: on).",
    )
    p.add_argument(
        "--no-legend",
        dest="legend",
        action="store_false",
        help="Disable legends.",
    )
    p.add_argument(
        "--legend-loc",
        type=str,
        default="best",
        help="Legend location (matplotlib loc string, default: best).",
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

    # Apply hardcoded styling
    try:
        plt.rcParams.update(MATPLOTLIB_RCPARAMS)
    except Exception:
        pass

    # Parse rename mappings (for labels only)
    rename_map: dict[str, str] = {}
    if args.rename:
        for token in args.rename:
            s = str(token).strip()
            if not s:
                continue
            if "=" in s:
                old, new = s.split("=", 1)
            elif ":" in s:
                old, new = s.split(":", 1)
            else:
                print(f"Error: --rename expects OLD=NEW (or OLD:NEW), got: {s!r}", file=sys.stderr)
                sys.exit(1)
            old = old.strip()
            new = new.strip()
            if not old or not new:
                print(f"Error: invalid --rename mapping: {s!r}", file=sys.stderr)
                sys.exit(1)
            rename_map[old] = new

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

    required_cols = (flat_cols_unique if args.hist else ([args.ref] + flat_cols_unique))
    if df is not None:
        try:
            validate_columns(df, required_cols)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        df = df.dropna(subset=required_cols)
        x = np.asarray(df[args.ref], dtype=float) if not args.hist else None
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
        x = arrs[args.ref][mask] if not args.hist else None
        y_arrays = {c: arrs[c][mask] for c in flat_cols_unique}

    colors = list(COLOR_CYCLE)

    n_plots = len(plot_groups)
    # one tall figure, height grows with number of plots
    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(float(FIG_WIDTH_IN), float(FIG_HEIGHT_PER_PLOT_IN) * n_plots),
        sharex=True,
    )

    # if only one subplot, axes is not a list; make it iterable
    if n_plots == 1:
        axes = [axes]

    for gi, (group, ax) in enumerate(zip(plot_groups, axes)):
        for li, col in enumerate(group):
            try:
                yv = y_arrays[col]
                med = float(np.median(yv))
                q25 = float(np.quantile(yv, 0.25))
                q75 = float(np.quantile(yv, 0.75))
                iqr = float(q75 - q25)
                std = float(np.std(yv))
                base = float(args.dropout_base) if args.dropout_base is not None else float(med)
                mult = float(args.dropout_mult)
                thr = float(mult * base)
                # Note: y_arrays already filters to finite values, but keep this robust.
                yv_f = yv[np.isfinite(yv)]
                n = int(yv_f.size)
                n_drop = int(np.sum(yv_f > thr)) if n > 0 else 0
                drop_rate = (float(n_drop) / float(n)) if n > 0 else float("nan")
                print(
                    f"Column {col}, "
                    f"avg: {float(np.average(yv))}, "
                    f"std: {std}, "
                    f"median: {med}, "
                    f"iqr: {iqr}, "
                    f"q25: {q25}, "
                    f"q75: {q75}, "
                    f"max: {float(np.max(yv))}, "
                    f"min: {float(np.min(yv))}, "
                    f"dropout_rate: {drop_rate} "
                    f"(dropouts: {n_drop}/{n}, threshold: {thr} = {mult}*{base})"
                )
            except Exception:
                pass

            color = colors[(gi + li) % len(colors)]
            label = rename_map.get(col, col)
            if args.hist:
                ax.hist(
                    y_arrays[col],
                    bins=int(max(1, args.bins)),
                    density=bool(args.density),
                    alpha=0.45,
                    color=color,
                    label=label,
                )
            else:
                ax.plot(x, y_arrays[col], color=color, label=label)

        # Y label precedence:
        #   --ylabel (global) > --ylabels (per subplot) > default
        if args.ylabel:
            ax.set_ylabel(str(args.ylabel))
        elif args.ylabels and gi < len(args.ylabels) and str(args.ylabels[gi]).strip():
            ax.set_ylabel(str(args.ylabels[gi]).strip())
        else:
            if len(group) == 1:
                ax.set_ylabel(rename_map.get(group[0], group[0]))
            else:
                ax.set_ylabel("value")

        if args.legend:
            ax.legend(loc=str(args.legend_loc))

        ax.grid(True, linestyle=str(GRID_STYLE), alpha=float(GRID_ALPHA))

    # common x-label on the last subplot
    if args.xlabel:
        axes[-1].set_xlabel(str(args.xlabel))
    else:
        axes[-1].set_xlabel("value" if args.hist else rename_map.get(args.ref, args.ref))

    if args.title:
        fig.suptitle(args.title, y=float(TITLE_Y))

    # Make plots use as much of the figure canvas as possible.
    # Use tight_layout but reserve a small band for the title if present.
    top_rect = float(MARGIN_TOP_WITH_TITLE) if args.title else float(MARGIN_TOP_NO_TITLE)
    try:
        fig.tight_layout(rect=(float(MARGIN_LEFT), float(MARGIN_BOTTOM), float(MARGIN_RIGHT), float(top_rect)))
    except Exception:
        pass
    try:
        fig.subplots_adjust(
            left=float(MARGIN_LEFT),
            right=float(MARGIN_RIGHT),
            bottom=float(MARGIN_BOTTOM),
            top=float(top_rect),
            hspace=float(HSPACE),
        )
    except Exception:
        pass

    if args.out:
        try:
            fig.savefig(args.out, dpi=int(DPI))
            print(f"Saved plot: {args.out}")
        except Exception as e:
            print(f"Error saving figure: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        plt.show()


if __name__ == "__main__":
    main()

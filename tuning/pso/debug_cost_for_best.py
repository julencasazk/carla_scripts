"""
Debug utility to evaluate a single PID candidate against the cost function.

What it does
  - Simulates a fixed plant + PID and prints step metrics.
  - Computes full cost decomposition and prints each term.

Outputs
  - Console metrics and optional CSV step response (if --out is provided).

Run (example)
  python3 tuning/pso/debug_cost_for_best.py --out step.csv
"""

import argparse
import csv
import numpy as np
import control as ctl
from lib.pso_pid_tuning import pid_cost_custom_with_Gd, sim_closed_loop_pid, step_metrics

def _write_csv(path, t, y, u, ref):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "ref", "y", "u"])
        for i in range(len(t)):
            writer.writerow([f"{t[i]:.6f}", f"{ref:.6f}", f"{y[i]:.6f}", f"{u[i]:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--out",
        "-f",
        dest="out",
        default=None,
        help="Write step response to CSV (t_s, ref, y, u).",
    )
    args = ap.parse_args()

    Ts = 0.01
    # Middle-range fitted model from control_plant_test_pure_python.py (newer)
    G0 = ctl.TransferFunction(
        [0, 56.3703, 155.5811, 2.6035],
        [1.0000, 15.5755, 2.6333, 0.0852],
    )
    Gd = ctl.c2d(G0, Ts, method='tustin')

    N_filt = 20.0
    u_min, u_max = 0.0, 1.0
    kb_aw = 1.0
    T_sim = 80.0
    ref = 15.0

    # Ziegler Nichols
    # Ku = 1.89
    # Kp = 1.1151
    # Ki = 0.005
    # Kd = 0.00125
    x_bad = np.array([1.1151, 0.005, 0.00125])

    # First: raw metrics with same sim/metrics used in cost
    t_nom, y_nom, u_nom = sim_closed_loop_pid(
        x_bad[0], x_bad[1], x_bad[2],
        N_filt, Ts,
        Gd,
        u_min, u_max, kb_aw,
        t_final=T_sim, ref=ref,
        disturb_time=None,
        disturb_value=None
    )
    SSE_nom, OS_nom, Tr_nom, Ts__nom = step_metrics(t_nom, y_nom, ref)
    print("Raw metrics for bad PID:")
    print(f"  SSE_nom = {SSE_nom:.3f}")
    print(f"  OS_nom  = {OS_nom:.3f} %")
    print(f"  Tr_nom  = {Tr_nom:.3f} s")
    print(f"  Ts_nom  = {Ts__nom:.3f} s")

    # Second: full cost decomposition
    J, info = pid_cost_custom_with_Gd(
        x_bad,
        Gd=Gd,
        N=N_filt, Ts=Ts,
        u_min=u_min, u_max=u_max, kb_aw=kb_aw,
        t_final=T_sim, ref=ref,
        bounded=True
    )

    print("\nFrom cost function:")
    print(f"  J1_SSE_norm = {info['J1_SSE_norm']:.6f}")
    print(f"  J2_OS_norm  = {info['J2_OS_norm']:.6f}")
    print(f"  J3_Ts_Tr    = {info['J3_Ts_Tr_norm']:.6f}")
    print(f"  Total_J     = {info['Total_J']:.6f}")
    if args.out:
        _write_csv(args.out, t_nom, y_nom, u_nom, ref)
        print(f"\nWrote step response CSV: {args.out}")

if __name__ == '__main__':
    main()

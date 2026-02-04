"""
Randomly samples PID gains to build raw metric distributions for scaling.

Outputs
  - JSON file with raw metrics (metrics_scan.json by default).
  - Optional histogram plots.

Run (example)
  python3 tuning/pso/debug_pso_metrics_scan.py
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import control as ctl

from lib.pso_pid_tuning import pid_cost_custom_with_Gd, sim_closed_loop_pid

def build_plant(Ts):
    Td = 0.15
    s = ctl.TransferFunction.s
    G0 = (12.68) / (s**2 + 1.076*s + 0.2744)
    num_delay, den_delay = ctl.pade(Td, 1)
    H_delay = ctl.tf(num_delay, den_delay)
    Gc = G0 * H_delay
    Gd = ctl.c2d(Gc, Ts, method='tustin')
    return Gd

def random_scan(num_samples=200,
                Ts=0.01,
                T_sim=80.0,
                setpoint=33.33, # Maximum allowed speed in Europe highways (Not Germany lol who cares)
                bounds=((0.0, 20.0), (0.0, 5.0), (0.0, 5.0)),
                out_json="metrics_scan.json",
                seed=0):
    np.random.seed(seed)

    u_min, u_max = 0.0, 1.0
    kb_aw = 1.0
    N_filt = 20.0

    Gd = build_plant(Ts)
    print("Using Gd:", Gd)

    logs = []

    for i in range(num_samples):
        kp = np.random.uniform(bounds[0][0], bounds[0][1])
        ki = np.random.uniform(bounds[1][0], bounds[1][1])
        kd = np.random.uniform(bounds[2][0], bounds[2][1])

        x = np.array([kp, ki, kd], dtype=float)

        # bounded=False => get *raw* metrics in info
        J, info = pid_cost_custom_with_Gd(
            x,
            Gd=Gd,
            N=N_filt,
            Ts=Ts,
            u_min=u_min,
            u_max=u_max,
            kb_aw=kb_aw,
            t_final=T_sim,
            ref=setpoint,
            bounded=False
        )

        logs.append(info)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"Sample {i+1}/{num_samples}: J={J:.3f}  "
                  f"Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f}")

    with open(out_json, "w") as f:
        json.dump(logs, f, indent=2, default=float)
    print(f"Saved raw metrics to {out_json}")

    return logs

def plot_hist(data, title, xlabel):
    plt.figure()
    plt.hist(data, bins=30, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.grid(True)

def main():
    Ts = 0.01
    T_sim = 80.0
    setpoint = 50.0
    bounds = ((0.0, 20.0), (0.0, 5.0), (0.0, 5.0))

    logs = random_scan(
        num_samples=200,
        Ts=Ts,
        T_sim=T_sim,
        setpoint=setpoint,
        bounds=bounds,
        out_json="metrics_scan.json",
        seed=42
    )

    # Extract raw metrics from logs
    SSE_vals   = [log["J1_contrib"] for log in logs]   # SSE_nom (since bounded=False)
    OS_vals    = [log["J2_contrib"] for log in logs]   # OS_nom
    TsTr_vals  = [log["J3_contrib"] for log in logs]   # TsTr_nom
    RVG_vals   = [log["J4_contrib"] for log in logs]   # RVG_raw
    ODJ_vals   = [log["J5_contrib"] for log in logs]   # ODJ_raw
    CEC_vals   = [log["J6_contrib"] for log in logs]   # CEC_raw

    print("\nSummary statistics (raw metrics):")
    def stats(name, arr):
        arr = np.array(arr)
        print(f"{name:6s}  min={arr.min():.4f}  p25={np.percentile(arr,25):.4f}  "
              f"p50={np.median(arr):.4f}  p75={np.percentile(arr,75):.4f}  "
              f"max={arr.max():.4f}")

    stats("SSE",  SSE_vals)
    stats("OS",   OS_vals)
    stats("TsTr", TsTr_vals)
    stats("RVG",  RVG_vals)
    stats("ODJ",  ODJ_vals)
    stats("CEC",  CEC_vals)

    # Histograms
    plot_hist(SSE_vals,  "SSE distribution (raw)",   "SSE")
    plot_hist(OS_vals,   "Overshoot distribution",   "OS [%]")
    plot_hist(TsTr_vals, "Ts-Tr distribution",       "Ts-Tr [s]")
    plot_hist(ODJ_vals,  "ODJ (disturbance) distr.", "ODJ_raw")
    plot_hist(CEC_vals,  "CEC (effort) distr.",      "CEC_raw")

    plt.show()

if __name__ == "__main__":
    main()

"""
Evaluates a single PID against normalized cost metrics.

Outputs
  - Console printout of raw metrics and scaled contributions.

Run (example)
  python3 tuning/pso/debug_single_pid_cost.py
"""

import numpy as np
import control as ctl

from lib.pid_tuning_offline_norm_mp import (
    pid_cost_normalized,
    load_metric_scales,
)

def build_plant(Ts: float):
    """Build the same Gd plant used in pid_tuning_offline_norm_mp.py."""
    Td = 0.15
    s = ctl.TransferFunction.s
    G0 = 12.68 / (s**2 + 1.076*s + 0.2744)
    num_delay, den_delay = ctl.pade(Td, 1)
    H_delay = ctl.tf(num_delay, den_delay)
    Gc = G0 * H_delay          # NOTE: no 1.4 gain here
    Gd = ctl.c2d(Gc, Ts, method="tustin")
    return Gd

def main():
    # ----- simulation + cost settings (must match tuner) -----
    Ts = 0.01
    T_sim = 80.0
    u_min, u_max = 0.0, 1.0
    setpoint = 33.33

    # PID to test (replace with PSO best result)
    kp = 0.63585695
    ki = 0.85758202
    kd = 3.63651481

    N_filt = 20.0
    kb_aw = 1.0
    x = np.array([kp, ki, kd], dtype=float)

    print("Testing PID:")
    print(f"  Kp = {kp}")
    print(f"  Ki = {ki}")
    print(f"  Kd = {kd}\n")

    # Build plant and load scales
    Gd = build_plant(Ts)
    scales = load_metric_scales("metric_scales.json")

    # Use default weights from pid_cost_normalized (pass weights=None)
    J, metrics = pid_cost_normalized(
        x,
        Gd,
        N=N_filt,
        Ts=Ts,
        u_min=u_min,
        u_max=u_max,
        kb_aw=kb_aw,
        t_final=T_sim,
        ref=setpoint,
        scales=scales,
        weights=None,
    )

    # Replicate the default weights here for printing
    weights = {
        "SSE":    1.0,
        "OS":     1.0,
        "TsTr":   0.8,
        "RVG":    0.2,
        "ODJ":    0.2,
        "u_rms":  0.1,
        "du_rms": 0.8,
        "sat_ratio": 0.3,
    }

    print("Raw metrics:")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v:.6f}")

    print("\nScaled contributions (val / sigma) and weighted terms:")
    total = 0.0
    for k, v in metrics.items():
        sigma = float(scales.get(k, 1.0))
        sigma = max(sigma, 1e-6)
        g = v / sigma
        w = weights.get(k, 0.0)
        contrib = w * g
        total += contrib
        print(f"  {k:8s}: val={v:10.6f}  sigma={sigma:10.6f}  g=val/sigma={g:10.6f}  w={w:5.2f}  w*g={contrib:10.6f}")

    print(f"\nJ reported by pid_cost_normalized: {J:.6f}")
    print(f"Sum of printed weighted terms    : {total:.6f}")

if __name__ == "__main__":
    main()

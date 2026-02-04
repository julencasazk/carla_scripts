"""
Evaluates control-effort metrics (u_rms, du_rms, saturation) for candidate PIDs.

Outputs
  - Console table of effort metrics and J6 contributions.

Run (example)
  python3 tuning/pso/debug_j6_cec_scan.py
"""

import numpy as np
import control as ctl
from lib.pso_pid_tuning import pid_cost_custom_with_Gd, sim_closed_loop_pid

def main():
    Ts = 0.01
    Td = 0.15
    s = ctl.TransferFunction.s
    G0 = (12.68) / (s**2 + 1.076*s + 0.2744)
    num_delay, den_delay = ctl.pade(Td, 1)
    H_delay = ctl.tf(num_delay, den_delay)
    Gc = G0 * H_delay
    Gd = ctl.c2d(Gc, Ts, method='tustin')

    N_filt = 20.0
    u_min, u_max = 0.0, 100.0
    kb_aw = 1.0
    T_sim = 80.0
    ref = 50.0

    # Some example PIDs: clearly bad, medium, and your "somewhat stable" one
    candidates = [
        (1.0, 0.0, 0.0),                      # very weak
        (5.0, 0.5, 0.1),                      # medium
        (10.0, 1.0, 0.2),                     # stronger
        (2.65994624, 4.08895623, 4.66907485)  # your earlier gains
    ]

    for kp, ki, kd in candidates:
        x = np.array([kp, ki, kd])

        # First, run the nominal closed-loop sim to get u(t)
        t_nom, y_nom, u_nom = sim_closed_loop_pid(
            kp, ki, kd,
            N_filt, Ts,
            Gd,
            u_min, u_max, kb_aw,
            t_final=T_sim, ref=ref,
            disturb_time=None,
            disturb_value=None
        )

        u_rms = np.sqrt(np.mean(u_nom**2))
        du = np.diff(u_nom)
        du_rms = np.sqrt(np.mean(du**2)) if len(du) > 0 else 0.0
        sat_ratio = np.mean(u_nom >= (u_max - 1e-3))

        # Then compute full cost to see J6 as implemented
        J, info = pid_cost_custom_with_Gd(
            x,
            Gd=Gd,
            N=N_filt, Ts=Ts,
            u_min=u_min, u_max=u_max, kb_aw=kb_aw,
            t_final=T_sim, ref=ref,
            bounded=True
        )

        print(f"Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")
        print(f"  u_rms     = {u_rms:.4f}")
        print(f"  du_rms    = {du_rms:.4f}")
        print(f"  sat_ratio = {sat_ratio:.4f}")
        print(f"  J6_norm   = {info['J6_CEC_norm']:.6f}")
        print(f"  J6_contrib= {info['J6_contrib']:.6f}")
        print(f"  Total_J   = {info['Total_J']:.6f}")
        print("-" * 50)

if __name__ == '__main__':
    main()

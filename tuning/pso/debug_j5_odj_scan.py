"""
Evaluates the ODJ term (output disturbance rejection) for candidate PIDs.

Outputs
  - Console table of ODJ contributions.

Run (example)
  python3 tuning/pso/debug_j5_odj_scan.py
"""

import numpy as np
import control as ctl
from lib.pso_pid_tuning import pid_cost_custom_with_Gd

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

candidates = [
    (1.0, 0.0, 0.0),          # clearly bad
    (5.0, 0.5, 0.1),          # ok-ish
    (2.65994624, 4.08895623, 4.66907485),  # your good PID
]

for kp, ki, kd in candidates:
    J, info = pid_cost_custom_with_Gd(
        np.array([kp, ki, kd]),
        Gd=Gd,
        N=N_filt, Ts=Ts,
        u_min=u_min, u_max=u_max, kb_aw=kb_aw,
        t_final=T_sim, ref=ref,
        bounded=True
    )
    print(f"Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f} -> "
          f"ODJ_raw={info['J5_contrib']:.5f}, J5_norm={info['J5_ODJ_norm']:.5f}, Total_J={info['Total_J']:.5f}")

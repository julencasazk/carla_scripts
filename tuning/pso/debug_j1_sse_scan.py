"""
Quick scan of a few PID candidates and prints SSE/OS/Tr/Ts metrics.

Outputs
  - Console table of metrics per candidate.

Run (example)
  python3 tuning/pso/debug_j1_sse_scan.py
"""

import numpy as np
import control as ctl
from lib.pso_pid_tuning import sim_closed_loop_pid, step_metrics

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

def test_pid(kp, ki, kd):
    t, y, u = sim_closed_loop_pid(
        kp, ki, kd,
        N_filt, Ts,
        Gd,
        u_min, u_max, kb_aw,
        t_final=T_sim, ref=ref,
        disturb_time=None,
        disturb_value=None
    )
    SSE, OS, Tr, Ts_ = step_metrics(t, y, ref)
    return SSE, OS, Tr, Ts_

candidates = [
    (1.0, 0.0, 0.0),
    (2.0, 0.1, 0.0),
    (5.0, 0.5, 0.1),
    (10.0, 1.0, 0.2),
    (15.0, 1.5, 0.5),
    (20.0, 2.0, 1.0),
    (2.65994624,
4.08895623,
4.66907485),
]

for kp, ki, kd in candidates:
    SSE, OS, Tr, Ts_ = test_pid(kp, ki, kd)
    print(f"Kp={kp:5.2f} Ki={ki:5.2f} Kd={kd:5.2f}  "
          f"SSE={SSE:7.3f}  OS={OS:7.3f}  Tr={Tr:7.3f}  Ts={Ts_:7.3f}")

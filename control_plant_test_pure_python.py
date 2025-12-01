import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from PID import PID


# Simulation / sampling
Ts = 0.01
T_sim = 80.0
N_steps = int(T_sim / Ts)

# Real plant with delay (same as in pso_pid_tuning.py)
Td = 0.15
s = ctl.TransferFunction.s
G0 = (12.68) / (s**2 + 1.076*s + 0.2744)
num_delay, den_delay = ctl.pade(Td, 1)
H_delay = ctl.tf(num_delay, den_delay)
Gc = G0 * H_delay
Gd = ctl.c2d(Gc, Ts, method='tustin')

print(f"Gd: {Gd}")

# Convert to discrete state-space
Ad, Bd, Cd, Dd = ctl.ssdata(ctl.ss(Gd))
Ad = np.asarray(Ad)
Bd = np.asarray(Bd).reshape(-1)
Cd = np.asarray(Cd)
Dd = np.asarray(Dd).reshape(-1)

# PID gains (from PSO)
kp = 0.63585695
ki = 0.85758202
kd = 3.63651481

N_d = 20
kb_aw = 1.0
u_min, u_max = 0.0, 1.0   # throttle 0–1

pid = PID(kp, ki, kd, N_d, Ts, u_min, u_max, kb_aw, der_on_meas=True)

# Time and reference
time = np.arange(N_steps + 1) * Ts   # store y[0..N_steps]
r = np.zeros_like(time)
r[time >= 5.0] = 40.0
r[time >= 50.0] = 0.0

y = np.zeros_like(time)
u = np.zeros_like(time)

# Initial plant state
x = np.zeros(Ad.shape[0])

for k in range(N_steps):

    y_meas = y[k]


    u[k] = pid.step(r[k], y_meas)


    x = Ad @ x + Bd * u[k]
    y[k+1] = (Cd @ x + Dd * u[k]).item()


u[-1] = u[-2]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, r, '--', label='r')
plt.plot(time, y, label='y')
plt.grid(True)
plt.legend()
plt.ylabel('Speed [m/s]')

plt.subplot(2, 1, 2)
plt.plot(time, u, label='u (throttle)')
plt.grid(True)
plt.legend()
plt.xlabel('t [s]')

plt.tight_layout()
plt.show()

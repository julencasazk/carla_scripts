import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from PID import PID


# Simulation / sampling
Ts = 0.01
T_sim = 160
N_steps = int(T_sim / Ts)

# Real plant with delay (same as in pso_pid_tuning.py)
s = ctl.TransferFunction.s

G0 = (8.7129*s + 0.05263) / (s**2 + 0.1953*s + 0.0001874) 
Gd = ctl.c2d(G0, Ts, method='tustin')

print(f"Gd: {Gd}")

# Convert to discrete state-space
Ad, Bd, Cd, Dd = ctl.ssdata(ctl.ss(Gd))
Ad = np.asarray(Ad)
Bd = np.asarray(Bd).reshape(-1)
Cd = np.asarray(Cd)
Dd = np.asarray(Dd).reshape(-1)

# PID gains (from PSO)
kp = 2.59397071e-01
ki = 1.27381733e-01
kd = 6.21160744e-03
N_d = 5.0
kb_aw = 1.0
u_min, u_max = 0.0, 1.0   # throttle 0–1


pid = PID(kp, ki, kd, N_d, Ts, u_min, u_max, kb_aw, der_on_meas=True)

# Time and reference
time = np.arange(N_steps + 1) * Ts   # store y[0..N_steps]
r = np.zeros_like(time)
'''
# Small changes test
#   for checking robustness
r[time >= 0.0] = 33.33 
r[time >= 15.0] = 32.00
r[time >= 30.0] = 33.33 

r[time >= 45.0] = 15.00
r[time >= 60.0] = 14.00
r[time >= 75.0] = 16.00

r[time >= 90.0] = 5.00
r[time >= 105.0] = 4.00
r[time >= 120.0] = 6.00

r[time >= 135.0] = 0.00
r[time >= 150.0] = 20.00
'''

# Standard step change test
r[time >= 0.0] = 33.33
r[time >= 60.0] = 0.0


y = np.zeros_like(time)
u = np.zeros_like(time)

meas_noise_std = 0.05

# Initial plant state
x = np.zeros(Ad.shape[0])

for k in range(N_steps):

    y_meas = y[k]
    
    # Gaussian measurement noise
    y_meas += np.random.normal(0.0, meas_noise_std)


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

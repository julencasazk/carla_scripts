import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from PID import PID


# Simulation / sampling
Ts = 0.01
T_sim = 80.0
N_steps = int(T_sim / Ts)

# Continuous-time plant without delay

# Plant from paper, not representative
'''
# Params
k = 0.156
wn = 0.396
zeta = 0.661
Td = 0.146

s = ctl.TransferFunction.s
G0 = k / (s**2 + 2*zeta*wn*s + wn**2)

# Delay as Pade(1)
num_delay, den_delay = ctl.pade(Td, 1)
H_delay = ctl.tf(num_delay, den_delay)

# Full continuous plant (with Pade delay), then discretize
Gc = G0 * H_delay
Gd = ctl.c2d(Gc, Ts, method='tustin')
'''


# Real discrete plant (2p1z + delay)
Td = 0.15
s = ctl.TransferFunction.s
G0 = (12.68) / (s**2 + 1.076*s + 0.2744)
num_delay, den_delay = ctl.pade(Td, 1)
H_delay = ctl.tf(num_delay, den_delay)

# Full continuous plant (with Pade delay), then discretize

Gc = G0 * H_delay
Gd = ctl.c2d(Gc, Ts, method='tustin')
    
print(f"Gd: {Gd}")
numd, dend = ctl.tfdata(Gd)
numd = np.squeeze(numd)
dend = np.squeeze(dend)
a = dend / dend[0]
b = numd / dend[0]


#a = np.array([1.0, -1.991, 0.991])
#b = np.array([0.0, 0.001032])

na = len(a) - 1
nb = len(b) - 1


# PID gains
# For CARLA car fitted plant
kp = 1.05730764
ki = 1.3201751
kd = 5.0

# For OLD plant - from paper
#kp = 1.3354087
#ki = 0.35239362
#kd = 4.11808942

N_d = 20
kb_aw = 1.0
u_min, u_max = 0.0, 100.0

pid = PID(kp, ki, kd, N_d, Ts, u_min, u_max, kb_aw, der_on_meas=True)

time = np.arange(N_steps) * Ts
r = np.zeros(N_steps)
r[time >= 5.0] = 50
r[time >= 50] = 0
y = np.zeros(N_steps)
u = np.zeros(N_steps)
y_hist = np.zeros(na)
u_hist = np.zeros(nb)

for k_idx in range(N_steps):
    y_meas = y[k_idx-1] if k_idx > 0 else 0.0 
    u_k = pid.step(r[k_idx], y_meas)
    u[k_idx] = u_k

    y_from_past = -np.dot(a[1:], y_hist) if na > 0 else 0.0
    if nb > 0:
        u_vec = np.concatenate(([u_k], u_hist))
    else:
        u_vec = np.array([u_k])

    y_k = y_from_past + np.dot(b, u_vec)
    y[k_idx] = y_k

    if na > 0:
        y_hist = np.concatenate(([y_k], y_hist[:-1]))
    if nb > 0:
        u_hist = np.concatenate(([u_k], u_hist[:-1]))

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, r, '--', label='r')
plt.plot(time, y, label='y')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, u, label='u')
plt.grid(True)
plt.legend()
plt.xlabel('t [s]')

plt.tight_layout()
plt.show()

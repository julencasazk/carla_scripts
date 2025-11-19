
import numpy as np
import matplotlib.pyplot as plt
import control as ctl

# Plant params
k = 0.156
wn = 0.396
zeta = 0.661
Td = 0.146

# Simulation / sampling
Ts = 0.001
T_sim = 80.0
N_steps = int(T_sim / Ts)

# Continuous-time plant without delay
s = ctl.TransferFunction.s
G0 = k / (s**2 + 2*zeta*wn*s + wn**2)

# Delay as Pade(1)
num_delay, den_delay = ctl.pade(Td, 1)
H_delay = ctl.tf(num_delay, den_delay)

# Full continuous plant (with Pade delay), then discretize
Gc = G0 * H_delay
Gd = ctl.c2d(Gc, Ts, method='tustin')

numd, dend = ctl.tfdata(Gd)
numd = np.squeeze(numd)
dend = np.squeeze(dend)
a = dend / dend[0]
b = numd / dend[0]
na = len(a) - 1
nb = len(b) - 1


time = np.arange(N_steps) * Ts
r = np.zeros(N_steps)
r[time >= 5.0] = 30

y = np.zeros(N_steps)
u = np.zeros(N_steps)
y_hist = np.zeros(na)
u_hist = np.zeros(nb)

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

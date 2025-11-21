import numpy as np
import matplotlib.pyplot as plt
import control as ctl

# Plant params
k = 0.156
wn = 0.396
zeta = 0.661
Td = 0.146

# Simulation / sampling
Ts = 0.01
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


class PID:
    def __init__(self, kp, ki, kd, N, Ts, u_min, u_max, kb_aw):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.n = N
        self.ts = Ts
        self.min_output = u_min
        self.max_output = u_max
        self.kb_aw = kb_aw

        self.i_state = 0.0
        self.e_prev = 0.0
        self.d_prev = 0.0
        self.w_prev = 0.0

        self._update_derivative_coeffs()

    def _update_derivative_coeffs(self):
        Kd = self.kd
        N = self.n
        Ts = self.ts
        if Kd <= 0.0 or N <= 0.0:
            self.k_u = 0.0
            self.k_w = 0.0
        else:
            self.k_u = -((N * Ts - 2.0) / (N * Ts + 2.0))     # Tustin
            self.k_w = (2.0 * Kd) / (N * Ts + 2.0)

    def step(self, setpoint, measurement):
        e = setpoint - measurement

        u_p = self.kp * e

        w = e  # derivative on error
        u_d = self.k_u * self.d_prev + self.k_w * w - self.k_w * self.w_prev
        self.d_prev = u_d
        self.w_prev = w

        dI_base = self.ki * self.ts * e  # backward Euler
        u_i = self.i_state + dI_base

        u_unsat = u_p + u_i + u_d
        u = max(self.min_output, min(self.max_output, u_unsat))

        aw = self.kb_aw * (u - u_unsat)  # back-calculation
        self.i_state = u_i + aw

        self.e_prev = e
        return u


# PID gains
kp = 1.5
ki = 0.3
kd = 0.0
N_d = 15.0
kb_aw = 1.0
u_min, u_max = 0, 100.0

pid = PID(kp, ki, kd, N_d, Ts, u_min, u_max, kb_aw)

time = np.arange(N_steps) * Ts
r = np.zeros(N_steps)
r[time >= 5.0] = 50

y = np.zeros(N_steps)
u = np.zeros(N_steps)
y_hist = np.zeros(na)
u_hist = np.zeros(nb)

for k_idx in range(N_steps):
    y_meas = y[k_idx - 1] if k_idx > 0 else 0.0
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

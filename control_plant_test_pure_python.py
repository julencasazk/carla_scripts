import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from PID import PID


# Simulation / sampling
Ts = 0.01
T_sim = 360
N_steps = int(T_sim / Ts)

# Real plant with delay (same as in pso_pid_tuning.py)
s = ctl.TransferFunction.s
z = ctl.TransferFunction.z
#G0 = 7.841 / (s + 0.3321) # Low speed range
#G0 = (56.37*s**2 + 155.6*s + 2.604) / (s**3 + 15.58*s**2 + 2.633*s + 0.08517) # middle range
G0 = ctl.TransferFunction([0,45.5849,23.7842,0.5994],
                          [1.0000, 2.4052, 0.2322, 0.0144])
Gd = ctl.c2d(G0, Ts, method='tustin')

print(f"G0: {G0}")
print(f"Zeros: {G0.zeros()}")
print(f"Poles: {G0.poles()}")
print(f"DC Gain: {G0.dcgain()}")

# Convert to discrete state-space
Ad, Bd, Cd, Dd = ctl.ssdata(ctl.ss(Gd))
Ad = np.asarray(Ad)
Bd = np.asarray(Bd).reshape(-1)
Cd = np.asarray(Cd)
Dd = np.asarray(Dd).reshape(-1)

# PID gains (from PSO)
kp =   0.07619123
ki =   0.00952621
kd =   0.0
N_d =  6.27890559
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

#r[time >= 0.0] = 0.0
#r[time >= 50.0] = 5.0
#r[time >= 100.0] = 0.0
#r[time >= 150.0] = 27.222
#r[time >= 200.0] = 27.222 + 5.0
#r[time >= 250.0] = 27.222


y = np.zeros_like(time)
u = np.zeros_like(time)
u[time >= 0.0] = 0.7
u[time >= 100.0] = 0.6
u[time >= 200.0] = 0.7
u[time >= 250.0] = 0.0

meas_noise_std = 0.05

# Initial plant state
x = np.zeros(Ad.shape[0])

for k in range(N_steps):

    y_meas = y[k]
    
    # Gaussian measurement noise
    y_meas += np.random.normal(0.0, meas_noise_std)


    #u[k] = pid.step(r[k], y_meas)


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

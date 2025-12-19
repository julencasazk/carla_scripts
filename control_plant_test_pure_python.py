import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from PID import PID


RANGE = "high"


# Simulation / sampling
Ts = 0.01
T_sim = 360
N_steps = int(T_sim / Ts)

# Plants for each range, only leave one uncommented, obviously
if (RANGE == "low"):
    G0 = ctl.TransferFunction([0, 7.8409],[1.0000, 0.3321]) # Low speed range
elif (RANGE == "mid"):
    G0 = ctl.TransferFunction([0, 56.3703, 155.5811, 2.6035],[1.0000, 15.5755, 2.6333, 0.0852]) # middle range
else:
    G0 = ctl.TransferFunction([0.0751,0.0010,   -0.0741],[1.0000,   -1.9870,    0.9871], Ts) # High speed range
#Gd = ctl.c2d(G0, Ts, method='tustin')
Gd = G0
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

u_min, u_max = 0.0, 1.0   # throttle 0–1
kb_aw = 1.0

if (RANGE == "low"):
    # Low speed range PID
    kp =   0.31919206
    ki =   0.24294889 
    kd =   0.01506719
    N_d =  15.0
elif (RANGE == "mid"):
    # Mid speed range PID
    kp =   0.17077964
    ki =   0.085938
    kd =   0.0
    N_d =   5.53169821
else:
    # High speed range PID
    kp =   0.57619798
    ki =   0.05383481
    kd =   0.13576069
    N_d =  14.00557397

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

# 
if (RANGE == "low"):
    bot = 0.0
    top = 11.11111 
elif (RANGE == "mid"):
    bot = 11.11111 
    top = 22.22222 
else:
    bot = 22.22222
    top = 33.33333 
mid = ((top - bot) / 2.0) + bot

r[time >= 0.0] = bot
r[time >= 50.0] = mid
r[time >= 100.0] = top
r[time >= 150.0] = mid
r[time >= 200.0] = bot
r[time >= 250.0] = top
r[time >= 300.0] = bot

u_raw = np.zeros_like(time)
u_raw[time >= 0.0] = 0.63
u_raw[time >= 50.0] = 0.68
u_raw[time >= 100.0] = 0.75
u_raw[time >= 150.0] = 0.68
u_raw[time >= 200.0] = 0.63
u_raw[time >= 250.0] = 0.75
u_raw[time >= 300.0] = 0.63


y = np.zeros_like(time)
u = np.zeros_like(time)


meas_noise_std = 0.05

# Initial plant state
x = np.zeros(Ad.shape[0])

for k in range(N_steps):

    y_meas = y[k]
    
    # Gaussian measurement noise
    y_meas += np.random.normal(0.0, meas_noise_std)


    #u[k] = pid.step(r[k], y_meas)
    u[k] = u_raw[k]

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

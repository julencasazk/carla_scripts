import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from PID import PID


RANGE = "low"

# -----------------------------
# User-defined operating point
# -----------------------------
if (RANGE == "low"):
    u0 = 0.35   
    v0 = 7.77777
elif (RANGE == "mid"):
    u0 = 0.55
    v0 = 16.666666
else:
    u0=0.68
    v0 = 27.7777
       

# Simulation / sampling
Ts = 0.01
T_sim = 360
N_steps = int(T_sim / Ts)

# Plants for each range
if RANGE == "low":
    G0 = ctl.TransferFunction([0, 8.5777, 54.0770], # Low speeds
                              [1.0000, 7.0533, 2.1718])
elif RANGE == "mid":
    G0 = ctl.TransferFunction([0,   12.3082,   67.2242],
                              [1.0000,    5.2957,    1.0404]) 
else:
    G0 = ctl.TransferFunction([ 0,15.0224,   20.4896],
                              [1.0000,    1.3026,    0.1813])

# Ensure discrete plant at Ts
Gd = G0 if G0.dt not in (None, 0) else ctl.c2d(G0, Ts, method='tustin')

print(f"Gd: {Gd}")
print(f"Zeros: {Gd.zeros()}")
print(f"Poles: {Gd.poles()}")
print(f"DC Gain: {Gd.dcgain()}")

# Convert to discrete state-space
Ad, Bd, Cd, Dd = ctl.ssdata(ctl.ss(Gd))
Ad = np.asarray(Ad)
Bd = np.asarray(Bd).reshape(-1)
Cd = np.asarray(Cd)
Dd = np.asarray(Dd).reshape(-1)

u_min, u_max = 0.0, 1.0
kb_aw = 1.0

# PID params
if RANGE == "low":
    kp, ki, kd, N_d =  0.43127789,  0.43676547,  0.0,     14.64609183
elif RANGE == "mid":
    kp, ki, kd, N_d =  0.11675119, 0.0514106, 0.0, 14.90530836
else:
    kp, ki, kd, N_d =  0.13408096 , 0.07281374 , 0.0, 12.16810135

pid = PID(kp, ki, kd, N_d, Ts, u_min, u_max, kb_aw, der_on_meas=True)

# Time and reference
time = np.arange(N_steps + 1) * Ts
r = np.zeros_like(time)

if RANGE == "low":
    bot, top = 0.0, 11.11111
elif RANGE == "mid":
    bot, top = 11.11111, 22.22222
else:
    bot, top = 22.22222, 33.33333
mid = ((top - bot) / 2.0) + bot

r[time >= 0.0] = bot
r[time >= 50.0] = mid
r[time >= 100.0] = top
r[time >= 150.0] = mid
r[time >= 200.0] = bot
r[time >= 250.0] = top
r[time >= 300.0] = bot

# Absolute throttle profile (as before)
u_raw = np.full_like(time, 0.7)
u_raw[time >= 30]  = 0.75
u_raw[time >= 80]  = 0.7
u_raw[time >= 130] = 0.63
u_raw[time >= 180] = 0.75
u_raw[time >= 230] = 0.63
u_raw[time >= 280] = 0.7
u_raw[time >= 350] = 0

# -----------------------------
# Incremental I/O construction
# -----------------------------
du = u_raw - u0            # model input is Δu
dv = np.zeros_like(time)   # model output is Δv (what we simulate)

# Optional: if you want an incremental reference for a controller
# dr = r - v0

u = np.zeros_like(time)    # will store absolute throttle actually applied (for plotting)
y = np.zeros_like(time)    # will store absolute speed reconstructed (for plotting)

meas_noise_std = 0.05

# Initial plant state (incremental state)
x = np.zeros(Ad.shape[0])

for k in range(N_steps):
    # Current absolute output is reconstructed from dv
    y[k] = v0 + dv[k]

    # Measurement noise on absolute measurement (if you use PID)
    y_meas = y[k] + np.random.normal(0.0, meas_noise_std)

    # If using PID, it should act on absolute (r, y_meas) but drive incremental plant:
    du_cmd = pid.step(r[k], y_meas) - u0
    du[k] = du_cmd
    u[k] = np.clip(u0 + du[k], u_min, u_max)
    
    du_k = u[k] - u0

    # Incremental plant update (Δu -> Δv)
    x = Ad @ x + Bd * du_k
    dv[k + 1] = (Cd @ x + Dd * du_k).item()

# Fill last samples for plotting
u[-1] = u[-2]
y[-1] = v0 + dv[-1]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, r, '--', label='r (abs)')
plt.plot(time, y, label='y (abs reconstructed)')
plt.grid(True)
plt.legend()
plt.ylabel('Speed [m/s]')

plt.subplot(2, 1, 2)
plt.plot(time, u, label='u (abs throttle)')
plt.grid(True)
plt.legend()
plt.xlabel('t [s]')

plt.tight_layout()
plt.show()

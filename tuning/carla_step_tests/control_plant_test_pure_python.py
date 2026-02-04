"""
Offline control-library simulation of fitted plants + PID response.

Outputs
  - Console prints of poles/zeros/gain and matplotlib plots.

Run (example)
  python3 tuning/carla_step_tests/control_plant_test_pure_python.py
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from lib.PID import PID


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
    #G0 = ctl.TransferFunction([0, 8.5777, 54.0770], # Low speeds # Old values
    #                          [1.0000, 7.0533, 2.1718])

    G0 = ctl.TransferFunction([0, 7.8409], # Low speeds # New values 1p continuous
                              [ 1.0000, 0.3321])
elif RANGE == "mid":
    #G0 = ctl.TransferFunction([0,   12.3082,   67.2242],        # Old values
    #                          [1.0000,    5.2957,    1.0404]) 

    G0 = ctl.TransferFunction([0,56.3703, 155.5811,2.6035], # New values, 3p2z continuous
                             [1.0000,    15.5755,    2.6333,    0.0852])
else:

    #G0 = ctl.TransferFunction([ 0,15.0224,   20.4896],         # Old values    
    #                          [1.0000,    1.3026,    0.1813])
    G0 = ctl.TransferFunction([0,45.5849,23.7842,0.5994],      # New values, 3p2z continuous
                              [1.0000,2.4052,0.2322,0.0144])
    #G0 = ctl.TransferFunction([ 0, 16.0307, 241.7231],         # New values 2p1z discrete, seems to be the same as old one
    #                          [1.0000, 13.8711, 2.1097])


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

    #kp, ki, kd, N_d =  0.43127789,0.43676547,0.5,14.64609183
    kp, ki, kd, N_d = 0.1089194,0.04906409, 0.0, 7.71822214 
elif RANGE == "mid":
    #kp, ki, kd, N_d =  0.11675119, 0.0514106, 0.5, 14.90530836
    kp, ki, kd, N_d =  0.11494122, 0.0489494,  0.0, 5.0        
    
else:
    kp, ki, kd, N_d =  0.13408096 , 0.07281374 , 0.0, 12.16810135 # Original from way back
    #kp, ki, kd, N_d =  0.28508729, 0.45692171, 0.0,15.0 # Aggresive with absolute values again
    #kp, ki, kd, N_d =  0.19211881, 0.18328656, 0.0, 9.29094921 # Aggresive with absolute values again

# Discretization variants to compare. (User label, PID discretization_method)
pid_methods = [
    ("tustin", "tustin"),
    ("backeuler", "backeuler"),
    ("fordwardeuler", "forwardeuler"),
]


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

meas_noise_std = 0.0005

results = {}

for label, discretization_method in pid_methods:
    pid = PID(
        kp,
        ki,
        kd,
        N_d,
        Ts,
        u_min,
        u_max,
        kb_aw,
        der_on_meas=True,
        derivative_disc_method=discretization_method,
        integral_disc_method=discretization_method
    )

    # Incremental I/O construction
    dv = np.zeros_like(time)   # model output is Deltav
    u = np.zeros_like(time)    # absolute throttle actually applied
    y = np.zeros_like(time)    # absolute speed reconstructed

    # Initial plant state (incremental state)
    x = np.zeros(Ad.shape[0])

    for k in range(N_steps):
        y[k] = v0 + dv[k]
        y_meas = y[k] + np.random.normal(0.0, meas_noise_std)

        # PID operates on absolute (r, y_meas) and outputs absolute throttle.
        u_cmd = pid.step(r[k], y_meas)
        u[k] = np.clip(u_cmd, u_min, u_max)
        du_k = u[k] - u0

        # Incremental plant update (Deltau -> Deltav)
        x = Ad @ x + Bd * du_k
        dv[k + 1] = (Cd @ x + Dd * du_k).item()

    # Fill last samples for plotting
    u[-1] = u[-2]
    y[-1] = v0 + dv[-1]
    results[label] = {"u": u, "y": y}


fig, (ax_y, ax_u) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax_y.plot(time, r, "k--", linewidth=1.2, label="r")
for label, _ in pid_methods:
    ax_y.plot(time, results[label]["y"], label=f"y ({label})")
ax_y.grid(True)
ax_y.set_ylabel("Speed [m/s]")
ax_y.legend(loc="best")

for label, _ in pid_methods:
    ax_u.plot(time, results[label]["u"], label=f"u ({label})")
ax_u.axhline(u_min, color="k", linewidth=0.8, alpha=0.3)
ax_u.axhline(u_max, color="k", linewidth=0.8, alpha=0.3)
ax_u.grid(True)
ax_u.set_ylabel("Throttle [-]")
ax_u.set_xlabel("t [s]")
ax_u.legend(loc="best")

plt.tight_layout()
plt.show()

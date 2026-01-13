import numpy as np
import matplotlib.pyplot as plt
import control as ctl

from PID import PID


np.random.seed(0)


RANGE = "high"

# -----------------------------
# User-defined operating point
# -----------------------------
if RANGE == "low":
	u0 = 0.35
	v0 = 7.77777
elif RANGE == "mid":
	u0 = 0.55
	v0 = 16.666666
else:
	u0 = 0.68
	v0 = 27.7777

# Simulation / sampling
Ts = 0.01
T_sim = 720
N_steps = int(T_sim / Ts)

# Plants for each range
if RANGE == "low":
	G0 = ctl.TransferFunction([0, 8.5777, 54.0770],  # Low speeds
							  [1.0000, 7.0533, 2.1718])
elif RANGE == "mid":
	G0 = ctl.TransferFunction([0, 12.3082, 67.2242],
							  [1.0000, 5.2957, 1.0404])
else:
	G0 = ctl.TransferFunction([0, 15.0224, 20.4896],
							  [1.0000, 1.3026, 0.1813])

# Ensure discrete plant at Ts
Gd = G0 if G0.dt not in (None, 0) else ctl.c2d(G0, Ts, method="tustin")

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
	kp, ki, kd, N_d = 0.43127789, 0.43676547, 0.5, 14.64609183
elif RANGE == "mid":
	kp, ki, kd, N_d = 0.11675119, 0.0514106, 0.5, 14.90530836
else:
	kp, ki, kd, N_d = 0.13408096, 0.07281374, 0.5, 12.16810135

# Discretization variants to compare. (User label, PID method name)
pid_methods = [
	("tustin", "tustin"),
	("backeuler", "backeuler"),
	("fordwardeuler", "forwardeuler"),
]

# Time and reference
time = np.arange(N_steps + 1) * Ts
r_step = np.zeros_like(time)
r = np.zeros_like(time)

if RANGE == "low":
	bot, top = 0.0, 11.11111
elif RANGE == "mid":
	bot, top = 11.11111, 22.22222
else:
	bot, top = 22.22222, 33.33333
mid = ((top - bot) / 2.0) + bot

r[time >= 0.0] = bot

# Base step schedule (mirrors the original test), repeated to cover the longer sim.
events = [
	(0.0, bot),
	(50.0, mid),
	(100.0, top),
	(150.0, mid),
	(200.0, bot),
	(250.0, top),
	(300.0, bot),
	(350.0, mid),
	(400.0, top),
	(450.0, mid),
	(500.0, bot),
	(550.0, top),
	(600.0, bot),
	(650.0, mid),
	(700.0, bot),
]

r_step[:] = bot
for t_evt, val_evt in events:
	r_step[time >= t_evt] = val_evt

# Reference profile:
# - [0, t_step_end): keep hard steps
# - [t_step_end, t_sine_start): use aggressively low-pass filtered steps
# - [t_sine_start, end]: 2 Hz sine about an offset
t_step_end = 300.0
t_sine_start = 540.0

tau_ref = 8.0
alpha_ref = Ts / (tau_ref + Ts)

sine_hz = 2.0
sine_amp = 0.75
sine_offset = 0.0

k_step_end = int(t_step_end / Ts)
k_sine_start = int(t_sine_start / Ts)
k_step_end = max(0, min(N_steps, k_step_end))
k_sine_start = max(0, min(N_steps, k_sine_start))

# 1) Steps
r[: k_step_end + 1] = r_step[: k_step_end + 1]

# 2) Smoothed transitions (LPF)
if k_step_end < N_steps:
	r[k_step_end] = r_step[k_step_end]
for k in range(k_step_end, min(k_sine_start, N_steps)):
	r[k + 1] = r[k] + alpha_ref * (r_step[k + 1] - r[k])

# 3) 2 Hz sine about a continuous offset
if k_sine_start <= N_steps:
	base = r[k_sine_start] + sine_offset
	for k in range(k_sine_start, N_steps):
		t = time[k + 1] - time[k_sine_start]
		r[k + 1] = base + sine_amp * np.sin(2.0 * np.pi * sine_hz * t)

meas_noise_std = 0.0005

results = {}

for label, disc_method in pid_methods:
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
		derivative_disc_method=disc_method,
		integral_disc_method=disc_method,
	)

	dv = np.zeros_like(time)  # model output is Δv
	u = np.zeros_like(time)   # absolute throttle actually applied
	y = np.zeros_like(time)   # absolute speed reconstructed

	x = np.zeros(Ad.shape[0])

	for k in range(N_steps):
		y[k] = v0 + dv[k]
		y_meas = y[k] + np.random.normal(0.0, meas_noise_std)

		u_cmd = pid.step(r[k], y_meas)
		u[k] = np.clip(u_cmd, u_min, u_max)
		du_k = u[k] - u0

		x = Ad @ x + Bd * du_k
		dv[k + 1] = (Cd @ x + Dd * du_k).item()

	u[-1] = u[-2]
	y[-1] = v0 + dv[-1]

	results[label] = {"u": u, "y": y}


# -----------------------------
# Plots
# -----------------------------
fig, (ax_y, ax_u, ax_dy) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

ax_y.plot(time, r_step, color="0.7", linewidth=1.0, linestyle="--", label="r_step")
ax_y.plot(time, r, "k-", linewidth=1.2, label="r_smooth")
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
ax_u.legend(loc="best")

# Pairwise output differences (y)
y_tustin = results["tustin"]["y"]
y_back = results["backeuler"]["y"]
y_fwd = results["fordwardeuler"]["y"]

dy_back_tustin = y_back - y_tustin
dy_fwd_back = y_fwd - y_back
dy_fwd_tustin = y_fwd - y_tustin

ax_dy.plot(time, dy_back_tustin, label="backeuler - tustin")
ax_dy.plot(time, dy_fwd_back, label="fordwardeuler - backeuler")
ax_dy.plot(time, dy_fwd_tustin, label="fordwardeuler - tustin")
ax_dy.axhline(0.0, color="k", linewidth=0.8, alpha=0.3)
ax_dy.grid(True)
ax_dy.set_ylabel("Δy [m/s]")
ax_dy.set_xlabel("t [s]")
ax_dy.legend(loc="best")

dy_max = float(np.max(np.abs([dy_back_tustin, dy_fwd_back, dy_fwd_tustin])))
print(f"Max |Δy| across methods: {dy_max:.6g} m/s")
if dy_max > 0.0:
	ax_dy.set_ylim(-1.1 * dy_max, 1.1 * dy_max)

plt.tight_layout()
plt.show()


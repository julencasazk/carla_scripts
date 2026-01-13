"""
PSO-based PID tuning (absolute I/O), using an incremental (mean-removed) plant model internally.

Goal:
  - The controller and all performance metrics are computed in ABSOLUTE units:
      input:  v_ref_abs [m/s], v_meas_abs [m/s]
      output: u_abs (throttle) in [u_abs_min, u_abs_max]
  - The identified plant model is incremental (mean-removed):
      Δv = Gd * Δu
    so inside the simulation we:
      Δu = u_abs - u0
      Δv -> v_abs = v0 + Δv

Operating point (from your dataset stats):
  v0 = 29.1063989 m/s
  u0 = 0.6933333333333334

Actuator limits (absolute throttle):
  u_abs_min = 0.63
  u_abs_max = 0.75

Step tests:
  - Default step is "aggressive" but capped to remain inside [v_abs_min, v_abs_max].
  - You can adjust STEP_DV_ABS and/or STEP_TARGET_MODE below.

Metrics (absolute-domain):
  SSE: |v_ref_abs - v_final_abs|
  OS (%): sign-robust and valid for up/down steps using normalized response:
          OS = max(0, max(s) - 1)*100 with s = (v - v0)/(v_ref - v0) for ref != v0
  Tr: rise time from 10% to 90% in normalized space
  Ts: settling time into ±tol*|step| band around final value
  TsTr: Ts - Tr, but "hard-penalizes" weird/unreached cases by setting TsTr=t_final
        (this is still a metric, not an extra penalty term)

Determinism:
  MEAS_NOISE_STD defaults to 0.0 during tuning/scales.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import control as ctl
import multiprocessing as mp

from PID import PID  # your existing PID implementation


JSON_FILE = "metric_scales_abs.json"
FIXED_KAW = 1.0

# Keep objective deterministic by default
MEAS_NOISE_STD = 0.0

# Aggressive step magnitude in absolute speed, but applied as Δ around v0 then capped.
STEP_DV_ABS = 3.0

# Choose how to set the absolute target (default: v0 + STEP_DV_ABS capped)
# Options: "UP_FROM_V0", "CENTER_OF_RANGE", "CUSTOM"
STEP_TARGET_MODE = "UP_FROM_V0"
CUSTOM_V_TARGET_ABS = 27.7778  # used only if STEP_TARGET_MODE == "CUSTOM"


# ----------------------------
# Multiprocessing helper
# ----------------------------
def _pso_worker(args):
    x, objective, objective_args = args
    return objective(x, *objective_args)


# ----------------------------
# PSO
# ----------------------------
def pso(objective,
        bounds,
        num_particles=30,
        max_it=50,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=None,
        verbose=True,
        log_path=None,
        num_workers=1,
        objective_args=None):

    if seed is not None:
        np.random.seed(seed)

    if objective_args is None:
        objective_args = ()

    dim = len(bounds)
    bounds_min = np.array([b[0] for b in bounds], dtype=float)
    bounds_max = np.array([b[1] for b in bounds], dtype=float)

    particles = np.random.uniform(bounds_min, bounds_max, size=(num_particles, dim))
    velocities = np.random.uniform(-1.0, 1.0, size=(num_particles, dim))

    pbest_positions = particles.copy()
    pbest_values = np.full(num_particles, np.inf, dtype=float)

    gbest_position = None
    gbest_value = np.inf

    logs = []
    history = []

    pool = None
    if num_workers > 1:
        pool = mp.Pool(processes=num_workers)

    def eval_particles(particles_batch):
        if pool is None:
            return [objective(x, *objective_args) for x in particles_batch]
        jobs = [(x, objective, objective_args) for x in particles_batch]
        return pool.map(_pso_worker, jobs)

    try:
        # initial
        results = eval_particles(particles)
        for i in range(num_particles):
            J, metrics = results[i]
            pbest_values[i] = J

            logs.append({
                "iter": 0,
                "particle": int(i),
                "J": float(J),
                **{k: float(v) for k, v in metrics.items()}
            })

        best_idx = int(np.argmin(pbest_values))
        gbest_position = pbest_positions[best_idx].copy()
        gbest_value = float(pbest_values[best_idx])
        history.append(gbest_value)

        if verbose:
            print(f"Iteration 0 - best cost: {gbest_value:.6f}")

        for it in range(1, max_it + 1):
            J_values = np.zeros(num_particles, dtype=float)

            for i in range(num_particles):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)

                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (pbest_positions[i] - particles[i])
                    + c2 * r2 * (gbest_position - particles[i])
                )
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], bounds_min, bounds_max)

            results = eval_particles(particles)

            for i in range(num_particles):
                J, metrics = results[i]
                J_values[i] = J

                if J < pbest_values[i]:
                    pbest_values[i] = J
                    pbest_positions[i] = particles[i].copy()

                if J < gbest_value:
                    gbest_value = J
                    gbest_position = particles[i].copy()

                logs.append({
                    "iter": it,
                    "particle": int(i),
                    "J": float(J),
                    **{k: float(v) for k, v in metrics.items()},
                    "best_x": gbest_position.tolist(),
                    "best_cost": float(gbest_value),
                })

            history.append(gbest_value)

            if verbose:
                print(f"Iteration {it} - best cost: {gbest_value:.6f}")
                print(f"  best_x = {gbest_position}")
                print(f"  mean J = {np.mean(J_values):.4f}")

    finally:
        if pool is not None:
            pool.close()
            pool.join()

        if log_path is not None:
            with open(log_path, "w") as f:
                json.dump(logs, f, indent=2, default=float)
            if verbose:
                print(f"Saved log to {log_path}")

    return gbest_position, gbest_value, history


# ----------------------------
# Plant scaling
# ----------------------------
def scale_plant(Gd, gain):
    num, den = ctl.tfdata(Gd)
    num = np.squeeze(num) * gain
    den = np.squeeze(den)
    return ctl.TransferFunction(num, den, Gd.dt)


# ----------------------------
# ABS simulation with incremental plant inside
# ----------------------------
def sim_closed_loop_pid_abs(kp, ki, kd, N, Ts,
                            Gd,
                            v0, u0,
                            u_abs_min, u_abs_max,
                            kb_aw,
                            t_final=5.0,
                            v_ref_abs=29.0,
                            disturb_time=None,
                            disturb_value_abs=0.0,
                            meas_noise_std=MEAS_NOISE_STD):
    """
    Absolute-domain closed-loop simulation:

      PID sees:     v_ref_abs, v_meas_abs
      PID outputs:  u_abs (saturated to [u_abs_min, u_abs_max])

    Internal incremental plant:
      Δu = u_abs - u0
      Δv = Gd * Δu
      v_abs = v0 + Δv

    Disturbance (if enabled) is applied in ABS units as an additive offset to v_meas_abs.
    """

    # Create PID that saturates in ABS units
    pid = PID(kp, ki, kd, N, Ts, u_abs_min, u_abs_max, kb_aw, der_on_meas=True)

    n_steps = int(t_final / Ts)
    t = np.arange(n_steps + 1) * Ts
    v_ref = np.full_like(t, v_ref_abs, dtype=float)

    # plant state-space
    Ad, Bd, Cd, Dd = ctl.ssdata(ctl.ss(Gd))
    Ad = np.asarray(Ad, dtype=float)
    Bd = np.asarray(Bd, dtype=float).reshape(-1)
    Cd = np.asarray(Cd, dtype=float)
    Dd = np.asarray(Dd, dtype=float).reshape(-1)

    x = np.zeros(Ad.shape[0], dtype=float)

    # store ABS signals
    v_abs = np.full_like(t, v0, dtype=float)   # start at operating point
    u_abs = np.full_like(t, u0, dtype=float)

    # store incremental output too (optional/debug)
    dv = np.zeros_like(t, dtype=float)
    du = np.zeros_like(t, dtype=float)

    for k in range(n_steps):
        v_meas = v_abs[k]

        if disturb_time is not None and t[k] >= disturb_time:
            v_meas = v_meas + disturb_value_abs

        if meas_noise_std > 0:
            v_meas += np.random.normal(0.0, meas_noise_std)

        # PID in ABS: outputs u_abs in [u_abs_min, u_abs_max]
        u_abs[k] = pid.step(v_ref[k], v_meas)

        # internal incremental input
        du[k] = u_abs[k] - u0

        # plant update in incremental coordinates
        x = Ad @ x + Bd * du[k]
        dv[k + 1] = (Cd @ x + Dd * du[k]).item()

        # reconstruct ABS output
        v_abs[k + 1] = v0 + dv[k + 1]

    return t, v_abs, u_abs, dv, du


# ----------------------------
# Robust absolute metrics (works for steps up/down)
# ----------------------------
def step_metrics_abs(t, v_abs, v_ref_abs, v0, tol=0.02):
    """
    Metrics for absolute signals, but normalized using incremental step size:

      step = v_ref_abs - v0
      y    = v_abs - v0

      s = y/step (dimensionless). Works for step>0 and step<0.

    Returns: SSE_abs, OS_pct, Tr, Ts_
    """

    t = np.asarray(t, dtype=float)
    v_abs = np.asarray(v_abs, dtype=float)
    t_final = float(t[-1])

    step = float(v_ref_abs - v0)
    y = v_abs - v0

    # SSE in ABS units
    SSE = abs(v_ref_abs - v_abs[-1])

    # If step is ~0, treat as regulation to v0: OS/Tr not meaningful
    if abs(step) < 1e-9:
        # Settling in absolute band around v_ref_abs (=v0)
        band = tol  # absolute band
        outside = np.abs(v_abs - v_ref_abs) > band
        if np.any(outside):
            last_out = int(np.where(outside)[0][-1])
            Ts_ = float(t[last_out + 1]) if (last_out + 1) < len(t) else t_final
        else:
            Ts_ = 0.0
        Tr = t_final
        OS = 0.0
        return float(SSE), float(OS), float(Tr), float(Ts_)

    # Normalized response
    s = y / step

    # Overshoot % in normalized domain
    OS = max(0.0, (np.max(s) - 1.0) * 100.0)

    # Rise time: 10% to 90% in s-domain; if not reached, Tr=t_final
    reached_10 = np.where(s >= 0.1)[0]
    reached_90 = np.where(s >= 0.9)[0]
    if reached_10.size > 0 and reached_90.size > 0:
        idx_10 = int(reached_10[0])
        idx_90 = int(reached_90[0])
        Tr = float(t[idx_90] - t[idx_10])
    else:
        Tr = t_final

    # Settling band around v_ref_abs: ± tol * |step|
    band = tol * abs(step)
    outside = np.abs(v_abs - v_ref_abs) > band
    if np.any(outside):
        last_out = int(np.where(outside)[0][-1])
        Ts_ = float(t[last_out + 1]) if (last_out + 1) < len(t) else t_final
    else:
        Ts_ = 0.0

    return float(SSE), float(OS), float(Tr), float(Ts_)


# ----------------------------
# Raw metrics for cost (ABS world)
# ----------------------------
def pid_metrics_raw_abs(x,
                        Gd,
                        Ts,
                        v0, u0,
                        v_ref_abs,
                        u_abs_min, u_abs_max,
                        t_final=5.0,
                        tol=0.02):
    """
    x = [kp, ki, kd, N]
    Returns metrics dict:
      SSE, OS, TsTr, RVG, ODJ, u_rms, du_rms, sat_ratio
    All computed in ABS domain except internal incremental stepping.
    """
    kp, ki, kd, N = x
    kb_aw = FIXED_KAW

    # Nominal
    t_nom, v_nom, u_nom, _, _ = sim_closed_loop_pid_abs(
        kp, ki, kd, N, Ts,
        Gd,
        v0, u0,
        u_abs_min, u_abs_max,
        kb_aw,
        t_final=t_final,
        v_ref_abs=v_ref_abs,
        disturb_time=None,
        disturb_value_abs=0.0,
        meas_noise_std=MEAS_NOISE_STD
    )

    SSE_nom, OS_nom, Tr_nom, Ts__nom = step_metrics_abs(t_nom, v_nom, v_ref_abs, v0, tol=tol)

    # TsTr with "hard penalty" baked into the metric:
    # if never reaches 90% OR never settles -> TsTr = t_final
    if (Tr_nom >= t_final - 1e-12) or (Ts__nom >= t_final - 1e-12):
        TsTr_nom = float(t_final)
    else:
        TsTr_nom = float(max(0.0, Ts__nom - Tr_nom))

    # Robustness: gain scaling (ABS metric comparisons)
    Gd_low = scale_plant(Gd, 0.7)
    t_low, v_low, _, _, _ = sim_closed_loop_pid_abs(
        kp, ki, kd, N, Ts,
        Gd_low,
        v0, u0,
        u_abs_min, u_abs_max,
        kb_aw,
        t_final=t_final,
        v_ref_abs=v_ref_abs,
        disturb_time=None,
        disturb_value_abs=0.0,
        meas_noise_std=MEAS_NOISE_STD
    )
    SSE_low, OS_low, Tr_low, Ts__low = step_metrics_abs(t_low, v_low, v_ref_abs, v0, tol=tol)
    if (Tr_low >= t_final - 1e-12) or (Ts__low >= t_final - 1e-12):
        TsTr_low = float(t_final)
    else:
        TsTr_low = float(max(0.0, Ts__low - Tr_low))

    Gd_high = scale_plant(Gd, 1.3)
    t_high, v_high, _, _, _ = sim_closed_loop_pid_abs(
        kp, ki, kd, N, Ts,
        Gd_high,
        v0, u0,
        u_abs_min, u_abs_max,
        kb_aw,
        t_final=t_final,
        v_ref_abs=v_ref_abs,
        disturb_time=None,
        disturb_value_abs=0.0,
        meas_noise_std=MEAS_NOISE_STD
    )
    SSE_high, OS_high, Tr_high, Ts__high = step_metrics_abs(t_high, v_high, v_ref_abs, v0, tol=tol)
    if (Tr_high >= t_final - 1e-12) or (Ts__high >= t_final - 1e-12):
        TsTr_high = float(t_final)
    else:
        TsTr_high = float(max(0.0, Ts__high - Tr_high))

    dSSE = max(0.0, max(SSE_high, SSE_low) - SSE_nom)
    dOS = max(0.0, max(OS_high, OS_low) - OS_nom)
    dTsTr = max(0.0, max(TsTr_high, TsTr_low) - TsTr_nom)
    RVG = dSSE + dOS + dTsTr

    # Disturbance rejection: inject a speed offset into measured speed (ABS)
    disturb_time = t_final * 0.6
    disturb_value_abs = 0.1 * (v_ref_abs - v0)  # 10% of step size in ABS
    t_dist, v_dist, _, _, _ = sim_closed_loop_pid_abs(
        kp, ki, kd, N, Ts,
        Gd,
        v0, u0,
        u_abs_min, u_abs_max,
        kb_aw,
        t_final=t_final,
        v_ref_abs=v_ref_abs,
        disturb_time=disturb_time,
        disturb_value_abs=disturb_value_abs,
        meas_noise_std=MEAS_NOISE_STD
    )

    mask = t_dist >= disturb_time
    if np.any(mask):
        e_dist = v_ref_abs - v_dist[mask]
        ODJ = float(np.mean(e_dist ** 2))
    else:
        ODJ = 0.0

    # Effort metrics in ABS throttle
    u_rms = float(np.sqrt(np.mean(u_nom ** 2)))
    du = np.diff(u_nom)
    du_rms = float(np.sqrt(np.mean(du ** 2))) if len(du) > 0 else 0.0
    sat_ratio = float(np.mean(u_nom >= (u_abs_max - 1e-3)))

    return {
        "SSE": float(SSE_nom),
        "OS": float(OS_nom),
        "TsTr": float(TsTr_nom),
        "RVG": float(RVG),
        "ODJ": float(ODJ),
        "u_rms": float(u_rms),
        "du_rms": float(du_rms),
        "sat_ratio": float(sat_ratio),
    }


# ----------------------------
# Scales: median + IQR
# ----------------------------
def robust_scale(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 1.0
    med = np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    return float(max(med + iqr, 1e-3))


def compute_metric_scales_abs(Gd,
                              Ts,
                              bounds,
                              v0, u0,
                              v_ref_abs,
                              u_abs_min, u_abs_max,
                              T_sim,
                              num_samples=300,
                              seed=42,
                              out_path=JSON_FILE):
    rng = np.random.default_rng(seed)

    SSE_vals, OS_vals, TsTr_vals = [], [], []
    RVG_vals, ODJ_vals = [], []
    u_vals, du_vals, sat_vals = [], [], []

    print(f"Compute scaling factors with {num_samples} random samples...")

    for i in range(num_samples):
        sample = np.array([rng.uniform(*b) for b in bounds], dtype=float)

        m = pid_metrics_raw_abs(
            sample, Gd, Ts,
            v0=v0, u0=u0,
            v_ref_abs=v_ref_abs,
            u_abs_min=u_abs_min, u_abs_max=u_abs_max,
            t_final=T_sim,
            tol=0.02
        )

        SSE_vals.append(m["SSE"])
        OS_vals.append(m["OS"])
        TsTr_vals.append(m["TsTr"])
        RVG_vals.append(m["RVG"])
        ODJ_vals.append(m["ODJ"])
        u_vals.append(m["u_rms"])
        du_vals.append(m["du_rms"])
        sat_vals.append(m["sat_ratio"])

        if (i + 1) % 50 == 0:
            print(f"Scale sample {i+1}/{num_samples}")

    scales = {
        "SSE": robust_scale(np.array(SSE_vals)),
        "OS": robust_scale(np.array(OS_vals)),
        "TsTr": robust_scale(np.array(TsTr_vals)),
        "RVG": robust_scale(np.array(RVG_vals)),
        "ODJ": robust_scale(np.array(ODJ_vals)),
        "u_rms": robust_scale(np.array(u_vals)),
        "du_rms": robust_scale(np.array(du_vals)),
        "sat_ratio": robust_scale(np.array(sat_vals)),
    }

    print("Computed metric scales (median+IQR):")
    for k, v in scales.items():
        print(f"  {k}: {v:.4f}")

    with open(out_path, "w") as f:
        json.dump(scales, f, indent=2)
    print(f"Saved scales to {out_path}")

    return scales


def load_metric_scales(path=JSON_FILE):
    default = {
        "SSE": 1.0,
        "OS": 10.0,
        "TsTr": 10.0,
        "RVG": 1.0,
        "ODJ": 1.0,
        "u_rms": 1.0,
        "du_rms": 1.0,
        "sat_ratio": 1.0,
    }
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            default[k] = float(v)
        return default
    except Exception:
        return default


# ----------------------------
# Cost (normalized)
# ----------------------------
def pid_cost_normalized_abs(x,
                            Gd,
                            Ts,
                            v0, u0,
                            v_ref_abs,
                            u_abs_min, u_abs_max,
                            t_final,
                            scales,
                            weights=None):
    metrics = pid_metrics_raw_abs(
        x, Gd, Ts,
        v0=v0, u0=u0,
        v_ref_abs=v_ref_abs,
        u_abs_min=u_abs_min, u_abs_max=u_abs_max,
        t_final=t_final,
        tol=0.02
    )

    if weights is None:
        weights = {
            "SSE":      5.0,
            "OS":       3.0,
            "TsTr":     2.0,   # was 0.0: give some dynamics incentive
            "RVG":      0.5,
            "ODJ":      1.0,
            "u_rms":    1.0,
            "du_rms":   4.0,
            "sat_ratio": 1.0,  # small push away from living at saturation
        }

    J = 0.0
    for name, val in metrics.items():
        sigma = float(scales.get(name, 1.0))
        sigma = max(sigma, 1e-6)
        J += weights.get(name, 0.0) * (val / sigma)

    return float(J), metrics


def objective_pid_global_abs(x,
                             Gd,
                             Ts,
                             v0, u0,
                             v_ref_abs,
                             u_abs_min, u_abs_max,
                             T_sim,
                             scales,
                             weights=None):
    return pid_cost_normalized_abs(
        x, Gd, Ts,
        v0=v0, u0=u0,
        v_ref_abs=v_ref_abs,
        u_abs_min=u_abs_min, u_abs_max=u_abs_max,
        t_final=T_sim,
        scales=scales,
        weights=weights
    )


# ----------------------------
# Main
# ----------------------------
def main(args):
    json_file = args.o

    Ts = 0.01
    T_sim = 80.0

    # Operating point (ABS)
    v0 = 29.1063989
    u0 = 0.6933333333333334

    # ABS limits from your test
    v_abs_min = 22.681
    v_abs_max = 35.6147
    u_abs_min = 0.63
    u_abs_max = 0.75

    # Choose v_ref_abs based on mode
    if STEP_TARGET_MODE == "CENTER_OF_RANGE":
        v_ref_abs = 0.5 * (v_abs_min + v_abs_max)
    elif STEP_TARGET_MODE == "CUSTOM":
        v_ref_abs = float(CUSTOM_V_TARGET_ABS)
    else:
        # UP_FROM_V0
        v_ref_abs = min(v0 + STEP_DV_ABS, v_abs_max)

    # Plant (your high-speed TF)
    s = ctl.TransferFunction.s
    G0 = (45.58*s**2 + 23.78*s + 2.604) / (s**3 + 2.405*s**2 + 0.2322*s + 0.01443)
    Gd = ctl.c2d(G0, Ts, method='tustin')

    print("ABS tuning configuration:")
    print(f"  v0={v0:.6f}, u0={u0:.6f}")
    print(f"  v_ref_abs={v_ref_abs:.6f}  (step size Δv={v_ref_abs - v0:+.6f} m/s)")
    print(f"  throttle limits: [{u_abs_min:.6f}, {u_abs_max:.6f}]")
    print(f"  MEAS_NOISE_STD={MEAS_NOISE_STD}")

    # Bounds: [kp, ki, kd, N]
    bounds = [
        (0.0, 2.0),
        (0.0, 1.0),
        (0.0, 1.5),
        (5.0, 15.0),
    ]

    if args.compute_scales:
        compute_metric_scales_abs(
            Gd, Ts,
            bounds=bounds,
            v0=v0, u0=u0,
            v_ref_abs=v_ref_abs,
            u_abs_min=u_abs_min, u_abs_max=u_abs_max,
            T_sim=T_sim,
            num_samples=args.scale_samples,
            seed=args.seed if args.seed != 0 else 42,
            out_path=JSON_FILE
        )
        return

    scales = load_metric_scales(JSON_FILE)

    seed = args.seed
    if (seed is None) or (seed == 0):
        seed = np.random.randint(0, 10000)

    best_x, best_J, history = pso(
        objective_pid_global_abs,
        bounds,
        num_particles=50,
        max_it=100,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=seed,
        verbose=args.v,
        log_path=json_file,
        num_workers=args.workers,
        objective_args=(Gd, Ts, v0, u0, v_ref_abs, u_abs_min, u_abs_max, T_sim, scales, None),
    )

    print(f"Best X: {best_x}")
    print(f"Best J: {best_J}")
    print(f"Seed:   {seed}")

    plt.figure()
    plt.plot(history)
    plt.xlabel("iteration")
    plt.ylabel("best J")
    plt.title("Best cost history")
    plt.grid(True)

    kp, ki, kd, N = best_x
    t_best, v_best, u_best, dv_best, du_best = sim_closed_loop_pid_abs(
        kp, ki, kd, N, Ts,
        Gd,
        v0=v0, u0=u0,
        u_abs_min=u_abs_min, u_abs_max=u_abs_max,
        kb_aw=FIXED_KAW,
        t_final=T_sim,
        v_ref_abs=v_ref_abs,
        disturb_time=None,
        disturb_value_abs=0.0,
        meas_noise_std=MEAS_NOISE_STD
    )

    # Plot ABS speed
    plt.figure()
    plt.plot(t_best, v_best, label="v_abs(t)")
    plt.plot(t_best, np.full_like(t_best, v_ref_abs), "k--", label="v_ref_abs")
    plt.axhline(v0, linestyle="--", linewidth=1, label="v0")
    plt.xlabel("time [s]")
    plt.ylabel("speed [m/s]")
    plt.title("Step response (ABS) with incremental plant inside")
    plt.grid(True)
    plt.legend()

    # Plot ABS throttle
    plt.figure()
    plt.plot(t_best, u_best, label="u_abs(t)")
    plt.axhline(u_abs_min, linestyle="--", linewidth=1, label="u_abs_min")
    plt.axhline(u_abs_max, linestyle="--", linewidth=1, label="u_abs_max")
    plt.axhline(u0, linestyle="--", linewidth=1, label="u0")
    plt.xlabel("time [s]")
    plt.ylabel("throttle")
    plt.title("Control signal (ABS throttle)")
    plt.grid(True)
    plt.legend()

    # Optional: plot internal deltas
    plt.figure()
    plt.plot(t_best, dv_best, label="Δv(t)")
    plt.plot(t_best, np.full_like(t_best, v_ref_abs - v0), "k--", label="Δv_ref")
    plt.xlabel("time [s]")
    plt.ylabel("Δspeed [m/s]")
    plt.title("Internal incremental output (for debugging)")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(t_best, du_best, label="Δu(t)")
    plt.axhline(u_abs_min - u0, linestyle="--", linewidth=1, label="Δu_min")
    plt.axhline(u_abs_max - u0, linestyle="--", linewidth=1, label="Δu_max")
    plt.xlabel("time [s]")
    plt.ylabel("Δthrottle")
    plt.title("Internal incremental control (for debugging)")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSO PID Tuning (ABS I/O; incremental plant internally)")

    parser.add_argument("-v", action="store_true", help="Enable verbose output")
    parser.add_argument("-o", type=str, default="pso_log_abs.json", help="Path to log JSON file")
    parser.add_argument("--seed", type=int, default=0, help="Numpy RNG seed (0 -> random)")
    parser.add_argument("--compute-scales", action="store_true", help="Compute metric scales offline and exit")
    parser.add_argument("--scale-samples", type=int, default=300, help="Number of samples for offline scales")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes (1=no multiprocessing)")

    args = parser.parse_args()
    main(args)

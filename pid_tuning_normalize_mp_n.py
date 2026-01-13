"""
PSO-based PID tuning with offline normalization of metric scales,
tuning kp, ki, kd, N (derivative filter).

Decision variables:
    x = [kp, ki, kd, N]

Metrics:
    J1: SSE        (steady-state error, absolute in Δ-units)
    J2: OS         (overshoot, %, sign-robust via normalized response y/ref)
    J3: TsTr       (Ts - Tr, settling minus rise time)
    J4: RVG        (robustness to gain variation)
    J5: ODJ        (output disturbance rejection)
    J6: u_rms      (RMS control effort)
    J7: du_rms     (RMS control rate)
    J8: sat_ratio  (fraction of time at upper saturation)


Note:
  - Measurement noise is set to 0 by default to keep objective deterministic.
    
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import control as ctl
import multiprocessing as mp

from PID import PID  


JSON_FILE = "metric_scales_n.json"
FIXED_KAW = 1.0  # anti-windup gain is fixed, not tuned

# Determinism: keep noise off during tuning/scales unless explicitly enabled
MEAS_NOISE_STD = 0.0

# Aggressive step size in speed (m/s) relative to operating point v0.
# This will be capped to stay within v_range_max below.
STEP_DV = 3.0


# ---------------------------------------------------------------------------
# Helper for multiprocessing workers
# ---------------------------------------------------------------------------

def _pso_worker(args):
    """
    Helper for multiprocessing.Pool.map.

    args = (x, objective, objective_args)
    """
    x, objective, objective_args = args
    return objective(x, *objective_args)


# ---------------------------------------------------------------------------
# PSO implementation using scalar cost J
# ---------------------------------------------------------------------------

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
    """Standard PSO minimizing scalar J(x), with optional multiprocessing."""

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
        # initial evaluation
        results = eval_particles(particles)
        for i in range(num_particles):
            J, metrics = results[i]
            pbest_values[i] = J

            logs.append({
                "iter": 0,
                "particle": int(i),
                "J": float(J),
                "SSE": float(metrics["SSE"]),
                "OS": float(metrics["OS"]),
                "TsTr": float(metrics["TsTr"]),
                "RVG": float(metrics["RVG"]),
                "ODJ": float(metrics["ODJ"]),
                "u_rms": float(metrics["u_rms"]),
                "du_rms": float(metrics["du_rms"]),
                "sat_ratio": float(metrics["sat_ratio"]),
            })

        best_idx = int(np.argmin(pbest_values))
        gbest_position = pbest_positions[best_idx].copy()
        gbest_value = float(pbest_values[best_idx])
        history.append(gbest_value)

        if verbose:
            print(f"Iteration 0 - best cost: {gbest_value:.6f}")

        # main PSO loop
        for it in range(1, max_it + 1):
            try:
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
                        "SSE": float(metrics["SSE"]),
                        "OS": float(metrics["OS"]),
                        "TsTr": float(metrics["TsTr"]),
                        "RVG": float(metrics["RVG"]),
                        "ODJ": float(metrics["ODJ"]),
                        "u_rms": float(metrics["u_rms"]),
                        "du_rms": float(metrics["du_rms"]),
                        "sat_ratio": float(metrics["sat_ratio"]),
                        "best_x": gbest_position.tolist(),
                        "best_cost": float(gbest_value)
                    })

                history.append(gbest_value)

                if verbose:
                    print(f"Iteration {it} - best cost: {gbest_value:.6f}")
                    print(f"  best_x = {gbest_position}")
                    print(f"  mean J = {np.mean(J_values):.4f}")

            except KeyboardInterrupt:
                print(f"INTERRUPTED at iteration {it}, best cost so far: {gbest_value:.6f}")
                break

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


# ---------------------------------------------------------------------------
# Plant and simulation
# ---------------------------------------------------------------------------

def scale_plant(Gd, gain):
    """Scale the discrete plant gain."""
    num, den = ctl.tfdata(Gd)
    num = np.squeeze(num) * gain
    den = np.squeeze(den)
    return ctl.TransferFunction(num, den, Gd.dt)


def sim_closed_loop_pid(kp, ki, kd,
                        N, Ts,
                        Gd,
                        u_min, u_max, kb_aw,
                        t_final=5.0,
                        ref=1.0,
                        disturb_time=None,
                        disturb_value=0.0,
                        meas_noise_std=MEAS_NOISE_STD):
    """
    Closed-loop simulation with discrete PID and discrete plant Gd
    using state-space stepping.

    IMPORTANT:
      - This simulation is in incremental coordinates (Δy, Δu) if Gd is identified
        on mean-removed data. Then ref is Δy_ref.
    """
    pid = PID(kp, ki, kd, N, Ts, u_min, u_max, kb_aw, der_on_meas=True)

    n_steps = int(t_final / Ts)
    t = np.arange(n_steps + 1) * Ts
    r = np.full_like(t, ref, dtype=float)

    Ad, Bd, Cd, Dd = ctl.ssdata(ctl.ss(Gd))
    Ad = np.asarray(Ad, dtype=float)
    Bd = np.asarray(Bd, dtype=float).reshape(-1)
    Cd = np.asarray(Cd, dtype=float)
    Dd = np.asarray(Dd, dtype=float).reshape(-1)

    x = np.zeros(Ad.shape[0], dtype=float)
    y = np.zeros_like(t, dtype=float)
    u = np.zeros_like(t, dtype=float)

    for k in range(n_steps):
        y_meas = y[k]
        if disturb_time is not None and t[k] >= disturb_time:
            y_meas = y_meas + disturb_value

        if meas_noise_std > 0:
            y_meas += np.random.normal(0.0, meas_noise_std)

        u[k] = pid.step(r[k], y_meas)

        x = Ad @ x + Bd * u[k]
        y[k + 1] = (Cd @ x + Dd * u[k]).item()

    return t, y, u


def step_metrics(t, y, ref=1.0, tol=0.02):
    """
    Robust metrics for positive/negative steps, without constant OS injection.

    Returns: SSE, OS(%), Tr, Ts_
      - SSE = |ref - y_final|
      - OS computed in normalized space s=y/ref (sign-robust). Always computed.
      - Tr: time to go from 10% to 90% of target in normalized space.
            If never reaches 90% -> Tr = t_final (flag of failure).
      - Ts_: settling time into |y-ref| <= tol*|ref| band.
            If never settles -> Ts_ = t_final (flag of failure).
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    t_final = float(t[-1])

    # Regulation case (ref ~ 0): treat separately
    if abs(ref) < 1e-9:
        SSE = abs(y[-1])
        band = tol  # absolute band around 0
        outside = np.abs(y) > band
        if np.any(outside):
            last_out = int(np.where(outside)[0][-1])
            Ts_ = float(t[last_out + 1]) if (last_out + 1) < len(t) else t_final
        else:
            Ts_ = 0.0
        Tr = t_final  # undefined
        # OS% not meaningful; keep a bounded indicator
        OS = 0.0
        return float(SSE), float(OS), float(Tr), float(Ts_)

    # Normalize by ref to make sign-robust
    s = y / ref

    SSE = abs(ref - y[-1])

    # Overshoot always defined from peak of normalized response
    OS = max(0.0, (np.max(s) - 1.0) * 100.0)

    # Rise time: need reach 10% AND 90% in normalized space
    reached_10 = np.where(s >= 0.1)[0]
    reached_90 = np.where(s >= 0.9)[0]

    if reached_10.size > 0 and reached_90.size > 0:
        idx_10 = int(reached_10[0])
        idx_90 = int(reached_90[0])
        Tr = float(t[idx_90] - t[idx_10])
    else:
        # Did not reach 90% (or even 10%): mark as failure via Tr=t_final
        Tr = t_final

    # Settling time in absolute band around ref
    band = tol * abs(ref)
    outside = np.abs(y - ref) > band
    if np.any(outside):
        last_out = int(np.where(outside)[0][-1])
        Ts_ = float(t[last_out + 1]) if (last_out + 1) < len(t) else t_final
    else:
        Ts_ = 0.0

    return float(SSE), float(OS), float(Tr), float(Ts_)


# ---------------------------------------------------------------------------
# Raw metrics (no normalization)
# ---------------------------------------------------------------------------

def pid_metrics_raw(x,
                    Gd,
                    Ts,
                    u_min, u_max,
                    t_final=5.0,
                    ref=1.0):
    """
    x = [kp, ki, kd, N]
    Compute raw metrics for cost:
      SSE_nom, OS_nom, TsTr_nom, RVG, ODJ, u_rms, du_rms, sat_ratio
    """
    kp, ki, kd, N = x
    kb_aw = FIXED_KAW

    # Nominal response
    t_nom, y_nom, u_nom = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd,
        u_min, u_max, kb_aw,
        t_final, ref,
        disturb_time=None,
        disturb_value=0.0,
        meas_noise_std=MEAS_NOISE_STD
    )

    SSE_nom, OS_nom, Tr_nom, Ts__nom = step_metrics(t_nom, y_nom, ref)
    TsTr_nom = max(0.0, Ts__nom - Tr_nom)

    # Robustness via gain scaling
    Gd_low = scale_plant(Gd, 0.7)
    t_low, y_low, _ = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd_low,
        u_min, u_max, kb_aw,
        t_final, ref,
        disturb_time=None,
        disturb_value=0.0,
        meas_noise_std=MEAS_NOISE_STD
    )
    SSE_low, OS_low, Tr_low, Ts__low = step_metrics(t_low, y_low, ref)

    Gd_high = scale_plant(Gd, 1.3)
    t_high, y_high, _ = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd_high,
        u_min, u_max, kb_aw,
        t_final, ref,
        disturb_time=None,
        disturb_value=0.0,
        meas_noise_std=MEAS_NOISE_STD
    )
    SSE_high, OS_high, Tr_high, Ts__high = step_metrics(t_high, y_high, ref)

    TsTr_low = max(0.0, Ts__low - Tr_low)
    TsTr_high = max(0.0, Ts__high - Tr_high)

    dSSE = max(0.0, max(SSE_high, SSE_low) - SSE_nom)
    dOS = max(0.0, max(OS_high, OS_low) - OS_nom)
    dTsTr = max(0.0, max(TsTr_high, TsTr_low) - TsTr_nom)

    RVG = dSSE + dOS + dTsTr

    # disturbance rejection
    disturb_time = t_final * 0.6
    disturb_value = ref * 0.1

    t_dist, y_dist, _ = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd,
        u_min, u_max, kb_aw,
        t_final, ref,
        disturb_time=disturb_time,
        disturb_value=disturb_value,
        meas_noise_std=MEAS_NOISE_STD
    )

    mask = t_dist >= disturb_time
    if np.any(mask):
        e_dist = ref - y_dist[mask]
        ODJ = float(np.mean(e_dist ** 2))
    else:
        ODJ = 0.0

    # effort metrics
    u_rms = float(np.sqrt(np.mean(u_nom ** 2)))
    du = np.diff(u_nom)
    du_rms = float(np.sqrt(np.mean(du ** 2))) if len(du) > 0 else 0.0
    sat_ratio = float(np.mean(u_nom >= (u_max - 1e-3)))

    metrics = {
        "SSE": float(SSE_nom),
        "OS": float(OS_nom),
        "TsTr": float(TsTr_nom),
        "RVG": float(RVG),
        "ODJ": float(ODJ),
        "u_rms": float(u_rms),
        "du_rms": float(du_rms),
        "sat_ratio": float(sat_ratio),
    }
    return metrics


# ---------------------------------------------------------------------------
# Offline scale computation (median + IQR)
# ---------------------------------------------------------------------------

def robust_scale(arr: np.ndarray) -> float:
    """Return median+IQR as robust scale for array."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 1.0
    med = np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    return float(max(med + iqr, 1e-3))


def compute_metric_scales(Gd,
                          Ts,
                          bounds,
                          u_min,
                          u_max,
                          T_sim,
                          setpoint,
                          num_samples=300,
                          seed=42,
                          out_path=JSON_FILE):
    """Randomly sample PID params in bounds, compute metrics, derive median+IQR scales."""
    rng = np.random.default_rng(seed)

    SSE_vals, OS_vals, TsTr_vals = [], [], []
    RVG_vals, ODJ_vals = [], []
    u_vals, du_vals, sat_vals = [], [], []

    print(f"Compute scaling factors with {num_samples} random samples...")

    for i in range(num_samples):
        sample = np.array([rng.uniform(*b) for b in bounds], dtype=float)

        metrics = pid_metrics_raw(
            sample, Gd,
            Ts=Ts,
            u_min=u_min, u_max=u_max,
            t_final=T_sim, ref=setpoint
        )

        SSE_vals.append(metrics["SSE"])
        OS_vals.append(metrics["OS"])
        TsTr_vals.append(metrics["TsTr"])
        RVG_vals.append(metrics["RVG"])
        ODJ_vals.append(metrics["ODJ"])
        u_vals.append(metrics["u_rms"])
        du_vals.append(metrics["du_rms"])
        sat_vals.append(metrics["sat_ratio"])

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
    """Load metric scales from JSON, with safe defaults if missing."""
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


# ---------------------------------------------------------------------------
# Normalized numeric cost
# ---------------------------------------------------------------------------

def pid_cost_normalized(x,
                        Gd,
                        Ts,
                        u_min, u_max,
                        t_final, ref,
                        scales,
                        weights=None):
    """
    x = [kp, ki, kd, N]
    Compute scalar cost J and metrics dict, using fixed scales:
        g_j = metric_j / sigma_j
        J = sum_j w_j * g_j
    """
    metrics = pid_metrics_raw(
        x, Gd,
        Ts=Ts,
        u_min=u_min, u_max=u_max,
        t_final=t_final, ref=ref
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
        g = val / sigma
        J += weights.get(name, 0.0) * g

    return float(J), metrics


# ---------------------------------------------------------------------------
# Top-level objective for multiprocessing
# ---------------------------------------------------------------------------

def objective_pid_global(x,
                         Gd,
                         Ts,
                         u_min,
                         u_max,
                         T_sim,
                         setpoint,
                         scales,
                         weights=None):
    """Top-level objective for multiprocessing."""
    return pid_cost_normalized(
        x, Gd,
        Ts=Ts,
        u_min=u_min, u_max=u_max,
        t_final=T_sim, ref=setpoint,
        scales=scales,
        weights=weights,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    json_file = args.o

    # General settings
    Ts = 0.01
    T_sim = 80.0

    # ---- Operating point from dataset (mean values) ----
    v0 = 29.1063989
    u0 = 0.6933333333333334

    # ---- Observed test extremes ----
    v_test_min = 22.681
    v_test_max = 35.6147
    u_test_min = 0.63
    u_test_max = 0.75

    # ---- Define setpoint as Δv_ref (aggressive step, capped within range) ----
    
    v_target = min(v0 + STEP_DV, v_test_max)
    setpoint = v_target - v0  # Δv_ref in m/s

    # ---- Actuator saturation in Δu around u0 ----
    u_min = u_test_min - u0
    u_max = u_test_max - u0

    print("Operating point / limits used:")
    print(f"  v0 = {v0:.6f} m/s, u0 = {u0:.6f}")
    print(f"  v_target = {v_target:.6f} -> Δv_ref(setpoint) = {setpoint:.6f} m/s")
    print(f"  Δu limits: u_min = {u_min:.6f}, u_max = {u_max:.6f}")

    # Plant
    s = ctl.TransferFunction.s
    G0 = (45.58*s**2 + 23.78*s + 2.604) / (s**3 + 2.405*s**2 + 0.2322*s + 0.01443)  # High speed range
    Gd = ctl.c2d(G0, Ts, method='tustin')

    # PID search bounds: [kp, ki, kd, N]
    bounds = [
        (0.0, 2.0),    # Kp
        (0.0, 1.0),    # Ki
        (0.0, 1.5),    # Kd
        (5.0, 15.0),   # N
    ]

    # Optional: offline scales computation mode
    if args.compute_scales:
        compute_metric_scales(
            Gd, Ts,
            bounds=bounds,
            u_min=u_min, u_max=u_max,
            T_sim=T_sim, setpoint=setpoint,
            num_samples=args.scale_samples,
            seed=args.seed if args.seed != 0 else 42,
            out_path=JSON_FILE
        )
        return

    # Normal PSO run: load scales
    scales = load_metric_scales(JSON_FILE)

    seed = args.seed
    if (seed is None) or (seed == 0):
        seed = np.random.randint(0, 10000)

    best_x, best_J, history = pso(
        objective_pid_global,
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
        objective_args=(Gd, Ts, u_min, u_max, T_sim, setpoint, scales, None),
    )

    print(f"Best X: {best_x}")
    print(f"Best J: {best_J}")
    print(f"Seed:   {seed}")

    # Plot best cost history
    plt.figure()
    plt.plot(history)
    plt.xlabel("iteration")
    plt.ylabel("best J")
    plt.title("Best cost history")
    plt.grid(True)

    # simulate and plot step response for best controller (Δ-coordinates)
    kp, ki, kd, N = best_x
    kb_aw = FIXED_KAW

    t_best, y_best, u_best = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd,
        u_min, u_max, kb_aw,
        t_final=T_sim, ref=setpoint,
        disturb_time=None,
        disturb_value=0.0,
        meas_noise_std=MEAS_NOISE_STD
    )

    # Output vs reference (Δy)
    plt.figure()
    plt.plot(t_best, y_best, label="Output Δv(t)")
    plt.plot(t_best, np.full_like(t_best, setpoint), "k--", label="Reference Δv_ref")
    plt.xlabel("time [s]")
    plt.ylabel("Δspeed [m/s]")
    plt.title("Step response of best PID (incremental model)")
    plt.grid(True)
    plt.legend()

    # Control signal (Δu)
    plt.figure()
    plt.plot(t_best, u_best, label="Δu(t)")
    plt.axhline(u_min, linestyle="--", linewidth=1, label="Δu min")
    plt.axhline(u_max, linestyle="--", linewidth=1, label="Δu max")
    plt.xlabel("time [s]")
    plt.ylabel("Δthrottle")
    plt.title("Control signal of best PID (Δu)")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSO PID Tuning (kp,ki,kd,N; offline normalized)")

    parser.add_argument(
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-o",
        type=str,
        default="pso_log_n.json",
        help="Path to log JSON file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Numpy random number generator seed (0 -> random)"
    )
    parser.add_argument(
        "--compute-scales",
        action="store_true",
        help="Run offline random scan to compute metric scales and exit"
    )
    parser.add_argument(
        "--scale-samples",
        type=int,
        default=300,
        help="Number of random PIDs to sample when computing scales"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel evaluation (1 = no multiprocessing)"
    )

    args = parser.parse_args()
    main(args)

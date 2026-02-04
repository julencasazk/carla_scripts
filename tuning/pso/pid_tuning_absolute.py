"""
CURRENT TUNING SCRIPT (use this one)

PSO-based PID tuning for an incremental (mean-removed) plant:
    Deltav = G(z) * Deltau

How it evaluates the model
  - Simulates the controller in ABSOLUTE variables and applies Deltau to the plant.

Decision variables
  - x = [kp, ki, kd, N]

Cost metrics (same format as the original PSO script)
  J1 = |SSE|   # steady-state error
  J2 = OS      # overshoot
  J3 = Ts-Tr   # settling time - rise time
  J4 = RVG     # robustness to gain variation
  J5 = ODJ     # output disturbance rejection
  J6 = u_rms   # control effort
  J7 = du_rms  # control rate
  J8 = sat_ratio # fraction of time at saturation

Important setup
  - Provide fitted incremental plant Gd (Deltau -> Deltav) at Ts.
  - Provide operating point (u0, v0) and actuator bounds.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import control as ctl
import multiprocessing as mp

from lib.PID import PID  # your existing PID implementation

JSON_FILE = "metric_scales_n.json"
FIXED_KAW = 1.0  # anti-windup gain fixed (not tuned)

# ---------------------------------------------------------------------------
# Helper for multiprocessing workers
# ---------------------------------------------------------------------------

def _pso_worker(args):
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
                **{k: float(v) for k, v in metrics.items()}
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

                # update velocities and positions
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

                # evaluate all particles
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
    """Scale the discrete incremental plant gain (Deltav/Deltau)."""
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
                        base_ref=1.0,
                        u0=0.0,
                        disturb_time=None,
                        disturb_value=0.0):
    """
    Closed-loop simulation with discrete PID and INCREMENTAL discrete plant Gd (Deltau -> Deltav).

    Absolute signals:
        y_abs = base_ref + dv
        u_abs = clip( PID(r_abs, y_abs_meas), [u_min, u_max] )

    Plant update uses du = u_abs - u0.
    """
    pid = PID(kp, ki, kd, N, Ts, u_min, u_max, kb_aw, der_on_meas=True, integral_disc_method="backeuler", derivative_disc_method="tustin")

    n_steps = int(t_final / Ts)
    t = np.arange(n_steps + 1) * Ts
    r = np.full_like(t, ref, dtype=float)  # absolute reference

    Ad, Bd, Cd, Dd = ctl.ssdata(ctl.ss(Gd))
    Ad = np.asarray(Ad, dtype=float)
    Bd = np.asarray(Bd, dtype=float).reshape(-1)
    Cd = np.asarray(Cd, dtype=float)
    Dd = np.asarray(Dd, dtype=float).reshape(-1)

    x = np.zeros(Ad.shape[0], dtype=float)

    dv = np.zeros_like(t, dtype=float)                 # incremental output
    y = np.full_like(t, base_ref, dtype=float)         # absolute output
    u = np.full_like(t, u0, dtype=float)               # absolute throttle

    meas_noise_std = 0.05

    for k in range(n_steps):
        # reconstruct absolute output
        y[k] = base_ref + dv[k]

        # absolute measurement + disturbance + noise
        y_meas = y[k]
        if disturb_time is not None and t[k] >= disturb_time:
            y_meas = y_meas + disturb_value
        if meas_noise_std > 0:
            y_meas += np.random.normal(0.0, meas_noise_std)

        # PID output interpreted as ABSOLUTE throttle
        u_abs = float(pid.step(r[k], y_meas))
        u_abs = float(np.clip(u_abs, u_min, u_max))

        # incremental plant input
        du = u_abs - u0

        # update incremental plant
        x = Ad @ x + Bd * du
        dv[k + 1] = (Cd @ x + Dd * du).item()

        # log
        u[k] = u_abs

    y[-1] = base_ref + dv[-1]
    u[-1] = u[-2]
    return t, y, u

def step_metrics(t, y, ref=1.0, tol=0.02, base_ref=1.0, tol_abs_min=0.05):
    """
    Compute:
      SSE (abs)
      OS (% of step size)
      Tr (10-90% of step size)
      Ts_ (settling time within band around ref, band based on step size)

    tol: relative band w.r.t. step magnitude
    tol_abs_min: minimum absolute settling band
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    step = ref - base_ref
    step_mag = max(abs(step), 1e-6)

    SSE = abs(ref - y[-1])

    # overshoot relative to step magnitude
    OS = max(0.0, (np.max(y) - ref) / step_mag * 100.0)

    # rise time (10-90%) relative to step
    y10 = base_ref + 0.1 * step
    y90 = base_ref + 0.9 * step
    try:
        if step >= 0:
            idx_10 = np.where(y >= y10)[0][0]
            idx_90 = np.where(y >= y90)[0][0]
        else:
            idx_10 = np.where(y <= y10)[0][0]
            idx_90 = np.where(y <= y90)[0][0]
        Tr = t[idx_90] - t[idx_10]
    except IndexError:
        Tr = t[-1]
        OS += 100.0

    # settling band around ref based on step magnitude (with absolute floor)
    band = max(tol * step_mag, tol_abs_min)
    lower = ref - band
    upper = ref + band

    Ts_ = t[-1]
    for i in range(len(t) - 1, -1, -1):
        if y[i] < lower or y[i] > upper:
            Ts_ = t[i + 1] if (i + 1) < len(t) else t[-1]
            break

    return SSE, OS, Tr, Ts_

# ---------------------------------------------------------------------------
# Raw metrics (no normalization)
# ---------------------------------------------------------------------------

def pid_metrics_raw(x,
                    Gd,
                    Ts,
                    u_min, u_max,
                    t_final=5.0,
                    ref=1.0,
                    base_ref=1.0,
                    u0=0.0):
    """
    x = [kp, ki, kd, N]
    Compute raw metrics for cost:
      SSE, OS, TsTr, RVG, ODJ, u_rms, du_rms, sat_ratio
    """
    kp, ki, kd, N = x
    kb_aw = FIXED_KAW

    # Nominal response (ABS evaluation, incremental plant)
    t_nom, y_nom, u_nom = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd,
        u_min, u_max, kb_aw,
        t_final, ref,
        base_ref,
        u0=u0,
        disturb_time=None,
        disturb_value=0.0
    )

    SSE_nom, OS_nom, Tr_nom, Ts__nom = step_metrics(t_nom, y_nom, ref, base_ref=base_ref)
    TsTr_nom = max(0.0, Ts__nom - Tr_nom)

    # Robustness via gain scaling
    Gd_low = scale_plant(Gd, 0.7)
    t_low, y_low, _ = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd_low,
        u_min, u_max, kb_aw,
        t_final, ref, base_ref,
        u0=u0,
        disturb_time=None,
        disturb_value=0.0
    )
    SSE_low, OS_low, Tr_low, Ts__low = step_metrics(t_low, y_low, ref, base_ref=base_ref)

    Gd_high = scale_plant(Gd, 1.3)
    t_high, y_high, _ = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd_high,
        u_min, u_max, kb_aw,
        t_final, ref, base_ref,
        u0=u0,
        disturb_time=None,
        disturb_value=0.0
    )
    SSE_high, OS_high, Tr_high, Ts__high = step_metrics(t_high, y_high, ref, base_ref=base_ref)

    TsTr_low = max(0.0, Ts__low - Tr_low)
    TsTr_high = max(0.0, Ts__high - Tr_high)

    dSSE = max(0.0, max(SSE_high, SSE_low) - SSE_nom)
    dOS = max(0.0, max(OS_high, OS_low) - OS_nom)
    dTsTr = max(0.0, max(TsTr_high, TsTr_low) - TsTr_nom)

    RVG = dSSE + dOS + dTsTr

    # disturbance rejection (output disturbance on measurement)
    disturb_time = t_final * 0.6
    disturb_value = (ref - base_ref) * 0.1  # 10% of step magnitude

    t_dist, y_dist, _ = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd,
        u_min, u_max, kb_aw,
        t_final, ref, base_ref,
        u0=u0,
        disturb_time=disturb_time,
        disturb_value=disturb_value
    )

    mask = t_dist >= disturb_time
    if np.any(mask):
        e_dist = ref - y_dist[mask]
        ODJ = float(np.mean(e_dist**2))
    else:
        ODJ = 0.0

    # effort metrics
    u_rms = float(np.sqrt(np.mean(u_nom**2)))
    du = np.diff(u_nom)
    du_rms = float(np.sqrt(np.mean(du**2))) if len(du) > 0 else 0.0
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
                          ref_pv,
                          u0,
                          num_samples=300,
                          seed=42,
                          out_path=JSON_FILE):
    """Randomly sample PID params in bounds, compute metrics, derive median+IQR scales."""
    rng = np.random.default_rng(seed)

    buckets = {k: [] for k in ["SSE","OS","TsTr","RVG","ODJ","u_rms","du_rms","sat_ratio"]}

    for i in range(num_samples):
        sample = np.array([rng.uniform(*b) for b in bounds], dtype=float)

        metrics = pid_metrics_raw(
            sample, Gd,
            Ts=Ts,
            u_min=u_min, u_max=u_max,
            t_final=T_sim, ref=setpoint,
            base_ref=ref_pv,
            u0=u0
        )

        for k in buckets:
            buckets[k].append(metrics[k])

        if (i + 1) % 50 == 0:
            print(f"Scale sample {i+1}/{num_samples}")

    scales = {k: robust_scale(np.array(v)) for k, v in buckets.items()}

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

# ---------------------------------------------------------------------------
# Normalized numeric cost
# ---------------------------------------------------------------------------

def pid_cost_normalized(x,
                        Gd,
                        Ts,
                        u_min, u_max,
                        t_final, ref,
                        base_ref,
                        u0,
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
        t_final=t_final, ref=ref,
        base_ref=base_ref,
        u0=u0
    )

    if weights is None:
        weights = {
            "SSE":      3.0,
            "OS":       3.0,
            "TsTr":     1.0,
            "RVG":      2.0,
            "ODJ":      2.0,
            "u_rms":    0.5,
            "du_rms":   2.0,
            "sat_ratio":0.5,
        }

    J = 0.0
    for name, val in metrics.items():
        sigma = float(scales.get(name, 1.0))
        sigma = max(sigma, 1e-6)
        J += weights.get(name, 0.0) * (val / sigma)

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
                         ref_pv,
                         u0,
                         scales,
                         weights=None):
    return pid_cost_normalized(
        x, Gd,
        Ts=Ts,
        u_min=u_min, u_max=u_max,
        t_final=T_sim, ref=setpoint,
        base_ref=ref_pv,
        u0=u0,
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
    u_min = 0.0
    u_max = 1.0
    T_sim = 80.0

    # ------------------------------------------------------------------
    # Operating point for the SPEED RANGE YOU ARE TUNING
    # v0 = base_ref (speed operating point)
    # u0 = throttle operating point
    # ------------------------------------------------------------------

    ref_pv = 6.1416430166666665
    setpoint = 10.0 
    u0 = 0.266666666666

    # ------------------------------------------------------------------
    # IMPORTANT: Gd must represent the INCREMENTAL plant Deltav/Deltau (Remove Means)
    # and it must be discrete with dt=Ts.
    # ------------------------------------------------------------------

    # Incremental Plant
    #G0 = ctl.TransferFunction([0,   16.0307,  241.7231], # High speeds
    #                          [1.0000, 13.8711, 2.1097])

    #G0 = ctl.TransferFunction([0,45.5849,23.7842,0.5994],      # New values, 3p2z continuous
    #                          [1.0000,2.4052,0.2322,0.0144])

    #G0 = ctl.TransferFunction([0, 12.3082, 67.2242], # Mid Speeds
    #                         [1.0000, 5.2957, 1.0404])

    #G0 = ctl.TransferFunction([0,56.3703, 155.5811,2.6035], # New values, 3p2z continuous
    #                         [1.0000,    15.5755,    2.6333,    0.0852])

    #G0 = ctl.TransferFunction([0,    8.5777,   54.0770], # Low speeds
    #                          [1.0000,    7.0533,    2.1718])

    G0 = ctl.TransferFunction([0, 7.8409], # Low speeds # New values 1p continuous
                              [ 1.0000, 0.3321])

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
            T_sim=T_sim, setpoint=setpoint, ref_pv=ref_pv,
            u0=u0,
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
        objective_args=(Gd, Ts, u_min, u_max, T_sim, setpoint, ref_pv, u0, scales, None),
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

    # Simulate and plot best controller
    kp, ki, kd, N = best_x
    kb_aw = FIXED_KAW

    t_best, y_best, u_best = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd,
        u_min, u_max, kb_aw,
        t_final=T_sim, ref=setpoint,
        base_ref=ref_pv,
        u0=u0,
        disturb_time=None,
        disturb_value=0.0
    )

    # Output vs reference
    plt.figure()
    plt.plot(t_best, y_best, label="Speed y(t) [abs]")
    plt.plot(t_best, np.full_like(t_best, setpoint), "k--", label="Reference")
    plt.xlabel("time [s]")
    plt.ylabel("speed [m/s]")
    plt.title("Step response of best PID (abs, incremental plant underneath)")
    plt.grid(True)
    plt.legend()

    # Control signal
    plt.figure()
    plt.plot(t_best, u_best, label="Throttle u(t) [abs]")
    plt.xlabel("time [s]")
    plt.ylabel("throttle")
    plt.title("Control signal of best PID (absolute)")
    plt.grid(True)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSO PID Tuning for incremental plants (kp,ki,kd,N)")

    parser.add_argument("-v", action="store_true", help="Enable verbose output")
    parser.add_argument("-o", type=str, default="pso_log_n.json", help="Path to log JSON file")
    parser.add_argument("--seed", type=int, default=0, help="Numpy seed (0 -> random)")

    parser.add_argument("--compute-scales", action="store_true",
                        help="Run offline random scan to compute metric scales and exit")
    parser.add_argument("--scale-samples", type=int, default=300,
                        help="Number of random PIDs to sample when computing scales")

    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes for parallel evaluation (1 = no multiprocessing)")

    args = parser.parse_args()
    main(args)

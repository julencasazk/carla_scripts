"""
PSO-based PID tuning with offline normalization of metric scales.

What it does
  - Computes metric scales from random PID samples (offline scan).
  - Uses normalized metrics to form a scalar cost and runs PSO.

Outputs
  - JSON scale file and optional PSO logs.

Run (example)
  python3 lib/pid_tuning_offline_norm_mp.py --help
"""

import argparse
import json
import os
import numpy as np
import control as ctl
import multiprocessing as mp

from lib.PID import PID  # your existing PID implementation

JSON_FILE = "metric_scales_kpkikd.json"
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
        num_particles=50,
        max_it=100,
        w=1.3,
        c1=2.1,
        c2=2.1,
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
            # single-process
            return [objective(x, *objective_args) for x in particles_batch]
        # multi-process: pack each particle with objective + args
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

                # evaluate all particles (possibly in parallel)
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
                        disturb_value=0.0):
    """
    Closed-loop simulation with discrete PID and discrete plant Gd
    using state-space stepping.
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

        u[k] = pid.step(r[k], y_meas)

        x = Ad @ x + Bd * u[k]
        y[k+1] = (Cd @ x + Dd * u[k]).item()

    return t, y, u


def step_metrics(t, y, ref=1.0, tol=0.02):
    """Compute SSE, overshoot (OS%), rise time Tr, settling time Ts_."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    SSE = abs(ref - y[-1])
    OS = max(0.0, (np.max(y) - ref) / max(ref, 1e-6) * 100.0)

    y10 = 0.1 * ref
    y90 = 0.9 * ref
    try:
        idx_10 = np.where(y > y10)[0][0]
        idx_90 = np.where(y > y90)[0][0]
        Tr = t[idx_90] - t[idx_10]
    except IndexError:
        Tr = t[-1]
        OS += 100.0

    lower = ref * (1 - tol)
    upper = ref * (1 + tol)
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
                    N, Ts,
                    u_min, u_max, kb_aw,
                    t_final=5.0,
                    ref=1.0):
    """
    Compute raw metrics for cost:
      SSE_nom, OS_nom, TsTr_nom, RVG, ODJ, u_rms, du_rms, sat_ratio
    """
    kp, ki, kd = x

    # Nominal response
    t_nom, y_nom, u_nom = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd,
        u_min, u_max, kb_aw,
        t_final, ref,
        disturb_time=None,
        disturb_value=0.0
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
        disturb_value=0.0
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
        disturb_value=0.0
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
        disturb_value=disturb_value
    )

    mask = t_dist >= disturb_time
    if np.any(mask):
        e_dist = ref - y_dist[mask]
        ODJ = np.trapezoid(e_dist**2, t_dist[mask])
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
                          N_filt,
                          u_min,
                          u_max,
                          kb_aw,
                          T_sim,
                          setpoint,
                          bounds,
                          num_samples=300,
                          seed=42,
                          out_path=JSON_FILE):
    """Randomly sample PIDs in bounds, compute metrics, derive median+IQR scales."""
    rng = np.random.default_rng(seed)

    SSE_vals = []
    OS_vals = []
    TsTr_vals = []
    RVG_vals = []
    ODJ_vals = []
    u_vals = []
    du_vals = []
    sat_vals = []

    for i in range(num_samples):
        kp = rng.uniform(*bounds[0])
        ki = rng.uniform(*bounds[1])
        kd = rng.uniform(*bounds[2])
        x = np.array([kp, ki, kd], dtype=float)

        metrics = pid_metrics_raw(
            x, Gd,
            N=N_filt, Ts=Ts,
            u_min=u_min, u_max=u_max, kb_aw=kb_aw,
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

    SSE_vals = np.array(SSE_vals)
    OS_vals = np.array(OS_vals)
    TsTr_vals = np.array(TsTr_vals)
    RVG_vals = np.array(RVG_vals)
    ODJ_vals = np.array(ODJ_vals)
    u_vals = np.array(u_vals)
    du_vals = np.array(du_vals)
    sat_vals = np.array(sat_vals)

    scales = {
        "SSE": robust_scale(SSE_vals),
        "OS": robust_scale(OS_vals),
        "TsTr": robust_scale(TsTr_vals),
        "RVG": robust_scale(RVG_vals),
        "ODJ": robust_scale(ODJ_vals),
        "u_rms": robust_scale(u_vals),
        "du_rms": robust_scale(du_vals),
        "sat_ratio": robust_scale(sat_vals),
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
                        N, Ts,
                        u_min, u_max, kb_aw,
                        t_final, ref,
                        scales,
                        weights=None):
    """
    Compute scalar cost J and metrics dict, using fixed scales:
        g_j = metric_j / sigma_j
        J = sum_j w_j * g_j
    """
    metrics = pid_metrics_raw(
        x, Gd,
        N=N, Ts=Ts,
        u_min=u_min, u_max=u_max, kb_aw=kb_aw,
        t_final=t_final, ref=ref
    )

    if weights is None:
        weights = {
            "SSE": 1.0,
            "OS": 1.0,
            "TsTr": 0.8,
            "RVG": 0.2,
            "ODJ": 0.2,
            "u_rms": 0.1,
            "du_rms": 0.8,
            "sat_ratio": 0.3,
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
                         N_filt,
                         Ts,
                         u_min,
                         u_max,
                         kb_aw,
                         T_sim,
                         setpoint,
                         scales,
                         weights=None):
    """
    Top-level objective for multiprocessing.
    x = [kp, ki, kd]
    """
    return pid_cost_normalized(
        x, Gd,
        N=N_filt, Ts=Ts,
        u_min=u_min, u_max=u_max, kb_aw=kb_aw,
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
    N_filt = 20.0
    Ts = 0.01
    u_min = 0.0
    u_max = 1.0
    kb_aw = 1.0
    T_sim = 80.0

    # Plant with delay (same structure as before)
    Td = 0.15
    s = ctl.TransferFunction.s
    G0 = 12.68 / (s**2 + 1.076*s + 0.2744)
    num_delay, den_delay = ctl.pade(Td, 1)
    H_delay = ctl.tf(num_delay, den_delay)
    Gc = G0 * H_delay
    Gd = ctl.c2d(Gc, Ts, method="tustin")

    setpoint = 33.33

    # PID search bounds (tune as desired)
    bounds = [
        (0.0, 2.0),   # Kp
        (0.0, 1.0),   # Ki
        (0.0, 10.0),  # Kd
    ]

    # Optional: offline scales computation mode
    if args.compute_scales:
        compute_metric_scales(
            Gd, Ts, N_filt,
            u_min, u_max, kb_aw,
            T_sim, setpoint,
            bounds=bounds,
            num_samples=args.scale_samples,
            seed=args.seed if args.seed != 0 else 42,
            out_path=JSON_FILE,
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
        max_it=300,
        w=1.3,
        c1=2.1,
        c2=2.1,
        seed=seed,
        verbose=args.v,
        log_path=json_file,
        num_workers=args.workers,
        objective_args=(Gd, N_filt, Ts, u_min, u_max, kb_aw, T_sim, setpoint, scales, None),
    )

    print(f"Best X: {best_x}")
    print(f"Best J: {best_J}")
    print(f"Seed:   {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSO PID Tuning (offline normalized)")

    parser.add_argument(
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-o",
        type=str,
        default="pso_log.json",
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

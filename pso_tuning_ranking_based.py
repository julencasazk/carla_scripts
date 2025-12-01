"""
This script is used to calculate Kp Ki and Kd parameters for a PID
controller, given a plant to control in closed-loop negative feedback
using Particle Swarm Optimization.

The cost function is calculated following the implmementation at:

        https://ieeexplore.ieee.org/document/8440927

Where:

    J1(X) = |SSE|   # Steady-State Error
    J2(X) = OS      # Overshoot
    J3(X) = Ts-Tr   # Settling time - Rise time  
    J4(X) = RVG     # Robustness for gain change
    J5(X) = ODJ     # Output disturbance rejection
    J6(X) = CEC     # Control Effort Limit
"""
import control as ctl
from PID import PID
import numpy as np
import json

import argparse


JSON_FILE = "new_targets.json"


def pso(objetive, bounds, num_particles=30, max_it=50, w=0.7, c1=1.5, c2=1.5,
        seed=None, verbose=True, log_path=None):
    if seed is not None:
        np.random.seed(seed)

    dim = len(bounds)
    bounds_min = np.array([b[0] for b in bounds])
    bounds_max = np.array([b[1] for b in bounds])

    particles = np.random.uniform(bounds_min, bounds_max, size=(num_particles, dim))
    velocities = np.random.uniform(-1, 1, size=(num_particles, dim))

    pbest_positions = particles.copy()
    pbest_values = np.full(num_particles, np.inf, dtype=float)

    gbest_position = None
    gbest_value = np.inf

    logs = []
    history = []

    w_sse = 1.0
    w_os = 1.0
    w_ts_tr = 0.45
    w_rvg = 0.8
    w_odj = 0.8
    w_u = 0.0
    w_du = 0.0
    w_sat = 0.3

    weights = np.array([
        w_sse,
        w_os,
        w_ts_tr,
        w_rvg,
        w_odj,
        w_u,
        w_du,
        w_sat
    ])

    metrics_list = []

    for i in range(num_particles):
        SSE_nom, OS_nom, Ts__nom, Tr_nom, RVG, ODJ, u_rms, du_rms, sat_ratio = objetive(particles[i])
        TsTr_nom = max(0.0, Ts__nom - Tr_nom)
        metrics_list.append([
            SSE_nom,
            OS_nom,
            TsTr_nom,
            RVG,
            ODJ,
            u_rms,
            du_rms,
            sat_ratio
        ])

    metrics_arr = np.array(metrics_list)
    num_metrics = metrics_arr.shape[1]
    J_values = np.zeros(num_particles)

    for j in range(num_metrics):
        col = metrics_arr[:, j]
        order = np.argsort(col)
        ranks = np.empty(num_particles, dtype=int)
        ranks[order] = np.arange(num_particles)
        if num_particles > 1:
            g = ranks / (num_particles - 1)
        else:
            g = np.zeros_like(ranks, dtype=float)
        J_values += weights[j] * g

    for i in range(num_particles):
        J = J_values[i]
        pbest_values[i] = J
        logs.append({
            "iter": 0,
            "particle": int(i),
            "J": float(J),
            "SSE": float(metrics_arr[i, 0]),
            "OS": float(metrics_arr[i, 1]),
            "TsTr": float(metrics_arr[i, 2]),
            "RVG": float(metrics_arr[i, 3]),
            "ODJ": float(metrics_arr[i, 4]),
            "u_rms": float(metrics_arr[i, 5]),
            "du_rms": float(metrics_arr[i, 6]),
            "sat_ratio": float(metrics_arr[i, 7])
        })

    best_idx = np.argmin(pbest_values)
    gbest_position = pbest_positions[best_idx].copy()
    gbest_value = float(pbest_values[best_idx])

    history.append(gbest_value)

    if verbose:
        print(f"Iteration 0 - best cost: {gbest_value:.6f}")

    for it in range(1, max_it + 1):
        try:
            metrics_list = []
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

                SSE_nom, OS_nom, Ts__nom, Tr_nom, RVG, ODJ, u_rms, du_rms, sat_ratio = objetive(particles[i])
                TsTr_nom = max(0.0, Ts__nom - Tr_nom)
                metrics_list.append([
                    SSE_nom,
                    OS_nom,
                    TsTr_nom,
                    RVG,
                    ODJ,
                    u_rms,
                    du_rms,
                    sat_ratio
                ])

            metrics_arr = np.array(metrics_list)
            J_values = np.zeros(num_particles)

            for j in range(num_metrics):
                col = metrics_arr[:, j]
                order = np.argsort(col)
                ranks = np.empty(num_particles, dtype=int)
                ranks[order] = np.arange(num_particles)
                if num_particles > 1:
                    g = ranks / (num_particles - 1)
                else:
                    g = np.zeros_like(ranks, dtype=float)
                J_values += weights[j] * g

            for i in range(num_particles):
                J = J_values[i]

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
                    "SSE": float(metrics_arr[i, 0]),
                    "OS": float(metrics_arr[i, 1]),
                    "TsTr": float(metrics_arr[i, 2]),
                    "RVG": float(metrics_arr[i, 3]),
                    "ODJ": float(metrics_arr[i, 4]),
                    "u_rms": float(metrics_arr[i, 5]),
                    "du_rms": float(metrics_arr[i, 6]),
                    "sat_ratio": float(metrics_arr[i, 7])
                })

            history.append(gbest_value)
            if verbose:
                print(f"Iteration {it} - Best cost: {gbest_value:.6f}")
        except KeyboardInterrupt:
            print(f"INTERRUPTED Iteration {it}, best values:")
            break

    if log_path is not None:
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2, default=float)
        if verbose:
            print(f"Saved log to {log_path}")

    return gbest_position, gbest_value, history


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
                        ref=1.0, disturb_time=20.0, disturb_value=5.0):
    """
    Closed-loop simulation with discretized PID and discrete plant Gd,
    using state-space stepping (same dynamics as control.step_response(Gd)).
    """

    pid = PID(kp, ki, kd, N, Ts, u_min, u_max, kb_aw, der_on_meas=True)

    n_steps = int(t_final / Ts)
    t = np.arange(n_steps + 1) * Ts

    r = np.full_like(t, ref, dtype=float)

    Ad, Bd, Cd, Dd = ctl.ssdata(ctl.ss(Gd))
    Ad = np.asarray(Ad)
    Bd = np.asarray(Bd).reshape(-1)
    Cd = np.asarray(Cd)
    Dd = np.asarray(Dd).reshape(-1)

    x = np.zeros(Ad.shape[0])
    y = np.zeros_like(t)
    u = np.zeros_like(t)

    for k in range(n_steps):
        
        y_meas = y[k]
        if disturb_time is not None and t[k] >= disturb_time:
            y_meas = y_meas + disturb_value

   
        u[k] = pid.step(r[k], y_meas)

        x = Ad @ x + Bd * u[k]

        y[k+1] = (Cd @ x + Dd * u[k]).item()

    return t, y, u


def step_metrics(t, y, ref=1.0, tol=0.02):
    """
    Compute SSE, overshoot, rise time, settling time.
    """
    y = np.asarray(y)
    t = np.asarray(t)

    SSE = abs(ref - y[-1])
    OS = max(0.0, (np.max(y) - ref) / ref * 100)

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


def pid_cost_custom_with_Gd(x,
                            Gd,
                            N, Ts,
                            u_min, u_max, kb_aw,
                            t_final=5.0,
                            ref=1.0,
                            w_sse=0.7,
                            w_os=0.7,
                            w_ts_tr=0.45,
                            w_rvg=0.35,
                            w_odj=0.35,
                            w_cec=0.3,
                            bounded: bool = True):
    """
    Cost for PSO iteration. If bounded=True, uses normalized/clipped J1..J6.
    If bounded=False, uses raw/unbounded contributions to inspect magnitudes.
    """
    kp, ki, kd = x

    t_nom, y_nom, u_nom = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd,
        u_min, u_max, kb_aw,
        t_final, ref,
        disturb_time=None,
        disturb_value=None
    )

    SSE_nom, OS_nom, Tr_nom, Ts__nom = step_metrics(t_nom, y_nom, ref)

    TsTr_nom = max(0.0, Ts__nom - Tr_nom)

    Gd_low = scale_plant(Gd, 0.7)
    t_low, y_low, _ = sim_closed_loop_pid(
        kp, ki, kd,
        N, Ts,
        Gd_low,
        u_min, u_max, kb_aw,
        t_final, ref,
        disturb_time=None,
        disturb_value=None
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
        disturb_value=None
    )
    SSE_high, OS_high, Tr_high, Ts__high = step_metrics(t_high, y_high, ref)

    TsTr_low = max(0.0, Ts__low - Tr_low)
    TsTr_high = max(0.0, Ts__high - Tr_high)

    dSSE = max(0, max(SSE_high, SSE_low) - SSE_nom) / max(SSE_nom, 1e-6)
    dOS = max(0, max(OS_high, OS_low) - OS_nom) / max(OS_nom, 1e-6)
    dTsTr = max(0, max(TsTr_high, TsTr_low) - TsTr_nom) / max(TsTr_nom, 1e-6)

    RVG = dSSE + dOS + dTsTr

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
    e_dist = ref - y_dist[mask]

    if np.any(mask):
        ODJ = np.trapezoid(e_dist**2, t_dist[mask])
    else:
        ODJ = 0.0

    u_range = max(u_max - u_min, 1e-6)

    u_rms = np.sqrt(np.mean(u_nom**2))
    du = np.diff(u_nom)
    du_rms = np.sqrt(np.mean(du**2)) if len(du) > 0 else 0.0
    sat_ratio = np.mean(u_nom >= (u_max - 1e-3))

    return SSE_nom, OS_nom, Ts__nom, Tr_nom, RVG, ODJ, u_rms, du_rms, sat_ratio


def main(args):

    JSON_FILE = args.o

    N_filt = 20.0
    Ts = 0.01
    u_min = 0.0
    u_max = 1.0
    kb_aw = 1.0
    T_sim = 80.0

    Td = 0.15
    s = ctl.TransferFunction.s
    G0 = (12.68) / (s**2 + 1.076*s + 0.2744)
    num_delay, den_delay = ctl.pade(Td, 1)
    H_delay = ctl.tf(num_delay, den_delay)

    Gc = G0 * H_delay
    Gd = ctl.c2d(Gc, Ts, method='tustin')

    setpoint = 33.33

    def objective_pid(x):
        return pid_cost_custom_with_Gd(
            x, Gd=Gd,
            N=N_filt, Ts=Ts,
            u_min=u_min, u_max=u_max, kb_aw=kb_aw,
            t_final=T_sim, ref=setpoint,
            bounded=True
        )

    bounds = [
        (0.0, 20.0),
        (0.0, 5.0),
        (0.0, 5.0)
    ]

    seed = args.seed
    if ( seed is None ) or (seed == 0.0):
        seed = np.random.randint(0, 10000)
    best_x, best_J, history = pso(
        objective_pid,
        bounds,
        num_particles=50,
        max_it=100,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=seed,
        verbose=args.v,
        log_path=JSON_FILE
    )
    print(f"Best X: {best_x}\nBest J: {best_J}\nSeed: {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSO PID Tuning")
    parser.add_argument("-v", action="store_true", help="Enable verbose output")
    parser.add_argument("-o", type=str, default="pso_log.json")
    parser.add_argument("--seed", type=int, default=0, help="Numpy random number geneartor seed")
    args = parser.parse_args()
    main(args)

"""
==================================================================

DEPRECATED SCRIPT!!!!
Please use pid_tuning_absolute.py instead

This file is kept for archival purpouses

==================================================================

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

Run (example)
  python3 tuning/pso/pso_pid_tuning.py --help

Outputs
  - Prints PSO progress and best gains to stdout.
  - Optional JSON log file if a log path is provided in the CLI.
"""

import control as ctl
from lib.PID import PID
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

    # Initialize particles and speeds
    particles = np.random.uniform(bounds_min, bounds_max, size=(num_particles, dim))
    velocities = np.random.uniform(-1, 1, size=(num_particles, dim))

    # Personal best
    pbest_positions = particles.copy()
    pbest_values = np.zeros(num_particles, dtype=float)

    # Initialize log list
    logs = []
    for i in range(num_particles):
        J, info = objetive(particles[i])
        pbest_values[i] = J
        logs.append({
            "iter": 0,
            "particle": int(i),
            **info
        })

    # Global best
    best_idx = np.argmin(pbest_values)
    gbest_position = pbest_positions[best_idx].copy()
    gbest_value = float(pbest_values[best_idx])

    history = [gbest_value]

    print(f"Iteration 0 - best cost: {gbest_value:.6f}")
    print(f"    Best position: {gbest_position}")

    # Main loop
    for it in range(1, max_it + 1):
        try:
            for i in range(num_particles):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)

                # Update velocity
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (pbest_positions[i] - particles[i])
                    + c2 * r2 * (gbest_position - particles[i])
                )
                # Update position
                particles[i] += velocities[i]

                # Apply boundaries
                particles[i] = np.clip(particles[i], bounds_min, bounds_max)

                # Evaluate new cost
                J, info = objetive(particles[i])

                logs.append({
                    "iter": it,
                    "particle": int(i),
                    **info
                })

                # Update personal and global bests
                if J < pbest_values[i]:
                    pbest_values[i] = J
                    pbest_positions[i] = particles[i].copy()

                if J < gbest_value:
                    gbest_value = J
                    gbest_position = particles[i].copy()

            history.append(gbest_value)
            print(f"Iteration {it} - Best cost: {gbest_value:.6f}")
            print(f"    Best position: {gbest_position}")
        except KeyboardInterrupt:
            print(f"INTERRUPTED Iteration {it}, best values:")
            break

    if log_path is not None:
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2, default=float)
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

    # constant reference array
    r = np.full_like(t, ref, dtype=float)

    # state-space from Gd
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

        # Plant update: x_{k+1} = A x_k + B u_k
        x = Ad @ x + Bd * u[k]

        # Output: y_{k+1} = C x_{k+1} + D u_k
        y[k+1] = (Cd @ x + Dd * u[k]).item()

    return t, y, u


def step_metrics(t, y, ref=1.0, tol=0.02):
    """
    Compute SSE, overshoot, rise time, settling time.
    """
    y = np.asarray(y)
    t = np.asarray(t)

    # Steady State Error
    SSE = abs(ref - y[-1])
    # Overshoot in %
    OS = max(0.0, (np.max(y) - ref) / ref * 100)

    # Rise Time
    y10 = 0.1 * ref
    y90 = 0.9 * ref
    try:
        idx_10 = np.where(y > y10)[0][0]
        idx_90 = np.where(y > y90)[0][0]
        Tr = t[idx_90] - t[idx_10]
    except IndexError:
        Tr = t[-1]
        OS += 100.0

    # Settling time
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
                            w_cec=0.8,
                            bounded: bool = True,
                            verbose=False):
    """
    Cost for PSO iteration. If bounded=True, uses normalized/clipped J1..J6.
    If bounded=False, uses raw/unbounded contributions to inspect magnitudes.
    """
    kp, ki, kd = x

    if kp < 0.0 or ki < 0.0 or kd < 0.0:
        return 1e9, {}

    # ==================================================
    # J1 - SSE - Steady State Error
    # ==================================================
    SSE_target = 0.0 * ref       
    SSE_max =  0.05* ref           

    # Nominal response, used in other costs too
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

    if SSE_nom <= SSE_target:
        j1 = 0.0
    else:
        j1 = (SSE_nom - SSE_target) / (SSE_max - SSE_target)
        j1 = min(j1, 3.0)

    # ==================================================
    # J2 - OS - Overshoot
    # ==================================================
    OS_target = 0.0   # %
    OS_max = 5.0     # %

    if OS_nom <= OS_target:
        j2 = 0.0
    else:
        j2 = (OS_nom - OS_target) / (OS_max - OS_target)
        j2 = min(j2, 3.0)

    # ==================================================
    # J3 - Ts - Tr - Speed
    # ==================================================
    TsTr_nom = max(0.0, Ts__nom - Tr_nom)
    TsTr_target = 3.0
    TsTr_max = 10.0 

    if TsTr_nom <= TsTr_target:
        j3 = 0.0
    else:
        j3 = (TsTr_nom - TsTr_target) / (TsTr_max - TsTr_target)
        j3 = min(j3, 3.0)

    # ==================================================
    # J4 - RVG - Robustness to Variation in Gain
    # ==================================================
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

    #dSSE_ref = max(SSE_max - SSE_target, 1e-6)
    #dOS_ref = max(OS_max - OS_target, 1e-6)
    #TsTr_range = max(TsTr_max - TsTr_target, 1e-6)
    
    # More leanient references, as system gain change is less common
    # than regular operation
    dSSE_ref = ref * 0.10
    dOS_ref = 20.0
    TsTr_range = 10.0

    dSSE = max(SSE_low - SSE_nom, SSE_high - SSE_nom, 0.0) / dSSE_ref
    dOS = max(OS_low - OS_nom, OS_high - OS_nom, 0.0) / dOS_ref
    dTsTr = max(TsTr_low - TsTr_nom, TsTr_high - TsTr_nom, 0.0) / TsTr_range

    w_d_sse = 1.0
    w_d_os = 1.0
    w_d_ts = 1.0

    RVG_raw = w_d_sse * dSSE + w_d_os * dOS + w_d_ts * dTsTr

    RVG_target = 0.0 
    RVG_max = 5.0

    if RVG_raw <= RVG_target:
        j4 = 0.0
    else:
        j4 = (RVG_raw - RVG_target) / (RVG_max - RVG_target)
        j4 = min(j4, 3.0)

    # ==================================================
    # J5 - ODJ - Output Disturbance Rejection
    # ==================================================
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
        ISE = np.trapezoid(e_dist**2, t_dist[mask])
        T_dist = t_dist[mask][-1] - t_dist[mask][0]
        ODJ_raw = ISE / (ref**2 * max(T_dist, 1e-6))
    else:
        ODJ_raw = 0.0

    ODJ_target = 0.0
    ODJ_max = 0.1
    if ODJ_raw <= ODJ_target:
        j5 = 0.0
    else:
        j5 = (ODJ_raw - ODJ_target) / (ODJ_max - ODJ_target)
        j5 = min(j5, 3.0)

    # ==================================================
    # J6 - CEC - Control Effort Constraint
    # ==================================================
    u_range = max(u_max - u_min, 1e-6)


    u_rms_target = 0.0 * u_range    # 3% of range
    u_rms_max    = 0.5 * u_range    # 10% of range

    du_rms_target = 0.0 * u_range  # 0.5% of range
    du_rms_max    = 0.2  * u_range  # 2% of range

    sat_target = 0.00
    sat_max    = 0.5

    u_rms = np.sqrt(np.mean(u_nom**2))
    du = np.diff(u_nom)
    du_rms = np.sqrt(np.mean(du**2)) if len(du) > 0 else 0.0
    sat_ratio = np.mean(u_nom >= (u_max - 1e-3))

    # u_rms -> j6_u in [0,3]
    if u_rms <= u_rms_target:
        j6_u = 0.0
    else:
        j6_u = (u_rms - u_rms_target) / max(u_rms_max - u_rms_target, 1e-6)
        j6_u = min(j6_u, 3.0)

    # du_rms -> j6_du in [0,3]
    if du_rms <= du_rms_target:
        j6_du = 0.0
    else:
        j6_du = (du_rms - du_rms_target) / max(du_rms_max - du_rms_target, 1e-6)
        j6_du = min(j6_du, 3.0)

    # saturation -> j6_sat in [0,3]
    if sat_ratio <= sat_target:
        j6_sat = 0.0
    else:
        j6_sat = (sat_ratio - sat_target) / max(sat_max - sat_target, 1e-6)
        j6_sat = min(j6_sat, 3.0)

    w_cec_u = 1.0
    w_cec_du = 2.0
    w_cec_sat = 1.0

    CEC_raw = w_cec_u * j6_u + w_cec_du * j6_du + w_cec_sat * j6_sat
    j6 = min(CEC_raw, 3.0)

    # ==================================================
    # Select bounded vs unbounded contributions
    # ==================================================
    if bounded:
        J1_contrib = j1
        J2_contrib = j2
        J3_contrib = j3
        J4_contrib = j4
        J5_contrib = j5
        J6_contrib = j6
    else:
        # Raw/unbounded contributions
        J1_contrib = SSE_nom
        J2_contrib = OS_nom
        J3_contrib = TsTr_nom
        J4_contrib = RVG_raw
        J5_contrib = ODJ_raw
        J6_contrib = CEC_raw

    # ==================================================
    # Combine contributions with global weights
    # ==================================================
    j = (
        w_sse * J1_contrib +
        w_os * J2_contrib +
        w_ts_tr * J3_contrib +
        w_rvg * J4_contrib +
        w_odj * J5_contrib +
        w_cec * J6_contrib
    )
    if verbose:
        print(f"Costs for this iteration (bounded={bounded}):")
        print(f"  J1: {J1_contrib:.6f}")
        print(f"  J2: {J2_contrib:.6f}")
        print(f"  J3: {J3_contrib:.6f}")
        print(f"  J4: {J4_contrib:.6f}")
        print(f"  J5: {J5_contrib:.6f}")
        print(f"  J6: {J6_contrib:.6f}")

        print("Weighted contributions:")
        print(f"  w_sse*J1:   {w_sse * J1_contrib:.6f}")
        print(f"  w_os*J2:    {w_os * J2_contrib:.6f}")
        print(f"  w_ts_tr*J3: {w_ts_tr * J3_contrib:.6f}")
        print(f"  w_rvg*J4:   {w_rvg * J4_contrib:.6f}")
        print(f"  w_odj*J5:   {w_odj * J5_contrib:.6f}")
        print(f"  w_cec*J6:   {w_cec * J6_contrib:.6f}")
        print(f"  Total cost J: {j:.6f}\n")

    if bounded:
        for name, val in [("J1_SSE", j1), ("J2_OS", j2), ("J3_Ts_Tr", j3),
                          ("J4_RVG", j4), ("J5_ODJ", j5), ("J6_CEC", j6)]:
            assert 0.0 <= val <= 3.0 + 1e-6, f"{name} out of range: {val}"

        for name, val, w in [
            ("J1_weighted", j1, w_sse),
            ("J2_weighted", j2, w_os),
            ("J3_weighted", j3, w_ts_tr),
            ("J4_weighted", j4, w_rvg),
            ("J5_weighted", j5, w_odj),
            ("J6_weighted", j6, w_cec),
        ]:
            assert 0.0 <= w * val <= 3.0 + 1e-6, f"{name} out of range: {w * val}"

    info = {
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),

        # Normalized Js
        "J1_SSE_norm": float(j1),
        "J2_OS_norm": float(j2),
        "J3_Ts_Tr_norm": float(j3),
        "J4_RVG_norm": float(j4),
        "J5_ODJ_norm": float(j5),
        "J6_CEC_norm": float(j6),

        # Actual contribution used in cost (raw or normalized)
        "J1_contrib": float(J1_contrib),
        "J2_contrib": float(J2_contrib),
        "J3_contrib": float(J3_contrib),
        "J4_contrib": float(J4_contrib),
        "J5_contrib": float(J5_contrib),
        "J6_contrib": float(J6_contrib),

        "J1_weighted": float(w_sse * J1_contrib),
        "J2_weighted": float(w_os * J2_contrib),
        "J3_weighted": float(w_ts_tr * J3_contrib),
        "J4_weighted": float(w_rvg * J4_contrib),
        "J5_weighted": float(w_odj * J5_contrib),
        "J6_weighted": float(w_cec * J6_contrib),
        "Total_J": float(j)
    }

    return j, info


def main(args):

    JSON_FILE = args.o

    N_filt = 20.0
    Ts = 0.01
    u_min = 0.0
    u_max = 1.0
    kb_aw = 1.0
    T_sim = 80.0

    # Real discrete plant (2p1z + delay)
    Td = 0.15
    s = ctl.TransferFunction.s
    G0 = (12.68) / (s**2 + 1.076*s + 0.2744)
    num_delay, den_delay = ctl.pade(Td, 1)
    H_delay = ctl.tf(num_delay, den_delay)

    # Full continuous plant (with Pade delay), then discretize
    Gc = G0 * H_delay
    Gd = ctl.c2d(Gc, Ts, method='tustin')

    setpoint = 33.33 # Max speed in most European conntries


    def objective_pid(x):
        return pid_cost_custom_with_Gd(
            x, Gd=Gd,
            N=N_filt, Ts=Ts,
            u_min=u_min, u_max=u_max, kb_aw=kb_aw,
            t_final=T_sim, ref=setpoint,
            bounded=True,
            verbose=args.v
        )

    bounds = [
        (0.0, 2.0),  # Kp
        (0.0, 1.0),   # Ki
        (0.0, 10.0)    # Kd
    ]

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

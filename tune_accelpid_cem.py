#!/usr/bin/env python3
"""
Iterative tuning (CEM) for the *inner* accel PID.

We bypass the outer speed PID and directly apply a desired longitudinal
acceleration schedule (both +accel and -accel) to exercise throttle and braking.

Algorithm: Cross-Entropy Method (elite sampling).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Any

import rclpy

from PID import PID
from following_ros_cam_accelpid import FollowingRosAccelPidBridge


@dataclass(frozen=True)
class Gains:
    accel_kp: float
    accel_ki: float
    accel_kd: float
    accel_n: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.accel_kp, self.accel_ki, self.accel_kd, self.accel_n)


@dataclass
class TrialResult:
    gains: Gains
    cost: float
    avg_abs_a_err_throttle: float
    avg_abs_a_err_brake: float
    avg_abs_u: float
    avg_abs_du: float
    samples: int
    throttle_samples: int
    brake_samples: int
    aborted: bool
    abort_reason: str


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    var = sum((float(v) - float(mean)) ** 2 for v in values) / float(len(values) - 1)
    return float(math.sqrt(max(0.0, var)))


def _sample_gains(
    rng: random.Random,
    mean: Gains,
    std: Gains,
    bounds: dict[str, tuple[float, float]],
) -> Gains:
    def s(name: str, mu: float, sigma: float) -> float:
        lo, hi = bounds[name]
        return _clip(rng.gauss(mu, sigma), lo, hi)

    return Gains(
        accel_kp=s("accel_kp", mean.accel_kp, std.accel_kp),
        accel_ki=s("accel_ki", mean.accel_ki, std.accel_ki),
        accel_kd=s("accel_kd", mean.accel_kd, std.accel_kd),
        accel_n=s("accel_n", mean.accel_n, std.accel_n),
    )


def _update_distribution(
    old_mean: Gains,
    old_std: Gains,
    elites: list[TrialResult],
    alpha: float,
    min_std: float,
) -> tuple[Gains, Gains]:
    def col(selector) -> list[float]:
        return [float(selector(r.gains)) for r in elites]

    cols = {
        "accel_kp": col(lambda g: g.accel_kp),
        "accel_ki": col(lambda g: g.accel_ki),
        "accel_kd": col(lambda g: g.accel_kd),
        "accel_n": col(lambda g: g.accel_n),
    }

    elite_means = {k: _mean(v) for k, v in cols.items()}
    elite_stds = {k: _std(cols[k], elite_means[k]) for k in cols}

    def blend(mu_old: float, mu_new: float) -> float:
        return float((1.0 - alpha) * float(mu_old) + alpha * float(mu_new))

    def blend_std(s_old: float, s_new: float) -> float:
        out = float((1.0 - alpha) * float(s_old) + alpha * float(s_new))
        return float(max(float(min_std), out))

    new_mean = Gains(
        accel_kp=blend(old_mean.accel_kp, elite_means["accel_kp"]),
        accel_ki=blend(old_mean.accel_ki, elite_means["accel_ki"]),
        accel_kd=blend(old_mean.accel_kd, elite_means["accel_kd"]),
        accel_n=blend(old_mean.accel_n, elite_means["accel_n"]),
    )
    new_std = Gains(
        accel_kp=blend_std(old_std.accel_kp, elite_stds["accel_kp"]),
        accel_ki=blend_std(old_std.accel_ki, elite_stds["accel_ki"]),
        accel_kd=blend_std(old_std.accel_kd, elite_stds["accel_kd"]),
        accel_n=blend_std(old_std.accel_n, elite_stds["accel_n"]),
    )
    return new_mean, new_std


def _build_accel_schedule(args) -> list[tuple[float, float]]:
    sched: list[tuple[float, float]] = []
    for _ in range(int(args.cycles)):
        sched.extend(
            [
                (float(args.hold_s), float(args.a_pos)),
                (float(args.rest_s), 0.0),
                (float(args.hold_s), float(args.a_neg)),
                (float(args.rest_s), 0.0),
            ]
        )
    return sched


def _run_one_trial(
    *,
    bridge: FollowingRosAccelPidBridge,
    gains: Gains,
    init_speed_mps: float,
    duration_s: float,
    print_every_s: float,
    a_schedule: list[tuple[float, float]],
    accel_deadband_on: float,
    accel_deadband_off: float,
    abort_abs_v_err: float,
    abort_speed_mps: float,
    min_mode_samples: int,
    w_v: float,
    w_a_throttle: float,
    w_a_brake: float,
    w_u: float,
    w_du: float,
) -> TrialResult:
    bridge.reset_test()
    ego_name = bridge.ros_names[bridge.ego_index]

    def set_cmd(throttle: float, brake: float) -> None:
        bridge._last_cmd[ego_name]["throttle"] = float(throttle)
        bridge._last_cmd[ego_name]["brake"] = float(brake)

    # Pre-condition: accelerate/brake to reach an initial speed (not part of tuning objective).
    pre_timeout_s = 15.0
    pre_start = float(bridge.sim_time)
    while float(bridge.sim_time) < float(pre_start + pre_timeout_s):
        bridge.step_simulation()
        speed = float(bridge._last_speeds[bridge.ego_index]) if bridge._last_speeds else 0.0
        err = float(init_speed_mps - speed)
        if abs(err) < 0.25 and speed > 1.0:
            break
        if err > 0.0:
            throttle = _clip(0.15 + 0.08 * err, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = _clip(0.10 + 0.10 * (-err), 0.0, 1.0)
        set_cmd(throttle, brake)
        bridge.apply_controls()

    # Controller being tuned: accel setpoint -> effort u in [-1, 1]
    pid = PID(
        float(gains.accel_kp),
        float(gains.accel_ki),
        float(gains.accel_kd),
        float(gains.accel_n),
        float(bridge.Ts),
        -1.0,
        1.0,
        1.0,
        der_on_meas=True,
        integral_disc_method="backeuler",
        derivative_disc_method="tustin",
    )

    start_sim = float(bridge.sim_time)
    end_sim = float(start_sim + float(duration_s))
    next_print_sim = float(start_sim + float(print_every_s)) if float(print_every_s) > 0.0 else float("inf")

    sum_abs_v_err = 0.0
    sum_abs_a_err_throttle = 0.0
    sum_abs_a_err_brake = 0.0
    sum_abs_u = 0.0
    sum_abs_du = 0.0
    prev_u = 0.0
    samples = 0
    throttle_samples = 0
    brake_samples = 0
    aborted = False
    abort_reason = ""

    # schedule pointer
    if not a_schedule:
        a_schedule = [(duration_s, 0.0)]
    sched_idx = 0
    seg_end = float(start_sim + float(a_schedule[0][0]))
    a_sp = float(a_schedule[0][1])

    mode = "coast"

    while float(bridge.sim_time) < float(end_sim):
        bridge.step_simulation()

        while float(bridge.sim_time) >= float(seg_end) and (sched_idx + 1) < len(a_schedule):
            sched_idx += 1
            seg_end = float(seg_end + float(a_schedule[sched_idx][0]))
            a_sp = float(a_schedule[sched_idx][1])

        v = float(bridge._last_speeds[bridge.ego_index]) if bridge._last_speeds else 0.0
        a_meas = float(bridge._last_accels_long[bridge.ego_index]) if bridge._last_accels_long else 0.0
        u_effort = float(pid.step(a_sp, a_meas))

        # mode selection with hysteresis (same structure as PlatoonMember)
        if mode == "coast":
            if u_effort > float(accel_deadband_on):
                mode = "throttle"
            elif u_effort < -float(accel_deadband_on):
                mode = "brake"
        elif mode == "throttle":
            if u_effort < float(accel_deadband_off):
                mode = "coast"
        else:  # brake
            if u_effort > -float(accel_deadband_off):
                mode = "coast"

        throttle_cmd = _clip(u_effort, 0.0, 1.0) if mode == "throttle" else 0.0
        brake_cmd = _clip(-u_effort, 0.0, 1.0) if mode == "brake" else 0.0
        set_cmd(throttle_cmd, brake_cmd)
        bridge.apply_controls()

        u = float(throttle_cmd) - float(brake_cmd)
        du = float(u - prev_u)
        prev_u = u

        v_err = float(init_speed_mps - v)
        a_err = float(a_sp - a_meas)
        sum_abs_v_err += abs(v_err)
        if a_sp > 0.05:
            sum_abs_a_err_throttle += abs(a_err)
            throttle_samples += 1
        elif a_sp < -0.05:
            sum_abs_a_err_brake += abs(a_err)
            brake_samples += 1
        sum_abs_u += abs(u)
        sum_abs_du += abs(du)
        samples += 1

        if abs(v_err) > float(abort_abs_v_err):
            aborted = True
            abort_reason = f"|v_err|>{abort_abs_v_err}"
            break
        if v > float(abort_speed_mps):
            aborted = True
            abort_reason = f"speed>{abort_speed_mps}"
            break

        if float(bridge.sim_time) >= float(next_print_sim):
            print(
                f"  t={bridge.sim_time:7.2f}s v={v:5.2f} v_err={v_err:6.3f} "
                f"a_sp={a_sp:6.3f} a={a_meas:6.3f} a_err={a_err:6.3f} "
                f"mode={mode:7s} thr={throttle_cmd:5.3f} brk={brake_cmd:5.3f}"
            )
            next_print_sim = float(bridge.sim_time) + float(print_every_s)

    avg_abs_v_err = float(sum_abs_v_err / max(1, samples))
    avg_abs_a_err_throttle = float(sum_abs_a_err_throttle / max(1, throttle_samples))
    avg_abs_a_err_brake = float(sum_abs_a_err_brake / max(1, brake_samples))
    avg_abs_u = float(sum_abs_u / max(1, samples))
    avg_abs_du = float(sum_abs_du / max(1, samples))

    if throttle_samples < int(min_mode_samples) or brake_samples < int(min_mode_samples):
        aborted = True
        abort_reason = f"insufficient mode samples (thr={throttle_samples}, brk={brake_samples})"

    cost = float(
        w_v * avg_abs_v_err
        + w_a_throttle * avg_abs_a_err_throttle
        + w_a_brake * avg_abs_a_err_brake
        + w_u * avg_abs_u
        + w_du * avg_abs_du
    )
    if aborted:
        cost = float(cost + 1e6)

    return TrialResult(
        gains=gains,
        cost=cost,
        avg_abs_a_err_throttle=avg_abs_a_err_throttle,
        avg_abs_a_err_brake=avg_abs_a_err_brake,
        avg_abs_u=avg_abs_u,
        avg_abs_du=avg_abs_du,
        samples=int(samples),
        throttle_samples=int(throttle_samples),
        brake_samples=int(brake_samples),
        aborted=bool(aborted),
        abort_reason=str(abort_reason),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CEM tuning for inner accel PID (accel setpoints).")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("-f", type=str, default="out_ros_accelpid_cem.csv", help="Bridge log CSV (same format as following_ros_cam_accelpid.py).")
    parser.add_argument("--plen", type=int, default=1)
    parser.add_argument("--mcu-index", type=int, default=None)
    parser.add_argument("--teleport-dist", type=float, default=320.0)
    # Dashcam is disabled for tuning (run as fast as possible, no GUI).
    parser.add_argument(
        "--dashcam",
        action="store_true",
        help="Ignored (dashcam is disabled in this tuning script).",
    )
    parser.add_argument(
        "--no-dashcam",
        action="store_true",
        help="Ignored (dashcam is disabled in this tuning script).",
    )
    parser.add_argument("--lane-keep", action="store_true")
    parser.add_argument("--lane-lookahead", type=float, default=12.0)
    parser.add_argument("--lane-gain", type=float, default=1.0)
    parser.add_argument("--wall-dt", type=float, default=None)

    parser.add_argument("--init-speed", type=float, default=10.0, help="Initial speed [m/s] before accel schedule.")
    parser.add_argument("--trial-duration-s", type=float, default=40.0, help="Duration per trial [s] (sim time).")
    parser.add_argument("--print-every-s", type=float, default=2.0, help="Live debug print period [s] (sim time).")

    parser.add_argument("--a-pos", type=float, default=1.0, help="Positive accel setpoint [m/s^2] (throttle).")
    parser.add_argument("--a-neg", type=float, default=-1.0, help="Negative accel setpoint [m/s^2] (brake).")
    parser.add_argument("--hold-s", type=float, default=4.0, help="Hold duration per accel step [s].")
    parser.add_argument("--rest-s", type=float, default=2.0, help="Rest duration between accel steps [s].")
    parser.add_argument("--cycles", type=int, default=2, help="Number of (+a,0,-a,0) cycles per trial.")
    parser.add_argument("--accel-deadband-on", type=float, default=0.03)
    parser.add_argument("--accel-deadband-off", type=float, default=0.01)

    parser.add_argument("--iters", type=int, default=6, help="CEM iterations.")
    parser.add_argument("--samples", type=int, default=30, help="Trials per CEM iteration.")
    parser.add_argument("--elite-frac", type=float, default=0.2, help="Fraction of elites to keep per iteration.")
    parser.add_argument("--alpha", type=float, default=0.7, help="EMA update weight for mean/std.")
    parser.add_argument("--min-std", type=float, default=0.01, help="Lower bound for per-parameter std.")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed.")

    parser.add_argument("--abort-abs-v-err", type=float, default=20.0, help="Abort trial if |v_err| exceeds this.")
    parser.add_argument("--abort-speed-mps", type=float, default=60.0, help="Abort trial if speed exceeds this.")
    parser.add_argument("--min-mode-samples", type=int, default=100, help="Minimum samples in BOTH +a and -a segments.")

    parser.add_argument("--w-v", type=float, default=1.0, help="Weight for avg|v_err|.")
    parser.add_argument("--w-a-throttle", type=float, default=1.0, help="Weight for avg|a_err| during +a segments.")
    parser.add_argument("--w-a-brake", type=float, default=1.0, help="Weight for avg|a_err| during -a segments.")
    parser.add_argument("--w-u", type=float, default=0.05, help="Weight for avg|u| (effort).")
    parser.add_argument("--w-du", type=float, default=0.05, help="Weight for avg|du| (smoothness).")

    parser.add_argument("--out-json", type=str, default="cem_tuning_results.json")
    args = parser.parse_args()
    # Bridge expects args.no_dashcam; always disable dashcam for tuning.
    args.no_dashcam = True

    rng = random.Random(int(args.seed))

    # Bounds for sampling (keep stable-ish by design)
    bounds: dict[str, tuple[float, float]] = {
        "accel_kp": (0.01, 4.00),
        "accel_ki": (0.00, 2.00),
        "accel_kd": (0.00, 1.00),
        "accel_n": (1.00, 40.00),
    }

    # Initial distribution (roughly centered in the bounds with a wide std)
    mean = Gains(
        accel_kp=1.0,
        accel_ki=0.10,
        accel_kd=0.05,
        accel_n=12.0,
    )
    std = Gains(
        accel_kp=0.8,
        accel_ki=0.15,
        accel_kd=0.08,
        accel_n=6.0,
    )

    rclpy.init()
    bridge = FollowingRosAccelPidBridge(args)

    # Disable rendering so CARLA can run faster than real-time while keeping Ts=0.01.
    try:
        settings = bridge.world.get_settings()
        settings.no_rendering_mode = True
        bridge.world.apply_settings(settings)
        try:
            bridge.get_logger().info("no_rendering_mode ON (tuning)")
        except Exception:
            pass
    except Exception as e:
        try:
            bridge.get_logger().warn(f"Failed to enable no_rendering_mode: {e}")
        except Exception:
            pass

    a_schedule = _build_accel_schedule(args)

    history: dict[str, Any] = {
        "meta": {
            "init_speed": float(args.init_speed),
            "trial_duration_s": float(args.trial_duration_s),
            "Ts": float(bridge.Ts),
            "iters": int(args.iters),
            "samples": int(args.samples),
            "elite_frac": float(args.elite_frac),
            "alpha": float(args.alpha),
            "seed": int(args.seed),
            "weights": {
                "w_v": float(args.w_v),
                "w_a_throttle": float(args.w_a_throttle),
                "w_a_brake": float(args.w_a_brake),
                "w_u": float(args.w_u),
                "w_du": float(args.w_du),
            },
            "a_schedule": a_schedule,
            "deadbands": {"on": float(args.accel_deadband_on), "off": float(args.accel_deadband_off)},
        },
        "iterations": [],
        "best": None,
    }

    best_overall: TrialResult | None = None

    try:
        for it in range(int(args.iters)):
            elite_count = max(1, int(math.ceil(float(args.elite_frac) * float(args.samples))))
            print(
                f"\n=== CEM Iteration {it+1}/{int(args.iters)} "
                f"(mean={mean.as_tuple()} std={std.as_tuple()}) ==="
            )

            results: list[TrialResult] = []
            for k in range(int(args.samples)):
                g = _sample_gains(rng, mean, std, bounds)
                print(f"\n-- Trial {k+1}/{int(args.samples)} gains: accel={g.as_tuple()}")
                tr = _run_one_trial(
                    bridge=bridge,
                    gains=g,
                    init_speed_mps=float(args.init_speed),
                    duration_s=float(args.trial_duration_s),
                    print_every_s=float(args.print_every_s),
                    a_schedule=a_schedule,
                    accel_deadband_on=float(args.accel_deadband_on),
                    accel_deadband_off=float(args.accel_deadband_off),
                    abort_abs_v_err=float(args.abort_abs_v_err),
                    abort_speed_mps=float(args.abort_speed_mps),
                    min_mode_samples=int(args.min_mode_samples),
                    w_v=float(args.w_v),
                    w_a_throttle=float(args.w_a_throttle),
                    w_a_brake=float(args.w_a_brake),
                    w_u=float(args.w_u),
                    w_du=float(args.w_du),
                )
                results.append(tr)
                print(
                    f"Trial done: cost={tr.cost:.4f} "
                    f"avg|a_err|thr={tr.avg_abs_a_err_throttle:.4f} avg|a_err|brk={tr.avg_abs_a_err_brake:.4f} "
                    f"avg|u|={tr.avg_abs_u:.4f} avg|du|={tr.avg_abs_du:.4f} "
                    f"samples={tr.samples} thr_n={tr.throttle_samples} brk_n={tr.brake_samples} "
                    f"aborted={tr.aborted} {tr.abort_reason}"
                )

                if best_overall is None or tr.cost < best_overall.cost:
                    best_overall = tr

            results_sorted = sorted(results, key=lambda r: float(r.cost))
            elites = results_sorted[:elite_count]
            best_it = results_sorted[0]

            print(
                f"\nIteration {it+1} best: cost={best_it.cost:.4f} accel={best_it.gains.as_tuple()}"
            )
            print(f"Elites kept: {elite_count}/{len(results)}")

            mean, std = _update_distribution(
                old_mean=mean,
                old_std=std,
                elites=elites,
                alpha=float(args.alpha),
                min_std=float(args.min_std),
            )

            history["iterations"].append(
                {
                    "iter": it + 1,
                    "mean": mean.__dict__,
                    "std": std.__dict__,
                    "best": {
                        "gains": best_it.gains.__dict__,
                        "cost": best_it.cost,
                        "avg_abs_a_err_throttle": best_it.avg_abs_a_err_throttle,
                        "avg_abs_a_err_brake": best_it.avg_abs_a_err_brake,
                        "avg_abs_u": best_it.avg_abs_u,
                        "avg_abs_du": best_it.avg_abs_du,
                        "throttle_samples": best_it.throttle_samples,
                        "brake_samples": best_it.brake_samples,
                        "aborted": best_it.aborted,
                        "abort_reason": best_it.abort_reason,
                    },
                    "results": [
                        {
                            "gains": r.gains.__dict__,
                            "cost": r.cost,
                            "avg_abs_a_err_throttle": r.avg_abs_a_err_throttle,
                            "avg_abs_a_err_brake": r.avg_abs_a_err_brake,
                            "avg_abs_u": r.avg_abs_u,
                            "avg_abs_du": r.avg_abs_du,
                            "samples": r.samples,
                            "throttle_samples": r.throttle_samples,
                            "brake_samples": r.brake_samples,
                            "aborted": r.aborted,
                            "abort_reason": r.abort_reason,
                        }
                        for r in results_sorted
                    ],
                }
            )

            if best_overall is not None:
                history["best"] = {
                    "gains": best_overall.gains.__dict__,
                    "cost": best_overall.cost,
                    "avg_abs_a_err_throttle": best_overall.avg_abs_a_err_throttle,
                    "avg_abs_a_err_brake": best_overall.avg_abs_a_err_brake,
                    "avg_abs_u": best_overall.avg_abs_u,
                    "avg_abs_du": best_overall.avg_abs_du,
                    "throttle_samples": best_overall.throttle_samples,
                    "brake_samples": best_overall.brake_samples,
                    "aborted": best_overall.aborted,
                    "abort_reason": best_overall.abort_reason,
                }

            try:
                with open(str(args.out_json), "w") as f:
                    json.dump(history, f, indent=2)
                print(f"Wrote progress to {args.out_json}")
            except Exception as e:
                print(f"Failed to write {args.out_json}: {e}")

    except KeyboardInterrupt:
        pass
    finally:
        if best_overall is not None:
            print("\n=== Best Overall ===")
            print(f"cost={best_overall.cost:.6f}")
            print(f"accel gains (kp,ki,kd,N)={best_overall.gains.as_tuple()}")
            print(
                f"avg|a_err|thr={best_overall.avg_abs_a_err_throttle:.4f} "
                f"avg|a_err|brk={best_overall.avg_abs_a_err_brake:.4f} "
                f"avg|u|={best_overall.avg_abs_u:.4f} avg|du|={best_overall.avg_abs_du:.4f}"
            )

        try:
            bridge.shutdown()
        except Exception:
            pass
        try:
            bridge.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()

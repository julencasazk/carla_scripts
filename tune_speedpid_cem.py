#!/usr/bin/env python3
"""
CEM tuning for the *outer* speed PID (speed -> desired acceleration).

This script:
- Runs CARLA in synchronous mode at Ts=0.01 s (sim time).
- Disables rendering (no_rendering_mode=True) and does NOT use the dashcam.
- Runs faster-than-real-time (no wall-clock pacing): world.tick() as fast as possible.

Control architecture for the trial:
  speed PID: v_sp -> a_des (clipped to [a_min, a_max])
  accel PID (fixed gains): a_sp=a_des -> effort u in [-1,1] -> throttle/brake (with hysteresis)

We tune ONLY the speed PID gains (kp, ki, kd, N) using Cross-Entropy Method (elite sampling).
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
    speed_kp: float
    speed_ki: float
    speed_kd: float
    speed_n: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.speed_kp, self.speed_ki, self.speed_kd, self.speed_n)


@dataclass
class TrialResult:
    gains: Gains
    cost: float
    avg_abs_v_err: float
    overshoot_mps: float
    avg_abs_u: float
    avg_abs_du: float
    a_sat_ratio: float
    u_sat_ratio: float
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
        speed_kp=s("speed_kp", mean.speed_kp, std.speed_kp),
        speed_ki=s("speed_ki", mean.speed_ki, std.speed_ki),
        speed_kd=s("speed_kd", mean.speed_kd, std.speed_kd),
        speed_n=s("speed_n", mean.speed_n, std.speed_n),
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
        "speed_kp": col(lambda g: g.speed_kp),
        "speed_ki": col(lambda g: g.speed_ki),
        "speed_kd": col(lambda g: g.speed_kd),
        "speed_n": col(lambda g: g.speed_n),
    }

    elite_means = {k: _mean(v) for k, v in cols.items()}
    elite_stds = {k: _std(cols[k], elite_means[k]) for k in cols}

    def blend(mu_old: float, mu_new: float) -> float:
        return float((1.0 - alpha) * float(mu_old) + alpha * float(mu_new))

    def blend_std(s_old: float, s_new: float) -> float:
        out = float((1.0 - alpha) * float(s_old) + alpha * float(s_new))
        return float(max(float(min_std), out))

    new_mean = Gains(
        speed_kp=blend(old_mean.speed_kp, elite_means["speed_kp"]),
        speed_ki=blend(old_mean.speed_ki, elite_means["speed_ki"]),
        speed_kd=blend(old_mean.speed_kd, elite_means["speed_kd"]),
        speed_n=blend(old_mean.speed_n, elite_means["speed_n"]),
    )
    new_std = Gains(
        speed_kp=blend_std(old_std.speed_kp, elite_stds["speed_kp"]),
        speed_ki=blend_std(old_std.speed_ki, elite_stds["speed_ki"]),
        speed_kd=blend_std(old_std.speed_kd, elite_stds["speed_kd"]),
        speed_n=blend_std(old_std.speed_n, elite_stds["speed_n"]),
    )
    return new_mean, new_std


def _build_speed_schedule(args) -> list[tuple[float, float]]:
    """
    Speed schedule designed to exercise both throttle and braking:
      v_base -> v_high -> v_base -> v_low -> v_base (repeat cycles)
    """
    sched: list[tuple[float, float]] = []
    for _ in range(int(args.cycles)):
        sched.extend(
            [
                (float(args.hold_s), float(args.v_base)),
                (float(args.hold_s), float(args.v_high)),
                (float(args.hold_s), float(args.v_base)),
                (float(args.hold_s), float(args.v_low)),
                (float(args.hold_s), float(args.v_base)),
            ]
        )
        if float(args.rest_s) > 0.0:
            sched.append((float(args.rest_s), float(args.v_base)))
    return sched


def _run_one_trial(
    *,
    bridge: FollowingRosAccelPidBridge,
    gains: Gains,
    duration_s: float,
    print_every_s: float,
    v_schedule: list[tuple[float, float]],
    # fixed accel PID (inner loop)
    accel_gains: tuple[float, float, float, float],
    accel_deadband_on: float,
    accel_deadband_off: float,
    # limits
    a_min: float,
    a_max: float,
    abort_abs_v_err: float,
    abort_speed_mps: float,
    min_mode_samples: int,
    # weights
    w_v: float,
    w_os: float,
    w_u: float,
    w_du: float,
    w_a_sat: float,
    w_u_sat: float,
) -> TrialResult:
    ego_name = bridge.ros_names[bridge.ego_index]

    def set_cmd(throttle: float, brake: float) -> None:
        bridge._last_cmd[ego_name]["throttle"] = float(throttle)
        bridge._last_cmd[ego_name]["brake"] = float(brake)

    # Reset rig
    bridge.reset_test()
    bridge.apply_controls()

    Ts = float(bridge.Ts)

    speed_pid = PID(
        float(gains.speed_kp),
        float(gains.speed_ki),
        float(gains.speed_kd),
        float(gains.speed_n),
        Ts,
        float(a_min),
        float(a_max),
        1.0,
        der_on_meas=True,
        integral_disc_method="backeuler",
        derivative_disc_method="tustin",
    )

    accel_pid = PID(
        float(accel_gains[0]),
        float(accel_gains[1]),
        float(accel_gains[2]),
        float(accel_gains[3]),
        Ts,
        -1.0,
        1.0,
        1.0,
        der_on_meas=True,
        integral_disc_method="backeuler",
        derivative_disc_method="tustin",
    )

    # Ensure schedule exists
    if not v_schedule:
        v_schedule = [(duration_s, 0.0)]

    start_sim = float(bridge.sim_time)
    end_sim = float(start_sim + float(duration_s))
    next_print_sim = float(start_sim + float(print_every_s)) if float(print_every_s) > 0.0 else float("inf")

    # schedule pointer
    sched_idx = 0
    seg_end = float(start_sim + float(v_schedule[0][0]))
    v_sp = float(v_schedule[0][1])

    # mode state (inner effort -> throttle/brake)
    mode = "coast"

    sum_abs_v_err = 0.0
    max_overshoot = 0.0
    sum_abs_u = 0.0
    sum_abs_du = 0.0
    prev_u = 0.0
    samples = 0
    throttle_samples = 0
    brake_samples = 0
    a_sat_hits = 0
    u_sat_hits = 0
    aborted = False
    abort_reason = ""

    while float(bridge.sim_time) < float(end_sim):
        bridge.step_simulation()

        while float(bridge.sim_time) >= float(seg_end) and (sched_idx + 1) < len(v_schedule):
            sched_idx += 1
            seg_end = float(seg_end + float(v_schedule[sched_idx][0]))
            v_sp = float(v_schedule[sched_idx][1])

        v = float(bridge._last_speeds[bridge.ego_index]) if bridge._last_speeds else 0.0
        a_meas = float(bridge._last_accels_long[bridge.ego_index]) if bridge._last_accels_long else 0.0

        a_des = float(speed_pid.step(float(v_sp), float(v)))
        a_des = _clip(a_des, float(a_min), float(a_max))
        if abs(a_des - float(a_min)) < 1e-9 or abs(a_des - float(a_max)) < 1e-9:
            a_sat_hits += 1

        u_effort = float(accel_pid.step(float(a_des), float(a_meas)))

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

        if throttle_cmd >= 0.999 or brake_cmd >= 0.999:
            u_sat_hits += 1

        set_cmd(throttle_cmd, brake_cmd)
        bridge.apply_controls()

        v_err = float(v_sp - v)
        sum_abs_v_err += abs(v_err)
        if v > float(v_sp):
            max_overshoot = max(float(max_overshoot), float(v - v_sp))

        u = float(throttle_cmd) - float(brake_cmd)
        du = float(u - prev_u)
        prev_u = u
        sum_abs_u += abs(u)
        sum_abs_du += abs(du)

        if throttle_cmd > 0.02:
            throttle_samples += 1
        if brake_cmd > 0.02:
            brake_samples += 1

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
                f"  t={bridge.sim_time:7.2f}s v_sp={v_sp:5.2f} v={v:5.2f} v_err={v_err:6.3f} "
                f"a_des={a_des:6.3f} a={a_meas:6.3f} mode={mode:7s} "
                f"thr={throttle_cmd:5.3f} brk={brake_cmd:5.3f}"
            )
            next_print_sim = float(bridge.sim_time) + float(print_every_s)

    avg_abs_v_err = float(sum_abs_v_err / max(1, samples))
    avg_abs_u = float(sum_abs_u / max(1, samples))
    avg_abs_du = float(sum_abs_du / max(1, samples))
    a_sat_ratio = float(a_sat_hits / max(1, samples))
    u_sat_ratio = float(u_sat_hits / max(1, samples))

    if throttle_samples < int(min_mode_samples) or brake_samples < int(min_mode_samples):
        aborted = True
        abort_reason = f"insufficient mode samples (thr={throttle_samples}, brk={brake_samples})"

    cost = float(
        w_v * avg_abs_v_err
        + w_os * float(max_overshoot)
        + w_u * avg_abs_u
        + w_du * avg_abs_du
        + w_a_sat * a_sat_ratio
        + w_u_sat * u_sat_ratio
    )
    if aborted:
        cost = float(cost + 1e6)

    return TrialResult(
        gains=gains,
        cost=cost,
        avg_abs_v_err=avg_abs_v_err,
        overshoot_mps=float(max_overshoot),
        avg_abs_u=avg_abs_u,
        avg_abs_du=avg_abs_du,
        a_sat_ratio=a_sat_ratio,
        u_sat_ratio=u_sat_ratio,
        samples=int(samples),
        throttle_samples=int(throttle_samples),
        brake_samples=int(brake_samples),
        aborted=bool(aborted),
        abort_reason=str(abort_reason),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CEM tuning for outer speed PID (speed setpoints).")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("-f", type=str, default="out_ros_speedpid_cem.csv", help="Bridge log CSV.")
    parser.add_argument("--plen", type=int, default=1)
    parser.add_argument("--mcu-index", type=int, default=None)
    parser.add_argument("--teleport-dist", type=float, default=320.0)
    parser.add_argument("--lane-keep", action="store_true")
    parser.add_argument("--lane-lookahead", type=float, default=12.0)
    parser.add_argument("--lane-gain", type=float, default=1.0)

    # Speed schedule for tuning
    parser.add_argument("--v-base", type=float, default=10.0)
    parser.add_argument("--v-high", type=float, default=14.0)
    parser.add_argument("--v-low", type=float, default=6.0)
    parser.add_argument("--hold-s", type=float, default=8.0)
    parser.add_argument("--rest-s", type=float, default=0.0)
    parser.add_argument("--cycles", type=int, default=1)

    parser.add_argument("--trial-duration-s", type=float, default=40.0)
    parser.add_argument("--print-every-s", type=float, default=2.0)

    # Fixed inner accel PID gains (from your tuned inner results)
    parser.add_argument("--accel-kp", type=float, default=1.0)
    parser.add_argument("--accel-ki", type=float, default=0.10)
    parser.add_argument("--accel-kd", type=float, default=0.05)
    parser.add_argument("--accel-n", type=float, default=12.0)
    parser.add_argument("--accel-deadband-on", type=float, default=0.03)
    parser.add_argument("--accel-deadband-off", type=float, default=0.01)

    # Output limits for speed->accel
    parser.add_argument("--a-min", type=float, default=-6.0)
    parser.add_argument("--a-max", type=float, default=3.0)

    # CEM settings
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--elite-frac", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--min-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)

    # Abort + coverage
    parser.add_argument("--abort-abs-v-err", type=float, default=20.0)
    parser.add_argument("--abort-speed-mps", type=float, default=60.0)
    parser.add_argument("--min-mode-samples", type=int, default=100)

    # Cost weights
    parser.add_argument("--w-v", type=float, default=1.0, help="avg|v_err|")
    parser.add_argument("--w-os", type=float, default=2.0, help="max overshoot [m/s]")
    parser.add_argument("--w-u", type=float, default=0.05, help="avg|u|")
    parser.add_argument("--w-du", type=float, default=0.05, help="avg|du|")
    parser.add_argument("--w-a-sat", type=float, default=0.5, help="a_des saturation ratio")
    parser.add_argument("--w-u-sat", type=float, default=0.5, help="throttle/brake saturation ratio")

    parser.add_argument("--out-json", type=str, default="cem_speedpid_results.json")
    args = parser.parse_args()

    # Bridge requires args.no_dashcam; always disable dashcam for tuning.
    args.no_dashcam = True

    rng = random.Random(int(args.seed))

    bounds: dict[str, tuple[float, float]] = {
        "speed_kp": (0.01, 3.00),
        "speed_ki": (0.00, 1.00),
        "speed_kd": (0.00, 1.00),
        "speed_n": (1.00, 40.00),
    }

    mean = Gains(speed_kp=0.7, speed_ki=0.05, speed_kd=0.05, speed_n=12.0)
    std = Gains(speed_kp=0.6, speed_ki=0.08, speed_kd=0.08, speed_n=6.0)

    rclpy.init()
    bridge = FollowingRosAccelPidBridge(args)

    # Disable rendering (faster-than-real-time).
    try:
        settings = bridge.world.get_settings()
        settings.no_rendering_mode = True
        bridge.world.apply_settings(settings)
        try:
            bridge.get_logger().info("no_rendering_mode ON (speed PID tuning)")
        except Exception:
            pass
    except Exception as e:
        try:
            bridge.get_logger().warn(f"Failed to enable no_rendering_mode: {e}")
        except Exception:
            pass

    v_schedule = _build_speed_schedule(args)
    accel_gains = (float(args.accel_kp), float(args.accel_ki), float(args.accel_kd), float(args.accel_n))

    history: dict[str, Any] = {
        "meta": {
            "trial_duration_s": float(args.trial_duration_s),
            "Ts": float(bridge.Ts),
            "iters": int(args.iters),
            "samples": int(args.samples),
            "elite_frac": float(args.elite_frac),
            "alpha": float(args.alpha),
            "seed": int(args.seed),
            "v_schedule": v_schedule,
            "accel_gains": {"kp": args.accel_kp, "ki": args.accel_ki, "kd": args.accel_kd, "n": args.accel_n},
            "accel_deadbands": {"on": float(args.accel_deadband_on), "off": float(args.accel_deadband_off)},
            "a_limits": {"min": float(args.a_min), "max": float(args.a_max)},
            "weights": {
                "w_v": float(args.w_v),
                "w_os": float(args.w_os),
                "w_u": float(args.w_u),
                "w_du": float(args.w_du),
                "w_a_sat": float(args.w_a_sat),
                "w_u_sat": float(args.w_u_sat),
            },
        },
        "iterations": [],
        "best": None,
    }

    best_overall: TrialResult | None = None

    try:
        for it in range(int(args.iters)):
            elite_count = max(1, int(math.ceil(float(args.elite_frac) * float(args.samples))))
            print(f"\n=== CEM Iteration {it+1}/{int(args.iters)} (mean={mean.as_tuple()} std={std.as_tuple()}) ===")

            results: list[TrialResult] = []
            for k in range(int(args.samples)):
                g = _sample_gains(rng, mean, std, bounds)
                print(f"\n-- Trial {k+1}/{int(args.samples)} gains: speed={g.as_tuple()}")
                tr = _run_one_trial(
                    bridge=bridge,
                    gains=g,
                    duration_s=float(args.trial_duration_s),
                    print_every_s=float(args.print_every_s),
                    v_schedule=v_schedule,
                    accel_gains=accel_gains,
                    accel_deadband_on=float(args.accel_deadband_on),
                    accel_deadband_off=float(args.accel_deadband_off),
                    a_min=float(args.a_min),
                    a_max=float(args.a_max),
                    abort_abs_v_err=float(args.abort_abs_v_err),
                    abort_speed_mps=float(args.abort_speed_mps),
                    min_mode_samples=int(args.min_mode_samples),
                    w_v=float(args.w_v),
                    w_os=float(args.w_os),
                    w_u=float(args.w_u),
                    w_du=float(args.w_du),
                    w_a_sat=float(args.w_a_sat),
                    w_u_sat=float(args.w_u_sat),
                )
                results.append(tr)

                print(
                    f"Trial done: cost={tr.cost:.4f} avg|v_err|={tr.avg_abs_v_err:.4f} "
                    f"os={tr.overshoot_mps:.3f} avg|u|={tr.avg_abs_u:.4f} avg|du|={tr.avg_abs_du:.4f} "
                    f"a_sat={tr.a_sat_ratio:.3f} u_sat={tr.u_sat_ratio:.3f} "
                    f"thr_n={tr.throttle_samples} brk_n={tr.brake_samples} "
                    f"aborted={tr.aborted} {tr.abort_reason}"
                )

                if best_overall is None or tr.cost < best_overall.cost:
                    best_overall = tr

            results_sorted = sorted(results, key=lambda r: float(r.cost))
            elites = results_sorted[:elite_count]
            best_it = results_sorted[0]

            print(f"\nIteration {it+1} best: cost={best_it.cost:.4f} speed={best_it.gains.as_tuple()}")
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
                        "avg_abs_v_err": best_it.avg_abs_v_err,
                        "overshoot_mps": best_it.overshoot_mps,
                        "avg_abs_u": best_it.avg_abs_u,
                        "avg_abs_du": best_it.avg_abs_du,
                        "a_sat_ratio": best_it.a_sat_ratio,
                        "u_sat_ratio": best_it.u_sat_ratio,
                        "throttle_samples": best_it.throttle_samples,
                        "brake_samples": best_it.brake_samples,
                        "aborted": best_it.aborted,
                        "abort_reason": best_it.abort_reason,
                    },
                    "results": [
                        {
                            "gains": r.gains.__dict__,
                            "cost": r.cost,
                            "avg_abs_v_err": r.avg_abs_v_err,
                            "overshoot_mps": r.overshoot_mps,
                            "avg_abs_u": r.avg_abs_u,
                            "avg_abs_du": r.avg_abs_du,
                            "a_sat_ratio": r.a_sat_ratio,
                            "u_sat_ratio": r.u_sat_ratio,
                            "throttle_samples": r.throttle_samples,
                            "brake_samples": r.brake_samples,
                            "samples": r.samples,
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
                    "avg_abs_v_err": best_overall.avg_abs_v_err,
                    "overshoot_mps": best_overall.overshoot_mps,
                    "avg_abs_u": best_overall.avg_abs_u,
                    "avg_abs_du": best_overall.avg_abs_du,
                    "a_sat_ratio": best_overall.a_sat_ratio,
                    "u_sat_ratio": best_overall.u_sat_ratio,
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
            print(f"speed gains (kp,ki,kd,N)={best_overall.gains.as_tuple()}")
            print(
                f"avg|v_err|={best_overall.avg_abs_v_err:.4f} os={best_overall.overshoot_mps:.3f} "
                f"avg|u|={best_overall.avg_abs_u:.4f} avg|du|={best_overall.avg_abs_du:.4f} "
                f"a_sat={best_overall.a_sat_ratio:.3f} u_sat={best_overall.u_sat_ratio:.3f}"
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


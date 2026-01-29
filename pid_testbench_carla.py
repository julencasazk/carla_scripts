#!/usr/bin/env python3
"""
Single-car CARLA testbench for the full 2-loop speed controller:

  speed PID: v_sp -> a_des (clipped)
  accel PID: a_sp=a_des -> effort u in [-1,1] -> throttle/brake (hysteresis)

Only CLI args:
  --host, --port, -f

Everything else (setpoints, gains, durations, lane keep, dashcam) is hardcoded
below so you can edit the test by editing this file.
"""

from __future__ import annotations

import argparse
import time

import rclpy

from PID import PID
from following_ros_cam_accelpid import FollowingRosAccelPidBridge


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _mode_from_effort(mode: str, u_effort: float, db_on: float, db_off: float) -> str:
    if mode == "coast":
        if u_effort > db_on:
            return "throttle"
        if u_effort < -db_on:
            return "brake"
        return "coast"
    if mode == "throttle":
        return "coast" if u_effort < db_off else "throttle"
    return "coast" if u_effort > -db_off else "brake"

def _update_speed_range(current: str, speed_abs: float, v1: float, v2: float, h: float) -> tuple[str, bool]:
    prev = current
    if current == "low":
        if speed_abs > v1 + h:
            current = "mid"
    elif current == "mid":
        if speed_abs > v2 + h:
            current = "high"
        elif speed_abs < v1 - h:
            current = "low"
    else:  # high
        if speed_abs < v2 - h:
            current = "mid"
    return current, (current != prev)


# ---------------------------------------------------------------------------
# Edit these constants to change the test
# ---------------------------------------------------------------------------

# Speed-range scheduling (match Platooning.py / following_ros_cam.py)
# Ranges are 0–40 km/h, 40–80 km/h, 80–120 km/h with hysteresis h.
V1_MPS = 11.11  # 40 km/h
V2_MPS = 22.22  # 80 km/h
RANGE_HYST_MPS = 1.0

# Speed setpoints [m/s] and how long to hold each [s]
SPEED_SETPOINTS_MPS = [5.0, 10.0, 15.0, 10.0, 5.0, 4.0, 5.0, 15.0, 20.0,22.2,13.0, 15.0, 18.0, 15.0, 26.0, 24.0, 33.33, 29.0, 26.0,33.33, 22.22,0.0, 15.0, 0.0, 33.33, 0.0 ]
HOLD_TIME_S = 15.0

# Optional initial settle time at first setpoint [s]
WARMUP_S = 2.0

# Outer PID (speed -> desired accel) gain scheduling: low/mid/high
SPEED_PID_GAINS_LOW = {"kp": 1.84842, "ki": 0.96428, "kd": 0.00, "n": 14.38}
SPEED_PID_GAINS_MID = {"kp": 0.91421, "ki": 0.35309, "kd": 0.00, "n": 18.59}
SPEED_PID_GAINS_HIGH = {"kp": 0.93406, "ki": 0.37149, "kd": 0.00, "n": 20.0}

OUTER_A_MIN = -6.0
OUTER_A_MAX = 3.0

# Inner PID (desired accel -> signed effort u) gain scheduling: low/mid/high
USE_SPLIT_INNER_PID = False  # True => separate throttle/brake inner PIDs

# Single inner PID per range (u in [-1,1])
ACCEL_PID_GAINS_LOW = {"kp": 0.4638, "ki": 0.10, "kd": 0.00, "n": 14.17}

ACCEL_PID_GAINS_MID = dict(ACCEL_PID_GAINS_LOW)
ACCEL_PID_GAINS_HIGH = dict(ACCEL_PID_GAINS_LOW)

# Split inner PIDs per range:
# - throttle PID output: [0, 1]
# - brake PID output:    [-1, 0]
THROTTLE_ACCEL_PID_GAINS_LOW = {"kp": 0.1898, "ki": 1.3466, "kd": 0.0028, "n": 20.00}
THROTTLE_ACCEL_PID_GAINS_MID = {"kp": 0.2362, "ki": 1.3336, "kd": 0.0056, "n": 20.00}
THROTTLE_ACCEL_PID_GAINS_HIGH = {"kp": 0.2142, "ki": 1.5168, "kd": 0.0049, "n": 20.00}

BRAKE_ACCEL_PID_GAINS_LOW = {"kp": 0.1869, "ki": 1.8861, "kd": 0.0002, "n": 12.39}
BRAKE_ACCEL_PID_GAINS_MID = {"kp": 0.0613, "ki": 1.9646, "kd": 0.0165, "n": 13.44}
BRAKE_ACCEL_PID_GAINS_HIGH = {"kp": 0.2721, "ki": 4.4198, "kd": 0.0490, "n": 17.24}



ACCEL_DEADBAND_ON = 0.5
ACCEL_DEADBAND_OFF = 0.25

# Bridge behavior
LANE_KEEP = True
LANE_LOOKAHEAD_M = 12.0
LANE_GAIN = 1.0
TELEPORT_DIST_M = 320.0

# Dashcam
DASHCAM = True

# Real-time pacing for nicer viewing (True), or run as fast as possible (False)
PACE_WALL_TIME = True

# Print status every N seconds (sim time)
PRINT_EVERY_S = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="CARLA PID testbench (hardcoded gains + setpoints).")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("-f", type=str, default="pid_testbench.csv")
    args = parser.parse_args()

    # Bridge required fields
    args.plen = 1
    args.mcu_index = None
    args.teleport_dist = float(TELEPORT_DIST_M)
    args.no_dashcam = bool(not DASHCAM)
    args.lane_keep = bool(LANE_KEEP)
    args.lane_lookahead = float(LANE_LOOKAHEAD_M)
    args.lane_gain = float(LANE_GAIN)
    args.wall_dt = None
    # Extra debug fields to log into the bridge CSV (for plotting)
    args.extra_csv_fields = [
        "speed_range",
        "active_inner",
        "mode",
        "a_cmd",
        "a_des",
        "a_meas",
        "u_raw",
        "u_eff",
    ]

    rclpy.init()
    bridge = FollowingRosAccelPidBridge(args)

    Ts = float(bridge.Ts)
    dt_wall_s = float(bridge.dt)

    speed_pids = {
        "low": PID(
            float(SPEED_PID_GAINS_LOW["kp"]),
            float(SPEED_PID_GAINS_LOW["ki"]),
            float(SPEED_PID_GAINS_LOW["kd"]),
            float(SPEED_PID_GAINS_LOW["n"]),
            Ts,
            float(OUTER_A_MIN),
            float(OUTER_A_MAX),
            1.0,
            der_on_meas=True,
            integral_disc_method="tustin",
            derivative_disc_method="tustin",
        ),
        "mid": PID(
            float(SPEED_PID_GAINS_MID["kp"]),
            float(SPEED_PID_GAINS_MID["ki"]),
            float(SPEED_PID_GAINS_MID["kd"]),
            float(SPEED_PID_GAINS_MID["n"]),
            Ts,
            float(OUTER_A_MIN),
            float(OUTER_A_MAX),
            1.0,
            der_on_meas=True,
            integral_disc_method="tustin",
            derivative_disc_method="tustin",
        ),
        "high": PID(
            float(SPEED_PID_GAINS_HIGH["kp"]),
            float(SPEED_PID_GAINS_HIGH["ki"]),
            float(SPEED_PID_GAINS_HIGH["kd"]),
            float(SPEED_PID_GAINS_HIGH["n"]),
            Ts,
            float(OUTER_A_MIN),
            float(OUTER_A_MAX),
            1.0,
            der_on_meas=True,
            integral_disc_method="tustin",
            derivative_disc_method="tustin",
        ),
    }

    accel_pid_signed = {
        "low": PID(
            float(ACCEL_PID_GAINS_LOW["kp"]),
            float(ACCEL_PID_GAINS_LOW["ki"]),
            float(ACCEL_PID_GAINS_LOW["kd"]),
            float(ACCEL_PID_GAINS_LOW["n"]),
            Ts,
            -1.0,
            1.0,
            1.0,
            der_on_meas=True,
            integral_disc_method="tustin",
            derivative_disc_method="tustin",
        ),
        "mid": PID(
            float(ACCEL_PID_GAINS_MID["kp"]),
            float(ACCEL_PID_GAINS_MID["ki"]),
            float(ACCEL_PID_GAINS_MID["kd"]),
            float(ACCEL_PID_GAINS_MID["n"]),
            Ts,
            -1.0,
            1.0,
            1.0,
            der_on_meas=True,
            integral_disc_method="tustin",
            derivative_disc_method="tustin",
        ),
        "high": PID(
            float(ACCEL_PID_GAINS_HIGH["kp"]),
            float(ACCEL_PID_GAINS_HIGH["ki"]),
            float(ACCEL_PID_GAINS_HIGH["kd"]),
            float(ACCEL_PID_GAINS_HIGH["n"]),
            Ts,
            -1.0,
            1.0,
            1.0,
            der_on_meas=True,
            integral_disc_method="tustin",
            derivative_disc_method="tustin",
        ),
    }

    accel_pid_throttle = {
        "low": PID(
            float(THROTTLE_ACCEL_PID_GAINS_LOW["kp"]),
            float(THROTTLE_ACCEL_PID_GAINS_LOW["ki"]),
            float(THROTTLE_ACCEL_PID_GAINS_LOW["kd"]),
            float(THROTTLE_ACCEL_PID_GAINS_LOW["n"]),
            Ts,
            0.0,
            1.0,
            1.0,
            der_on_meas=True,
            integral_disc_method="tustin",
            derivative_disc_method="tustin",
        ),
        "mid": PID(
            float(THROTTLE_ACCEL_PID_GAINS_MID["kp"]),
            float(THROTTLE_ACCEL_PID_GAINS_MID["ki"]),
            float(THROTTLE_ACCEL_PID_GAINS_MID["kd"]),
            float(THROTTLE_ACCEL_PID_GAINS_MID["n"]),
            Ts,
            0.0,
            1.0,
            1.0,
            der_on_meas=True,
            integral_disc_method="tustin",
            derivative_disc_method="tustin",
        ),
        "high": PID(
            float(THROTTLE_ACCEL_PID_GAINS_HIGH["kp"]),
            float(THROTTLE_ACCEL_PID_GAINS_HIGH["ki"]),
            float(THROTTLE_ACCEL_PID_GAINS_HIGH["kd"]),
            float(THROTTLE_ACCEL_PID_GAINS_HIGH["n"]),
            Ts,
            0.0,
            1.0,
            1.0,
            der_on_meas=True,
            integral_disc_method="tustin",
            derivative_disc_method="tustin",
        ),
    }
    accel_pid_brake = {
        "low": PID(
            float(BRAKE_ACCEL_PID_GAINS_LOW["kp"]),
            float(BRAKE_ACCEL_PID_GAINS_LOW["ki"]),
            float(BRAKE_ACCEL_PID_GAINS_LOW["kd"]),
            float(BRAKE_ACCEL_PID_GAINS_LOW["n"]),
            Ts,
            -1.0,
            0.0,
            1.0,
            der_on_meas=True,
            integral_disc_method="tustin",
            derivative_disc_method="tustin",
        ),
        "mid": PID(
            float(BRAKE_ACCEL_PID_GAINS_MID["kp"]),
            float(BRAKE_ACCEL_PID_GAINS_MID["ki"]),
            float(BRAKE_ACCEL_PID_GAINS_MID["kd"]),
            float(BRAKE_ACCEL_PID_GAINS_MID["n"]),
            Ts,
            -1.0,
            0.0,
            1.0,
            der_on_meas=True,
            integral_disc_method="tustin",
            derivative_disc_method="tustin",
        ),
        "high": PID(
            float(BRAKE_ACCEL_PID_GAINS_HIGH["kp"]),
            float(BRAKE_ACCEL_PID_GAINS_HIGH["ki"]),
            float(BRAKE_ACCEL_PID_GAINS_HIGH["kd"]),
            float(BRAKE_ACCEL_PID_GAINS_HIGH["n"]),
            Ts,
            -1.0,
            0.0,
            1.0,
            der_on_meas=True,
            integral_disc_method="tustin",
            derivative_disc_method="tustin",
        ),
    }

    ego_name = bridge.ros_names[bridge.ego_index]
    mode = "coast"
    speed_range = "low"

    segments: list[tuple[float, float]] = []
    if float(WARMUP_S) > 0.0 and len(SPEED_SETPOINTS_MPS) > 0:
        segments.append((float(WARMUP_S), float(SPEED_SETPOINTS_MPS[0])))
    for sp in SPEED_SETPOINTS_MPS:
        segments.append((float(HOLD_TIME_S), float(sp)))

    total_time_s = float(sum(d for d, _ in segments))
    print(f"Test duration: {total_time_s:.1f}s (sim time), Ts={Ts}")
    print(f"Speed setpoints: {SPEED_SETPOINTS_MPS} hold={HOLD_TIME_S}s warmup={WARMUP_S}s")
    print(
        "Speed PID scheduling (low/mid/high)\n"
        f"  low : (kp,ki,kd,N)=({SPEED_PID_GAINS_LOW['kp']},{SPEED_PID_GAINS_LOW['ki']},{SPEED_PID_GAINS_LOW['kd']},{SPEED_PID_GAINS_LOW['n']})\n"
        f"  mid : (kp,ki,kd,N)=({SPEED_PID_GAINS_MID['kp']},{SPEED_PID_GAINS_MID['ki']},{SPEED_PID_GAINS_MID['kd']},{SPEED_PID_GAINS_MID['n']})\n"
        f"  high: (kp,ki,kd,N)=({SPEED_PID_GAINS_HIGH['kp']},{SPEED_PID_GAINS_HIGH['ki']},{SPEED_PID_GAINS_HIGH['kd']},{SPEED_PID_GAINS_HIGH['n']})\n"
        f"  thresholds v1={V1_MPS:.2f} v2={V2_MPS:.2f} h={RANGE_HYST_MPS:.2f} a_lim=[{OUTER_A_MIN},{OUTER_A_MAX}]"
    )
    if USE_SPLIT_INNER_PID:
        print(
            "Inner PID scheduling: SPLIT (throttle + brake) per speed range\n"
            f"  deadband(on/off)=({ACCEL_DEADBAND_ON},{ACCEL_DEADBAND_OFF})"
        )
    else:
        print(
            "Inner PID scheduling: SINGLE per speed range "
            f"deadband(on/off)=({ACCEL_DEADBAND_ON},{ACCEL_DEADBAND_OFF})"
        )

    bridge.reset_test()
    bridge.apply_controls()
    # Prime state so first control isn't based on zeros.
    bridge.step_simulation()

    seg_idx = 0
    seg_start_sim = float(bridge.sim_time)
    seg_end_sim = float(seg_start_sim + float(segments[0][0]))
    v_sp = float(segments[0][1])
    bridge.lead_speed_setpoints = [v_sp]
    bridge.lead_sp_idx = 0

    next_print_sim = float(seg_start_sim + float(PRINT_EVERY_S)) if float(PRINT_EVERY_S) > 0 else float("inf")
    tick_due_wall_s = time.perf_counter()

    try:
        v = float(bridge._last_speeds_long[bridge.ego_index]) if getattr(bridge, "_last_speeds_long", None) else 0.0
        a_meas = float(bridge._last_accels_long[bridge.ego_index]) if bridge._last_accels_long else 0.0

        while float(bridge.sim_time) < float(seg_start_sim + total_time_s):
            if PACE_WALL_TIME:
                while True:
                    now_s = time.perf_counter()
                    if now_s >= tick_due_wall_s:
                        break
                    time.sleep(min(0.0005, max(0.0, tick_due_wall_s - now_s)))

            if float(bridge.sim_time) >= float(seg_end_sim) and (seg_idx + 1) < len(segments):
                seg_idx += 1
                seg_end_sim = float(seg_end_sim + float(segments[seg_idx][0]))
                v_sp = float(segments[seg_idx][1])
                bridge.lead_speed_setpoints = [v_sp]
                bridge.lead_sp_idx = 0

            speed_range, changed = _update_speed_range(
                speed_range, abs(float(v)), float(V1_MPS), float(V2_MPS), float(RANGE_HYST_MPS)
            )
            if changed:
                try:
                    speed_pids[speed_range].reset()
                except Exception:
                    pass
                try:
                    accel_pid_signed[speed_range].reset()
                    accel_pid_throttle[speed_range].reset()
                    accel_pid_brake[speed_range].reset()
                except Exception:
                    pass

            a_cmd = float(speed_pids[speed_range].step(float(v_sp), float(v)))
            a_des = _clip(float(a_cmd), float(OUTER_A_MIN), float(OUTER_A_MAX))

            active_inner = "single"
            if USE_SPLIT_INNER_PID:
                if float(a_des) >= 0.0:
                    active_inner = "thr"
                    u_raw = float(accel_pid_throttle[speed_range].step(float(a_des), float(a_meas)))
                    u_effort = _clip(float(u_raw), 0.0, 1.0)
                else:
                    active_inner = "brk"
                    u_raw = float(accel_pid_brake[speed_range].step(float(a_des), float(a_meas)))
                    u_effort = _clip(float(u_raw), -1.0, 0.0)
            else:
                u_raw = float(accel_pid_signed[speed_range].step(float(a_des), float(a_meas)))
                u_effort = _clip(float(u_raw), -1.0, 1.0)
            mode = _mode_from_effort(mode, float(u_effort), float(ACCEL_DEADBAND_ON), float(ACCEL_DEADBAND_OFF))

            throttle_cmd = _clip(u_effort, 0.0, 1.0) if mode == "throttle" else 0.0
            brake_cmd = _clip(-u_effort, 0.0, 1.0) if mode == "brake" else 0.0

            bridge._last_cmd[ego_name]["throttle"] = float(throttle_cmd)
            bridge._last_cmd[ego_name]["brake"] = float(brake_cmd)
            bridge.set_extra_csv_values(
                {
                    "speed_range": speed_range,
                    "active_inner": active_inner,
                    "mode": mode,
                    "a_cmd": f"{a_cmd:.6f}",
                    "a_des": f"{a_des:.6f}",
                    "a_meas": f"{a_meas:.6f}",
                    "u_raw": f"{u_raw:.6f}",
                    "u_eff": f"{u_effort:.6f}",
                }
            )
            bridge.apply_controls()
            bridge.step_simulation()
            v = float(bridge._last_speeds_long[bridge.ego_index]) if getattr(bridge, "_last_speeds_long", None) else 0.0
            a_meas = float(bridge._last_accels_long[bridge.ego_index]) if bridge._last_accels_long else 0.0

            if float(bridge.sim_time) >= float(next_print_sim):
                v_err = float(v_sp - v)
                print(
                    f"t={bridge.sim_time:7.2f}s seg={seg_idx+1:02d}/{len(segments)} "
                    f"v_sp={v_sp:5.2f} v={v:5.2f} v_err={v_err:6.3f} "
                    f"a_cmd={a_cmd:6.3f} a_des={a_des:6.3f} a={a_meas:6.3f} "
                    f"u_raw={u_raw:+6.3f} u_eff={u_effort:+6.3f} "
                    f"range={speed_range:4s} inner={active_inner:6s} mode={mode:7s} "
                    f"thr={throttle_cmd:5.3f} brk={brake_cmd:5.3f}"
                )
                next_print_sim = float(bridge.sim_time) + float(PRINT_EVERY_S)

            tick_due_wall_s += dt_wall_s

    except KeyboardInterrupt:
        pass
    finally:
        bridge.shutdown()
        try:
            bridge.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()

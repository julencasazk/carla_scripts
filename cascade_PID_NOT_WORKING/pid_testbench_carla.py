#!/usr/bin/env python3
"""
Single-car CARLA testbench for the full cascaded controller:
outer speed PID -> desired accel -> inner accel PID -> throttle/brake.

What it does
  - Runs a fixed speed setpoint schedule with the 2-loop controller.
  - Supports single inner PID or split throttle/brake inner PIDs.
  - Uses lane-keeping and teleport resets to keep the vehicle on road.

Outputs
  - CSV logs with speed, accel, control effort, and debug fields.

Run (example)
  python3 cascade_PID_NOT_WORKING/pid_testbench_carla.py     --host localhost --port 2000 -f out.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
from typing import Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor

from lib.PID import PID
from lib.following_ros_cam_accelpid import FollowingRosAccelPidBridge


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
# Ranges are 0-40 km/h, 40-80 km/h, 80-120 km/h with hysteresis h.
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


def _with_suffix(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(str(path))
    if not ext:
        ext = ".csv"
    return f"{root}{suffix}{ext}"


def _configure_bridge_args(args: argparse.Namespace, *, mcu_index: Optional[int]) -> None:
    # Bridge required fields
    args.plen = 1
    args.mcu_index = mcu_index
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


def _build_segments() -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    if float(WARMUP_S) > 0.0 and len(SPEED_SETPOINTS_MPS) > 0:
        segments.append((float(WARMUP_S), float(SPEED_SETPOINTS_MPS[0])))
    for sp in SPEED_SETPOINTS_MPS:
        segments.append((float(HOLD_TIME_S), float(sp)))
    return segments


def _read_bridge_speed_csv(path: str, vehicle_name: str) -> tuple[list[float], list[float]]:
    """
    Read a FollowingRosAccelPidBridge CSV and return (time_s, speed_long_mps) series.
    Falls back to speed magnitude if longitudinal speed isn't present.
    """
    time_s: list[float] = []
    speeds: list[float] = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return time_s, speeds

        try:
            time_idx = header.index("time_s")
        except ValueError:
            return time_s, speeds

        speed_long_key = f"speed_long_{vehicle_name}"
        speed_key = f"speed_{vehicle_name}"
        speed_idx = None
        if speed_long_key in header:
            speed_idx = header.index(speed_long_key)
        elif speed_key in header:
            speed_idx = header.index(speed_key)
        else:
            return time_s, speeds

        for row in reader:
            if len(row) <= max(time_idx, speed_idx):
                continue
            try:
                t = float(row[time_idx])
                v = float(row[speed_idx])
            except Exception:
                continue
            time_s.append(t)
            speeds.append(v)

    return time_s, speeds


def _read_csv_table(path: str) -> tuple[list[str], list[list[str]]]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        rows = [row for row in reader]
    return header, rows


def _setpoint_for_time(t_s: float, segments: list[tuple[float, float]]) -> float:
    acc = 0.0
    for dur_s, sp in segments:
        acc += float(dur_s)
        if float(t_s) < float(acc):
            return float(sp)
    return float(segments[-1][1]) if segments else 0.0


def _write_merged_csv(
    out_path: str,
    *,
    carla_csv: str,
    mcu_csv: str,
    vehicle_name: str,
) -> None:
    segments = _build_segments()

    h_carla, r_carla = _read_csv_table(carla_csv)
    h_mcu, r_mcu = _read_csv_table(mcu_csv)

    if not h_carla or not h_mcu:
        raise RuntimeError("Cannot merge: one of the run CSVs has no header.")

    try:
        t_idx_carla = h_carla.index("time_s")
    except ValueError as e:
        raise RuntimeError("Cannot merge: CARLA run CSV missing 'time_s' column.") from e

    try:
        t_idx_mcu = h_mcu.index("time_s")
    except ValueError as e:
        raise RuntimeError("Cannot merge: MCU run CSV missing 'time_s' column.") from e

    # Use the union of both headers (excluding time_s) so we keep *everything* we logged.
    cols: list[str] = []
    seen = set()
    for col in list(h_carla) + list(h_mcu):
        if col == "time_s":
            continue
        if col in seen:
            continue
        seen.add(col)
        cols.append(col)

    n = min(len(r_carla), len(r_mcu))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["time_s", "speed_setpoint_mps"]
        for col in cols:
            header.append(f"{col}_carla")
        for col in cols:
            header.append(f"{col}_mcu")
        w.writerow(header)

        idx_carla = {k: i for i, k in enumerate(h_carla)}
        idx_mcu = {k: i for i, k in enumerate(h_mcu)}

        for i in range(n):
            row_carla = r_carla[i]
            row_mcu = r_mcu[i]

            t_raw = row_carla[t_idx_carla] if len(row_carla) > t_idx_carla else ""
            try:
                t = float(t_raw)
            except Exception:
                # Fall back to MCU time if CARLA time is missing/unparseable.
                t_mcu_raw = row_mcu[t_idx_mcu] if len(row_mcu) > t_idx_mcu else ""
                t = float(t_mcu_raw) if t_mcu_raw else float(i) * 0.01

            merged = [f"{t:.4f}", f"{_setpoint_for_time(t, segments):.4f}"]

            for col in cols:
                j = idx_carla.get(col)
                merged.append(row_carla[j] if (j is not None and len(row_carla) > j) else "")

            for col in cols:
                j = idx_mcu.get(col)
                merged.append(row_mcu[j] if (j is not None and len(row_mcu) > j) else "")

            w.writerow(
                merged
            )


def _run_python_pid_test(args: argparse.Namespace) -> None:
    _configure_bridge_args(args, mcu_index=None)

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


def _run_mcu_reference_test(args: argparse.Namespace) -> None:
    # For reference: MCU is veh_0.
    _configure_bridge_args(args, mcu_index=0)

    rclpy.init()
    bridge = FollowingRosAccelPidBridge(args)

    Ts = float(bridge.Ts)
    dt_wall_s = float(bridge.dt)
    segments = _build_segments()
    total_time_s = float(sum(d for d, _ in segments))

    print(f"MCU reference run: duration={total_time_s:.1f}s (sim time), Ts={Ts}")
    print("Expecting MCU to publish /veh_0/command/throttle and /veh_0/command/brake.")

    executor = SingleThreadedExecutor()
    executor.add_node(bridge)

    def drain_executor(max_callbacks: int, max_wall_s: float) -> None:
        start = time.perf_counter()
        for _ in range(int(max_callbacks)):
            executor.spin_once(timeout_sec=0.0)
            if (time.perf_counter() - start) >= max_wall_s:
                break

    state_delivery_budget_s = min(0.002, dt_wall_s * 0.25) if dt_wall_s > 0.0 else 0.002

    bridge.reset_test()
    bridge.apply_controls()
    bridge.step_simulation()

    seg_idx = 0
    seg_start_sim = float(bridge.sim_time)
    seg_end_sim = float(seg_start_sim + float(segments[0][0]))
    v_sp = float(segments[0][1])
    bridge.lead_speed_setpoints = [v_sp]
    bridge.lead_sp_idx = 0

    next_print_sim = float(seg_start_sim + float(PRINT_EVERY_S)) if float(PRINT_EVERY_S) > 0 else float("inf")
    tick_due_wall_s = time.perf_counter()

    ego_name = bridge.ros_names[bridge.ego_index]

    try:
        while float(bridge.sim_time) < float(seg_start_sim + total_time_s):
            if PACE_WALL_TIME:
                while True:
                    now_s = time.perf_counter()
                    remaining_s = float(tick_due_wall_s - now_s)
                    if remaining_s <= 0.0:
                        break
                    executor.spin_once(timeout_sec=remaining_s)
            else:
                drain_executor(max_callbacks=500, max_wall_s=state_delivery_budget_s)

            tick_start_wall_s = time.perf_counter()

            if float(bridge.sim_time) >= float(seg_end_sim) and (seg_idx + 1) < len(segments):
                seg_idx += 1
                seg_end_sim = float(seg_end_sim + float(segments[seg_idx][0]))
                v_sp = float(segments[seg_idx][1])
                bridge.lead_speed_setpoints = [v_sp]
                bridge.lead_sp_idx = 0

            # Apply the latest received MCU controls, then advance the sim.
            bridge.apply_controls()
            bridge.step_simulation()

            # Give time for MCU to consume state/setpoint and publish next cmd.
            drain_executor(max_callbacks=500, max_wall_s=state_delivery_budget_s)

            if float(bridge.sim_time) >= float(next_print_sim):
                v = float(bridge._last_speeds_long[bridge.ego_index]) if getattr(bridge, "_last_speeds_long", None) else 0.0
                cmd = bridge._last_cmd.get(ego_name, {"throttle": 0.0, "brake": 0.0})
                print(
                    f"t={bridge.sim_time:7.2f}s seg={seg_idx+1:02d}/{len(segments)} "
                    f"v_sp={v_sp:5.2f} v={v:5.2f} "
                    f"thr={float(cmd.get('throttle', 0.0)):5.3f} brk={float(cmd.get('brake', 0.0)):5.3f}"
                )
                next_print_sim = float(bridge.sim_time) + float(PRINT_EVERY_S)

            tick_due_wall_s = float(tick_start_wall_s + dt_wall_s) if PACE_WALL_TIME else float(tick_start_wall_s)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.remove_node(bridge)
        except Exception:
            pass
        bridge.shutdown()
        try:
            bridge.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="CARLA PID testbench (hardcoded gains + setpoints).")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("-f", type=str, default="pid_testbench.csv")
    parser.add_argument(
        "--rerun-with-mcu",
        action="store_true",
        help="After the Python-PID run, rerun the same setpoint schedule with veh_0 controlled by the MCU (topics like following_ros_cam.py).",
    )
    args = parser.parse_args()

    if bool(getattr(args, "rerun_with_mcu", False)):
        vehicle_name = "veh_0"
        # Use temp files for the two runs, then merge into the requested output path.
        carla_tmp = tempfile.NamedTemporaryFile(prefix="pid_testbench_carla_", suffix=".csv", delete=False)
        mcu_tmp = tempfile.NamedTemporaryFile(prefix="pid_testbench_mcu_", suffix=".csv", delete=False)
        carla_tmp.close()
        mcu_tmp.close()

        carla_args = argparse.Namespace(**vars(args))
        carla_args.f = str(carla_tmp.name)
        _run_python_pid_test(carla_args)

        mcu_args = argparse.Namespace(**vars(args))
        mcu_args.f = str(mcu_tmp.name)
        _run_mcu_reference_test(mcu_args)

        _write_merged_csv(
            str(args.f),
            carla_csv=str(carla_tmp.name),
            mcu_csv=str(mcu_tmp.name),
            vehicle_name=vehicle_name,
        )

        try:
            os.remove(str(carla_tmp.name))
        except Exception:
            pass
        try:
            os.remove(str(mcu_tmp.name))
        except Exception:
            pass
    else:
        _run_python_pid_test(argparse.Namespace(**vars(args)))


if __name__ == "__main__":
    main()

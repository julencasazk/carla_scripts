#!/usr/bin/env python3
"""
Runs the same speed-setpoint test in CARLA and offline plant models,
writing both to a single CSV for overlay plotting.

Outputs
  - Combined CSV log for CARLA vs model comparison.

Run (example)
  python3 tuning/carla_step_tests/compare_speedpid_carla_vs_plants.py
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

try:
    import carla  # type: ignore
except ModuleNotFoundError:  # Allows running offline mode without CARLA.
    carla = None  # type: ignore

from lib.PID import PID

SpeedRange = Literal["low", "mid", "high"]


# -----------------------------------------------------------------------------
# Hardcoded test configuration (edit these for thesis figures)
# -----------------------------------------------------------------------------

TS = 0.01

# Initial settle time at setpoint 0.0 (both CARLA + model), before the step schedule.
PRE_ROLL_S = 30.0

# Gain scheduling thresholds (m/s) and hysteresis
V1_MPS = 11.11
V2_MPS = 22.22
HYST_MPS = 1.0

# Speed low-pass filter (matches following_ros_cam.py / step_simulation)
SPEED_LPF_ALPHA = 0.4

# CARLA scene
TOWN = "Town04"
SPAWN_INDEX = 115
VEHICLE_FILTER = "vehicle.tesla.model3"

# Teleport the CARLA vehicle back to the spawn if it travels too far.
TELEPORT_DISTANCE_M = 250.0

# Physics substepping (keeps control tick at TS, improves integration)
USE_SUBSTEPPING = True
SUBSTEP_DT = 0.001
MAX_SUBSTEPS = 10

# Optional: disable rendering for faster-than-wallclock CARLA
NO_RENDERING = False

# Speed PID gains per range: (kp, ki, kd, N)
PID_GAINS: Dict[SpeedRange, Tuple[float, float, float, float]] = {
    # Matches the ACTIVE gains in following_ros_test.py (slow_pid/mid_pid/fast_pid).
    "low": (0.1089194, 0.04906409, 0.0, 7.71822214),
    "mid": (0.11494122, 0.0489494, 0.0, 5.0),
    "high": (0.19021246, 0.15704399, 0.0, 12.70886655),
}

# Operating points used by the incremental plant tests (u0, v0) per range.
# These are the center points of each speed range.
OP_POINTS: Dict[SpeedRange, Tuple[float, float]] = {
    "low": (0.35, 7.77777),
    "mid": (0.55, 16.666666),
    "high": (0.68, 27.7777),
}


@dataclass(frozen=True)
class Segment:
    speed_range: SpeedRange
    initial_speed_mps: float
    steps: List[Tuple[float, float]]  # (duration_s, setpoint_mps)


# Test schedule:
# - 4 plateaus per speed range (15s each)
# - includes up/down within each range (no cross-range setpoints)
SCHEDULE: List[Segment] = [
    Segment(
        speed_range="low",
        initial_speed_mps=6.0,
        steps=[(15.0, 6.0), (15.0, 10.0), (15.0, 7.0), (15.0, 9.0)],
    ),
    Segment(
        speed_range="mid",
        initial_speed_mps=14.0,
        steps=[(15.0, 14.0), (15.0, 21.0), (15.0, 16.0), (15.0, 20.0)],
    ),
    Segment(
        speed_range="high",
        initial_speed_mps=25.0,
        steps=[(15.0, 25.0), (15.0, 32.0), (15.0, 27.0), (15.0, 31.0)],
    ),
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _speed_abs(vehicle: "carla.Vehicle") -> float:
    v = vehicle.get_velocity()
    return float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))

def _try_set_target_velocity(actor: "carla.Actor", v: "carla.Vector3D") -> None:
    """
    CARLA API differences:
      - Some versions have Actor.set_target_velocity / set_target_angular_velocity
      - Some examples use enable_constant_velocity on Vehicle
      - Some older code used set_velocity (not available in newer CARLA)
    """
    if hasattr(actor, "set_target_velocity"):
        actor.set_target_velocity(v)  # type: ignore[attr-defined]
        return
    if hasattr(actor, "enable_constant_velocity"):
        actor.enable_constant_velocity(v)  # type: ignore[attr-defined]
        return


def _try_set_target_angular_velocity(actor: "carla.Actor", w: "carla.Vector3D") -> None:
    if hasattr(actor, "set_target_angular_velocity"):
        actor.set_target_angular_velocity(w)  # type: ignore[attr-defined]
        return
    if hasattr(actor, "disable_constant_velocity"):
        # best-effort: clear constant velocity mode if it exists
        try:
            actor.disable_constant_velocity()  # type: ignore[attr-defined]
        except Exception:
            pass


def _update_speed_range(current: SpeedRange, speed_abs: float) -> SpeedRange:
    if current == "low":
        if speed_abs > V1_MPS + HYST_MPS:
            return "mid"
        return "low"
    if current == "mid":
        if speed_abs > V2_MPS + HYST_MPS:
            return "high"
        if speed_abs < V1_MPS - HYST_MPS:
            return "low"
        return "mid"
    # high
    if speed_abs < V2_MPS - HYST_MPS:
        return "mid"
    return "high"


def _pid_for_range(rng: SpeedRange, u_min: float, u_max: float) -> PID:
    kp, ki, kd, N = PID_GAINS[rng]
    return PID(
        kp,
        ki,
        kd,
        N,
        TS,
        u_min,
        u_max,
        kb_aw=1.0,
        der_on_meas=True,
        derivative_disc_method="tustin",
        integral_disc_method="tustin",
        anti_windup="clamping"
    )


def _build_incremental_plant_ss(rng: SpeedRange):
    import numpy as np  # type: ignore
    import control as ctl  # type: ignore

    if rng == "low":
        # 2-pole solution (legacy/2nd-order fit)
        G0 = ctl.TransferFunction([0, 8.5777, 54.0770], [1.0000, 7.0533, 2.1718])
    elif rng == "mid":
        # 2-pole solution (legacy/2nd-order fit)
        G0 = ctl.TransferFunction([0, 12.3082, 67.2242], [1.0000, 5.2957, 1.0404])
    else:
        # 2-pole solution (legacy/2nd-order fit)
        G0 = ctl.TransferFunction([0, 15.0224, 20.4896], [1.0000, 1.3026, 0.1813])

    Gd = ctl.c2d(G0, TS, method="tustin")
    Ad, Bd, Cd, Dd = ctl.ssdata(ctl.ss(Gd))
    Ad = np.asarray(Ad)
    Bd = np.asarray(Bd).reshape(-1)
    Cd = np.asarray(Cd)
    Dd = np.asarray(Dd).reshape(-1)
    return Gd, Ad, Bd, Cd, Dd


def _iter_schedule_steps() -> List[Tuple[int, SpeedRange, float]]:
    """
    Returns a flat list of (n_steps, segment_range, setpoint_mps) entries.
    """
    flat: List[Tuple[int, SpeedRange, float]] = []
    # Pre-roll at setpoint 0.0 so the system can settle to rest.
    if PRE_ROLL_S > 0.0:
        n0 = int(round(float(PRE_ROLL_S) / TS))
        flat.append((max(1, n0), "low", 0.0))
    for seg in SCHEDULE:
        for dur_s, sp in seg.steps:
            n = int(round(float(dur_s) / TS))
            flat.append((max(1, n), seg.speed_range, float(sp)))
    return flat


# -----------------------------------------------------------------------------
# CARLA run
# -----------------------------------------------------------------------------

def run_carla_collect(host: str, port: int) -> Dict[str, List]:
    if carla is None:
        raise RuntimeError("CARLA Python API (module `carla`) not available.")

    client = carla.Client(host, int(port))
    client.set_timeout(20.0)

    client.load_world(TOWN)
    time.sleep(1.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(TS)
    settings.no_rendering_mode = bool(NO_RENDERING)
    if USE_SUBSTEPPING:
        settings.substepping = True
        settings.max_substep_delta_time = float(SUBSTEP_DT)
        settings.max_substeps = int(MAX_SUBSTEPS)
    world.apply_settings(settings)

    actors = []
    try:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points in map.")

        sp_idx = SPAWN_INDEX if 0 <= SPAWN_INDEX < len(spawn_points) else 0
        spawn_tf = spawn_points[sp_idx]
        spawn_tf.location.z += 0.25

        bp = bp_lib.filter(VEHICLE_FILTER)[0]
        vehicle = world.try_spawn_actor(bp, spawn_tf)
        if vehicle is None:
            # fall back to first available spawn
            for sp in spawn_points:
                sp2 = carla.Transform(sp.location, sp.rotation)
                sp2.location.z += 0.25
                vehicle = world.try_spawn_actor(bp, sp2)
                if vehicle is not None:
                    spawn_tf = sp2
                    break
        if vehicle is None:
            raise RuntimeError("Failed to spawn vehicle.")

        vehicle.set_autopilot(False)
        actors.append(vehicle)

        # Set an initial speed once (do not reset/teleport during the test).
        fwd0 = vehicle.get_transform().get_forward_vector()
        v_init = float(SCHEDULE[0].initial_speed_mps) if SCHEDULE else 0.0
        _try_set_target_velocity(vehicle, carla.Vector3D(fwd0.x * v_init, fwd0.y * v_init, fwd0.z * v_init))

        # Build PIDs per range (throttle-only controller)
        u_min, u_max = 0.0, 1.0
        pid = _pid_for_range("low", u_min=u_min, u_max=u_max)
        u_prev = 0.0

        test_t = 0.0

        schedule = _iter_schedule_steps()

        current_range: SpeedRange = "low"

        # Precompute a list of setpoints per tick, with segment range
        per_tick: List[Tuple[SpeedRange, float]] = []
        for n, rng, sp in schedule:
            per_tick.extend([(rng, sp)] * n)

        out: Dict[str, List] = {
            "time_s": [],
            "segment_range": [],
            "active_range": [],
            "setpoint_mps": [],
            "speed_mps": [],
            "throttle": [],
            "brake": [],
            "u": [],
        }

        # Distance tracking for periodic teleport (keep speed).
        dist_accum = 0.0
        last_loc = vehicle.get_transform().location
        speed_filt: float | None = None

        for k in range(len(per_tick)):
            desired_seg_range, sp = per_tick[k]

            world.tick()
            test_t += TS

            loc = vehicle.get_transform().location
            dx = float(loc.x - last_loc.x)
            dy = float(loc.y - last_loc.y)
            dz = float(loc.z - last_loc.z)
            dist_accum += math.sqrt(dx * dx + dy * dy + dz * dz)
            last_loc = loc

            if TELEPORT_DISTANCE_M > 0.0 and dist_accum >= float(TELEPORT_DISTANCE_M):
                # Teleport back but preserve instantaneous speed magnitude.
                v_now = _speed_abs(vehicle)
                vehicle.set_transform(spawn_tf)
                fwd = vehicle.get_transform().get_forward_vector()
                _try_set_target_velocity(vehicle, carla.Vector3D(fwd.x * v_now, fwd.y * v_now, fwd.z * v_now))
                dist_accum = 0.0
                last_loc = vehicle.get_transform().location

            v_raw = _speed_abs(vehicle)
            if speed_filt is None:
                speed_filt = float(v_raw)
            else:
                speed_filt = float(SPEED_LPF_ALPHA) * float(v_raw) + (1.0 - float(SPEED_LPF_ALPHA)) * float(speed_filt)

            v_meas = float(speed_filt)

            new_range = _update_speed_range(current_range, v_meas)
            if new_range != current_range:
                current_range = new_range
                kp, ki, kd, N = PID_GAINS[current_range]
                pid.set_gains_bumpless(kp, ki, kd, N, float(sp), float(v_meas), float(u_prev))

            u = float(pid.step(float(sp), float(v_meas)))
            throttle = float(max(0.0, min(1.0, u)))
            brake = 0.0
            u_prev = float(throttle)

            vehicle.apply_control(
                carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0, hand_brake=False, reverse=False)
            )

            out["time_s"].append(float(test_t))
            out["segment_range"].append(str(desired_seg_range))
            out["active_range"].append(str(current_range))
            out["setpoint_mps"].append(float(sp))
            out["speed_mps"].append(float(v_meas))
            out["throttle"].append(float(throttle))
            out["brake"].append(float(brake))
            out["u"].append(float(u))

        return out
    finally:
        for a in actors[::-1]:
            try:
                a.destroy()
            except Exception:
                pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Offline plant run
# -----------------------------------------------------------------------------

def run_plants_collect() -> Dict[str, List]:
    import numpy as np  # type: ignore

    # Build plants and pids (throttle-only; u in [0,1])
    u_min, u_max = 0.0, 1.0
    pid = _pid_for_range("low", u_min=u_min, u_max=u_max)
    u_prev = 0.0

    plants = {}
    for rng in ("low", "mid", "high"):
        _Gd, Ad, Bd, Cd, Dd = _build_incremental_plant_ss(rng)
        plants[rng] = (Ad, Bd, Cd, Dd)

    test_t = 0.0

    # Flatten schedule and replay continuously (no resets between segments).
    schedule = _iter_schedule_steps()

    # Expand to per tick list so we can align exactly with CARLA output length.
    per_tick: List[Tuple[SpeedRange, float]] = []
    for n, rng, sp in schedule:
        per_tick.extend([(rng, sp)] * n)

    # Initialize near the first segment's initial speed.
    current_range: SpeedRange = "low"
    x = np.zeros(plants[current_range][0].shape[0])
    u0, v0 = OP_POINTS[current_range]
    y = float(SCHEDULE[0].initial_speed_mps) if SCHEDULE else float(v0)
    dv = float(y - v0)
    y_filt: float | None = None

    out: Dict[str, List] = {
        "time_s": [],
        "segment_range": [],
        "active_range": [],
        "setpoint_mps": [],
        "speed_mps": [],
        "throttle": [],
        "brake": [],
        "u": [],
    }

    for k, (desired_seg_range, sp) in enumerate(per_tick):
        # Filter speed measurement for scheduling + PID feedback (match CARLA script behavior).
        if y_filt is None:
            y_filt = float(y)
        else:
            y_filt = float(SPEED_LPF_ALPHA) * float(y) + (1.0 - float(SPEED_LPF_ALPHA)) * float(y_filt)

        # Range selection follows the simulated vehicle speed (like gain scheduling in CARLA),
        # not the desired segment label.
        next_range = _update_speed_range(current_range, abs(float(y_filt)))
        if next_range != current_range:
            current_range = next_range
            # Switch plant operating point and keep absolute speed continuous:
            u0, v0 = OP_POINTS[current_range]
            Ad, Bd, Cd, Dd = plants[current_range]
            x = np.zeros(Ad.shape[0])
            dv = float(y - v0)
            kp, ki, kd, N = PID_GAINS[current_range]
            pid.set_gains_bumpless(kp, ki, kd, N, float(sp), float(y_meas), float(u_prev))
        else:
            Ad, Bd, Cd, Dd = plants[current_range]

        y = float(v0 + dv)
        # Use filtered speed as measurement into the PID.
        y_meas = float(y_filt) if y_filt is not None else float(y)

        u = float(pid.step(float(sp), float(y_meas)))
        u = float(max(u_min, min(u_max, u)))
        du = float(u - u0)
        u_prev = float(u)

        x = Ad @ x + Bd * du
        dv = float((Cd @ x + Dd * du).item())

        test_t += TS

        out["time_s"].append(float(test_t))
        out["segment_range"].append(str(desired_seg_range))
        out["active_range"].append(str(current_range))
        out["setpoint_mps"].append(float(sp))
        out["speed_mps"].append(float(y_meas))
        out["throttle"].append(float(u))
        out["brake"].append(float(0.0))
        out["u"].append(float(u))

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("-o", "--out", default="compare_speedpid_carla_vs_plants.csv")
    args = ap.parse_args()

    fieldnames = [
        "time_s",
        "segment_range",
        "active_range_carla",
        "active_range_model",
        "setpoint_carla_mps",
        "setpoint_model_mps",
        "speed_carla_mps",
        "speed_model_mps",
        "throttle_carla",
        "throttle_model",
        "brake_carla",
        "brake_model",
        "u_carla",
        "u_model",
    ]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print("=== Running CARLA test (first) ===")
        print(f"  host={args.host} port={args.port} town={TOWN} spawn={SPAWN_INDEX} Ts={TS}")
        carla_out = run_carla_collect(args.host, args.port)

        print("=== Running offline plant test (second) ===")
        model_out = run_plants_collect()

        n = min(len(carla_out["time_s"]), len(model_out["time_s"]))
        if len(carla_out["time_s"]) != len(model_out["time_s"]):
            print(
                "WARNING: CARLA and model produced different lengths: "
                f"carla={len(carla_out['time_s'])} model={len(model_out['time_s'])}. "
                f"Writing min length={n}."
            )

        for i in range(n):
            # Align by sample index; both use TS increments from 0 with no pre-teleport settling logged.
            writer.writerow(
                {
                    "time_s": f"{float(carla_out['time_s'][i]):.4f}",
                    "segment_range": str(carla_out["segment_range"][i]),
                    "active_range_carla": str(carla_out["active_range"][i]),
                    "active_range_model": str(model_out["active_range"][i]),
                    "setpoint_carla_mps": f"{float(carla_out['setpoint_mps'][i]):.4f}",
                    "setpoint_model_mps": f"{float(model_out['setpoint_mps'][i]):.4f}",
                    "speed_carla_mps": f"{float(carla_out['speed_mps'][i]):.4f}",
                    "speed_model_mps": f"{float(model_out['speed_mps'][i]):.4f}",
                    "throttle_carla": f"{float(carla_out['throttle'][i]):.6f}",
                    "throttle_model": f"{float(model_out['throttle'][i]):.6f}",
                    "brake_carla": f"{float(carla_out['brake'][i]):.6f}",
                    "brake_model": f"{float(model_out['brake'][i]):.6f}",
                    "u_carla": f"{float(carla_out['u'][i]):.6f}",
                    "u_model": f"{float(model_out['u'][i]):.6f}",
                }
            )

        f.flush()

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()

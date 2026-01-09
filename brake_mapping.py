import argparse
import csv
import json
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import carla
import cv2
import numpy as np

from PID import PID


@dataclass
class PulseResult:
	mode: str  # "throttle" | "brake"
	cmd: float
	target_speed: float
	bin_name: str
	avg_long_accel: float
	avg_speed: float


def closest_spawn_point(spawn_points, loc: carla.Location) -> carla.Transform:
	"""Return the spawn point (Transform) whose location is closest to loc."""
	best_sp = None
	best_dist2 = float("inf")
	for sp in spawn_points:
		dx = sp.location.x - loc.x
		dy = sp.location.y - loc.y
		dz = sp.location.z - loc.z
		d2 = dx * dx + dy * dy + dz * dz
		if d2 < best_dist2:
			best_dist2 = d2
			best_sp = sp
	return best_sp


class TeleportManager:
	def __init__(self, spawn_points, base_tf: carla.Transform, teleport_dist_m: float) -> None:
		self._spawn_points = spawn_points
		self._base_tf = base_tf
		self._teleport_dist_m = float(teleport_dist_m)
		self._dist_since_tp_m = 0.0
		self._last_loc: Optional[carla.Location] = None

	def enabled(self) -> bool:
		return self._teleport_dist_m > 0.0

	def after_tick(self, vehicle: carla.Vehicle) -> None:
		if not self.enabled():
			return

		loc = vehicle.get_transform().location
		if self._last_loc is None:
			self._last_loc = loc
			return

		self._dist_since_tp_m += math.dist(
			(loc.x, loc.y, loc.z),
			(self._last_loc.x, self._last_loc.y, self._last_loc.z),
		)
		self._last_loc = loc

		if self._dist_since_tp_m < self._teleport_dist_m:
			return

		print(
			f"[brake_mapping] Teleport triggered after {self._dist_since_tp_m:.2f} m "
			f"(threshold {self._teleport_dist_m:.2f} m)"
		)

		# Snap z to the nearest spawn point to avoid ground penetration.
		sp = closest_spawn_point(self._spawn_points, self._base_tf.location)
		target = carla.Transform(
			location=carla.Location(
				x=self._base_tf.location.x,
				y=self._base_tf.location.y,
				z=sp.location.z + 0.1,
			),
			rotation=self._base_tf.rotation,
		)
		vehicle.set_transform(target)

		self._dist_since_tp_m = 0.0
		self._last_loc = vehicle.get_transform().location


def _vehicle_speed_mps(vehicle: carla.Vehicle) -> float:
	vel = vehicle.get_velocity()
	return float(math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))


def _longitudinal_accel_mps2(vehicle: carla.Vehicle) -> float:
	"""Project world acceleration onto the vehicle's forward axis."""
	accel = vehicle.get_acceleration()
	forward = vehicle.get_transform().get_forward_vector()
	return float(accel.x * forward.x + accel.y * forward.y + accel.z * forward.z)


def _accel_from_speed(prev_speed_mps: float, speed_mps: float, dt: float) -> float:
	"""Finite-difference acceleration from speed magnitude."""
	if dt <= 0.0:
		return 0.0
	return float((speed_mps - prev_speed_mps) / dt)


def _select_pid(speed_mps: float, v1: float, v2: float, slow: PID, mid: PID, fast: PID) -> PID:
	if speed_mps >= v2:
		return fast
	if speed_mps >= v1:
		return mid
	return slow


class DashcamViewer:
	def __init__(
		self,
		world: carla.World,
		attach_to: carla.Actor,
		image_size: Tuple[int, int] = (640, 480),
		fov: int = 90,
		transform: Optional[carla.Transform] = None,
		window_name: str = "Brake/throttle mapping",
	) -> None:
		self._world = world
		self._attach_to = attach_to
		self._sensor = None
		self._queue: "queue.Queue[Optional[Tuple[bytes, int, int]]]" = queue.Queue(maxsize=1)
		self._stop = threading.Event()
		self._thread: Optional[threading.Thread] = None
		self._window_name = window_name

		bp_lib = world.get_blueprint_library()
		cam_bp = bp_lib.find("sensor.camera.rgb")
		cam_bp.set_attribute("image_size_x", str(int(image_size[0])))
		cam_bp.set_attribute("image_size_y", str(int(image_size[1])))
		cam_bp.set_attribute("fov", str(int(fov)))

		if transform is None:
			transform = carla.Transform(
				carla.Location(x=0.0, y=10.0, z=10.0),
				carla.Rotation(yaw=-20.0, pitch=-35),
			)

		self._sensor = world.spawn_actor(cam_bp, transform, attach_to=attach_to)
		self._sensor.listen(self._on_image)

		# Viewer thread (non-blocking for simulation loop)
		self._thread = threading.Thread(target=self._viewer_loop, daemon=True)
		self._thread.start()

	def _on_image(self, image: carla.Image) -> None:
		payload = (bytes(image.raw_data), image.width, image.height)
		try:
			while True:
				self._queue.get_nowait()
		except queue.Empty:
			pass
		try:
			self._queue.put_nowait(payload)
		except queue.Full:
			pass

	def _viewer_loop(self) -> None:
		time.sleep(0.5)
		try:
			cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
		except Exception:
			return

		try:
			while not self._stop.is_set():
				try:
					item = self._queue.get(timeout=0.05)
				except queue.Empty:
					continue
				if item is None:
					continue

				raw, width, height = item
				try:
					arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
					rgb = arr[:, :, :3]
					cv2.imshow(self._window_name, rgb)
					if (cv2.waitKey(1) & 0xFF) == ord("q"):
						self._stop.set()
						break
				except Exception:
					# Ignore occasional decode/render issues.
					pass
		finally:
			try:
				cv2.destroyAllWindows()
			except Exception:
				pass

	def should_stop(self) -> bool:
		return self._stop.is_set()

	def close(self) -> None:
		try:
			self._stop.set()
		except Exception:
			pass
		if self._sensor is not None:
			try:
				self._sensor.stop()
			except Exception:
				pass
			try:
				self._sensor.destroy()
			except Exception:
				pass
		if self._thread is not None:
			try:
				self._thread.join(timeout=1.0)
			except Exception:
				pass


def _apply_control(vehicle: carla.Vehicle, throttle: float, brake: float) -> None:
	vehicle.apply_control(
		carla.VehicleControl(
			throttle=float(max(0.0, min(1.0, throttle))),
			brake=float(max(0.0, min(1.0, brake))),
			steer=0.0,
			hand_brake=False,
			reverse=False,
		)
	)


def _tick(
	world: carla.World,
	wall_dt_s: Optional[float],
	tick_due_wall_s: float,
	teleport: Optional[TeleportManager] = None,
	vehicle: Optional[carla.Vehicle] = None,
) -> float:
	"""Tick synchronous CARLA; optionally pace wall-clock."""
	if wall_dt_s is not None:
		while True:
			now = time.perf_counter()
			remaining = tick_due_wall_s - now
			if remaining <= 0.0:
				break
			time.sleep(min(remaining, 0.002))
	world.tick()
	if teleport is not None and vehicle is not None:
		try:
			teleport.after_tick(vehicle)
		except Exception:
			pass
	return time.perf_counter()


def _hold_speed_with_pid(
	world: carla.World,
	vehicle: carla.Vehicle,
	target_speed: float,
	dt: float,
	slow_pid: PID,
	mid_pid: PID,
	fast_pid: PID,
	v1: float,
	v2: float,
	settle_band: float,
	settle_time_s: float,
	max_time_s: float,
	pid_brake_gain: float,
	wall_dt_s: Optional[float],
	tick_due_wall_s: float,
	teleport: Optional[TeleportManager],
) -> float:
	"""Run until speed is within band for settle_time_s (or max_time_s)."""
	within_s = 0.0
	elapsed = 0.0
	while elapsed < max_time_s:
		speed = _vehicle_speed_mps(vehicle)
		pid = _select_pid(speed, v1, v2, slow_pid, mid_pid, fast_pid)
		u = float(pid.step(target_speed, speed))  # [-1, 1]

		if u >= 0.0:
			throttle, brake = u, 0.0
		else:
			# Optional light brake for holding; can be set to 0 to disable.
			throttle, brake = 0.0, min(1.0, pid_brake_gain * abs(u))

		_apply_control(vehicle, throttle, brake)
		tick_due_wall_s = _tick(world, wall_dt_s, tick_due_wall_s, teleport=teleport, vehicle=vehicle)

		speed_after = _vehicle_speed_mps(vehicle)
		if abs(speed_after - target_speed) <= settle_band:
			within_s += dt
			if within_s >= settle_time_s:
				break
		else:
			within_s = 0.0

		elapsed += dt
		tick_due_wall_s += (wall_dt_s if wall_dt_s is not None else 0.0)

	# coast for one tick to reduce bias from brake/throttle pulses
	_apply_control(vehicle, 0.0, 0.0)
	tick_due_wall_s = _tick(world, wall_dt_s, tick_due_wall_s, teleport=teleport, vehicle=vehicle)
	return tick_due_wall_s


def _run_pulse_and_measure(
	world: carla.World,
	vehicle: carla.Vehicle,
	mode: str,
	cmd: float,
	pulse_time_s: float,
	avg_window_s: float,
	dt: float,
	wall_dt_s: Optional[float],
	tick_due_wall_s: float,
	teleport: Optional[TeleportManager],
	accel_alpha: float,
	csv_writer: csv.writer,
	bin_name: str,
	target_speed: float,
) -> Tuple[float, float, float]:
	"""Return (avg_long_accel, avg_speed, tick_due_wall_s)."""
	samples_needed = max(1, int(round(pulse_time_s / dt)))
	avg_needed = max(1, int(round(min(avg_window_s, pulse_time_s) / dt)))
	tail_acc: List[float] = []
	tail_speed: List[float] = []

	# Acceleration estimator state (from actor acceleration)
	acc_filt = 0.0
	alpha = float(max(0.0, min(1.0, accel_alpha)))

	for k in range(samples_needed):
		if mode == "throttle":
			_apply_control(vehicle, cmd, 0.0)
		elif mode == "brake":
			_apply_control(vehicle, 0.0, cmd)
		else:
			raise ValueError(f"Unknown mode: {mode}")

		tick_due_wall_s = _tick(world, wall_dt_s, tick_due_wall_s, teleport=teleport, vehicle=vehicle)

		speed = _vehicle_speed_mps(vehicle)
		acc_est = _longitudinal_accel_mps2(vehicle)
		acc_filt = alpha * acc_est + (1.0 - alpha) * acc_filt
		long_acc = float(acc_filt)
		sim_t = float(world.get_snapshot().timestamp.elapsed_seconds)
		csv_writer.writerow([
			f"{sim_t:.6f}",
			bin_name,
			f"{target_speed:.4f}",
			mode,
			f"{cmd:.4f}",
			f"{speed:.6f}",
			f"{long_acc:.6f}",
		])

		# Keep tail window
		tail_acc.append(long_acc)
		tail_speed.append(speed)
		if len(tail_acc) > avg_needed:
			tail_acc.pop(0)
			tail_speed.pop(0)

		tick_due_wall_s += (wall_dt_s if wall_dt_s is not None else 0.0)

	avg_long_acc = float(sum(tail_acc) / len(tail_acc))
	avg_speed = float(sum(tail_speed) / len(tail_speed))

	# Release actuators and tick a little
	_apply_control(vehicle, 0.0, 0.0)
	for _ in range(max(1, int(round(0.25 / dt)))):
		tick_due_wall_s = _tick(world, wall_dt_s, tick_due_wall_s, teleport=teleport, vehicle=vehicle)
		tick_due_wall_s += (wall_dt_s if wall_dt_s is not None else 0.0)

	return avg_long_acc, avg_speed, tick_due_wall_s


def main() -> None:
	parser = argparse.ArgumentParser(description="CARLA accel/brake/throttle mapping helper")
	parser.add_argument("--host", type=str, default="localhost")
	parser.add_argument("--port", type=int, default=2000)
	parser.add_argument("--town", type=str, default="Town04")
	parser.add_argument("--dt", type=float, default=0.01, help="World tick dt (fixed_delta_seconds) [s]")
	parser.add_argument(
		"--substep-dt",
		type=float,
		default=0.001,
		help="Physics substep delta time [s] (enable substepping when > 0)",
	)
	parser.add_argument(
		"--max-substeps",
		type=int,
		default=None,
		help="Max physics substeps per world tick (default: ceil(dt/substep_dt))",
	)
	parser.add_argument("--wall-dt", type=float, default=None, help="Optional wall-clock pacing per tick [s]")
	parser.add_argument("--spawn-index", type=int, default=125)
	parser.add_argument("--spawn-y-offset", type=float, default=-60.0)
	parser.add_argument(
		"--teleport-dist",
		type=float,
		default=250.0,
		help="Teleport the vehicle back to spawn after traveling this distance [m]; set <=0 to disable.",
	)

	parser.add_argument("--out-prefix", type=str, default="accel_brake_map")

	parser.add_argument("--v1", type=float, default=11.11)
	parser.add_argument("--v2", type=float, default=22.22)
	parser.add_argument(
		"--targets",
		type=float,
		nargs=3,
		default=[8.0, 16.0, 28.0],
		help="Target speeds (m/s) for low/mid/high bins",
	)

	parser.add_argument("--settle-band", type=float, default=0.15)
	parser.add_argument("--settle-time", type=float, default=2.0)
	parser.add_argument("--hold-timeout", type=float, default=20.0)
	parser.add_argument(
		"--pid-brake-gain",
		type=float,
		default=0.0,
		help="Optional light brake gain during PID speed-hold; keep 0 for throttle-only hold",
	)

	parser.add_argument("--pulse-time", type=float, default=2.0)
	parser.add_argument("--avg-window", type=float, default=0.75)
	parser.add_argument(
		"--accel-alpha",
		type=float,
		default=0.35,
		help="Low-pass alpha for accel from CARLA actor (0=no update, 1=no filtering).",
	)

	# Note: throttle mapping intentionally omitted by default; this script focuses on deceleration->brake mapping.
	parser.add_argument(
		"--brake-cmds",
		type=float,
		nargs="+",
		default=[0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
	)

	args = parser.parse_args()

	print(f"[brake_mapping] Connecting to CARLA at {args.host}:{int(args.port)} ...")
	client = carla.Client(args.host, int(args.port))
	client.set_timeout(10.0)

	print(f"[brake_mapping] Loading world: {args.town} ...")
	client.load_world(args.town)
	time.sleep(2.0)
	world = client.get_world()

	original_settings = world.get_settings()
	settings = world.get_settings()
	settings.synchronous_mode = True
	settings.fixed_delta_seconds = float(args.dt)
	# Physics substepping: more accurate integration without changing control tick.
	if float(args.substep_dt) > 0.0:
		settings.substepping = True
		settings.max_substep_delta_time = float(args.substep_dt)
		if args.max_substeps is None:
			settings.max_substeps = int(math.ceil(float(args.dt) / float(args.substep_dt)))
		else:
			settings.max_substeps = int(args.max_substeps)
		if settings.max_substeps < 1:
			settings.max_substeps = 1
	else:
		settings.substepping = False
		# CARLA may reject 0.0 here; set sane defaults.
		settings.max_substep_delta_time = 0.01
		settings.max_substeps = 1
	settings.no_rendering_mode = False
	world.apply_settings(settings)

	print(
		f"[brake_mapping] Synchronous mode ON, dt={float(args.dt):.4f}s, "
		f"substepping={bool(settings.substepping)}, substep_dt={float(settings.max_substep_delta_time):.4f}s, "
		f"max_substeps={int(settings.max_substeps)}"
	)

	bp_lib = world.get_blueprint_library()
	vehicle_bp = bp_lib.find("vehicle.tesla.model3")

	spawn_points = world.get_map().get_spawn_points()
	sp = spawn_points[int(args.spawn_index) % len(spawn_points)]
	sp.location.y += float(args.spawn_y_offset)
	print(f"[brake_mapping] Spawning vehicle at spawn_index={int(args.spawn_index)} y_offset={float(args.spawn_y_offset):.2f} ...")
	vehicle = world.spawn_actor(vehicle_bp, sp)
	vehicle.set_autopilot(False)

	teleport = TeleportManager(spawn_points=spawn_points, base_tf=sp, teleport_dist_m=float(args.teleport_dist))
	if teleport.enabled():
		print(f"[brake_mapping] Teleport enabled: every {float(args.teleport_dist):.2f} m")
	else:
		print("[brake_mapping] Teleport disabled")

	print(f"[brake_mapping] Vehicle spawned: id={vehicle.id} type={vehicle.type_id}")

	# Camera like following_ros_test.py
	dashcam = None
	try:
		dashcam = DashcamViewer(world, attach_to=vehicle)
		print("[brake_mapping] Dashcam started (press 'q' in window to stop).")
	except Exception:
		dashcam = None
		print("[brake_mapping] Dashcam failed to start; continuing headless.")

	# PID scheduling gains (copied from following_ros_test.py)
	Ts = float(args.dt)
	kb_aw = 1.0
	u_min, u_max = -1.0, 1.0
	slow_pid = PID(0.43127789, 0.43676547, 0.0, 15.0, Ts, u_min, u_max, kb_aw, der_on_meas=True)
	mid_pid = PID(0.11675119, 0.085938,   0.0, 14.90530836, Ts, u_min, u_max, kb_aw, der_on_meas=True)
	fast_pid = PID(0.13408096, 0.07281374, 0.0, 12.16810135, Ts, u_min, u_max, kb_aw, der_on_meas=True)

	out_csv = f"{args.out_prefix}_samples.csv"
	out_json = f"{args.out_prefix}_summary.json"

	print(f"[brake_mapping] Writing samples to: {out_csv}")
	print(f"[brake_mapping] Writing summary to: {out_json}")

	summary: Dict[str, Dict[str, object]] = {}
	results: List[PulseResult] = []

	tick_due_wall_s = time.perf_counter()
	wall_dt_s = float(args.wall_dt) if args.wall_dt is not None else None

	# Prime
	_apply_control(vehicle, 0.0, 0.0)
	tick_due_wall_s = _tick(world, wall_dt_s, tick_due_wall_s, teleport=teleport, vehicle=vehicle)

	try:
		with open(out_csv, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow([
				"sim_time_s",
				"bin",
				"target_speed_mps",
				"mode",
				"cmd",
				"speed_mps",
				"actor_long_accel_mps2",
			])

			bins = [
				("low", float(args.targets[0])),
				("mid", float(args.targets[1])),
				("high", float(args.targets[2])),
			]

			for bin_name, target_speed in bins:
				if dashcam is not None and dashcam.should_stop():
					break

				print(f"[brake_mapping] ==== Bin '{bin_name}' target_speed={target_speed:.3f} m/s ====")
				print("[brake_mapping] Holding speed with scheduled PID...")

				# Reset PID internal states between bins.
				slow_pid.reset()
				mid_pid.reset()
				fast_pid.reset()

				tick_due_wall_s = _hold_speed_with_pid(
					world=world,
					vehicle=vehicle,
					target_speed=target_speed,
					dt=float(args.dt),
					slow_pid=slow_pid,
					mid_pid=mid_pid,
					fast_pid=fast_pid,
					v1=float(args.v1),
					v2=float(args.v2),
					settle_band=float(args.settle_band),
					settle_time_s=float(args.settle_time),
					max_time_s=float(args.hold_timeout),
					pid_brake_gain=float(args.pid_brake_gain),
					wall_dt_s=wall_dt_s,
					tick_due_wall_s=tick_due_wall_s,
					teleport=teleport,
				)

				print("[brake_mapping] Speed hold done. Starting open-loop brake pulses...")

				# Brake mapping pulses (open-loop)
				for cmd in [float(x) for x in args.brake_cmds]:
					if dashcam is not None and dashcam.should_stop():
						break
					print(f"[brake_mapping] Brake pulse: cmd={cmd:.3f} for {float(args.pulse_time):.2f}s")
					tick_due_wall_s = _hold_speed_with_pid(
						world,
						vehicle,
						target_speed,
						float(args.dt),
						slow_pid,
						mid_pid,
						fast_pid,
						float(args.v1),
						float(args.v2),
						float(args.settle_band),
						float(args.settle_time),
						float(args.hold_timeout),
						float(args.pid_brake_gain),
						wall_dt_s,
						tick_due_wall_s,
						teleport,
					)

					avg_acc, avg_speed, tick_due_wall_s = _run_pulse_and_measure(
						world=world,
						vehicle=vehicle,
						mode="brake",
						cmd=cmd,
						pulse_time_s=float(args.pulse_time),
						avg_window_s=float(args.avg_window),
						dt=float(args.dt),
						wall_dt_s=wall_dt_s,
						tick_due_wall_s=tick_due_wall_s,
						teleport=teleport,
						accel_alpha=float(args.accel_alpha),
						csv_writer=writer,
						bin_name=bin_name,
						target_speed=target_speed,
					)
					results.append(
						PulseResult(
							mode="brake",
							cmd=cmd,
							target_speed=target_speed,
							bin_name=bin_name,
							avg_long_accel=avg_acc,
							avg_speed=avg_speed,
						)
					)

					print(
						f"[brake_mapping]   -> avg_long_accel={avg_acc:+.3f} m/s^2 (decel={-avg_acc:+.3f}) at avg_speed={avg_speed:.3f} m/s"
					)

					# Explicitly recover: release brake and return to target speed before next cmd.
					print("[brake_mapping] Recovering to target speed...")
					slow_pid.reset()
					mid_pid.reset()
					fast_pid.reset()
					tick_due_wall_s = _hold_speed_with_pid(
						world,
						vehicle,
						target_speed,
						float(args.dt),
						slow_pid,
						mid_pid,
						fast_pid,
						float(args.v1),
						float(args.v2),
						float(args.settle_band),
						float(args.settle_time),
						float(args.hold_timeout),
						float(args.pid_brake_gain),
						wall_dt_s,
						tick_due_wall_s,
						teleport,
					)

			# Build JSON summary
			for bin_name, target_speed in bins:
				brake_rows = [
					{
						"cmd": r.cmd,
						"decel_mps2": float(-r.avg_long_accel),
						"accel_mps2": r.avg_long_accel,
						"avg_speed_mps": r.avg_speed,
					}
					for r in results
					if r.bin_name == bin_name and r.mode == "brake"
				]
				summary[bin_name] = {
					"target_speed_mps": float(target_speed),
					"brake_map": brake_rows,
				}

		with open(out_json, "w") as jf:
			json.dump(
				{
					"meta": {
						"town": args.town,
						"dt": float(args.dt),
						"substepping": bool(settings.substepping),
						"max_substep_delta_time": float(settings.max_substep_delta_time),
						"max_substeps": int(settings.max_substeps),
						"accel_source": "actor_longitudinal",
						"v1": float(args.v1),
						"v2": float(args.v2),
						"pulse_time_s": float(args.pulse_time),
						"avg_window_s": float(args.avg_window),
						"brake_cmds": [float(x) for x in args.brake_cmds],
					},
					"bins": summary,
				},
				jf,
				indent=2,
			)

		print("[brake_mapping] Mapping complete.")
	finally:
		print("[brake_mapping] Stopping vehicle and cleaning up...")
		try:
			_apply_control(vehicle, 0.0, 1.0)
			for _ in range(int(round(1.0 / float(args.dt)))):
				tick_due_wall_s = _tick(world, wall_dt_s, tick_due_wall_s, teleport=teleport, vehicle=vehicle)
				tick_due_wall_s += (wall_dt_s if wall_dt_s is not None else 0.0)
		except Exception:
			pass

		if dashcam is not None:
			dashcam.close()

		try:
			vehicle.destroy()
		except Exception:
			pass

		try:
			world.apply_settings(original_settings)
		except Exception:
			pass

		print("[brake_mapping] Done.")


if __name__ == "__main__":
	main()


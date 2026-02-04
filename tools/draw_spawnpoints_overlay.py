#!/usr/bin/env python3
"""
Draws CARLA spawnpoint indices in the world for visual reference.

Outputs
  - On-screen debug text/arrows in the CARLA viewer.

Run (example)
  python3 tools/draw_spawnpoints_overlay.py --map Town04 --lifetime 600
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay CARLA spawn point indices and exit.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", type=str, default="Town04", help="Map name to load (e.g. Town04).")
    parser.add_argument(
        "--lifetime",
        type=float,
        default=3600.0,
        help="How long the overlay should remain visible [s].",
    )
    parser.add_argument(
        "--text-z",
        type=float,
        default=1.5,
        help="Text height above the spawn point [m].",
    )
    parser.add_argument(
        "--draw-arrows",
        action="store_true",
        help="Also draw an arrow showing spawn orientation.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Draw every Nth spawn point (useful to reduce clutter).",
    )
    args = parser.parse_args()

    try:
        import carla  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Failed to import the CARLA Python API (module 'carla').\n"
            "Run this from a shell where CARLA's PythonAPI is on PYTHONPATH.\n"
            f"Original error: {e}"
        )

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(10.0)

    client.load_world(str(args.map))
    time.sleep(1.0)
    world = client.get_world()

    # A couple of ticks helps ensure the map is fully loaded and the overlay shows up reliably.
    try:
        world.wait_for_tick(2.0)
        world.wait_for_tick(2.0)
    except Exception:
        pass

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points found on this map.")
        return

    try:
        green = carla.Color(0, 255, 0)
        cyan = carla.Color(0, 255, 255)
    except Exception:
        green = None
        cyan = None

    lifetime_s = float(args.lifetime)
    every = max(1, int(args.every))

    drawn = 0
    for i, tf in enumerate(spawn_points):
        if (i % every) != 0:
            continue

        loc = carla.Location(tf.location.x, tf.location.y, tf.location.z + float(args.text_z))
        world.debug.draw_string(
            loc,
            str(i),
            draw_shadow=True,
            color=green,
            life_time=lifetime_s,
            persistent_lines=(lifetime_s >= 3600.0),
        )
        drawn += 1

        if bool(args.draw_arrows):
            try:
                world.debug.draw_arrow(
                    tf.location + carla.Location(z=0.2),
                    tf.location + tf.get_forward_vector() * 2.0 + carla.Location(z=0.2),
                    thickness=0.1,
                    arrow_size=0.2,
                    color=cyan,
                    life_time=lifetime_s,
                    persistent_lines=(lifetime_s >= 3600.0),
                )
            except Exception:
                pass

    print(f"Loaded {args.map} and drew {drawn}/{len(spawn_points)} spawn point labels for {lifetime_s:.1f}s.")
    print("You can now use the CARLA viewer free camera to inspect spawn point numbers.")


if __name__ == "__main__":
    main()

"""Live viewer for a GEECS/LabVIEW IMAQ camera.

Subscribes to a camera device's TCP image stream, decodes each incoming NI IMAQ
"Flatten Image to String" payload, and displays the newest frame in a matplotlib
window that refreshes in a loop.

Usage
-----
    python extras/scripts/live_camera_viewer.py UC_ALineEBeam3
    python extras/scripts/live_camera_viewer.py UC_ALineEBeam3 --experiment Undulator
    python extras/scripts/live_camera_viewer.py --list-cameras

    # pin the display scale and refresh slower
    python extras/scripts/live_camera_viewer.py UC_VisaEBeam1 --scale fixed \
        --vmin 0 --vmax 4095 --fps 5

Interactive keys
----------------
    space   pause / resume the display (the subscription keeps running)
    s       save the displayed frame as .npy + .png in --save-dir
    q       quit

Notes
-----
Frames arrive on the device's TCP listener thread and are handed to the GUI
through a one-slot buffer: if the camera outruns the display, intermediate
frames are dropped rather than queued, so what you see is always the newest
frame. Dropped frames are reported in the window title.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from geecs_python_api.controls.devices.camera import latest_image, on_image
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interface.geecs_database import (
    GeecsDatabase,
    load_config,
)

LISTENER_NAME = "live_viewer"


class LatestFrame:
    """One-slot, thread-safe hand-off of the newest frame to the GUI thread.

    The producer (TCP listener thread) overwrites the slot on every frame; the
    consumer (matplotlib timer) takes it. Overwriting an untaken frame counts as
    a drop, which is the desired behaviour for a live view — never queue up a
    backlog of stale frames.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self.received = 0
        self.dropped = 0
        self.last_arrival: Optional[float] = None

    def put(self, image: np.ndarray) -> None:
        """Store the newest frame, dropping any frame not yet displayed."""
        with self._lock:
            if self._frame is not None:
                self.dropped += 1
            self._frame = image
            self.received += 1
            self.last_arrival = time.monotonic()

    def take(self) -> Optional[np.ndarray]:
        """Return the pending frame and clear the slot, or None if empty."""
        with self._lock:
            frame, self._frame = self._frame, None
            return frame


def resolve_experiment(explicit: Optional[str]) -> str:
    """Return the experiment name from the CLI flag or ~/.config config.ini."""
    if explicit:
        return explicit
    config = load_config()
    if config is None:
        sys.exit(
            "No ~/.config/geecs_python_api/config.ini found — pass --experiment "
            "explicitly, or copy a known-good config onto this machine."
        )
    try:
        return config.get("Experiment", "expt")
    except Exception:
        sys.exit(
            "config.ini has no [Experiment] expt entry — pass --experiment explicitly."
        )


def list_cameras(exp_info: dict) -> None:
    """Print every device in the experiment that exposes an ``image`` variable."""
    devices: dict = exp_info.get("devices", {})
    cameras = sorted(
        name for name, variables in devices.items() if "image" in (variables or {})
    )
    if not cameras:
        print("No devices with an 'image' variable found in this experiment.")
        return
    print(f"{len(cameras)} camera-like device(s):")
    for name in cameras:
        print(f"  {name}")


def compute_clim(
    image: np.ndarray, scale: str, vmin: Optional[float], vmax: Optional[float]
) -> tuple[float, float]:
    """Return the (vmin, vmax) display limits for one frame."""
    if scale == "fixed":
        lo = vmin if vmin is not None else float(image.min())
        hi = vmax if vmax is not None else float(image.max())
    elif scale == "dtype":
        info = np.iinfo(image.dtype) if np.issubdtype(image.dtype, np.integer) else None
        lo, hi = (0.0, float(info.max)) if info else (0.0, 1.0)
    elif scale == "percentile":
        lo, hi = (float(v) for v in np.percentile(image, (1.0, 99.5)))
    else:  # "frame"
        lo, hi = float(image.min()), float(image.max())
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def sensor_extent(image: np.ndarray) -> tuple[float, float, float, float]:
    """Return an imshow extent in full-resolution sensor pixels."""
    height, width = image.shape[:2]
    return (-0.5, width - 0.5, height - 0.5, -0.5)


def save_frame(image: np.ndarray, device_name: str, save_dir: Path, cmap: str,
               clim: tuple[float, float]) -> Path:
    """Write the frame to ``save_dir`` as both .npy (raw) and .png (as displayed)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{device_name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    npy_path = save_dir / f"{stem}.npy"
    np.save(npy_path, image)
    plt.imsave(
        save_dir / f"{stem}.png", image, cmap=cmap, vmin=clim[0], vmax=clim[1]
    )
    return npy_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Live display of a GEECS IMAQ camera stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "device", nargs="?", help="GEECS database device name, e.g. UC_ALineEBeam3"
    )
    parser.add_argument(
        "--experiment", help="Experiment name (default: [Experiment] expt in config.ini)"
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List devices exposing an 'image' variable and exit",
    )
    parser.add_argument(
        "--fps", type=float, default=10.0, help="Display refresh rate (Hz)"
    )
    parser.add_argument(
        "--scale",
        choices=("frame", "percentile", "dtype", "fixed"),
        default="frame",
        help="Colour scaling: per-frame min/max, 1-99.5%% percentiles, full dtype "
        "range, or fixed --vmin/--vmax",
    )
    parser.add_argument("--vmin", type=float, help="Lower display limit (--scale fixed)")
    parser.add_argument("--vmax", type=float, help="Upper display limit (--scale fixed)")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Display every Nth pixel in each axis (large sensors)",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Poll device.state instead of using the push callback (fallback for "
        "cameras whose image variable is aliased)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path.cwd() / "camera_grabs",
        help="Where the 's' key writes frames",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds without a frame before printing a diagnostic",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the viewer."""
    args = build_parser().parse_args(argv)
    # Default WARNING: at INFO the mysql-connector auth-plugin chatter buries
    # this script's own output. --verbose opts into the full stream.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    experiment = resolve_experiment(args.experiment)
    print(f"Loading experiment info for {experiment!r} ...")
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(experiment)

    if args.list_cameras:
        list_cameras(GeecsDevice.exp_info)
        return 0
    if not args.device:
        build_parser().error("a device name is required (or use --list-cameras)")

    camera = GeecsDevice(args.device)
    if not camera.is_valid():
        camera.close()
        print(
            f"Device {args.device!r} has no usable database entry in {experiment!r}. "
            "Run with --list-cameras to see valid names."
        )
        return 2
    if "image" not in (camera.dev_vars or {}):
        print(
            f"Warning: {args.device!r} has no 'image' variable in the database — "
            "this may not be a camera."
        )

    buffer = LatestFrame()
    state = {
        "paused": False,
        "image": None,
        "clim": (0.0, 1.0),
        "displayed": 0,
        "shape": None,
        "last_shot": None,
    }
    frame_times: deque[float] = deque(maxlen=30)

    listener: Optional[str] = None
    try:
        if not camera.subscribe_var_values(["image"]):
            camera.close()
            print(f"Failed to subscribe to {args.device!r} — is the device running?")
            return 2
        if not args.poll:
            listener = on_image(camera, buffer.put, name=LISTENER_NAME)

        print("Waiting for the first frame ... (space=pause, s=save, q=quit)")
        first = wait_for_first_frame(camera, buffer, args)
        if first is None:
            return 2

        run_viewer(camera, buffer, state, frame_times, first, args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if listener is not None:
            camera.unregister_update_listener(listener)
        camera.unsubscribe_var_values()
        camera.close()
        print("Disconnected.")
    return 0


def wait_for_first_frame(
    camera: GeecsDevice, buffer: LatestFrame, args: argparse.Namespace
) -> Optional[np.ndarray]:
    """Block until a frame arrives, or explain what is missing and give up."""
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        frame = latest_image(camera) if args.poll else buffer.take()
        if frame is not None:
            return frame
        time.sleep(0.05)

    print(
        f"No frame within {args.timeout:.0f} s. Device state keys: "
        f"{sorted(str(k) for k in camera.state)}"
    )
    print(
        "If an image-like key is present under a different name, the camera's "
        "image variable is aliased; decode it directly with "
        "geecs_data_utils.io.decode_imaq_image_string(camera.state[<key>])."
    )
    return None


def run_viewer(
    camera: GeecsDevice,
    buffer: LatestFrame,
    state: dict,
    frame_times: deque,
    first: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Open the matplotlib window and refresh it until it is closed."""
    step = max(1, args.downsample)
    shown = first[::step, ::step]
    clim = compute_clim(shown, args.scale, args.vmin, args.vmax)
    state["image"], state["clim"], state["shape"] = first, clim, shown.shape

    fig, ax = plt.subplots(figsize=(8, 6.5))
    if fig.canvas.manager is not None:  # None under headless/Agg backends
        fig.canvas.manager.set_window_title(f"{args.device} — live")
    handle = ax.imshow(
        shown,
        cmap=args.cmap,
        vmin=clim[0],
        vmax=clim[1],
        interpolation="nearest",
        # Axes always read *sensor* pixels, never display pixels, so coordinates
        # stay meaningful (and comparable to saved data) when --downsample > 1.
        extent=sensor_extent(first),
    )
    fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    title = ax.set_title("")

    def on_key(event) -> None:
        if event.key == " ":
            state["paused"] = not state["paused"]
            print("Paused." if state["paused"] else "Resumed.")
        elif event.key == "s" and state["image"] is not None:
            path = save_frame(
                state["image"], args.device, args.save_dir, args.cmap, state["clim"]
            )
            print(f"Saved {path}")

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame_index: int):
        if args.poll:
            # Polling re-reads the same cached frame until a new shot lands;
            # dedupe on shot number so the fps meter counts real frames only.
            shot = camera.state.get("shot number")
            stale = shot is not None and shot == state["last_shot"]
            frame = None if stale else latest_image(camera)
            state["last_shot"] = shot
        else:
            frame = buffer.take()

        if frame is None or state["paused"]:
            return (handle, title)

        frame_times.append(time.monotonic())
        state["image"] = frame
        state["displayed"] += 1
        shown = frame[::step, ::step]
        handle.set_data(shown)
        if shown.shape != state["shape"]:  # ROI changed mid-run
            extent = sensor_extent(frame)
            handle.set_extent(extent)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            state["shape"] = shown.shape

        clim = compute_clim(shown, args.scale, args.vmin, args.vmax)
        handle.set_clim(*clim)
        state["clim"] = clim

        fps = 0.0
        if len(frame_times) > 1:
            span = frame_times[-1] - frame_times[0]
            fps = (len(frame_times) - 1) / span if span > 0 else 0.0

        # Drop counts only mean something in push mode; polling never sees the
        # frames it skips, so reporting "0 dropped" there would be a lie.
        counters = f"{state['displayed']} shown"
        if not args.poll:
            counters += f" of {buffer.received} recv ({buffer.dropped} dropped)"

        title.set_text(
            f"{args.device}   shot {camera.state.get('shot number')}   "
            f"{frame.shape[1]}x{frame.shape[0]} {frame.dtype}\n"
            f"min {frame.min():g}  max {frame.max():g}  mean {frame.mean():.1f}   "
            f"{fps:.1f} fps   {counters}"
        )
        return (handle, title)

    interval_ms = max(1.0, 1000.0 / max(args.fps, 0.1))
    animation = FuncAnimation(  # noqa: F841 — must stay referenced while shown
        fig, update, interval=interval_ms, blit=False, cache_frame_data=False
    )
    plt.show()


if __name__ == "__main__":
    raise SystemExit(main())

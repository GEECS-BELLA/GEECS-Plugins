"""Record device streams as NDJSON fixtures: one ``DeviceFrame`` per line, as a listener receives them.

The web repository's fixture server replays these files, so a page can be
judged with nothing else running (``Planning/epics_estate.md`` §6a). The
recording is the bridge's own stream (:class:`~geecs_panels.bridge.stream.DeviceStreams`
over the real CA transport): the first line is the full frame a connect
gets, every later line a delta.

Against the geecs-epics sim host, from this package's environment (one
line; the three CA ports are the camera's, simmotor1's and eztest1's)::

    python fixtures/record.py --seconds 60 --ca "127.0.0.1:5066 127.0.0.1:5100 127.0.0.1:5068" --device UC_VisaEBeam6=ad_camera_chain:undulator:uc_visaebeam6: --device U_SimMotor1=motor_record:undulator:simmotor1::m1,m2 --device U_EzTest1=easydriver:undulator:eztest1:

A FAST-PS (``fastps:undulator:fpstest1:``, CA 127.0.0.1:5078) is added the
same way once its sim finishes ``iocInit`` (``fixtures/README.md``): a
device that never connected records a first frame that says so, which
``tests/test_fixtures.py`` refuses.

Both CA auto address lists stay off (the sims serve real PV names; on a
lab-attached machine an auto list would find the real IOC).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _device(spec: str) -> tuple[str, str, str, tuple[str, ...]]:
    """``NAME=FAMILY:PREFIX[:AXIS,AXIS]`` → (name, family, prefix, axes); the prefix ends in ':'."""
    name, _, rest = spec.partition("=")
    family, _, tail = rest.partition(":")
    prefix, sep, axes = tail.rpartition("::")
    if not sep:
        prefix, axes = tail, ""
    else:
        prefix += ":"
    return name, family, prefix, tuple(a for a in axes.split(",") if a)


async def _record(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    from geecs_schemas.epics.family import load_families
    from geecs_schemas.epics.panel_devices import PanelDevices

    from geecs_panels.bridge.catalog import build_catalog
    from geecs_panels.bridge.stream import DeviceStreams, frame_json
    from geecs_panels.ca import read_pvs, subscribe_reading

    entries = {}
    for spec in args.device:
        name, family, prefix, axes = _device(spec)
        entries[name] = {"family": family, "prefix": prefix}
        if axes:
            entries[name]["axes"] = list(axes)
    panels = PanelDevices.model_validate({"devices": entries})
    devices, errors = build_catalog("sim", [], panels, load_families())
    if errors:
        raise SystemExit("; ".join(errors))
    streams = DeviceStreams(
        {d.name: d for d in devices}, subscribe_reading, read=read_pvs
    )
    stats: dict[str, dict[str, float]] = {}

    async def one(name: str) -> None:
        lines: list[str] = []
        sizes: list[int] = []
        # Subscribe, and let the channels connect before the first frame: a
        # fixture opening on a cold feed would replay "connecting".
        streams.hub(name)
        await asyncio.sleep(args.settle)
        frames = streams.listen(name)
        end = time.monotonic() + args.seconds
        try:
            while time.monotonic() < end:
                try:
                    frame = await asyncio.wait_for(
                        frames.__anext__(), end - time.monotonic()
                    )
                except asyncio.TimeoutError:
                    break
                if frame is None:
                    continue
                text = frame_json(frame)
                lines.append(text)
                if not frame.full:
                    sizes.append(len(text))
        finally:
            await frames.aclose()
        out = Path(args.out) / f"{name}.ndjson"
        out.write_text("".join(line + "\n" for line in lines))
        first = json.loads(lines[0]) if lines else {}
        stats[name] = {
            "frames": len(lines),
            "deltas_per_s": len(sizes) / args.seconds,
            "mean_delta_bytes": sum(sizes) / len(sizes) if sizes else 0.0,
            "max_delta_bytes": max(sizes, default=0),
            "full_bytes": len(lines[0]) if lines else 0,
            "connected": bool(first.get("connected")),
        }

    await asyncio.gather(*(one(d.name) for d in devices))
    await streams.aclose()
    return stats


def main() -> None:
    """Parse the arguments, point CA at the sims, record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ca", required=True, help="EPICS_CA_ADDR_LIST")
    parser.add_argument("--device", action="append", required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args()
    # Before aioca is imported: libca reads the address list once.
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    os.environ["EPICS_CA_ADDR_LIST"] = args.ca
    os.environ.setdefault("EPICS_CA_MAX_ARRAY_BYTES", "20000000")
    stats = asyncio.run(_record(args))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

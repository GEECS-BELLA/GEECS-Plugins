r"""A loopback soft IOC serving the scope contract's names with synthetic captures: the trace page's bench.

geecs-epics has no scope simulator, so the trace page is developed against
this: every role of ``scope_contract.json`` at ``<prefix><suffix>`` (and
its write record), a capture of ``[channels, samples]`` float32, channels
fastest, at 1 Hz while acquiring, its time axis, the per-channel
statistics. The model has three channels (``D:Present_RBV`` 0, so the
array is ``[3, samples]``, as the contract has every channel the model has);
channel C is disabled (its row NaN), so the page's "off" series and hidden
group show. The counters post as the IOC's do: the driver's ``count``
first, the ``ARR`` capture, then the plugin's own ``arr_counter``. A write to ``X`` lands on ``X_RBV``. Bench only: never a production
device, never installed, served on 127.0.0.1.

    EPICS_CAS_SERVER_PORT=5094 EPICS_CA_SERVER_PORT=5094 \
        uvx --with caproto python GeecsControlRoom/scripts/scope_soft_ioc.py bench:scope1:

caproto is not a dependency of any package here: ``uvx`` brings it for the run.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np
from caproto import ChannelChar, ChannelData, ChannelDouble, ChannelEnum, ChannelInteger
from caproto.asyncio.server import start_server

FAMILY = Path(__file__).resolve().parents[2] / "GEECS-Schemas/geecs_schemas/epics/families/scope_contract.json"  # fmt: skip
CHANNELS, SAMPLES = 3, 2000  # the channels the model has (A, B, C)
#: Enum choices the IOC's records carry (the page reads them from the PV, never a list of its own).
ENUMS = {
    "acquire": ["Done", "Acquire"], "image_mode": ["Single", "Multiple", "Continuous"],
    "resolution": ["8 bit", "12 bit", "14 bit", "15 bit", "16 bit"], "state": ["Idle", "Acquire", "Readout", "Correct", "Saving", "Aborting", "Error", "Waiting"],
    "connected": ["Disconnected", "Connected"], "trigger_mode": ["Free run", "Auto", "Normal"],
    "trigger_source": ["A", "B", "C", "D", "External"], "trigger_edge": ["Rising", "Falling"],
    "range": ["10 mV", "20 mV", "50 mV", "100 mV", "200 mV", "500 mV", "1 V", "2 V", "5 V", "10 V", "20 V"],
    "coupling": ["AC", "DC"], "enable": ["Off", "On"], "present": ["No", "Yes"], "overrange": ["No", "Yes"],
}  # fmt: skip
START = {
    "acquire": 1, "image_mode": 2, "resolution": 1, "state": 1, "connected": 1, "trigger_mode": 1, "trigger_source": 4,
    "num_images": 1, "interval_ns": 10.0, "samples": SAMPLES, "pre_trigger": 20.0, "trigger_level": 0.5, "auto_ms": 100,
    "num_channels": CHANNELS, "model": "5242D (bench)", "manufacturer": "soft IOC", "serial": "BENCH-0", "firmware": "0",
    "status": "Acquiring (synthetic)", "last_error": "", "arr_channels": CHANNELS, "arr_samples": SAMPLES,
}  # fmt: skip


class Mirrored:
    """A write record whose value lands on its readback, as the scope contract's setpoints follow their ``_RBV``."""

    mirror: ChannelData | None = None

    async def verify_value(self, data):
        """The value as written, mirrored onto the readback."""
        value = await super().verify_value(data)  # type: ignore[misc]
        if self.mirror is not None:
            await self.mirror.write(value)
        return value


def channel(role: dict, start: object, write: bool = False) -> ChannelData:
    """A role's scalar record: an enum with its choices, text, a whole number or a double (``write``: mirrored)."""
    name, kind = role["name"], role["type"]
    key = name.split("_", 2)[-1] if name.startswith("ch_") else name
    base: type[ChannelData]
    if kind == "enum":
        base, kw = (
            ChannelEnum,
            {"enum_strings": ENUMS[key], "value": ENUMS[key][int(start or 0)]},
        )
    elif role.get("text"):
        base, kw = (
            ChannelChar,
            {
                "value": str(start or ""),
                "max_length": 256,
                "string_encoding": "latin-1",
            },
        )
    elif (
        role.get("integer")
        or kind == "counter"
        or name in ("num_channels", "arr_channels", "arr_samples")
    ):
        base, kw = ChannelInteger, {"value": int(start or 0)}
    else:
        base, kw = (
            ChannelDouble,
            {
                "value": float(start or 0.0),
                "units": role.get("units") or "",
                "precision": 4,
            },
        )
    return (
        type(f"M{base.__name__}", (Mirrored, base), {})(**kw) if write else base(**kw)
    )


def pvdb(prefix: str) -> tuple[dict[str, ChannelData], dict[str, ChannelData]]:
    """Every role's record (and write record) by PV name; the roles by name."""
    family, db, by_role = json.loads(FAMILY.read_text()), {}, {}
    for role in family["roles"]:
        name = role["name"]
        start = START.get(name)
        if name.startswith("ch_"):
            ch = "abcd".index(name[3])
            start = {"present": int(ch < CHANNELS), "enable": int(ch < 2), "range": 6, "coupling": 1, "offset": 0.0, "overrange": 0}.get(name.split("_", 2)[-1], start)  # fmt: skip
        if role["type"] == "array":
            n = CHANNELS * SAMPLES if name == "data" else SAMPLES
            ch_ = ChannelDouble(value=np.zeros(n, dtype=np.float32).tolist(), max_length=n, units=role.get("units") or "")  # fmt: skip
        elif role["type"] == "command":
            ch_ = ChannelInteger(value=0)
        else:
            ch_ = channel(role, start)
        db[prefix + role["suffix"]] = by_role[name] = ch_
        if role.get("write"):
            w = db[prefix + role["write"]] = channel(role, start, write=True)
            w.mirror = ch_
    return db, by_role


async def capture(by_role: dict[str, ChannelData]) -> None:
    """At 1 Hz while acquiring: a capture of every channel (a disabled one NaN), its statistics, the counter."""
    rng, n = np.random.default_rng(1), 0
    while True:
        await asyncio.sleep(1.0)
        if by_role["acquire"].value not in ("Acquire", 1):
            continue
        n += 1
        dt = by_role["interval_ns"].value * 1e-9
        pre = int(SAMPLES * by_role["pre_trigger"].value / 100)
        t = (np.arange(SAMPLES) - pre) * dt
        shapes = [
            0.8 * np.exp(-((t - 4e-6) / 1.5e-6) ** 2) * (1 + 0.05 * math.sin(n)),  # A: a pulse
            0.3 * np.sin(2 * np.pi * t / 6e-6 + n / 3),  # B: a sine
            np.zeros(SAMPLES), np.zeros(SAMPLES),
        ]  # fmt: skip
        rows = []
        for c, shape in enumerate(shapes[:CHANNELS]):
            on = by_role[f"ch_{'abcd'[c]}_enable"].value in ("On", 1) and by_role[f"ch_{'abcd'[c]}_present"].value in ("Yes", 1)  # fmt: skip
            row = (shape + rng.normal(0, 0.01, SAMPLES)).astype(np.float32) if on else np.full(SAMPLES, np.nan, np.float32)  # fmt: skip
            rows.append(row)
            if on:
                ch = "abcd"[c]
                for k, v in (("total", row.sum()), ("mean", row.mean()), ("min", row.min()), ("max", row.max()), ("sigma", row.std())):  # fmt: skip
                    await by_role[f"ch_{ch}_{k}"].write(float(v))
        await by_role["count"].write(
            n
        )  # the driver's counter, before the plugin publishes
        await by_role["time_axis"].write(t.astype(np.float32).tolist())
        # [channels, samples], channels fastest; then the plugin's own counter
        await by_role["data"].write(np.stack(rows, axis=1).ravel().tolist())
        await by_role["arr_counter"].write(n)
        await by_role["arr_uid"].write(float(n))


async def main(prefix: str) -> None:
    """Serve the records on 127.0.0.1 and capture at 1 Hz."""
    db, by_role = pvdb(prefix)
    print(f"serving {len(db)} PVs under {prefix} on 127.0.0.1", flush=True)
    asyncio.get_running_loop().create_task(capture(by_role))
    await start_server(db, interfaces=["127.0.0.1"])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "bench:scope1:"))

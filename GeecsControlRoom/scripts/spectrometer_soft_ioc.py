r"""A loopback soft IOC serving the spectrometer contract's names with synthetic spectra: the trace page's second bench.

The twin of ``scope_soft_ioc.py`` for ``spectrometer_contract.json``: every
role at ``<prefix><suffix>`` (and its write record), a 2048-pixel spectrum
over a wavelength polynomial (``Fit0..2``) at 1 Hz while acquiring, the
level readings, and the chain behind it as the IOC lays it out: ``ARR`` the
raw spectrum, ``PROC`` the background-subtracted one (``PROC:SaveBackground``
holds the current spectrum as if the light were blocked for the save, its
dark pattern and noise, ``PROC:EnableBackground`` subtracts it), and
``STAT1`` over ``PROC``. The counters post in the IOC's order: the driver's
``count``, the ``ARR`` spectrum, ``arr_counter``; then ``PROC``'s and
``STAT1``'s. Any role the family names that the chain produces
(``processed``, ``proc_counter``) is written when the family has it. Enum
words are the bench's, close to the driver's, never read by the page from
anywhere but the PV. A write to ``X`` lands on ``X_RBV``, clamped to the
range the bench's driver takes (``TAKES``: an integration time beyond 10 s
reads back 10 s while ``IntegrationTime`` keeps the request, as an asyn
driver's setpoint does): the setpoint control's ``differs``. Bench only: never
a production device, never installed, served on 127.0.0.1.

    EPICS_CAS_SERVER_PORT=5097 EPICS_CA_SERVER_PORT=5097 \
        uvx --with caproto --with numpy python GeecsControlRoom/scripts/spectrometer_soft_ioc.py bench:cct1:

caproto is not a dependency of any package here: ``uvx`` brings it for the run.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from caproto import ChannelChar, ChannelData, ChannelDouble, ChannelEnum, ChannelInteger
from caproto.asyncio.server import start_server

FAMILY = Path(__file__).resolve().parents[2] / "GEECS-Schemas/geecs_schemas/epics/families/spectrometer_contract.json"  # fmt: skip
PIXELS, FULL_SCALE = 2048, 65535
FIT = (
    350.0,
    0.25,
    -1.0e-5,
)  # nm = c0 + c1·i + c2·i², about 350..820 nm over the pixels
#: The bench's enum words (the page reads them from the PV, never a list of its own).
ENUMS = {
    "acquire": ["Done", "Acquire"], "image_mode": ["Single", "Multiple", "Continuous"],
    "connected": ["Disconnected", "Connected"], "state": ["Idle", "Acquire", "Readout", "Correct", "Saving", "Aborting", "Error", "Waiting"],
    "trigger_mode": ["Internal", "External"], "trigger_slope": ["Rising", "Falling"], "raw_mode": ["Processed", "Raw"],
    "direction": ["Forward", "Reverse"], "saturated": ["No", "Yes"], "board_trigger": ["Software", "Hardware"],
    "background_enable": ["Disable", "Enable"], "background_valid": ["Invalid", "Valid"],
}  # fmt: skip
START = {
    "acquire": 1, "image_mode": 2, "connected": 1, "state": 1, "trigger_mode": 0, "trigger_slope": 0, "raw_mode": 0,
    "num_images": 1, "integration_time": 0.01, "averages": 1, "acquire_period": 1.0, "temperature": 24.5,
    "manufacturer": "soft IOC", "model": "CCT (bench)", "serial": "BENCH-0", "firmware": "0", "sensor": "linear CCD",
    "board_serial": "BENCH-B", "spectrometer_serial": "BENCH-S", "status": "Acquiring (synthetic)", "last_error": "",
    "pixels": PIXELS, "arr_pixels": PIXELS, "fit0": FIT[0], "fit1": FIT[1], "fit2": FIT[2], "fit3": 0.0, "fit4": 0.0,
    "min_integration": 1e-5, "stored_integration": 0.01, "stored_averages": 1, "compute_statistics": 1, "compute_centroid": 1,
}  # fmt: skip
#: The range the bench's driver takes per role: a write outside it reads back clamped.
TAKES = {"integration_time": (1e-5, 10.0)}
WHOLE = {
    "pixels",
    "arr_pixels",
    "stored_averages",
    "compute_statistics",
    "compute_centroid",
}


class Mirrored:
    """A write record whose value lands on its readback, as the contract's setpoints follow their ``_RBV``."""

    mirror: ChannelData | None = None
    takes: tuple[float, float] | None = None

    async def verify_value(self, data):
        """The value as written, mirrored onto the readback (clamped to ``takes``)."""
        value = await super().verify_value(data)  # type: ignore[misc]
        if self.mirror is not None:
            low, high = self.takes or (value, value)
            await self.mirror.write(min(max(value, low), high))
        return value


class Command(ChannelInteger):
    """A command record: a write of 1 runs ``on_write`` (``PROC:SaveBackground`` holds the current spectrum)."""

    on_write = None

    async def verify_value(self, data):
        """Run the command on a 1; the record reads 0 again after."""
        value = await super().verify_value(data)
        if value and self.on_write is not None:
            self.on_write()
        return 0


def channel(role: dict, start: object, write: bool = False) -> ChannelData:
    """A role's scalar record: an enum with its choices, text, a whole number or a double (``write``: mirrored)."""
    name, kind = role["name"], role["type"]
    base: type[ChannelData]
    if kind == "enum":
        base, kw = (
            ChannelEnum,
            {"enum_strings": ENUMS[name], "value": ENUMS[name][int(start or 0)]},
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
    elif role.get("integer") or kind in ("counter", "bit") or name in WHOLE:
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
        if role["type"] == "array":
            ch: ChannelData = ChannelDouble(
                value=np.zeros(PIXELS).tolist(),
                max_length=PIXELS,
                units=role.get("units") or "",
            )
        elif role["type"] == "command":
            ch = Command(value=0)
        else:
            ch = channel(role, START.get(name))
        db[prefix + role["suffix"]] = by_role[name] = ch
        if role.get("write"):
            w = db[prefix + role["write"]] = channel(role, START.get(name), write=True)
            w.mirror, w.takes = ch, TAKES.get(name)
    return db, by_role


def on(record: ChannelData) -> bool:
    """An enum or bit record reads its second word or 1."""
    return record.value in (1, "Acquire", "Enable", "Valid", "Yes")


async def acquire(by_role: dict[str, ChannelData]) -> None:
    """At 1 Hz while acquiring: a spectrum, its levels, then PROC and STAT1 over it, the counters in the IOC's order."""
    rng, n, i = np.random.default_rng(1), 0, np.arange(PIXELS, dtype=np.float64)
    nm = FIT[0] + FIT[1] * i + FIT[2] * i**2
    await by_role["wavelength"].write(nm.tolist())
    held: dict[str, np.ndarray] = {}
    last = np.zeros(PIXELS)

    def save() -> None:
        """PROC:SaveBackground: the current spectrum with the light blocked held (the dark and its noise); valid."""
        held["background"] = last.copy()
        asyncio.get_running_loop().create_task(
            by_role["background_valid"].write("Valid")
        )

    by_role["background_save"].on_write = save  # type: ignore[attr-defined]
    while True:
        await asyncio.sleep(1.0)
        if not on(by_role["acquire"]):
            continue
        n += 1
        gain = max(by_role["integration_time"].value, 1e-5) / 0.01
        dark = 1000.0 + 40.0 * np.sin(
            i / 90.0
        )  # the sensor's fixed pattern: what Save background removes
        light = gain * (30000.0 * np.exp(-(((nm - 800.0) / 4.0) ** 2)) + 6000.0 * np.exp(-(((nm - 532.0) / 30.0) ** 2)) * (1 + 0.05 * np.sin(n)))  # fmt: skip
        noise = rng.normal(0, 25.0, PIXELS)
        raw = np.clip(dark + light + noise, 0, FULL_SCALE)
        last = dark + noise
        peak = float(raw.max())
        await by_role["count"].write(
            n
        )  # the driver's counter, before the plugins publish
        await by_role["level_max"].write(peak)
        await by_role["level_percent"].write(100.0 * peak / FULL_SCALE)
        await by_role["saturated"].write("Yes" if peak >= FULL_SCALE else "No")
        await by_role["spectrum"].write(
            raw.tolist()
        )  # ARR: the raw spectrum, then its own counter
        await by_role["arr_counter"].write(n)
        await by_role["arr_uid"].write(float(n))
        subtract = (
            on(by_role["background_enable"])
            and on(by_role["background_valid"])
            and "background" in held
        )
        proc = raw - held["background"] if subtract else raw
        if (
            "processed" in by_role
        ):  # PROC's own array and counter, when the family names them
            await by_role["processed"].write(proc.tolist())
        if "proc_counter" in by_role:
            await by_role["proc_counter"].write(n)
        if on(by_role["compute_statistics"]):  # STAT1 over PROC, in index units
            for k, v in (("total", proc.sum()), ("max", proc.max()), ("min", proc.min()), ("mean", proc.mean()), ("sigma", proc.std()), ("peak_index", float(proc.argmax()))):  # fmt: skip
                await by_role[k].write(float(v))
        if on(by_role["compute_centroid"]):
            w = np.clip(proc, 0, None)
            c = float((w * i).sum() / w.sum()) if w.sum() else 0.0
            await by_role["centroid_index"].write(c)
            await by_role["width_index"].write(
                float(np.sqrt((w * (i - c) ** 2).sum() / w.sum())) if w.sum() else 0.0
            )
        await by_role["stat_counter"].write(n)


async def main(prefix: str) -> None:
    """Serve the records on 127.0.0.1 and acquire at 1 Hz."""
    db, by_role = pvdb(prefix)
    print(f"serving {len(db)} PVs under {prefix} on 127.0.0.1", flush=True)
    asyncio.get_running_loop().create_task(acquire(by_role))
    await start_server(db, interfaces=["127.0.0.1"])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "bench:cct1:"))

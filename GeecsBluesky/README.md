# GeecsBluesky

Bridges the GEECS hardware control system to the [Bluesky](https://blueskyproject.io/)
experiment orchestration ecosystem via [ophyd-async](https://ophyd-async.readthedocs.io/).

GEECS devices speak a custom UDP/TCP protocol.  This package provides:

- `GeecsUdpClient` / `GeecsTcpSubscriber` — asyncio transport matching the real wire format
- `GeecsSignalBackend` — ophyd-async `SignalBackend` backed by GEECS UDP/TCP
- `GeecsDevice` — `StandardReadable` base class with shared-client lifecycle management
- `GeecsGenericDetector` — dynamically-signalled detector from a plain variable list
- `GeecsMotor` — settable motor device for any GEECS axis
- `FakeGeecsServer` — in-process fake device server for offline testing

## Current status

`BlueskyScanner` runs STANDARD scans and NOSCAN/statistics collection from the
`GEECS-Scanner-GUI` (`use_bluesky=True`), in two acquisition modes selected by
the `GEECS_BLUESKY_ACQUISITION_MODE` env var, with Tiled persistence and DG645
shot control:

- **`free_run_time_sync`** — external trigger free-runs; a reference device
  paces event rows and other devices contribute timestamp-matched data
  (tolerant of late/missing devices).
- **`strict_shot_control`** — every device required per shot; true plan-owned
  single-shot when the shot-control config defines an `ARMED` state, else a
  free-running `trigger_and_read` fallback.

Both modes write the same versioned event schema (see `EVENT_SCHEMA.md`) and are
hardware-verified.  For `save_nonscalar_data=True` devices, each event records
the detector `acq_timestamp` and the configured save directory; file names
remain hardware-native and should be joined by `acq_timestamp`.

Still open (features, not architecture): setup/closeout actions,
background/optimization modes, legacy s-file/TDMS output, and a reusable
shot-controller for notebook parity.  See `ROADMAP.md` and
`Planning/acquisition_modes/`.

## Requirements

- Python 3.11 (ophyd-async ≥ 0.16 requires ≥ 3.11; geecs-pythonapi requires < 3.12)
- Network access to the GEECS control system (or use `FakeGeecsServer` for offline work)

## Installation

### From this repo (recommended)

```bash
cd GeecsBluesky
poetry install
```

Poetry resolves `ophyd-async` and `bluesky` from PyPI.  If the resolver fails
(rare — can happen when pydantic-numpy's environment markers conflict), install
the runtime deps manually into the poetry venv:

```bash
VENV=$(poetry env info --path)
$VENV/bin/pip install "ophyd-async>=0.16,<1.0" "bluesky>=1.12" -q
# dev tools
$VENV/bin/pip install "pytest>=7" "pytest-asyncio>=0.23" -q
```

### DB lookup

`GeecsDevice.from_db()` resolves `(host, port)` from the GEECS MySQL database.
`mysql-connector-python` is included as a direct Poetry dependency and is
installed automatically by `poetry install`.

The package reads DB credentials from the standard GEECS user-data directory.
Configure the path in `~/.config/geecs_python_api/config.ini`:

```ini
[Paths]
geecs_data = /path/to/user data   # directory containing Configurations.INI
```

`Configurations.INI` in that directory provides the `[Database]` section
(host, port, name, user, password).

## Quick start

```python
import asyncio
from geecs_bluesky.devices.generic_detector import GeecsGenericDetector

async def main():
    # Resolve from GEECS database (requires mysql-connector-python):
    det = GeecsGenericDetector.from_db(
        "UC_Wavemeter", ["Wavelength (nm)", "Power (mW)"], name="wavemeter"
    )
    await det.connect()
    reading = await det.read()
    for key, val in reading.items():
        print(f"{key}: {val['value']}")
    await det.disconnect()

asyncio.run(main())
```

With the Bluesky RunEngine:

```python
from bluesky import RunEngine
import bluesky.plans as bp

RE = RunEngine()
RE(bp.count([det], num=5))
```

## Writing a new device

```python
from geecs_bluesky.devices.geecs_device import GeecsDevice
from geecs_bluesky.signals import geecs_signal_rw, geecs_signal_r
from geecs_bluesky.transport.udp_client import GeecsUdpClient

class MyDevice(GeecsDevice):
    def __init__(self, host: str, port: int, name: str = "my_device") -> None:
        udp = GeecsUdpClient(host, port)          # one shared client serialises I/O
        with self.add_children_as_readables():
            self.position = geecs_signal_rw(
                float, "U_MyDevice", "Position (mm)", host, port,
                units="mm", shared_udp=udp,
            )
            self.status = geecs_signal_r(
                int, "U_MyDevice", "Status", host, port,
                shared_udp=udp,
            )
        super().__init__(name=name, shared_udp=udp)
```

**Key rule:** all signals on the same device must share the same `GeecsUdpClient`
instance.  GEECS devices reject concurrent UDP commands; the shared client's
internal `asyncio.Lock` serialises them automatically.

## Running the tests

```bash
poetry run pytest tests/ -v
```

All tests run against `FakeGeecsServer` on localhost — no real hardware required.

A hardware integration test (`test_bluesky_scanner.py`) exercises NOSCAN, STANDARD
step scan, and DG645 shot control against real lab devices:

```bash
poetry run python test_bluesky_scanner.py
```

## Architecture notes

- **Wire protocol**: UDP two-stage (cmd → ACK on cmd port, exe response on cmd+1).
  Real hardware ACK format: `"get{var}>>>>accepted"`.  EXE status: `"no error,"` (not `"ok,"`).
- **TCP subscription**: framed (4-byte big-endian length prefix), pushed at 5 Hz.
- **Shared client + lock**: `GeecsUdpClient` holds an `asyncio.Lock`; all concurrent
  `get`/`set` calls on the same client are serialised automatically.
- **Local-IP detection**: `_detect_local_ip()` probes the OS routing table at
  `connect()` time via a no-op UDP socket — handles PPP/VPN lab links transparently.

# GEECS-Core

The GEECS access **library**: everything a Python consumer of GEECS devices
needs, and nothing else. It is the layer under both EPICS gateways and the
Bluesky engine — and it carries **`GeecsDevice`**, the entry-level way to talk
to a GEECS device from a script or notebook.

## Quickstart

```python
from geecs_core import GeecsDevice

with GeecsDevice("U_S1H") as dev:
    value = dev.get("Current")          # blocking read, typed result
    dev.set("Current", value)           # blocking write; returns the
                                        # device's reported readback
    dev.subscribe(["Current"])          # live push stream into dev.state
    ...
    print(dev.state)                    # {"Current": 0.103,
                                        #  "shot number": 117,
                                        #  "connected": True}
```

Construction looks the device up in the experiment database (one query, no
sockets). `get`/`set` block until the device answers; failures **raise**
(`GeecsCommandFailedError`, `GeecsCommandRejectedError`,
`GeecsConnectionError`) — a returned value is always a real device response.
Subscriptions feed `dev.state` at the device's push rate, add the reserved
keys `"shot number"` and `"connected"`, and auto-reconnect through device
restarts. See the [example notebook](examples/geecs_device_basics.ipynb) for
the full tour, and the [API reference](api/client.md) for signatures.

Prerequisite: the standard `~/.config/geecs_python_api/config.ini` (the
directory name is historical — the file is the fleet-wide contract). The
[Getting Started tutorial](../tutorials/getting_started.md) is the
key-by-key reference.

## Migrating from GEECS-PythonAPI

`GeecsDevice` succeeds the legacy `GeecsDevice`/`ScanDevice` objects (the
`GEECS-PythonAPI` package was removed 2026-08-20). The shape is deliberately
familiar:

```python
# old
from geecs_python_api.controls.devices.scan_device import ScanDevice
dev = ScanDevice("U_S1H")
dev.set("Current", 0)

# new
from geecs_core import GeecsDevice
dev = GeecsDevice("U_S1H")
dev.set("Current", 0)
```

What changed, and why it's better:

| Legacy behavior | New behavior |
|---|---|
| `get`/`set` return `None` on failure | Failures **raise** typed exceptions |
| `set` echoes back through a state cache | `set` returns the device's actual exe-response readback |
| `subscribe_var_values([...])` | `subscribe([...])` — same `state` feed, plus `"connected"` |
| Global `exp_info` + `collect_exp_info` setup | None needed — construction resolves the device directly |
| Variable aliases (`use_alias_in_TCP_subscription`) | Raw GEECS variable names only |
| One global lock serialized all devices | Devices command independently |
| Composite `ScanDevice(name, spec)` | No equivalent here — pseudo scan variables live in the Bluesky engine |

## What else is in the library

- **`geecs_core.transport`** — the asyncio UDP/TCP wire protocol
  (`GeecsUdpClient`, `GeecsTcpSubscriber`) that `GeecsDevice` and both
  gateways are built on. Use it directly if your application has its own
  event loop.
- **`geecs_core.db.GeecsDb`** — the experiment MySQL database: device
  endpoints, variable metadata, experiment rosters.
- **`geecs_core.pv_naming`** — the one shared GEECS→EPICS PV naming policy.
- **`geecs_core.testing`** — `FakeGeecsServer`, an in-process server speaking
  the real wire protocol, so your code can be tested with no hardware.

When should you *not* use `GeecsDevice`? For data-taking scans (use the GEECS
Console / Bluesky engine — scans need numbering, saving, and shot control),
and for monitoring dashboards at HTU (the [gateway PVs](../geecs_gateway/client_overview.md)
serve that better). `GeecsDevice` is the right tool for scripts, notebooks,
feedback loops, and facilities where the EPICS layer is not deployed.

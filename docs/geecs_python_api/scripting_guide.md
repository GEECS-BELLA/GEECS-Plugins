# Scripting Guide

Most users meet GEECS through the Scanner GUI. This page is for the other case — when you want to write a Python script (or notebook) that talks to one or more GEECS devices directly. Common reasons:

- Run a one-off measurement that doesn't fit a save-element-shaped scan.
- Script a calibration or alignment procedure.
- Drive an experiment from a notebook for interactive exploration.
- Read a device variable from outside the GUI for monitoring.

The path is short: configure your environment, instantiate a `GeecsDevice`, set or get variables. Everything below assumes you have `geecs-python-api` installed (it's a dependency of the Scanner GUI, so if you can run the GUI, you have it).

## Prerequisites

You need:

1. A working `~/.config/geecs_python_api/config.ini` pointing at the right experiment and database. The Scanner GUI sets this up the first time you run it. If you've never run the GUI on this machine, copy a known-good config from a colleague.
2. Network reachability to the GEECS database server and to the device(s) you want to talk to.
3. The device's name as it appears in the GEECS database. The Scanner GUI's "available devices" list is one way to find this; querying the database directly is another.

A quick smoke test from a Python prompt:

```python
from geecs_python_api.controls.interface.geecs_database import GeecsDatabase, load_config

expt = load_config().get("Experiment", "expt")
exp_info = GeecsDatabase.collect_exp_info(expt)
print(list(exp_info["devices"].keys())[:10])
```

If that prints a list of device names, your environment is set up correctly. If you get a database error, fix that first — the config file's `[Database]` section is the place to look.

## The basic pattern

```python
from geecs_python_api.controls.devices.scan_device import ScanDevice

# 1. Instantiate the device by its GEECS database name
device = ScanDevice("U_ESP_JetXYZ")

# 2. Read a variable
position = device.get("Position.Axis 1")
print(f"Current position: {position}")

# 3. Set a variable
device.set("Position.Axis 1", 4.0)

# 4. Read it back to confirm
print(f"New position: {device.get('Position.Axis 1')}")

# 5. Close the connection when done
device.close()
```

That's the whole API for simple cases. The `set` and `get` calls block until the device acknowledges (or until `exec_timeout=10` seconds elapse, whichever comes first). The variable name must match exactly what's in the GEECS database — case-sensitive, including spaces.

`ScanDevice` is the right class for almost all scripting. It inherits from `GeecsDevice` and adds support for composite variables; if you don't use composites, the inherited interface is identical.

## Subscribing to live updates

For continuous monitoring, subscribe to one or more variables:

```python
device = ScanDevice("U_ICT_Charge")

# Subscribe to one or more variables
device.subscribe_var_values(["charge"])

# The device's TCP listener now updates device.state in the background
# as new shots arrive. Read the latest cached value at any time:
import time
for _ in range(10):
    print(f"Charge: {device.state.get('charge')}")
    time.sleep(1)

device.unsubscribe_var_values()
device.close()
```

This is non-blocking — the TCP listener runs in a worker thread and updates `device.state` whenever the device pushes new data. Use this pattern for dashboards or "monitor for X seconds and log every value" scripts.

If you also want a callback fired on every update (rather than polling `device.state`), register one with `device.register_update_listener("my_listener", lambda value: ...)`.

## Composite scan variables

`ScanDevice` supports composite variables — a single logical variable that drives several underlying device variables through arbitrary mathematical relations. The same composite spec used by the Scanner GUI works in scripts:

```python
composite_spec = {
    "components": [
        {"device": "U_ESP_X", "variable": "Position",
         "relation": "composite_var * 1"},
        {"device": "U_ESP_Y", "variable": "Position",
         "relation": "composite_var * -1"},
    ],
    "mode": "relative",
}

dev = ScanDevice("balanced_xy", composite_spec_dict=composite_spec)
dev.set("balanced_xy", 0.5)
# X moves +0.5, Y moves -0.5, both relative to their pre-set positions
```

Modes are `"relative"` (offsets from the captured starting state) and `"absolute"` (the relations evaluate to absolute setpoints). For "get" composites, see `composite_spec_dict` documentation in `ScanDevice.__init__`.

## Handling errors

The three exceptions to know about, all in `geecs_python_api.controls.interface.geecs_errors`:

- **`GeecsDeviceCommandRejected`** — the device received the command but refused it (out of range, wrong mode). Often retryable.
- **`GeecsDeviceExeTimeout`** — the device didn't ack within the timeout. Usually means the device is hung; retrying typically makes it worse.
- **`GeecsDeviceCommandFailed`** — hardware-level failure. Not retryable.

```python
from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceCommandRejected,
    GeecsDeviceExeTimeout,
    GeecsDeviceCommandFailed,
)

try:
    device.set("Position.Axis 1", 100.0)
except GeecsDeviceCommandRejected as e:
    print(f"Device refused the command: {e}")
except GeecsDeviceExeTimeout:
    print("Device didn't respond — check network and device state")
except GeecsDeviceCommandFailed as e:
    print(f"Hardware failure: {e}")
```

If you're writing a script that needs the same retry-and-escalation policy the scan engine uses, look at `geecs_scanner.engine.device_command_executor.DeviceCommandExecutor`. It wraps `device.set` / `device.get` with a typed retry policy and calls a callback on unrecoverable failures. That's the same policy the Scanner GUI uses for its mid-scan dialogs.

## Cleaning up

Always call `device.close()` when you're done — it shuts down the UDP/TCP transports and unregisters the device from the global state. In a Jupyter notebook this gets less critical because cells are stateful, but in a script it matters: leaving devices open across script runs can lead to "device is busy" errors on the next run.

A context manager pattern works if you'd like:

```python
from contextlib import closing

with closing(ScanDevice("U_ESP_JetXYZ")) as dev:
    dev.set("Position.Axis 1", 4.0)
    # ... measurement logic ...
# device.close() runs automatically here
```

## A complete example: scripted scan

Putting it together — set a device variable to each of N values, record a reading from a different device at each step, write the result to a CSV.

```python
import time
import pandas as pd
from geecs_python_api.controls.devices.scan_device import ScanDevice

# Devices we'll use
motor = ScanDevice("U_ESP_JetXYZ")
charge_meter = ScanDevice("U_ICT_Charge")

# Subscribe so we can read live values from the charge meter
charge_meter.subscribe_var_values(["charge"])

records = []
positions = [0.0, 1.0, 2.0, 3.0, 4.0]

try:
    for pos in positions:
        # Move the motor and wait for the move to complete
        motor.set("Position.Axis 1", pos)

        # Settle, then average the next few shots
        time.sleep(2.0)
        readings = []
        for _ in range(10):
            readings.append(charge_meter.state.get("charge"))
            time.sleep(0.2)

        avg = sum(r for r in readings if r is not None) / len(readings)
        records.append({"position": pos, "charge_avg": avg})
        print(f"pos={pos:5.2f}  charge_avg={avg:.3f}")

finally:
    charge_meter.unsubscribe_var_values()
    motor.close()
    charge_meter.close()

df = pd.DataFrame(records)
df.to_csv("calibration.csv", index=False)
```

This is the script-shaped equivalent of a 1D scan. It's the right tool when:

- You don't want to write a full save element / scan config.
- Your acquisition logic doesn't fit the shot-counted-per-step model (you want fixed-count averages, or settling delays specific to one device).
- You're in a notebook and want immediate interactive results without a separate scan folder.

For anything that you'll run repeatedly or that needs to be reproducible, prefer the Scanner GUI — its scan folders are searchable and its s-files are the input to the broader analysis ecosystem. Use scripts for exploration; promote the things that work to save elements once they're stable.

## Where to read more

- **[Image Analysis](../image_analysis/overview.md)** — once you have raw images, this is how you process them.
- **[Data Utils — Basic Usage](../geecs_data_utils/examples/basic_usage.ipynb)** — for loading data that's already on disk.
- **[Scanner Architecture](../geecs_scanner/architecture.md)** — to understand how `DeviceCommandExecutor` and the engine use these primitives at scale.

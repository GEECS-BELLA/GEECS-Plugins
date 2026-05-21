# Save Elements

A **save element** tells the scanner two things: which devices to record from during a scan, and what to do at the start and end of the scan. They're YAML files that live alongside your experiment configuration, and you can compose multiple of them per scan — the scanner merges the device lists, validates compatibility, and runs the action sequences in order.

This page is the reference. If you're trying to add your first device to a scan, the [Tutorial](tutorial.md) walks through it more gently.

## File location

Save elements live at:

```
{experiment-config}/save_devices/{element_name}.yaml
```

The `experiment-config` path is whatever you have configured in `~/.config/geecs_python_api/config.ini` for your experiment. The GUI's **Save Element Editor** writes files here automatically. You can also edit them by hand; the YAML schema is below.

## Minimal example

```yaml
Devices:
  U_ModeImager:
    save_nonscalar_data: true
    synchronous: true
    variable_list:
      - exposure
      - gain
```

This says: during the scan, record the `exposure` and `gain` variables from device `U_ModeImager`, and pull any per-shot files the device produces (images) into the scan folder. The device is treated as synchronous, meaning the scanner waits for each shot's data to arrive before moving on.

## Full schema

```yaml
# Required: at least one device.
Devices:
  <device_name>:
    # Whether to record per-shot binary files (images, traces). Default: false.
    save_nonscalar_data: true

    # Whether the scanner waits for this device's data before declaring a shot done.
    # Use true for cameras and primary diagnostics; false for slow, asynchronous logs.
    # Default: false.
    synchronous: true

    # Variables to record. The names must match the device's variable list in the
    # GEECS database — the GUI's editor populates these from the database.
    variable_list:
      - <variable_a>
      - <variable_b>

    # If true, record every variable the device exposes. If both this and
    # variable_list are set, variable_list wins. Default: false.
    add_all_variables: false

    # Optional: per-device pre/post-scan setup. Each entry maps a variable name to
    # a [pre-scan value, post-scan value] pair. Applied at scan start and undone
    # at scan end.
    scan_setup:
      mode: ["scan", "standby"]

# Optional: free-form metadata written into scan_info.ini.
scan_info:
  notes: "anything you want recorded in the scan metadata"

# Optional: action sequence run before the scan starts.
setup_action:
  steps:
    - action: set
      device: U_ESP_JetXYZ
      variable: Position.Axis 1
      value: 4.0
    - action: wait
      wait: 1.0

# Optional: action sequence run after the scan ends.
# Runs even if the scan aborted, so use this for cleanup.
closeout_action:
  steps:
    - action: set
      device: U_ESP_JetXYZ
      variable: Position.Axis 1
      value: 0.0
```

Every field is optional except `Devices` (at least one entry required). The Pydantic model that validates this is `SaveDeviceConfig` in `geecs_scanner/engine/models/save_devices.py`.

## Composing multiple elements

The intent is that you keep one save element per "thing you might want to record" — `always_on_cameras.yaml`, `magspec_diagnostic.yaml`, `gas_jet_status.yaml` — and compose them per scan by selecting more than one in the GUI's **Selected Save Elements** list. The scanner merges them at scan time.

Two rules govern the merge:

1. **Variable lists union.** If two elements both list `U_ModeImager` with overlapping `variable_list` entries, the union is used.
2. **Boolean flags must agree.** If two elements list the same device with conflicting `save_nonscalar_data` or `synchronous` values, the scan refuses to start and shows a "Conflicting Save Elements" dialog. Resolve by editing one of the elements to match the other or by removing one from the selection.

Action sequences (`setup_action`, `closeout_action`) from every selected element run in selection order. The pattern: the always-on cameras element provides the "open shutter / close shutter" closeout; the diagnostic-of-the-day element provides the "set up that diagnostic" setup. They compose without you having to think about it.

## Device flags in detail

`save_nonscalar_data: true` configures the device to write per-shot files (typically images for cameras) and tells the scanner's `FileMover` to pull them into the scan folder under `Scan###/{device_name}/`. Use it for any device that has files associated with shots. If left false, only the variables in `variable_list` are recorded as scalars in the s-file.

`synchronous: true` means the scanner waits for this device's per-shot data before counting the shot as done. The shot timestamp comparison enforces alignment to within a configurable tolerance (default 50 ms). Use this for cameras and primary diagnostics where missing a shot means missing a measurement. `synchronous: false` is for devices that log asynchronously and are allowed to drift — the data still gets written but it's not used to gate scan progression.

`add_all_variables: true` is a shortcut: record every variable the device publishes. Convenient for new devices where you don't yet know which subset matters. In production, prefer an explicit `variable_list` so the s-file stays narrow and predictable.

`scan_setup` is for per-device pre/post-scan state. It runs before the `setup_action` sequence and is undone after the `closeout_action` sequence. It's the right place for "put the camera in scan mode" or "switch the trigger source to external" — anything that's specific to this device and should be reverted automatically at scan end.

## Action sequences

The `setup_action` and `closeout_action` blocks each contain a `steps:` list. Each step has an `action:` field that picks one of five step types:

### `wait` — pause for a fixed duration

```yaml
- action: wait
  wait: 2.5
```

Pauses the action sequence for 2.5 seconds. Useful between a `set` and a subsequent `get` to let hardware settle.

### `set` — write to a device variable

```yaml
- action: set
  device: U_ESP_JetXYZ
  variable: Position.Axis 1
  value: 4.0
  wait_for_execution: true   # optional, default true
```

Sends a command to set `Position.Axis 1` on `U_ESP_JetXYZ`. With `wait_for_execution: true` (the default), the sequence waits for the device's acknowledgement before proceeding. With `false`, it fires the command and moves on; this is rarely what you want.

If the command fails (rejected, timeout, or hardware error), the action's escalation policy kicks in — same Continue/Abort dialog as during a scan. An Abort propagates back and prevents the scan from starting.

### `get` — read a device variable, optionally validate

```yaml
- action: get
  device: U_ESP_JetXYZ
  variable: Position.Axis 1
  expected_value: 4.0
  tolerance: 0.05    # optional
```

Reads the current value of the variable. If `expected_value` is given, the read value is compared (with `tolerance` if numeric) and an `ActionError` is raised on mismatch. The error message names the device, variable, expected, and actual values so it's clear what went wrong.

### `execute` — run a named action from the action library

```yaml
- action: execute
  action_name: home_all_motors
```

Runs an action sequence defined in the action library (a separate YAML file managed by the **Action Library** dialog in the GUI). Use this for shared sequences that multiple save elements need.

### `run` — invoke a Python class

```yaml
- action: run
  file_name: my_pre_scan.py
  class_name: PreScanAlignment
```

Imports `my_pre_scan.py` from the action library directory and runs `PreScanAlignment.run()`. The escape hatch for custom logic that's hard to express as a sequence of set/get/wait steps. Use sparingly.

## A complete realistic example

```yaml
Devices:
  U_ModeImager:
    save_nonscalar_data: true
    synchronous: true
    variable_list:
      - exposure
      - gain

  U_ICT_Charge:
    save_nonscalar_data: false
    synchronous: true
    variable_list:
      - charge

  U_DG645_ShotControl:
    save_nonscalar_data: false
    synchronous: false
    variable_list:
      - delay_AB
      - delay_CD

setup_action:
  steps:
    - action: set
      device: U_BeamBlock
      variable: state
      value: open
    - action: wait
      wait: 0.5
    - action: get
      device: U_BeamBlock
      variable: state
      expected_value: open

closeout_action:
  steps:
    - action: set
      device: U_BeamBlock
      variable: state
      value: closed
```

Reads exposure/gain from a camera with images, charge from an ICT, and the DG645 trigger delays for context. Opens the beam block before the scan, verifies it's actually open, and closes it after the scan. The closeout runs even if the scan aborts.

## Validating a save element by hand

If you've edited a YAML file directly and want to check it before opening the GUI:

```python
from geecs_scanner.engine.models.save_devices import SaveDeviceConfig
import yaml

with open("my_save_element.yaml") as f:
    data = yaml.safe_load(f)

config = SaveDeviceConfig.model_validate(data)
print(config)
```

The Pydantic validator catches typos in field names, wrong types (e.g. `save_nonscalar_data: "true"` as a string instead of a bool), and structural problems. Errors include the path within the YAML where the problem is.

## Common mistakes

- **Variable name doesn't match the database.** The variable list comes from the GEECS device DB. If you typed `Position.Axis1` but the database has `Position.Axis 1` (with a space), the device will silently not record that variable. The GUI editor avoids this by populating from the DB; hand-edits are where this bites.
- **Two save elements list the same device with conflicting flags.** Resolve by editing one. The "Conflicting Save Elements" error message names the device and the conflicting flag.
- **Setup action depends on a device that's not in `Devices`.** Action steps can talk to any device the experiment knows about; they don't need to be in the save element's `Devices` block. But if you depend on a device's *recorded data*, that device does need to be in `Devices`.
- **Closeout that re-raises kills the scan.** Closeouts run inside a `try/except` that logs but doesn't fail the scan if a step raises. That's a feature — your data has already been written by the time closeout runs. But it means a closeout step that "must succeed" needs a `get` with `expected_value` to surface failure to the operator.

If you hit something not covered here, [Troubleshooting](troubleshooting.md) is a good next stop.

# GEECS Scanner GUI

GEECS Scanner is a PyQt5 application for running data-acquisition scans on a GEECS-controlled experiment. It connects to GEECS devices, coordinates them through a scan, records per-shot data and image files into a structured scan folder, and surfaces the lifecycle through a typed event stream that drives both the GUI and any other listener.

If you're new here, the [Tutorial](tutorial.md) walks through your first scan end to end.

## What it does

A scan in this system is a coordinated sequence: set a device variable to a value, wait for some number of shots while logging per-shot data from every device you've selected, set the variable to the next value, repeat. The Scanner GUI lets you configure that sequence — what to scan, what to record, how long to wait at each step — and then runs it on a background thread while the GUI shows progress and surfaces any device errors that come up.

The supported scan modes are NOSCAN (record N shots at a fixed configuration), 1D scan (sweep one device variable across a range), background (a NOSCAN tagged for use as a calibration reference), and optimization (Xopt drives the scan variable and uses scan results as objective values).

## What it gives you

- **Save elements.** Lists of devices and the variables you want recorded for them, written as YAML files. You compose them at scan time — the scanner merges the lists, validates compatibility, and configures every selected device to publish its variables for the duration of the scan. See [Save Elements](save_elements.md).
- **Composite scan variables.** Combine multiple device parameters into a single scan variable using arbitrary mathematical relations. The composite gets one entry in the GUI; under the hood the engine moves the underlying devices in lockstep.
- **Pre/post-scan action sequences.** A YAML-defined sequence of steps (set variable, wait, run nested action) that runs before the scan starts and after it ends. Used for calibration, gating, and "leave it as you found it" closeouts. See [Save Elements — Action Sequences](save_elements.md#action-sequences).
- **Multi-scanner.** Queue up several scans with different configurations and run them as a batch. Each one writes its own scan folder.
- **Optimization.** An optimization scan replaces the fixed step list with an Xopt-driven generator. Each iteration takes the latest scan result, computes an objective via a registered evaluator, and proposes the next set of variable values. See the [Optimization Example](examples/optimization/optimization_example.ipynb).
- **Scripted access.** The engine (`geecs_scanner.engine.ScanManager`) is independent of Qt; you can run a scan from a Python script without launching the GUI.

## Where things live

The package splits cleanly along the GUI/engine line:

- `geecs_scanner.app` — the PyQt5 layer: main window, save-element editor, multiscanner, action library editor. Imports Qt; depends on the engine through one adapter (`RunControl`).
- `geecs_scanner.engine` — the headless scan execution core: `ScanManager`, `ScanStepExecutor`, `DataLogger`, `FileMover`, `TriggerController`, `DeviceCommandExecutor`, `ScanLifecycleStateMachine`. No Qt imports.
- `geecs_scanner.engine.models` — Pydantic models for the GUI→engine contract: `ScanExecutionConfig`, `ScanOptions`, `SaveDeviceConfig`, action step types.
- `geecs_scanner.engine.scan_events` — the typed `ScanEvent` hierarchy. The single contract between the engine and any consumer (GUI, remote monitor, log writer, test harness).
- `geecs_scanner.utils` — shared exception hierarchy, retry helper, application paths.

The [Architecture page](architecture.md) goes into the lifecycle, event flow, and design rationale. It's worth reading once if you plan to extend the scanner; it's not required reading to use it.

## Getting started

1. **Install** — see [Installation & Setup](installation.md). Python 3.10, Poetry, and access to the GEECS database are the prerequisites.
2. **Run your first scan** — the [Tutorial](tutorial.md) walks through configuring an experiment, building a save element, and running a NOSCAN and a 1D scan.
3. **Understand the data layout** — the [Scan Output Structure](scan_output_structure.md) page documents what's in a scan folder and how to load each file.

## When something goes wrong

The [Troubleshooting](troubleshooting.md) page is a starter index for the errors you're likely to see and what they usually mean. The scanner emits structured logs that pair with the `geecs-log-triage` tool — `triage` parses scan logs into a markdown summary that classifies errors by source.

## Documentation map

| If you're trying to... | Read |
|---|---|
| Run your first scan | [Installation](installation.md) → [Tutorial](tutorial.md) |
| Add a new device to a scan | [Save Elements](save_elements.md) |
| Find a recorded data file | [Scan Output Structure](scan_output_structure.md) |
| Diagnose a failed scan | [Troubleshooting](troubleshooting.md) |
| Write a custom evaluator or analyzer | [Extending the Scanner](extending.md) |
| Understand the engine internals | [Architecture](architecture.md) |
| Run an optimization scan | [Optimization Example](examples/optimization/optimization_example.ipynb) |

---

*Originally developed for the HTU experiment, now used across BELLA. Designed to be portable to any GEECS-based experiment with minimal configuration.*

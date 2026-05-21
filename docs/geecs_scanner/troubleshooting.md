# Troubleshooting

This page is a starter index for the failures you're most likely to encounter and what to do about each. It's organized by where the problem becomes visible — the GUI, a scan that aborts, missing data, etc.

For a structured analysis of a specific scan's log, run the [`/triage` skill](../skills/overview.md) from a Claude Code session, or call the underlying CLI directly:

```bash
poetry run triage path/to/Scan042/Scan042.log
```

The output groups errors and warnings by category and points at the line in the log where each first appeared. It's the fastest way to figure out whether a scan failure was hardware, configuration, or code.

## At startup

### "No experiment selected" / status indicator is grey

The scanner couldn't create a `RunControl` for the experiment in your config file. Three common causes:

1. **The experiment field is blank.** Open `~/.config/geecs_python_api/config.ini` and verify the `[Experiment]` section has `expt = YourExperimentName`. The GUI will reload after a restart.
2. **The experiment isn't in the GEECS database.** If `expt` doesn't match a known experiment, the database lookup raises `KeyError` and the scanner sets `RunControl` to None. Use `Master Control` or query the database directly to confirm the experiment name.
3. **No database connection.** If your machine can't reach the MySQL database, you'll see `ConnectionError` or `ConnectionRefusedError` in the log. Check your network connection and the `[Database]` section of `config.ini`.

The status indicator stays grey and the Start Scan button is disabled until `RunControl` is successfully created.

### `GeecsDeviceInstantiationError: Check log for problem device(s)`

One or more devices in your experiment couldn't be instantiated when `RunControl` started. The error dialog now names the failing device. Either the device is offline, its TCP port is blocked, or its database entry is wrong (wrong IP, wrong port, wrong device class).

Try `ping <device_ip>` from the acquisition machine first. If that works, try connecting to the device port with telnet or netcat. If the network is fine, the device's GEECS server may need to be restarted.

### Poetry install fails on Windows

Two common patterns:

- **`poetry` is not on PATH.** The official installer drops it in `%APPDATA%\Python\Scripts` or `%USERPROFILE%\.local\bin`. Add the right directory to PATH and reopen the terminal.
- **`poetry env use` fails to find Python 3.10.** When multiple Python versions are installed, point Poetry at the right one explicitly: `poetry env use C:\Users\<you>\AppData\Local\Programs\Python\Python310\python.exe`.

If you see `pythoncom` or `pywin32` errors, run `poetry run python -m pip install pywin32` and then `poetry run python <path-to-pywin32_postinstall.py> -install`.

## During a scan

### "Conflicting Save Elements" dialog

Two save elements list the same device with different `save_nonscalar_data` or `synchronous` flags. The dialog names the device and the conflicting flag. Fix by editing one of the elements so they agree, or by deselecting one of them for this scan.

This is enforced before the scan starts so you can't end up with half a scan running with one set of flags and half with another.

### Continue / Abort dialog mid-scan

A device set or get failed. The dialog body lists the device, the variable that was being set, and the queued variables for that device (so you know what hardware state to verify). The error type tells you which flavor of failure:

- **`GeecsDeviceCommandRejected`** — the device received the command but refused it. Often means the value is out of range or the device is in the wrong mode. The scanner will already have retried up to a few times before showing the dialog.
- **`GeecsDeviceExeTimeout`** — the device didn't acknowledge in time. Usually means the device is hung or the network is slow. Retrying typically makes a hung device worse, so the dialog fires immediately on the first timeout.
- **`GeecsDeviceCommandFailed`** — the device reported a hardware-level failure. Same — no retry; immediate escalation.

**Continue** retries the same command (with the original retry policy) and proceeds. Pick this if the cause is transient — a reachable but momentarily slow device — or if you can fix the hardware while the dialog is up. **Abort** stops the scan cleanly. All data acquired so far is preserved (the s-file is sealed before any device interaction during teardown).

The status indicator turns yellow while you're deciding; it goes back to red on Continue or to red→green on Abort.

### Scan finishes with "Restore Failures" dialog

After a successful scan, the engine restores each scanned variable to its pre-scan value. If any of those restores fail, you get a single dialog at the end listing every device that failed. The data is fine — this is purely a notification that you may need to manually verify hardware state before the next scan.

Common causes: a device went offline between scan start and scan end, or the closeout action already left the device in a deliberate state and the restore conflicted. Inspect each named device manually.

## After a scan

### "I can't find my data"

Check, in order:

1. **The status indicator turned green at the end?** If it turned red briefly and then green, the scan aborted; data was still written. If it stayed orange or red and the GUI reports an error, no data was written.
2. **The experiment in `config.ini` matches what you expect?** Scans for experiment `Foo` go under `{base_path}/Foo/...`. If the config got changed mid-day, your scan might be in a different folder than yesterday's.
3. **The scan number matches what the GUI showed?** The GUI's "Current Scan" / "Previous Scan" line is the authoritative source.
4. **The base_path is reachable from where you're loading?** Acquisition machines often have a different mount than analysis machines. Use `geecs_data_utils.ScanPaths` rather than hardcoded paths to avoid this.

### "Some shots in the s-file have NaN for a device's variable"

The device didn't publish a value for that shot within the synchronization tolerance. With `synchronous: true`, this is logged at WARNING. Common causes are network jitter, an exposure time longer than the inter-shot interval, or a device that's restarted mid-scan.

If it's frequent and you can't fix it on the device side, you can widen the global time tolerance via the GUI's `Global Time Tolerance (ms)` option (under `Menu → Options`). Default is 50 ms.

### "A device's per-shot files are missing or incomplete"

When `save_nonscalar_data: true` is set, the scanner queues file-move tasks as the device writes shot files. If the queue can't drain in time, files end up in a post-scan orphan sweep that runs for up to 30 seconds. If they still aren't found, the WARNING `"Filesystem sweep tasks did not drain within 30 s"` lands in the log.

Most often this means the network share is slow or the device's local disk is full. Check disk space on the acquisition machine and the file server, and verify the share is mounted writable.

## Engine errors visible in the log

These typically don't surface as dialogs but show up in `Scan{NNN}.log` and trigger an aborted scan.

| Exception | Meaning | What to do |
|---|---|---|
| `ConfigError` / `ConflictingScanElements` | Save element validation failed | Edit the save element; the message names the offending field |
| `ScanSetupError` | Pre-logging setup couldn't finish | Look at the previous log line — usually a device set during setup |
| `TriggerError` | Shot-control device command failed | Hardware/comms issue with the trigger box (DG645 etc.) |
| `DeviceSynchronizationTimeout` | Devices didn't enter standby in 15 s | Check that all synchronous devices are running and reporting `acq_timestamp` |
| `ScanAbortedError` | Operator pressed Stop during setup | Not an error — this is the normal "stop before scan started" path |
| `DataFileError` | File move or write failed after retries | Network share blip, permission issue, or disk full |
| `OrphanProcessingTimeout` | File-mover queue didn't drain in 30 s | Slow share or many late-arriving files; look at the device file paths in the log |

The full hierarchy is in `geecs_scanner/utils/exceptions.py`.

## When the GUI freezes or won't quit

Symptoms: the window doesn't repaint, Stop Scan does nothing, the status indicator is stuck on red.

This used to happen when a worker thread blocked on a dialog that was supposed to come up on the Qt main thread. The architecture now uses a `pyqtSignal(object)` bridge for events and a `request.response_event.wait()` pattern for dialogs, so the worker can block safely while the main thread runs. If you're seeing a freeze on a current build, that's a regression worth reporting — open an issue with the scan log attached.

In the meantime: from the terminal that launched the GUI, Ctrl+C will hit the engine. If the scan thread is stuck on hardware, you may need to kill the Python process and look at the device that wasn't responding.

## When [`/triage`](../skills/overview.md) says it's a code bug

Open an issue with:

- The scan log (`Scan{NNN}.log`).
- The save element YAML(s) you used.
- Your `scan_info.ini`.
- The version (visible in the GUI title bar).
- A short description of what you were doing when it failed.

The first three pin down the configuration; the last two pin down which build and which feature path. With those four pieces, almost every reproducible bug can be diagnosed without lab access.

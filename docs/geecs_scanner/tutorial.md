# Tutorial — Your First Scan

This tutorial picks up where [Installation & Setup](installation.md) leaves off. By the end you'll have run a NOSCAN, run a 1D scan, and looked at the data both produced. Roughly 30 minutes if everything works on the first try.

## Before you start

You should already have:

- Python 3.10 installed.
- The `GEECS-Plugins` and `GEECS-Plugins-Configs` repositories cloned or mapped from the network drive.
- `poetry install` succeeded inside `GEECS-Scanner-GUI/`.
- A working GEECS database connection — your GEECS Master Control or equivalent control system is running and your `~/.config/geecs_python_api/config.ini` points at the right experiment.

If `poetry run python main.py` opens the GUI without crashing, you're ready. If it doesn't, see [Troubleshooting](troubleshooting.md).

## 1. Pick an experiment

The first time you launch, the GUI prompts you for an experiment name, a default repetition rate, and a timing configuration name. The experiment name must match an entry in your GEECS database — it's how the scanner knows which devices belong to which experiment.

Once selected, the experiment name is written to `~/.config/geecs_python_api/config.ini` and reloaded automatically next time. You can change it later from the GUI's experiment field; changing it triggers a reinitialization of the device connection layer.

If the experiment field is blank or set to a name the database doesn't know, the status indicator in the bottom-right of the window is grey and Start Scan is disabled. That's the signal that no `RunControl` could be created — fix the config file or pick a known experiment.

## 2. Build a save element

A **save element** is a YAML file that lists devices and the variables you want logged from each. The scanner GUI keeps these in `experiment-config/save_devices/`, one file per element. You can compose multiple save elements at scan time — this is how, for example, "always-on cameras" and "the diagnostic for today" become two files that get checked together when you start a scan.

Click **New Device Element** to open the editor. The editor walks you through three things:

1. **Pick devices** from the list of devices in your experiment's database. The list is queried at startup; if a device you expect is missing, it's missing from the database, not from this GUI.
2. **For each device, pick variables** to record. The variable list also comes from the database. Tick `save_nonscalar_data` for cameras and other devices that produce per-shot files (images, traces).
3. **Optional: set up `setup_action` and `closeout_action`** sequences for steps that should run before and after the scan. Useful for "open the shutter, run the scan, close the shutter."

Save the element with a memorable name. It will appear in the **Available Save Elements** list on the main window.

A minimal save element looks like this on disk:

```yaml
Devices:
  U_ModeImager:
    save_nonscalar_data: true
    synchronous: true
    variable_list:
      - exposure
      - gain
```

That's enough to record exposure and gain on `U_ModeImager` and pull camera images into the scan folder. See [Save Elements](save_elements.md) for the full schema.

## 3. Run a NOSCAN

A NOSCAN holds all variables fixed and records N shots. It's the simplest possible scan and the right thing to run first — it confirms that your save elements work, your devices are talking, and the data is landing where you expect.

On the main window:

1. **Move your save element from Available to Selected.** Double-click it, or select it and click the right-arrow. The Selected list is what's actually used for the scan.
2. **Pick "No Scan" mode** with the radio button (this is the default).
3. **Set the number of shots** you want. The wait time gets computed automatically from `(shots + 0.5) / repetition rate`.
4. **(Optional) Add a description.** It gets written into the scan's metadata file.
5. **Click Start Scan.**

The status indicator turns orange during initialization, then red while the scan is running, with the progress bar advancing as shots come in. When the scan finishes, the indicator turns green.

If a device raises an error mid-scan you'll get a Continue / Abort dialog — Continue retries the same command up to a few times before the scan moves on; Abort stops the scan cleanly with all data so far preserved.

## 4. Find your data

NOSCAN data lands at:

```
{base_path}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/scans/Scan{NNN}/
```

`base_path` comes from your `config.ini`. `Scan{NNN}` is whatever number this scan claimed (usually one more than the previous scan that day). Inside you'll find:

- `Scan{NNN}.tdms` — binary file with all logged scalar data (compatible with LabVIEW tools).
- `ScanData_scan.txt` — the *s-file*: tab-separated per-shot scalars, easy to load with pandas or `geecs_data_utils`.
- `scan_info.ini` — the scan's metadata (experiment, description, mode, parameters).
- One subfolder per device that has `save_nonscalar_data: true`, containing the per-shot files.

[Scan Output Structure](scan_output_structure.md) documents every file and how to load it.

A quick sanity check from a Python prompt:

```python
from geecs_data_utils import ScanPaths, ScanTag

tag = ScanTag(year=2026, month=5, day=8, number=42)  # use today's actual numbers
paths = ScanPaths(tag=tag, experiment="MyExperiment", read_mode=True)
df = paths.load_sfile()
print(df.head())
```

If that DataFrame has the columns you asked for, your scan worked.

## 5. Run a 1D scan

A 1D scan moves one device variable through a range while recording shots at each step.

1. **Switch to "1D Scan" mode.** The Start/Stop/Step boxes become editable.
2. **Pick the scan variable** in the dropdown. The list shows every variable that any device in your experiment exposes for setting, plus any composite scan variables you've configured.
3. **Set Start, Stop, and Step.** Watch the *number of steps* and *total shots* readouts update — if your numbers don't make sense, the readouts will tell you.
4. **Set Shots per Step.** Typical values are 10–100 depending on how much per-step averaging you need.
5. **Click Start Scan.**

The progress bar now reflects shots completed across all steps. Each step transition is logged.

The same scan folder structure is produced; the s-file additionally has a `scan_var` column recording the value of the scanned variable at each shot. Plotting `df.groupby('scan_var').mean()` is the typical first move.

## 6. (Optional) Save a preset

If you'll run the same scan repeatedly, click **Save Preset** to capture the current configuration — selected save elements, scan mode, ranges, descriptions — under a name. Presets show up in the list at the top right; double-click to load one back.

## What to read next

You now have a working data-acquisition workflow. The places to go from here:

- **[Save Elements](save_elements.md)** — the full YAML format including action sequences and composite variables.
- **[Scan Output Structure](scan_output_structure.md)** — every file produced by a scan and how to load it.
- **[Extending the Scanner](extending.md)** — write a custom evaluator for optimization scans, or a custom analyzer that runs after each scan completes.
- **[Troubleshooting](troubleshooting.md)** — what to check when things don't work.

If you want to script device control outside the GUI, see the [GEECS Python API Scripting Guide](../geecs_python_api/scripting_guide.md).

# GEECS Scanner

The GEECS Scanner is the operator front end for running scans on a GEECS
beamline, and it is a web page: open it in a browser, pick or compose a
scan, submit it, watch it run. Scans execute in the Bluesky queueserver
worker ([GeecsBluesky](../platform/fleet_map.md)); devices are reached
through the GEECS Channel-Access gateway; every scan is recorded both as a
classic GEECS scan folder and as a structured run in the Tiled catalog.

It replaced the PySide6 GEECS-Console in September 2026 (final state at
the tag `geecs-console-v0.32.1-final`). Like the console before it, the
scanner is a *client* of the queue, never an engine: everything it submits
can be submitted headlessly from a script with identical results.

## Where it runs

The scanner is a service on the worker host (systemd unit `geecs-scanner`,
port **8300**), beside the Data Portal (8200) and the logbook (8400) — one
bookmark each. Nothing is installed on an operator machine: open
`http://<worker-host>:8300/` in a browser on the lab network. The host and
the other services are on the [fleet map](../platform/fleet_map.md).

For development, the package runs anywhere from a checkout of
GEECS-Plugins, with or without a real worker:

```bash
cd GeecsScanner
poetry install
poetry run geecs-scanner --experiment <Experiment>   # the real RE Manager, from config.ini [qserver]
poetry run geecs-scanner --demo                      # an in-memory manager that runs scans by itself
```

`--demo` contacts nothing, so the page can be explored offline.

## A tour of the page

The rail on the left lists the page's sections; two **health chips** in the
header (`manager`, `doc stream`) say whether the RE Manager answers and
whether the worker's document stream is being heard.

- **Now** — the running scan: state, the claimed scan number, progress
  against the planned shots, and a live tail of the scan's own `scan.log`
  (the manager's console text sits behind a toggle). **Pause**, **Resume**
  and **Stop scan…** live here and act on the running item.
- **New scan** — the form. Start from a **preset** (a saved scan document
  from the experiment's configs repository, in a dropdown) or compose one:
  the shape is **No-scan** (N shots at a fixed configuration), **1D** (one
  variable, start → stop → steps, shots per step), **Grid** (two
  variables, outer product) or **Background** (a no-scan tagged as a
  reference); **Optimize** is greyed out until the worker has an
  optimization plan. Acquisition is **strict** (fire between trigger and
  wait, one row per shot) or **gated** (the trigger box in SCAN, cameras
  count a batch). Pick the trigger profile, then press **Start** — see
  [Running a scan](#running-a-scan) for what Start does. The current form
  can be saved back as a preset (**Save as preset…**).
- **Queue** — the running item, what waits behind it, and recent history;
  **Clear** empties what waits. The page refuses a second waiting item
  unless you say so.
- **Devices · move** — every numeric settable variable of the experiment,
  straight from the GEECS database, behind a type-ahead: type any fragment
  of the device, the variable or its alias to narrow the list. Pick one and
  the page shows its live **readback** (the gateway's readback PV, never the
  setpoint) with the reading's age; enter a value and **Move** queues a
  single move. Moves run only while the queue is idle — the panel's chip
  says `idle`, `a plan is running` or `items wait in the queue`.
- **Actions** — the experiment's action library: pick a plan, preview
  every step (nested plans inlined), **Arm**, then **Run**. Arming is never
  remembered — it is asked for on every run. Idle only.
- **Calibration** — the stored per-device shot offsets, with **Check** and
  **Measure** verbs over the devices in the New scan table and the chosen
  trigger profile. Idle only.

## Running a scan

1. Pick a preset in **New scan**, or compose a scan and pick a trigger
   profile.
2. Press **Start**. The page first runs the pre-submit **preflight**.
   Refusals (an unknown name, a variable no device serves, a worker that
   is not ready) come back as text and nothing is queued. Warnings come
   back as *questions* — stale cameras, a trigger that looks off — in a
   dialog; nothing is queued until every question is acknowledged and you
   press **Submit to queue**. The acknowledgements are stamped into the
   run's metadata. A clean preflight queues the item at once — Start is
   not a dry run (the dry run is the API's `POST /api/preflight`).
3. The item lands in the queue and, when the worker is idle, runs; **Now**
   follows it shot by shot.
4. **Stop scan…** aborts at the next safe point and restores the trigger
   to its standby state. An aborted scan's folder is kept, never deleted.

The preset *is* the submission document (`geecs_schemas.Preset`: a device
group, a plan call, a trigger profile). The scanner expands it with the
same `expand_preset` every client uses and binds no device itself, so a
preset submitted from a notebook through `geecs_bluesky.qs_client` — or
through the page's own JSON API, `POST /api/submit` — behaves exactly as
one submitted from the form.

## Where the data lands

Every scan is recorded twice, deliberately: the classic GEECS scan folder
on the data share (`scans/ScanNNN/` with `ScanInfoScanNNN.ini`, the
`ScanDataScanNNN.txt` s-file exported after the scan, `scan.log`, and each
device's natively saved per-shot files), so every existing analysis tool
keeps working; and a run in the Tiled catalog with the start/stop metadata
and one event row per shot. Only the worker ever creates a scan folder.

Read it back in the Data Portal (day → scan → metadata, scalar plots,
images, from any browser) or from Python through the `ScanCatalog` layer
and the s-file readers in [Data Utils](../geecs_data_utils/overview.md).
The column contract is `GeecsBluesky/EVENT_SCHEMA.md` in the repository.

## When something is wrong

- **`manager` chip down** — the RE Manager does not answer. The
  queueserver stack lives on the worker host; see the
  [fleet map](../platform/fleet_map.md) for the units and their probes.
- **Preflight says the worker is not ready**, or a submit is refused with
  *"not in the list of allowed plans"* — the worker's RunEngine
  environment is closed or its plan list is stale. That is a worker-side
  fix (the readiness unit reopens it); the page cannot do it.
- **`doc stream` chip down while a scan runs** — the page still shows the
  queue's state, but no shot-by-shot progress. Scans are unaffected.
- **Move, Actions and Calibration are held** — the move chip reads
  `a plan is running` or `items wait in the queue`; they open again when
  the queue is idle.
- **The scan folder is missing data** — start at the scan's `scan.log`
  (the **Now** tail, or the file in the scan folder) and the repository's
  `/triage` skill.

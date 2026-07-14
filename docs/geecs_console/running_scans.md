# Running Scans

This page covers the scan form: what each choice means physically, and
what the engine does with it.

## Scan modes

- **No-scan** — record N shots at a fixed configuration ("statistics
  collection"). Under the hood it is a step scan with no scan variable —
  one bin, N shots — so everything on this page applies to it too.
- **1D scan** — sweep one scan variable across start → stop in steps,
  recording *shots per step* at each position. Scan variables are named
  entries from the experiment's scan-variable catalog (see the Editors
  menu) — a device:variable pair, optionally with a *confirmation*
  readback (set X, confirm on Y) for devices whose setpoint register is
  not proof of physical motion.
- **Grid** — two variables, outer product; the first axis is the
  outermost (slowest). Only axes whose target changed are re-moved
  between points.
- **Optimization** — an Xopt-driven scan: each iteration evaluates an
  objective from the latest bin's data and proposes the next point. Pick
  an optimizer config (YAML in the experiment's `optimizer_configs/`);
  the save sets must include the objective's diagnostics.
- **Background** — a No-scan tagged as a calibration/background reference.

## Acquisition style: `free_run` vs `strict`

This is the most physics-relevant setting on the form. Both styles write
the same data schema; they differ in how a shot becomes a row.

**`free_run` (default)** — the external trigger free-runs at the machine
rep rate. One synchronous device acts as the **reference** (pacemaker):
each advance of its acquisition timestamp creates one event row, and every
other device contributes its latest data to that row, labeled with how
well it kept up (see `shot_offset`/`valid` in [Scan Data](scan_data.md)).
A slow or dropped device never blocks the scan — its cells are truthfully
labeled instead. Use this for normal rep-rated running.

**`strict`** — every listed device must be present on every shot. The
engine takes ownership of the trigger: arm, confirm the trigger is
quiescent, fire exactly one shot, wait for *all* devices to report it,
record the row. A device that misses a shot triggers a bounded re-fire; a
device that is genuinely down aborts with a clear message. Use this when
per-shot completeness matters more than rate — strict is structurally a
low-rate mode (each shot includes a blocking trigger fire; ~2 Hz is the
practical ceiling, whereas free-run keeps up with the 5 Hz system limit).

During a free-run scan the trigger is bracketed per step: on (SCAN) while
shots are taken, standby during moves — for experiments where the trigger
gates a physical system (e.g. a gas jet), that means "jet on during
shots, off during moves", matching long-standing operating practice.

## Shots, timing, and what "rep rate" does

*Shots per step* is literal: rows recorded per position. The session-bar
rep rate tells the engine the trigger period — it scales progress
estimates and the shot-numbering arithmetic. Get it right: it is recorded
into the scan's metadata and used to derive per-device shot IDs.

## Trigger profiles

The trigger profile (session bar) names a timing configuration — which
device is the shot controller and what its OFF / SCAN / STANDBY /
SINGLESHOT / ARMED states set. Profiles are YAML in the experiment
configs, edited via **Editors → Shot Control**. A profile can span
multiple devices with ordered writes. Without one, free-run scans still
run (the trigger is simply not managed — expect warnings about save
windowing), and strict scans are refused.

## Presets

**Save-as** stores the *entire current form* — mode, variables, save sets,
shots, acquisition, description — as a named YAML preset in the
experiment's configs repo. **Apply** restores it, validating first: a
preset that references configs the current experiment doesn't have is
rejected with the reason in the status bar, and your form is left
untouched. Presets are plain `ScanRequest` documents — diffable,
reviewable, and shareable through the configs repository like any other
config.

## While the scan runs

The Now panel tracks lifecycle (INITIALIZING → RUNNING → DONE/ABORTED),
progress against the expected shot total, and the claimed scan number.
Pre-flight problems (e.g. all cameras stale because the trigger is off)
surface as operator dialogs with Continue/Abort — the scan waits for your
answer. **Stop** aborts at the next safe point and always restores the
trigger state; an aborted scan's folder is kept (never deleted) and
clearly logged as incomplete.

## Under the hood (one paragraph)

Start builds a `ScanRequest`, the engine resolves all names fail-fast,
connects devices, claims `ScanNNN/`, and runs the plan through the Bluesky
RunEngine: per-step actions, native image saving windowed to the
trigger-active span, background telemetry riding along, and every event
document written both to the Tiled catalog and — post-scan — exported as
the classic s-file. The full engine architecture lives in
`GeecsBluesky/CLAUDE.md`; the data contract in
`GeecsBluesky/EVENT_SCHEMA.md`, summarized in [Scan Data](scan_data.md).

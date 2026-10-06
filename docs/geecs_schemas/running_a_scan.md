# Running a scan the new way

!!! warning "Status (2026-09-10)"
    This page describes the scan-request funnel.  On the native-Bluesky
    feature branch (GEECS-Plugins#807, phase 1 PR 2) a scan is a stock
    bluesky plan item — a **preset** (device group + plan call) expanded by
    the client — and save sets no longer exist; the web scanner submits
    presets today, and this page is rewritten with that foundation. The
    MCP is **not** being rewired onto it: its write verbs were deleted in
    geecs-mcp 0.9.0 (GEECS-Plugins#727) because that server is an
    experiment, not an operator surface.

This page explains, in plain language, how a scan is described and what happens
to its data when it runs on the new engine. If you want the map of the five
config kinds first, read [Scanner Configs, Explained](schemas_overview.md); for
the exact fields of each config, see the
[Schema reference](schema_reference.md). This page is about the *run*: what you
submit, what fills the gaps, and where the data lands.

## One document describes the whole scan

A **scan request** is the complete description of one scan — the single document
you submit. It says what kind of scan it is (sweep a variable, sit still and
collect shots, or let an optimizer drive), which positions to visit and how many
shots to take, and — by name — which save set, trigger profile, and action plans
to use. A saved preset *is* a scan request.

You rarely have to spell out everything. Whatever the request leaves silent is
filled from the experiment's standing defaults, and every value that gets filled
in this way is **recorded in the run's metadata** so the record shows what the
scan actually used, not what some defaults file happened to say at the time.

### The four layers that decide what a scan does

When a request is silent on something, the answer comes from a fixed stack of
sources. From the bottom up:

1. **The GEECS experiment database** (MySQL) — the per-device, per-variable
   facts: which variables are logged for scans (`get='yes'`), their types,
   units, and limits. This is the bedrock the configs sit on; it is not itself a
   config file you edit here.
2. **`experiment_defaults.yaml`** — the experiment's standing choices: the
   default trigger profile, the setup/closeout plans every scan runs, and
   whether background telemetry is on. Applied only where the request is silent;
   it never overrides a value the request states explicitly.
3. **Save-set entry rituals** — each required device can carry its own
   `setup`/`closeout` action plans, so a device's ritual travels with it into
   any scan whose save set includes that entry.
4. **The scan request's own fields** — the most specific layer; anything stated
   here wins.

Setup plans nest like context managers on the way in — defaults first, then the
save-set entry rituals, then the scan's own — and unwind in the exact reverse on
the way out, so an experiment-wide "return the machine to standby" always runs
last. The assembled order is recorded in the run metadata (`action_plans`).

## Everything happens through actions, not database writes

Scan-start, scan-end, and between-step operations are all expressed as **action
plans** — named checklists of steps (set a variable, wait, check a readback, run
another plan). A scan request points at them in three slots:

- **setup** — before the first step,
- **per_step** — at every position, right after the move and before the shots,
- **closeout** — once at the end, in reverse order, and even if the scan aborts.

Alongside actions, three other mechanisms shape the run, and none of them is a
database write:

- the **save set's entry rituals** (per-device setup/closeout, layer 3 above),
- **`experiment_defaults.yaml`** actions (layer 2),
- the **trigger profile** — the machine's OFF / STANDBY / SCAN / SINGLESHOT /
  ARMED states, each an ordered list of device writes, driven by the shot
  controller,
- the scanner's **save-windowing** — native camera saving is switched on only
  for the trigger-stopped part of the scan, so free-running frames are never
  saved as orphan images.

If you remember configuring scan behavior through the database's scan-start /
scan-end values, that is the part that has moved.

## The database set-side is intentionally disabled

The GEECS database can, in principle, write device values at scan boundaries
(the `set='yes'` rows with their `startvalue` / `endvalue`). **The engine does
not apply those writes** — the set-side is reserved, not honored. The reason is
concrete: triggering and camera saving are first-class engine features now, so
the database's boundary writes would race the shot controller on the DG645 (the
`set='yes'` rows are the very trigger and amplitude variables the shot
controller already drives). The reserved schema fields
(`SaveSetEntry.at_scan_start` / `at_scan_end`,
`ExperimentDefaults.apply_db_scan_defaults`) are kept for a possible future
re-enable; a config that still sets them logs one warning and is otherwise
inert.

The database **get-side** is very much live — it is what decides what gets
recorded, described next.

## The two-tier recording model

"Required" and "recorded" used to be the same decision. They are two decisions
now, split across two tiers.

### Tier 1 — the save set (required devices)

The save set is the list of devices the scan *requires*: they get completeness
guarantees, a dialog if one dies, their images saved when asked, and their
setup/closeout rituals run around the scan.

What gets recorded for a Tier-1 device is its **`db_scalars` resolution**: by
default (`db_scalars=true`) the recorded scalars are the device's database
`get='yes'` variables **∪** any explicit `scalars` you list; `all_scalars=true`
unions every database variable; `db_scalars=false` (the pin the legacy converter
emits) records the explicit list only. Images are always Tier-1 — file saving
needs coordination with the device, so there is no soft version of it.

Tier-1 data goes to **both** destinations: the legacy on-disk s-file
(`ScanDataScanNNN.txt` and the `analysis/sNNN.txt` copy) **and** the Tiled
catalog.

### Tier 2 — background telemetry

Every `get='yes'` variable the run does not record itself — the required
devices and the non-essential ones are the run's; of the scanned axis's
device only the axis column is — is still recorded, the way Master Control
did it: every such scalar is read into every row, softly (`BackgroundSnapshot` in GeecsBluesky,
GEECS-Plugins#1016). This tier is safe by construction:

- it is **read from the gateway's monitor cache and never waited on**, so it
  cannot slow or stall a shot;
- the set is decided **per run**: right before the run opens, every candidate
  is probed once within a bounded budget (about a second for the whole set),
  and a device that does not answer — a PV the gateway does not serve, a
  device that went away — is left out of *that* run with a log line and named
  in the run's start document (`background_dropped`; a scanned axis's device is
  admitted at the first step instead, and one that fails then is named in the
  log only); the next run probes it again, so nothing needs a restart to come
  back;
- a reading the gateway marks INVALID (a dead device's stale readbacks) reads
  `NaN`, and every column is in every row.

Background telemetry goes to **both** destinations, like Tier 1: the columns
carry their `Device Variable` headers, so the s-file has them (as the Master
Control s-file did) and Tiled has them in the row stream.

It is on by default for the experiment
(`ExperimentDefaults.background_telemetry`, read by the worker at every scan)
and a preset can override it (`Preset.background_telemetry`). The start
document records the switch (`background_telemetry`) and the devices left out.

### The one-question test for which tier a device belongs to

Does any analysis need the device **shot-by-shot**? If yes, it is Tier-1 by
definition — synchronicity means waiting, and only required devices are waited
on. If no, it can stay Tier-2 — softness means never waiting, and the two are
mutually exclusive.

## Where the data lands

- **The s-file** (`ScanDataScanNNN.txt`, copied to `analysis/sNNN.txt`) carries
  the Tier-1 scalars and the background columns. For every **triggered
  device** the device's **`acq_timestamp`** column appears in the s-file as
  well — the raw device acquisition timestamp that ties each saved frame back
  to its scan row: the run's own devices' as their shot stamp, and a triggered
  device outside the run's as its last frame's (background telemetry), so a
  row's alignment to it is checkable. Pure-scalar devices have no stamp.
- **Tiled** holds the full per-shot event stream: Tier-1 data *and* the
  background-telemetry columns, under the same `<device>-<variable>` keys as
  any other column. The worker writes the s-file from the run's own rows at
  the stop document — no Tiled round trip.

Join saved files to scan rows by a device's `acq_timestamp`, never by the
derived `shot_id` (which counts trigger opportunities, not rows). The full
per-column contract is `GeecsBluesky/EVENT_SCHEMA.md` in the source tree.

## Two habits worth keeping

- **Typos fail loudly.** Configs are validated when loaded — a misspelled key is
  an immediate error naming the bad field, resolved fail-fast *before* any
  hardware is touched or a scan number is used up.
- **You describe intent, not mechanics.** Timestamp bookkeeping, synchronization
  flags, and parallel laser-off files are gone on purpose — the engine derives
  them, and legacy files convert automatically on load.

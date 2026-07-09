# Running a scan the new way

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

Every live experiment device with a `get='yes'` variable that is **not** in the
save set is still recorded — as soft, read-only snapshot columns read straight
from the gateway's always-on monitor cache. This tier is safe by construction:

- it is **read-only** and **never waited on**, so it can never slow or stall a
  shot;
- it is **dtype-tolerant** — each column's type is inferred from its PV, so a
  numeric variable stays numeric while an enum or string variable (a plunger
  position, a digital-output label) is logged as its label. A telemetry column
  set can mix float and string columns; one awkward non-numeric channel never
  drops the device's other columns;
- a device that is **dead at scan start is dropped with a log line** — no
  dialog, no abort; a value that can't be read mid-scan degrades to a
  type-appropriate null cell.

Background telemetry is **Tiled-only — it does not go to the s-file.** That was
a deliberate decision: the s-file is the legacy Tier-1 scalar contract that
downstream GEECS analysis consumes, and Tier-2 telemetry is a new, best-effort
record that lives alongside it in Tiled rather than reshaping the legacy file.

Telemetry is on by default for the experiment
(`ExperimentDefaults.background_telemetry`) and can be overridden per scan
(`ScanRequest.background_telemetry`). The `{device: [variables]}` actually
selected is recorded in the run metadata.

### The one-question test for which tier a device belongs to

Does any analysis need the device **shot-by-shot**? If yes, it is Tier-1 by
definition — synchronicity means waiting, and only required devices are waited
on. If no, it can stay Tier-2 — softness means never waiting, and the two are
mutually exclusive.

## Where the data lands

- **The s-file** (`ScanDataScanNNN.txt`, copied to `analysis/sNNN.txt`) carries
  the Tier-1 scalars. For **image / file-saving devices**, the device's
  **`acq_timestamp`** column now appears in the s-file as well — the raw device
  acquisition timestamp that ties each saved frame back to its scan row. It is
  surfaced only for file-saving devices; pure-scalar devices don't get it.
- **Tiled** holds the full per-shot event stream: Tier-1 data *and* the Tier-2
  background-telemetry columns (keyed `telemetry_<device>-<variable>`), plus the
  schema-v1 companion columns. The s-file is exported from the Tiled run
  best-effort after the scan.

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

# Save Sets

A **save set** is a named, reusable answer to "what should this scan
record?" — a YAML document listing devices, which of their variables to
log as scalars, whether to save their images, and what role each device
plays in shot synchronization. Save sets live in the experiment's configs
repository and are edited with **Editors → Save Sets** (or any text
editor — they're plain YAML validated by `geecs-schemas`).

## Composing sets per scan

Select one or more sets in the console; the scan records their **union**.
The intended pattern: one always-on set per beamline area (e.g. the
standard cameras) plus a diagnostic-of-the-day set, mixed and matched per
scan without editing anything. Union rules are conservative: scalar lists
merge, image flags OR together, and *conflicting explicit roles for the
same device are refused* (roles define synchronization semantics, so
overlapping sets must not disagree).

## What an entry controls

- **`scalars`** — variables recorded as per-shot columns.
- **`db_scalars`** (default on) — additionally record every variable the
  GEECS database marks for scan logging (`get='yes'`) on that device, so
  a set stays honest even when someone adds a logging variable in the DB.
- **`images`** — native per-shot file saving: the device writes directly
  into `ScanNNN/<Device>/`, windowed to the trigger-active part of the
  scan so no orphan frames are saved during moves or setup.
- **`role`** — shot-synchronization semantics in free-run mode:
  - `reference` — the pacemaker whose shots define the event rows
    (exactly one per scan; the first synchronous device by default);
  - `contributor` — synchronized, expected on every shot, labeled with
    `shot_offset`/`valid` relative to the reference;
  - `snapshot` — asynchronous, sampled once per row, no shot labels.
  In strict mode every non-snapshot device is a triggered participant.
- **`setup` / `closeout`** — named action plans (the action library) run
  once when this set is part of a scan: the "open the shutter / close the
  shutter" ritual travels with the set that needs it, deduplicated if
  several sets share a ritual.

## Everything else is telemetry

Any live experiment device with DB logging variables that is *not* in the
selected save sets is recorded anyway, as soft **background telemetry**
columns (`telemetry_<device>-...`): sampled once per row, never waited on,
never able to slow or abort a scan. The division of labor is deliberate:

> **Save sets carry guarantees** (completeness, synchronization, dialogs,
> images). **Telemetry carries context** (best-effort, free). Anything
> that must be strictly synchronized belongs in a save set.

Telemetry can be disabled per scan (`background_telemetry` on the
request) or by experiment default. Details of what telemetry columns mean
— including shot attribution for telemetry devices that happen to be
triggered — are in [Scan Data](scan_data.md).

## Legacy save elements

Save sets are the schema-world successor of the legacy GUI's "save
elements". Existing save-element YAML is auto-converted on read (actions
become entry rituals, `synchronous: false` becomes `role: snapshot`), so
old configs keep working; new configs should be written as save sets.

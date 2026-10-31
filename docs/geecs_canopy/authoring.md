# Authoring screens and policy

The [Control Room](../web_services/control_room.md) draws everything it
shows from files in the experiment's folder of the configs repository
(`<configs root>/scanner_configs/experiments/<experiment>/`). A new
control screen, a launcher entry, a sequence behind a button or a write
grant is a file edit there, never code. This page shows each file.

| File | What it holds | Schema (GEECS-Schemas) |
|---|---|---|
| `screens/<name>.yaml` | One control screen | `Screen` |
| `home.yaml` | The home's launcher | `HomeLayout` |
| `action_library/actions.yaml` | Named action plans a button may run | `ActionPlanLibrary` |
| `operator_policy.yaml` | Who may write what, and which roles need the hazard confirm | `OperatorPolicy` |

The device names below (`SimMotor1`, `eztest1`, `U_PLC`, …) are the
simulator bench's and the reference deployment's, used as examples.

## How it fits together

Canopy is SLAC's display backend: it serves live PV values over a
WebSocket and stores snapshots (its Squirrel service). The **GEECS device
module** runs inside Canopy and knows the experiment's devices: the native
EPICS devices from `native_devices.yaml` and `panel_devices.yaml`, and the
LabVIEW devices the CA gateway serves, from the GEECS DB. It judges each
role's freshness, and it is the only path for a write: every write passes
its gates (the policy, the device's allow list, the hazard confirm, the run
lock, the limits) and leaves a journal row. The Control Room holds a read
key only; it never writes to Canopy directly.

## Writing a screen

A screen is a title and groups placed on a grid. Each group is a titled
block of widgets.

```yaml
schema_version: 1
title: Sim bench
groups:
  - title: Stage
    at: [1, 1]          # its top-left grid cell: row, column (from 1)
    size: [1, 1]        # rows and columns it spans (default 1 × 1)
    columns: 1          # widgets side by side inside it (1–8), filled row by row
    tone: trace-1       # accent, slate (default), muted, trace-1 … trace-4
    widgets:
      - {kind: setpoint, label: X, bind: {device: SimMotor1, role: val, axis: m1}, precision: 3}
```

If no group spans more than one column, the groups in each column pack top
to bottom, as a LabVIEW panel does. If any group spans columns, every
group sits exactly at its `at` cell on a strict grid, gaps included. A tone is a colour token of the theme with no status
meaning; there is no way to write a literal colour, an expression, a
script or a condition in a screen. A screen binds, groups and confirms;
it never computes.

Save the file as `screens/<name>.yaml` (letters, digits, `_` and `-`)
and it is served at `/screens/<name>`. A screen that does not fit the
experiment's devices is not drawn: the page lists every problem (a device
that does not exist, a role it lacks, a role that does not fit the widget,
an action the library does not hold). `GeecsControlRoom/examples/sim_bench.yaml`
is a complete worked example.

### Bindings

A widget's `bind` is exactly one of three forms:

| Form | Example | Meaning |
|---|---|---|
| Device role | `{device: eztest1, role: current_sp}` | A role from the device's family file. On a per-axis family (a motor), add `axis: m1`. On a LabVIEW device served through the gateway, the role is the GEECS variable name: `{device: U_PLC, role: DO.Ch1}`. |
| Bare PV | `{pv: "x:u_plc:di_ch5"}` | A PV by name, **read only**: a writing widget bound to one is drawn disabled. For a value that belongs to no device role (a timestamp, a derived channel). |
| Action | `{action: bench_park}` | A plan in `action_library/actions.yaml`. Only a `button` may bind one. |

Name the device by its GEECS name, never by its PV; the PV comes from the
module.

### Widget kinds

Every widget has a `kind` and a `label` (empty: the role's own label).
Optional fields: `units` (when the record carries none), `precision`
(digits after the point, 0–12) and `confirm` (a hazard note the operator
confirms before a write; writing kinds only).

**setpoint**: an input with the role's readback beside it. Takes a writable
`setpoint` role.

```yaml
- {kind: setpoint, label: Exposure, bind: {device: UC_VisaEBeam6, role: exposure}, units: s, precision: 4}
```

**toggle**: a two-state switch. Takes a writable `bit` or `enum` role.

```yaml
- kind: toggle
  label: Output
  bind: {device: eztest1, role: output_sp}
  confirm: Make sure nothing downstream is exposed before switching the output.
```

**flipper**: a toggle with up to two indicators, in then out. Takes a
writable `bit` or `enum` role; each indicator is a binding that can be
read.

```yaml
- kind: flipper
  label: Output
  bind: {device: ps, role: output_sp}
  indicators: [{device: ps, role: output}]
```

**button**: a latching command (a jog, a reset), or an action plan. Takes a
writable `command` or `bit` role, or `{action: …}`.

```yaml
- {kind: button, label: Reset faults, bind: {device: eztest1, role: reset}}
- {kind: button, label: Park, bind: {action: bench_park}}
```

**readout**: a read-only value. Takes a `readback`, `setpoint`, `counter`,
`bit` or `enum` role, or a bare PV.

```yaml
- {kind: readout, label: Frame rate, bind: {device: UC_VisaEBeam6, role: rate}, precision: 1}
```

**annotation**: text only; the `label` is the text, and it takes no
binding.

```yaml
- {kind: annotation, label: The output takes the hazard confirm (the family's).}
```

**image**: a picture from the screens folder, named by `src` (a `.png`,
`.jpg`, `.gif` or `.webp` file there); it takes no binding.

```yaml
- {kind: image, src: chicane_layout.png}
```

## The home's launcher: `home.yaml`

```yaml
schema_version: 1
areas:
  - title: Bench
    screens: [sim_bench, control_center]   # screen file stems, in order
device_pages:
  - {title: Cameras, families: [ad_camera_chain]}
links: [scanner, portal]                   # any of scanner, portal, logbook, config_editor
```

`areas` group screens under a heading; `device_pages` give a picker over
the module's devices of those families; `links` name the other services.
A link is a word, never a URL: each service's address is a deployment
setting (`GEECS_CONTROL_ROOM_SCANNER_BROWSER_URL`,
`GEECS_CONTROL_ROOM_PORTAL_URL`, `GEECS_CONTROL_ROOM_LOGBOOK_URL`,
`GEECS_CONTROL_ROOM_CONFIG_EDITOR_URL`). A screen file that does not exist,
a family that is not installed or a service with no address is drawn
disabled with the reason.

## An action plan for a button

A button bound `{action: <name>}` runs that plan from
`action_library/actions.yaml`, the same library the scanner reads for a
scan's setup and closeout. A plan is an ordered list of steps:

```yaml
schema_version: 1
plans:
  bench_park:
    description: Park the stage at zero and check it stopped.
    steps:
      - {do: set, device: SimMotor1, variable: m1, value: 0}
      - {do: wait, seconds: 1}
      - {do: check, device: SimMotor1, variable: m1.movn, expected: 0}
  closeout:
    steps:
      - {do: run, plan: bench_park}
```

| Step | Fields | Does |
|---|---|---|
| `set` | `device`, `variable`, `value`, `wait_for_execution` | Writes the value through the module's gates. From a button every set waits for the device; `wait_for_execution` is honoured in a scan only. |
| `wait` | `seconds` (> 0) | Pauses. |
| `check` | `device`, `variable`, `expected` | Reads the value and fails the run if it does not match **exactly** (no tolerance; an enum reads as its label). Check discrete values (a bit, an enum, a word), not a motor's position. |
| `run` | `plan` | Runs another plan of the library. |

On a per-axis device, `variable` is the axis (`m1`: its setpoint to set,
its readback to check) or `<axis>.<role>` (`m1.velo`). Before step 1 the
module checks every `set` (the grant, the allow list, the run lock, the
limits); if one would be refused, nothing runs. When any step writes a
hazardous role the page asks once, showing the plan's description and its
hazard steps. The first failing step ends the run; nothing is retried. A
sequence is never scripted in a screen.

## Who may write: `operator_policy.yaml`

No file, no grant: every write is denied.

```yaml
schema_version: 1
grants:
  - operators: [A. Operator, B. Operator]
    families: [motor_record]          # every motor, within each allow list
  - operators: [A. Operator]
    devices: [U_ChicaneInner]
    roles: [current_sp, reset]        # empty: every role the device allows
  - operators: [A. Operator]
    families: [PLC Digital Output]    # a LabVIEW device's family is its GEECS device type
    roles: [DO.Ch1]                   # and a role its GEECS variable name
hazards:                              # roles that need the hazard confirm
  - devices: [U_ShutterPLC]
    roles: [DO.Ch1]
```

A grant names operators and a scope (`families` and/or `devices`), and
optionally narrows it to `roles`. A grant never widens a device's allow
list. `hazards` marks roles that need the confirm. A native device's
family file can also mark a role `confirm: hazard`, a device's `hazard`
list in `panel_devices.yaml` marks roles of that one device, and a screen
widget's `confirm` note asks too. Any of the four makes the page ask; the
module refuses an unconfirmed write to a role the policy, the family or
the panel file marks (428 `confirm_required`).

!!! warning "The policy guards against mistakes, not against users"
    The operator is the name the request carries. Nothing binds it to a
    person: anyone with the display's read key can type a granted name.
    What holds for every caller is the allow list, the hazard confirm, the
    run lock and the limits.

A LabVIEW device needs a `min` and `max` on its GEECS DB row for every
setpoint written from here (without them the write is refused,
`bounds_unknown`), and it has no run lock.

## Presets: scope and `squirrel-sync`

A preset is a Squirrel snapshot of every **registered** setpoint. The
registry is derived from the module's devices, never edited by hand: one
row per writable, non-command role of every device, its setpoint with its
readback beside it, named by the GEECS device and role. Run the sync at the
first install, after every change to `native_devices.yaml` or
`panel_devices.yaml`, and after a GEECS DB change that adds or retires a
served setpoint:

```bash
dc run --rm api geecs-canopy squirrel-sync --check   # print the drift, write nothing
dc run --rm api geecs-canopy squirrel-sync           # reconcile
```

(`dc` is the deployment's compose wrapper, in the deploy runbook.) Removing
a registry row deletes its saved values from every snapshot, so the sync
only removes rows it owns, never a LabVIEW device's row while the DB is
off, and more than ten removals in one run need `--yes` after reading
`--check`.

A preset's **scope** is what it is for, stored with the snapshot: a list
of devices, families (every device of them on the module's roster) and
screens (every device bound on `screens/<name>.yaml`). The save always
records every registered setpoint; the scope decides what a diff shows,
what a restore writes and which baseline a golden mark sets (one golden
per scope). A restore is a run in the module, every row through the same
gates as a single write; Squirrel's own restore is never used.

## Deploying

Canopy, the device module and its keys are deployed by the GeecsCanopy
recipe; the runbook is
[`GeecsCanopy/DEPLOYMENT.md`](https://github.com/GEECS-BELLA/GEECS-Plugins/blob/master/GeecsCanopy/DEPLOYMENT.md)
(install, keys, the module's settings and routes, the GEECS DB source,
upgrade and rollback). The Control Room is the `geecs-control-room` unit,
configured by the `GEECS_CONTROL_ROOM_*` settings listed in
[`GeecsControlRoom/CLAUDE.md`](https://github.com/GEECS-BELLA/GEECS-Plugins/blob/master/GeecsControlRoom/CLAUDE.md);
its read key is a secret, never a `site.env` value. Where each facility
value lives is the [site profile](../platform/site_profile.md).

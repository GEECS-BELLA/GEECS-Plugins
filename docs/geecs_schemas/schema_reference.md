<!-- GENERATED FILE — do not edit by hand.
     Regenerate with:  python -m geecs_schemas.docgen
     (or GEECS-Schemas/tests/generate_schema_reference.py).
     A no-drift test (tests/test_schema_reference.py) fails CI if this
     file falls out of step with the schema field descriptions. -->

# GEECS config schema reference

## `optimizer_config`

### OptimizerConfig

An optimization: search space, live measurements, derived outputs and generator.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `vocs` | `VOCS` | yes | — | GEST variables, objectives, constraints, constants and observables; compact input and typed canonical output are accepted. |
| `measurements` | `dict[str, SignalMeasurement \| DiagnosticMeasurement]` | yes | — | Named live scalar or diagnostic measurements. |
| `derived` | `dict[str, str \| PythonDerived]` | no | empty | Expressions or worker callables, in dependency order. |
| `generator` | `OptimizerGenerator` | yes | — | The generator recipe used to propose positions. |
| `run` | `OptimizationRun` | no | OptimizationRun(on_finish='best', seed_dumps=[], shots_per_step=5, max_iterations=None) | Run defaults and finish policy. |

Example:

```yaml
schema_version: 1
vocs:
  variables: {"Motor:Current": [-1, 1]}
  objectives: {camera.image_total: MAXIMIZE}
measurements:
  camera: {diagnostic: ExampleCamera, frames: per_bin}
generator: {name: bayes_turbo_standard}
run: {shots_per_step: 5, max_iterations: 20}
```

### VOCS

Variables, Objectives, Constraints, and other Settings (VOCS) data structure to describe optimization problems.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `variables` | `VariableDict` | yes | — | variable names with bounds or discrete sets |
| `objectives` | `ObjectiveDict` | no | {} | objective names with type of objective |
| `constraints` | `ConstraintDict` | no | {} | constraint names with a list of constraint type and value |
| `constants` | `ConstantDict` | no | {} | constant names and values passed to evaluate function |
| `observables` | `ObservableDict` | no | {} | observables tracked alongside objectives and constraints |

### SignalMeasurement

A subscribed scalar in the shot reading.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `reduce` | `'mean' \| 'median' \| 'min' \| 'max' \| 'sum' \| 'std'` | no | 'mean' | Reduction of valid per-shot scalar values. |
| `min_shots` | `int` | no | 1 | Minimum valid shots needed for a finite measurement. |
| `signal` | `str` | yes | — | Device:Variable to read on every shot. |

### DiagnosticMeasurement

A diagnostic document applied to live frames.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `reduce` | `'mean' \| 'median' \| 'min' \| 'max' \| 'sum' \| 'std'` | no | 'mean' | Reduction of valid per-shot scalar values. |
| `min_shots` | `int` | no | 1 | Minimum valid shots needed for a finite measurement. |
| `diagnostic` | `str` | yes | — | Diagnostic document ID (file stem). |
| `frames` | `'per_bin' \| 'per_shot'` | no | 'per_bin' | Analyze the mean frame, or analyze each frame before reducing scalars. |
| `overrides` | `dict[str, JsonValue]` | no | empty | Steps/measure overrides deep-merged into the diagnostic's recipe. |

### PythonDerived

Worker callable over the reduced scalars, for non-expressible objectives.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `python` | `str` | yes | — | Importable module:function accepting a scalar mapping and returning a float. |

### OptimizerGenerator

Generator recipe and its algorithm-specific settings.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `name` | `str` | yes | — | Worker generator recipe name. |
| `options` | `dict[str, JsonValue]` | no | empty | Recipe-specific options validated by the worker before opening a run. |

### OptimizationRun

Run defaults; queue-item arguments may override the iteration and shot budgets.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `on_finish` | `'best' \| 'hold'` | no | 'best' | Move to the best feasible observation, or hold; no best restores initial positions. Relative pseudos always restore on unstage. |
| `seed_dumps` | `list[str]` | no | empty | Previous xopt_dump.yaml files; relative paths resolve beside this config. |
| `shots_per_step` | `int` | no | 5 | Successful strict acquisitions per iteration. |
| `max_iterations` | `int (optional)` | no | None | Iteration limit; required here or in the submitted plan arguments. |

## `preset`

### Preset

A saved scan: the device group plus the plan call.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `name` | `str` | yes | — | The name clients use to pick this preset. |
| `description` | `str` | no | '' | What this scan is for — becomes the run's description (ScanStartInfo in the legacy ScanInfo file). |
| `trigger_profile` | `str (optional)` | no | None | Name of the trigger profile driving the shots. Leave unset to use the experiment default (experiment_defaults.yaml). |
| `background` | `bool` | no | False | Flag this scan as a background measurement (metadata only: ScanMode 'background' in ScanInfo, 'background' in the run). |
| `native_image_save` | `bool (optional)` | no | None | Whether cameras whose frames the PVA gateway's file plugin captures also write their LabVIEW per-shot files (PNGs) this scan. Leave unset for the experiment default (experiment_defaults.yaml). Off keeps the plugin's HDF5 stack as those cameras' only record. A device without a file plugin (no PVA stream, a proprietary format) always writes its native files whatever this says, and a scalars-only entry (save_images off) is unaffected. |
| `background_telemetry` | `bool (optional)` | no | None | Whether every other logged scalar of the experiment — the variables the GEECS experiment database marks for scan logging (expt_device_variable, get='yes') on the devices not in this scan — is read into every row as well, softly: from the gateway's monitor cache, never waited on; a device that does not answer when the scan starts is left out of that scan with a log line, and one the gateway marks INVALID reads NaN. Leave unset for the experiment default (experiment_defaults.yaml). Off records this scan's own devices alone. |
| `devices` | `list[PresetDevice]` | no | empty | The devices recording this scan, one entry per device. Every subscribed scalar of each is a column of every row; each entry chooses whether its images are saved. An empty list scans with the motors' readbacks only. |
| `plan` | `PlanCall (optional)` | no | None | The stock plan call this preset submits. Unset means a device group with no scan attached yet. |

Example:

```yaml
schema_version: 1
name: emq1_scan
description: emq1 scan after bax alignment
trigger_profile: HTU-Normal          # omit to use the experiment default
devices:
  - device: UC_ALineEBeam3           # frames saved (the default)
  - device: UC_VisaEBeam1
    save_images: false               # scalars only, no frames on disk
  - device: U_BCaveICT
plan:
  name: scan                         # a stock bluesky plan the worker registers
  args: ["EMQ1 Current", 1.2, 1.7, 6]  # motor (catalog name or Device:Variable), start, stop, points
  kwargs: {shots_per_step: 20}       # rows recorded at every position
```

### PresetDevice

One device of the group: it records every shot of the scan.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `device` | `str` | yes | — | GEECS device name exactly as it appears in the GEECS experiment database (MySQL), e.g. 'UC_ALineEbeam1'. |
| `save_images` | `bool` | no | True | Save the device's images / non-scalar files (camera frames, traces) beside the scalar data. Off records the device's scalars only — its per-shot readings still land in every row; the frames stay off the disk. Meaningless for a scalar-only device (nothing to save either way). |
| `essential` | `bool` | no | True | Wait for this device on every shot (on, the default) — a shot is not complete without its reading. Off streams the device for the run's duration instead, at its own rate, joined to the shots by its timestamp: it never holds a shot or aborts the scan, so use it for a slow or unreliable device whose data are welcome but not required. The device needs a shot timestamp (acq_timestamp). |

### PlanCall

The stock plan the preset runs, with its arguments.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `name` | `str` | yes | — | The stock bluesky plan to run, e.g. 'count', 'scan', 'list_scan', 'grid_scan' — one of the names the worker registers. |
| `args` | `list[JsonValue]` | no | empty | Positional arguments after the detectors, in the plan's own order — e.g. ['EMQ1 Current', 1.2, 1.7, 6] for scan (motor, start, stop, number of points). A scan variable is a string: 'Device:Variable' or a scan-variable catalog name. |
| `kwargs` | `dict[str, JsonValue]` | no | empty | Keyword arguments — e.g. {num: 100} for count, {shots_per_step: 10} for the scan verbs (rows recorded at every position). |

## `scan_request`

### ScanRequest

One complete scan, ready to submit: what to do, what to save, how to trigger.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 4 | Format version of this config file. Leave at 4 — tools update this automatically when the file format changes. |
| `mode` | `ScanRequestMode` | yes | — | What kind of scan: 'step' sweeps one or more axes, 'noscan' collects shots without moving anything. |
| `axes` | `list[ScanAxis]` | no | empty | For step scans: what to sweep. One entry is a simple 1-D scan; several entries form a grid visiting every combination, with the first axis as the outermost (slowest) loop and the last as the innermost (fastest). Leave empty for noscan. |
| `capture` | `CaptureSettings` | no | CaptureSettings(shots_per_step=1, acquisition=<AcquisitionMode.STRICT: 'strict'>, save_sets=[], background_telemetry=None, trigger_profile=None) | How shots are taken and what gets recorded: shots per step, acquisition discipline, save sets, the telemetry toggle and the trigger profile. Omit for a one-shot strict capture with no named save sets. |
| `actions` | `ActionBindings` | no | ActionBindings(setup=[], per_step=[], closeout=[]) | Named action plans to run before the scan (setup), between steps (per_step), and after it (closeout). |
| `description` | `str` | no | '' | Free-text note about this scan; it ends up in the scan's metadata and the experiment log. |
| `background` | `bool` | no | False | Mark this scan's data as background/calibration shots so analysis can find them later. |

Example:

```yaml
schema_version: 4
mode: step
axes:
  - variable: jet_z
    positions: {start: 4.0, end: 6.0, step: 0.5}
  # add more axes to scan a grid — the first axis is the outermost
  # (slowest) loop, the last the innermost (fastest), e.g.:
  # - variable: gas_pressure
  #   positions: {values: [1.5, 2.0, 2.5]}
capture:
  shots_per_step: 10
  acquisition: free_run
  save_sets: [undulator_baseline, aux_diagnostics]  # unioned; a bare string also works
  trigger_profile: htu_shot_control
actions:
  setup: [pre_scan_ebeam]
  per_step: []
  closeout: []
description: "jet z scan with probe"
# v1 documents (the capture fields flat at the top level) still validate —
# they are lifted into this shape automatically.
```

### ScanAxis

One swept variable and the positions it visits.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `variable` | `str` | yes | — | The friendly name of the variable this axis sweeps (from the experiment's scan-variables catalog). |
| `positions` | `PositionRange \| PositionList` | yes | — | The positions this axis visits, either as {start, end, step} or as {values: [...]}. |

### PositionRange

Scan positions given as start / end / step size.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `start` | `float` | yes | — | First position of the sweep. |
| `end` | `float` | yes | — | Last position of the sweep. |
| `step` | `float` | yes | — | Spacing between positions. Its sign is ignored — the sweep direction comes from start and end. |

### PositionList

Scan positions given as an explicit list of values.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `values` | `list[float]` | yes | — | The exact positions to visit, in the order given. |

### CaptureSettings

How shots are taken and what gets recorded — the capture concern.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `shots_per_step` | `int` | no | 1 | How many shots to take at each scan position / grid point (or in total for a noscan). |
| `acquisition` | `AcquisitionMode` | no | 'strict' | 'strict' fires shot by shot and guarantees every device is in every row; 'free_run' lets the trigger run at the machine rate and matches devices up by timestamp. |
| `save_sets` | `list[str]` | no | empty | Names of the save sets — reusable named device groups — recorded for this scan; devices are unioned across them. Each names the devices that get guarantees (completeness, dialogs, images, rituals). A bare string is accepted and stored as a one-element list. Empty means no required devices beyond scan bookkeeping. |
| `background_telemetry` | `bool (optional)` | no | None | Also log every other live experiment device as best-effort snapshot columns — the variables the GEECS experiment database marks for scan logging (MySQL table expt_device_variable, get='yes') — read from the gateway's always-on monitor cache: read-only and never waited on, so it cannot slow or stall the scan; dead devices are dropped with a log line, never a dialog or abort. Leave unset to inherit the experiment default; set true/false to override for this scan. |
| `trigger_profile` | `str (optional)` | no | None | Name of the trigger profile that drives the shot trigger. Unset means the scan does not manage the trigger. |

### ActionBindings

Which named action plans run around (and inside) the scan.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `setup` | `list[str]` | no | empty | Plans to run once before the scan starts. |
| `per_step` | `list[str]` | no | empty | Plans to run between scan steps — after each move, before the shots at that position. |
| `closeout` | `list[str]` | no | empty | Plans to run once after the scan finishes (even on abort). |

## `scan_variables`

### ScanVariables

The experiment's catalog of scannable variables, keyed by friendly name.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `variables` | `dict[str, ScanVariable \| PseudoScanVariable]` | yes | — | All scannable variables, keyed by the friendly name shown when setting up a scan. |

Example:

```yaml
schema_version: 1
variables:
  jet_z:
    target: "U_ESP_JetXYZ:Position.Axis 3"
    kind: motor
  gas_pressure:
    target: "U_HP_Daq:AnalogOutput.Channel 1"
  e_beam_angle_x:
    kind: pseudo
    mode: relative
    targets:
      - target: "U_S3H:Current"
        forward: "composite_var * 1"
      - target: "U_S4H:Current"
        forward: "composite_var * -2"
```

### ScanVariable

A friendly name for one device variable you can scan.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `target` | `str` | yes | — | The device variable this name moves, written as 'Device:Variable', e.g. 'U_ESP_JetXYZ:Position.Axis 3'. |
| `kind` | `'motor' \| 'setpoint'` | no | 'setpoint' | 'setpoint' = write the value and wait for the device to accept it (the default). 'motor' = additionally poll the readback until the device reports it arrived — use for real positioners. |
| `description` | `str (optional)` | no | None | Free text for the people who edit this file: what the variable is for, units, anything the name does not say. |
| `confirm` | `str (optional)` | no | None | Optional 'Device:Variable' that *measures* the result when it differs from the variable being set — e.g. set a supply's current limit but confirm on its measured current. Leave unset when the set variable is also the readback (the common case). Declared but not yet enforced by the engine in v1. |

### PseudoScanVariable

A friendly name that moves several devices together from one number.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'pseudo'` | yes | — | Variable type. 'pseudo' moves several devices from one number. |
| `targets` | `list[PseudoComponent]` | yes | — | The devices this variable moves, each with its own formula. |
| `mode` | `CompositeMode` | yes | — | 'absolute' = each device goes exactly where its formula says. 'relative' = each device is offset from where it was when the scan started. |
| `description` | `str (optional)` | no | None | Free text for the people who edit this file — for a steering bump, the geometry and assumptions behind the coefficients (drift lengths, equal kick per ampere), so the numbers can be audited later. |
| `inverse` | `str (optional)` | no | None | Formula recovering the scanned number from the components' positions. Name a component by its device name when only one target uses that device ('U_ChicaneInner'), or always by its full target with every non-alphanumeric character replaced by '_' ('U_ESP302_02_Position_Axis_3'): e.g. '560968.636 * U_ChicaneInner**2 / 100**2'. Required when a 'forward' is not linear in the scanned value; leave unset for linear relations, which the software inverts itself. |

### PseudoComponent

One device a pseudo variable moves, and the formula for its value.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `target` | `str` | yes | — | The device variable to move, written as 'Device:Variable', e.g. 'U_S1H:Current'. |
| `forward` | `str` | yes | — | Formula for this device's value in terms of the scanned number: plain arithmetic with 'composite_var' (or its short alias 'x') as the scanned value — e.g. 'composite_var * -2', 'x * -2', or '8.5 + (composite_var-10)*2.5'. |

## `trigger_profile`

### TriggerProfile

The device writes that drive the machine through its trigger states.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 2 | Format version of this config file. Leave at 2 — tools update this automatically when the file format changes. |
| `name` | `str` | yes | — | The name scans use to refer to this trigger profile. |
| `states` | `dict[TriggerState, list[TriggerWrite]]` | no | empty | For each trigger state, the writes that put the machine into it, applied in order from top to bottom. A transition may write several devices. Omit a device variable from a state to leave it untouched. |
| `description` | `str` | no | '' | Optional note about what setup this profile is for. |
| `use_scan_mode` | `bool` | no | False | Run scans under this profile in GEECS scan mode, as Master Control drives the devices: the worker announces the scan to the experiment's devices (ScanStart), pushes each shot's save path to them before the fire, and ends the scan (ScanEnd) when the run closes. A device then acquires once per announced shot, saves shot-numbered files and applies its own database start/end values. Off by default. Strict acquisition only: a gated run and the calibration plans refuse such a profile. A profile with this flag and no writes at all is the box-less case — the shot message is the only trigger. |

Example:

```yaml
schema_version: 2
name: htu_shot_control
# each state lists its writes IN ORDER (top to bottom); a transition may
# touch several devices
states:
  OFF:
    - {device: U_DG645_ShotControl, variable: Amplitude.Ch AB, value: "0.5"}
    - {device: U_DG645_ShotControl, variable: Trigger.Source,
       value: Single shot external rising edges}
  STANDBY:
    - {device: U_DG645_ShotControl, variable: Amplitude.Ch AB, value: "0.5"}
    - {device: U_DG645_ShotControl, variable: Trigger.Source,
       value: External rising edges}
  SCAN:
    - {device: U_DG645_ShotControl, variable: Amplitude.Ch AB, value: "4.0"}
    - {device: U_DG645_ShotControl, variable: Trigger.Source,
       value: External rising edges}
    - {device: U_GasJetPLC, variable: DO.Jet, value: "on"}
  ARMED:
    - {device: U_DG645_ShotControl, variable: Amplitude.Ch AB, value: "4.0"}
    - {device: U_DG645_ShotControl, variable: Trigger.Source,
       value: Single shot external rising edges}
  SINGLESHOT:
    - {device: U_DG645_ShotControl, variable: Trigger.ExecuteSingleShot,
       value: "on"}
```

### TriggerWrite

One device variable set during a state transition.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `device` | `str` | yes | — | The device to write to, e.g. 'U_DG645_ShotControl' or a gas-jet controller — any settable device can take part in a transition. |
| `variable` | `str` | yes | — | Which variable on the device to set, e.g. 'Trigger.Source'. |
| `value` | `str` | yes | — | The value to send, exactly as the device expects it — a number as text ('4.0'), a word ('on'), or a device option name ('External rising edges'). |

## `action_plan`

### ActionPlan

An ordered checklist of steps the scanner runs for you.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `steps` | `list[SetStep \| WaitStep \| CheckStep \| RunPlanStep]` | yes | — | The steps to perform, in order from top to bottom. |
| `description` | `str` | no | '' | Optional note to your future self about what this plan does and when to use it. |

Example:

```yaml
schema_version: 1
description: "Zero the pressure voltage and confirm the PLC output"
steps:
  - do: set
    device: U_HP_Daq
    variable: AnalogOutput.Channel 1
    value: 0
  - do: wait
    seconds: 3
  - do: check
    device: U_148_PLC
    variable: DI.Ch17
    expected: "off"
  - do: run
    plan: close_gaia_internal_shutters
```

### SetStep

One step that sets a device variable to a value.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `do` | `'set'` | yes | — | Step type. 'set' writes a value to a device variable. |
| `device` | `str` | yes | — | Name of the device to command, e.g. 'U_148_PLC'. |
| `variable` | `str` | yes | — | Which variable on the device to set, e.g. 'DO.Ch9'. |
| `value` | `str \| float \| int` | yes | — | The value to write — a number, or a word the device understands such as 'on' or 'off'. |
| `wait_for_execution` | `bool` | no | True | Wait for the device to confirm the change before moving to the next step. Leave on unless you know the step is fire-and-forget. |

### WaitStep

One step that simply pauses for a number of seconds.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `do` | `'wait'` | yes | — | Step type. 'wait' pauses the plan for a fixed time. |
| `seconds` | `float` | yes | — | How long to pause, in seconds (must be greater than 0). |

### CheckStep

One step that reads a device variable and verifies its value.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `do` | `'check'` | yes | — | Step type. 'check' reads a device variable and stops the plan with an error if the value is not what you expected. |
| `device` | `str` | yes | — | Name of the device to read, e.g. 'U_GaiaSVEReader'. |
| `variable` | `str` | yes | — | Which variable on the device to read, e.g. 'InternalShutterA'. |
| `expected` | `str \| float \| int` | yes | — | The value the reading must match for the plan to continue — a number or a word such as 'on'. |

### RunPlanStep

One step that runs another named action plan.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `do` | `'run'` | yes | — | Step type. 'run' executes another named plan from the library. |
| `plan` | `str` | yes | — | Name of the plan to run, as listed in the action library. |

## `action_plan_library`

### ActionPlanLibrary

The collection of all named action plans for an experiment.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `plans` | `dict[str, ActionPlan]` | yes | — | All named plans, keyed by the name used to invoke them. |

Example:

```yaml
schema_version: 1
plans:
  zero_pressure_voltage:
    steps:
      - do: set
        device: U_HP_Daq
        variable: AnalogOutput.Channel 1
        value: 0
  experiment_closeout:
    steps:
      - do: run
        plan: zero_pressure_voltage
```

## `experiment_defaults`

### ExperimentDefaults

Per-experiment fallbacks applied where a scan request is silent.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `trigger_profile` | `str (optional)` | no | None | Name of the trigger profile to use when a scan doesn't name one. Leave unset if scans must always choose explicitly. |
| `native_image_save` | `bool` | no | True | Whether cameras whose frames the PVA gateway's file plugin captures also write their LabVIEW per-shot files (PNGs) when a scan does not say (the preset's 'native_image_save' unset). On by default: the dual-write is the rollout's parity evidence until PNG retirement (GEECS-Plugins#738). Set false to run the experiment on the plugin's HDF5 stacks alone; a device without a file plugin always keeps its native files. The worker reads this at every scan, so an edit takes effect at the next scan without reopening its environment. |
| `actions` | `DefaultActions` | no | DefaultActions(setup=[], closeout=[]) | Action plans every scan runs by default — setup plans run first (before the scan's own), closeout plans run last (after the scan's own). |
| `apply_db_scan_defaults` | `bool` | no | True | RESERVED AND NOT HONORED. The DB's scan start/end writes (MySQL table expt_device_variable: rows with set='yes', their startvalue/endvalue) are applied by the DEVICE itself on the ScanStart/ScanEnd commands of GEECS scan mode, never by the engine: under a trigger profile with use_scan_mode they take effect exactly as under Master Control, and without one they do not. This flag changes neither. Kept for a possible future engine-side application. This is the set-side only: the get-side 'db_scalars' (standard telemetry) and 'background_telemetry' are separate and remain active. |
| `background_telemetry` | `bool` | no | True | Read every logged scalar of the experiment that is not in a scan's device group — the variables the GEECS experiment database marks for scan logging (MySQL table expt_device_variable, get='yes') — into every row of the scan as well, the way Master Control did: softly, from the gateway's monitor cache, never waited on. A device that does not answer when the scan starts is left out of that scan with a log line (and named in the run's start document), one the gateway marks INVALID reads NaN; nothing here can stall or fail a scan. On by default so no data is silently lost; a preset can override it with its own 'background_telemetry'. The worker reads this at every scan, so an edit takes effect at the next scan without reopening its environment. |
| `description` | `str` | no | '' | Optional note about what these defaults are for. |

Example:

```yaml
schema_version: 1
# applied only where a scan request is silent: defaults run first,
# then the scan's own
trigger_profile: htu_shot_control
actions:
  setup: [pre_scan_checklist]
  closeout: [experiment_closeout]
background_telemetry: true   # soft-log every live device not in the save set
description: "HTU standing defaults"
```

### DefaultActions

The action plans every scan of the experiment runs by default.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `setup` | `list[str]` | no | empty | Names of action plans to run before every scan, ahead of any setup plans the scan itself lists. |
| `closeout` | `list[str]` | no | empty | Names of action plans to run after every scan, after any closeout plans the scan itself lists (teardown mirrors setup: these are the outermost bracket). |

## `interlocks`

### InterlockSet

The experiment's software interlocks, armed on the scan worker.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `interlocks` | `list[Interlock]` | no | empty | The interlocks, in display order. |

Example:

```yaml
schema_version: 1
# each interlock pauses the scan while unsafe, runs on_trip once, and
# resumes once safe for settle_s; a disconnected device is always unsafe
interlocks:
  - name: chamber vacuum
    condition:
      kind: threshold
      device: U_HP_Vacuum
      variable: Pressure
      operator: ">"
      limit: 1.0e-5
      hysteresis: 2.0e-6
    on_trip: close_shutters
    before_resume: open_shutters
    settle_s: 5
  - name: gauge alive
    condition:
      kind: stale
      device: U_HP_Vacuum
      max_age_s: 3
```

### Interlock

One interlock: a condition, and what to do when it trips.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `name` | `str` | yes | — | Short name shown to operators and on the status stream. |
| `description` | `str` | no | '' | Optional note on what this interlock protects. |
| `enabled` | `bool` | no | True | Set false to keep the entry without arming it. |
| `condition` | `ThresholdCondition \| AlignmentCondition \| StaleCondition` | yes | — | The condition that makes this interlock unsafe. |
| `on_trip` | `str (optional)` | no | None | Action plan (from the experiment's action library) to run once when the interlock trips, e.g. 'close_shutters'. |
| `before_resume` | `str (optional)` | no | None | Action plan to run once the condition is safe again, just before the scan resumes. |
| `settle_s` | `float` | no | 0.0 | Seconds to wait after the condition is safe again before resuming. |

### ThresholdCondition

Unsafe when a variable crosses a limit.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'threshold'` | yes | — | Condition type. 'threshold' compares one variable to a limit. |
| `device` | `str` | yes | — | Device to watch, e.g. 'U_HP_Vacuum'. |
| `variable` | `str` | yes | — | Variable on the device, e.g. 'Pressure'. |
| `operator` | `'<' \| '<=' \| '>' \| '>='` | yes | — | The reading is UNSAFE when 'value <operator> limit' holds, e.g. '>' trips above the limit. |
| `limit` | `float` | yes | — | The limit the reading is compared against. |
| `hysteresis` | `float` | no | 0.0 | How far back past the limit the reading must come before the interlock counts as safe again (0 = resume as soon as it is back). |

### AlignmentCondition

Unsafe when a variable strays from a target variable by ``tolerance`` or more.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'alignment'` | yes | — | Condition type. 'alignment' compares a variable with a target variable. |
| `device` | `str` | yes | — | Device carrying the watched variable. |
| `variable` | `str` | yes | — | The watched variable, e.g. 'CentroidX'. |
| `target_device` | `str (optional)` | no | None | Device carrying the target variable (default: the same device). |
| `target_variable` | `str` | yes | — | The target variable, e.g. 'TargetX'. |
| `tolerance` | `float` | yes | — | UNSAFE when \|value - target\| is this large or larger (in the variable's units). |
| `hysteresis` | `float` | no | 0.0 | How much closer than the tolerance the value must come before the interlock counts as safe again. |

### StaleCondition

Unsafe when a device stops publishing.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'stale'` | yes | — | Condition type. 'stale' trips when a device stops publishing. |
| `device` | `str` | yes | — | Device to watch. |
| `max_age_s` | `float (optional)` | no | None | Seconds without a new acq_timestamp before the device counts as stale. Leave empty to trip only on a disconnect. |

## `derived_channels`

### DerivedChannels

A file of computed read-only PVs for the CA gateway.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `derived_channels` | `list[DerivedChannel]` | no | empty | Derived PVs to expose. Each entry computes one read-only float PV from numeric push-frame values. Cross-device entries use latest-value semantics with stale_after. |

Example:

```yaml
schema_version: 1
derived_channels:
  - device: TargetChamberPressure
    variable: Pressure
    expression: "10**(v - 6)"
    inputs:
      - symbol: v
        device: U_VacuumGauge
        variable: "AI_mean.Channel 0"
    egu: Torr
    precision: 6
    description: "Convectron pressure from U_VacuumGauge analog input 0"
```

### DerivedChannel

One read-only float PV computed from a numeric expression.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `device` | `str` | yes | — | Device component of the output PV, e.g. 'U_ChamberVac' for 'undulator:u_chambervac:pressure'. This may be semantic and does not need to be a real GEECS hardware device. |
| `variable` | `str` | yes | — | Variable component of the output PV, e.g. 'Pressure'. The gateway normalizes it using the same rules as raw GEECS variables. |
| `expression` | `str` | yes | — | Numeric formula for the output value, using input symbols and the gateway's restricted arithmetic subset. Example: '10**(v - 5)'. |
| `inputs` | `list[DerivedInput]` | yes | — | Input variables available to the expression. Inputs from one source device are frame-coherent; inputs spanning devices use latest-value semantics and require stale_after. |
| `stale_after` | `float (optional)` | no | None | Maximum input age in seconds for latest-value derived channels. Required when inputs span more than one source device. Leave unset for same-device frame-coherent expressions. |
| `experiment` | `str (optional)` | no | None | Optional experiment prefix override for the output PV. Leave unset to use the gateway's launched experiment. |
| `pv` | `str (optional)` | no | None | Optional explicit output PV variable component. Leave unset to use the 'variable' field. |
| `egu` | `str` | no | '' | Engineering units displayed by CA clients, e.g. 'Torr'. |
| `precision` | `int` | no | 3 | Number of decimal places CA clients should display. |
| `lo` | `float (optional)` | no | None | Optional lower display limit for the output PV. This is metadata only; it is not an alarm or control limit. |
| `hi` | `float (optional)` | no | None | Optional upper display limit for the output PV. This is metadata only; it is not an alarm or control limit. |
| `deadband` | `float` | no | 0.0 | Monitor deadband for the computed float value. Leave at 0.0 to post every changed value and suppress only exact repeats. |
| `description` | `str` | no | '' | Operator-facing note describing what the derived PV represents, for example the gauge model or calibration provenance. |

### DerivedInput

One source variable bound to a symbol in a derived-channel formula.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `symbol` | `str` | yes | — | Python-style symbol used in the expression, e.g. 'v' for a voltage input. Must be a valid identifier and must not shadow a reserved math function or constant. |
| `device` | `str` | yes | — | GEECS source device that provides this input variable, e.g. 'U_DaqPad1'. Inputs may span devices only when the derived channel declares stale_after. |
| `variable` | `str` | yes | — | GEECS source variable on the input device, e.g. 'Analog Input 10'. The gateway subscribes to it even if it is not exposed as its own raw readback PV. |

## `archive_policy`

### ArchivePolicy

The experiment's archive-set exceptions and sampling defaults.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `exclude` | `list[str]` | no | empty | fnmatch globs over full lowercase PV names that the derived rule would include but this experiment does not archive. |
| `include` | `list[str]` | no | empty | Full PV names to archive beyond what the rule derives (e.g. a gateway diagnostic such as 'undulator:cagateway:uptime'). Listed PVs are never paused by onboarding; 'exclude' does not apply to them. |
| `include_setpoints` | `bool` | no | True | Archive the ':SP' setpoint PV of every settable variable. They change only on puts and carry the operator's intent and the refused-write alarm history. |
| `include_status` | `bool` | no | True | Archive every device's 'connected' status PV (state changes only) and the gateway's own heartbeat and devices_connected. |
| `include_derived` | `bool` | no | True | Archive the gateway's derived channels declared in the experiment's gateway/derived_channels.yaml. |
| `default_sampling_period` | `float` | no | 1.0 | Sampling period (seconds) for PVs no override matches. |
| `default_sampling_method` | `'MONITOR' \| 'SCAN'` | no | 'MONITOR' | Sampling method for PVs no override matches. |
| `sampling_overrides` | `list[SamplingOverride]` | no | empty | Per-glob sampling rules; the last matching entry wins per field. |

Example:

```yaml
schema_version: 1
# the archive set is derived from the DB (monitored readbacks, :SP, connected,
# derived channels); this file lists the exceptions
exclude:
  - "undulator:uc_*:image_size*"
include:
  - "undulator:cagateway:uptime"   # beyond the rule; never paused
include_setpoints: true
sampling_overrides:
  - match: "undulator:u_vacuumgauge:*"
    sampling_period: 10             # slow-moving: one stored sample per 10 s
```

### SamplingOverride

A sampling rule for the PVs matching one glob.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `match` | `str` | yes | — | fnmatch glob over the full lowercase PV name. |
| `sampling_period` | `float (optional)` | no | None | Sampling period in seconds (the appliance's samplingperiod). |
| `sampling_method` | `'MONITOR' \| 'SCAN' (optional)` | no | None | MONITOR (store every change the appliance is sent; the period only sizes its buffer) or SCAN (store the latest value once per period). |

## `shot_offsets`

### ShotOffsets

The experiment's measured per-device drain offsets.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `reference` | `str` | yes | — | The ophyd object name of the device the offsets are measured against — the one that stamped first. Its own offset_s is 0.0. Only differences matter, so which device this is carries no meaning beyond anchoring the numbers. |
| `devices` | `dict[str, DeviceOffset]` | no | empty | Ophyd object name (e.g. 'uc_amp3_ir_input') → that device's measured offset. A device absent from this mapping keeps the 0.0 default, which is correct only if it really stamps with the reference. |
| `measured_at` | `str (optional)` | no | None | ISO-8601 timestamp of the measurement, with offset. Informational, but the thing to look at when a join goes wrong: a calibration older than the last camera or server change is suspect. |
| `trigger_profile` | `str (optional)` | no | None | Trigger profile the measurement fired through. Recorded because a profile that drives a different trigger box would measure different latencies. |
| `trigger_rate_hz` | `float (optional)` | no | None | Rep rate the measurement was taken at, Hz. Load-bearing provenance, not decoration: a camera that pipelines — exposing the next frame while draining the last — has a rate-DEPENDENT offset. Measured on HTU 2026-09-12: an un-ROI'd camera's offset shifted 11.3 ms between 1 Hz and 5 Hz while an ROI'd one moved 0.2 ms. Calibrate at the rate you intend to run at, and compare this field against it before trusting the numbers. |
| `description` | `str` | no | '' | Optional note about this measurement. |

Example:

```yaml
schema_version: 1
# Written by the measure_shot_offsets calibration plan, not by hand.
reference: uc_amp3_ir_input   # stamped first; its own offset is 0.0 by definition
devices:
  uc_amp3_ir_input:
    offset_s: 0.0             # seconds after the reference that this device stamps
    scatter_s: 0.004          # peak-to-peak over the shots (host clock dither)
    shots: 10                 # complete shots that contributed to the mean
    geecs_device: UC_Amp3_IR_input
  uc_amp4_ir_input:
    offset_s: 0.036
    scatter_s: 0.009
    shots: 10
    geecs_device: UC_Amp4_IR_input
measured_at: "2026-09-13T18:22:04-07:00"
trigger_profile: HTU-LaserOFF
description: "after the Amp4 server rebuild"
# Only DIFFERENCES matter: the join subtracts each device's offset from its
# stamp before matching frames to rows, so adding a constant to every entry
# changes nothing. A device absent here keeps 0.0.
```

### DeviceOffset

One device's measured edge-to-stamp latency, relative to the reference.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `offset_s` | `float` | yes | — | Seconds after the reference device that this device stamps the same shot — the mean over the shots that contributed. The reference device's own value is 0.0. Subtracted from this device's acq_timestamp before frames are matched to shot rows. |
| `scatter_s` | `float` | no | 0.0 | Peak-to-peak spread of this device's per-shot offset across the measurement, seconds. Expect up to ~10 ms from ordinary host clock dither; markedly more than the other devices in the set means this machine's timekeeping is worth looking at. |
| `shots` | `int` | no | 1 | How many complete shots were averaged into this offset. A shot counts only when every device in the set delivered, so this is the same for every device of one measurement; shots that some device missed were discarded and retaken, and appear only in the calibration plan's log. |
| `geecs_device` | `str` | no | '' | GEECS device name this offset was measured for, e.g. 'UC_Amp3_IR_input'. Informational: the mapping key is the ophyd object name the runtime uses. |

## `analysis_recipe`

### AnalysisRecipe

One device's analysis: input, ordered steps, a measure, the draw, the summaries.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 3 | Format version of this recipe. Leave at 3 — tools update this automatically when the file format changes. |
| `device` | `str` | yes | — | The device whose data folder under scans/ScanNNN/ is analyzed. |
| `output_name` | `str (optional)` | no | None | Label for everything this recipe writes (s-file column prefix, output folder). Defaults to device; set it to run two recipes over one device with distinct outputs. |
| `scalar_suffix` | `str (optional)` | no | None | Suffix appended to every s-file column name; scalars only, never files or folders. |
| `description` | `str (optional)` | no | None | Free-text note about this recipe. |
| `metadata` | `dict[str, Any] (optional)` | no | None | Free-form documentary fields; nothing reads them. |
| `input` | `CameraInput \| LineInput` | yes | — | Where the frames come from and how one is read. |
| `inputs` | `dict[str, FrameInput]` | no | empty | Inputs loaded before the run and bound by name: frames for steps that take one (a background image) or a measure that compares against one (the haso reference), and tables for a measure that reads one (the magspec calibration surveys). |
| `steps` | `list[StepRef]` | no | empty | Processing steps in order; any order, repeats allowed. |
| `measure` | `MeasureRef` | no | MeasureRef(kind='none') | What is measured on each processed frame. |
| `scan` | `RecipeRuntime` | no | RecipeRuntime(priority=100, average_frames_first=False, save=True, workers=1) | How the recipe runs over a scan. |
| `figure` | `FigureStyle` | no | FigureStyle(imshow={}, pcolormesh={}, plot={}, colorbar={}, axes={}, fig={}, overlays={}) | The per-frame draw, reused by every summary kind. |
| `summaries` | `list[ImageGridSummary \| WaterfallSummary \| AverageSummary \| ScalarFitSummary]` | no | empty | Scan-level figures, each a frozen kind with its own options; an empty list draws no summary. |

Example:

```yaml
schema_version: 3
device: UC_TopView               # the device folder under scans/ScanNNN/; stems the products
output_name: UC_TopView_left     # optional: label outputs differently from the device
input: {kind: camera}            # or kind: line, with loading/x_scale/x_unit/label/storage_dtype
inputs:                          # frames the scan host loads and binds by name
  camera_background: {path: "{scan_dir}/computed_background.npy", fallback_level: 0}
steps:                           # in order, repeats allowed; the core's registry validates each
  - {step: background_frame, source: camera_background, alignment: samples}
  - {step: roi, bounds: [[350, 650], [0, 650]]}
  - {step: median, kernel: 3}
measure: {kind: beam, enabled_stats: [image_total, x_CoM, y_CoM]}
scan: {priority: 10, average_frames_first: false, save: true}
figure:                          # the per-frame draw: matplotlib keyword groups, overlays by id
  imshow: {cmap: plasma, vmin: 0}
  axes: {xlabel: x (px)}
  overlays: {projection_y: {hidden: true}}
summaries:                       # scan-level figures, each a frozen kind with its own options
  - {kind: image_grid, columns: 3}
  - {kind: average}
```

### CameraInput

Frames are camera images (2D).

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'camera'` | no | 'camera' | Image input. |
| `folder` | `str (optional)` | no | None | Data subfolder under scans/ScanNNN/ when it differs from the device (stitched or post-processed outputs in a sibling folder). |
| `file_tail` | `str (optional)` | no | None | Filename suffix that identifies this device's files ('.png', '.txt', '_postprocessed.tsv'); unset uses the kind's default. |
| `format` | `'per_shot_files' \| 'device_hdf5' (optional)` | no | None | 'device_hdf5' reads the per-device frame stack the PVA gateway's file plugin writes; a camera falls back to per-shot files when no stack can be mapped, a 'pva_stack' trace does not. |

### LineInput

Frames are traces (1D): how one file is read and scaled.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'line'` | no | 'line' | Trace input. |
| `folder` | `str (optional)` | no | None | Data subfolder under scans/ScanNNN/ when it differs from the device (stitched or post-processed outputs in a sibling folder). |
| `file_tail` | `str (optional)` | no | None | Filename suffix that identifies this device's files ('.png', '.txt', '_postprocessed.tsv'); unset uses the kind's default. |
| `format` | `'per_shot_files' \| 'device_hdf5' (optional)` | no | None | 'device_hdf5' reads the per-device frame stack the PVA gateway's file plugin writes; a camera falls back to per-shot files when no stack can be mapped, a 'pva_stack' trace does not. |
| `loading` | `Data1DLoading` | yes | — | How to read one trace file. |
| `x_scale` | `float` | no | 1.0 | Multiply the file's x values by this before processing. |
| `y_scale` | `float` | no | 1.0 | Multiply the file's y values by this before processing. |
| `x_unit` | `str` | no | '' | Unit of the scaled x axis (axis labels). |
| `y_unit` | `str` | no | '' | Unit of the scaled y values (axis labels). |
| `label` | `str` | no | '' | What the trace is (the y-axis label). |
| `storage_dtype` | `'float32' \| 'float64'` | no | 'float32' | Precision of the stored processed trace; statistics are taken from the stored values, so this rounds them too. |
| `siblings` | `list[str] (optional)` | no | None | Stitch: data folders of other devices whose same-shot traces are joined to this input's and sorted by x before processing (the segments of a multi-camera spectrometer). A shot missing from a sibling is stitched without it. |

### Data1DLoading

How to read one trace file into an x-vs-y array.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `data_type` | `Data1DType` | yes | — | File format: 'tek_scope_hdf5' or 'tdms_scope' for scope captures, 'csv' / 'tsv' for delimited text, 'npy' for a saved array, 'pva_stack' for one shot of the per-device capture stack a Bluesky scan writes (traces and spectra alike; the reader takes its axis from the stack and hands back the trace at its true length, never the padded one). |
| `trace_index` | `int` | no | 0 | Which trace / channel holds the y values (scope formats). |
| `x_trace_index` | `int (optional)` | no | None | Which trace holds the x values; unset derives x from the waveform properties (scope formats). |
| `delimiter` | `str (optional)` | no | None | Column delimiter for csv/tsv; unset uses the format's default. |
| `x_column` | `int` | no | 0 | Column index of x (text formats). |
| `y_column` | `int` | no | 1 | Column index of y (text formats). |
| `auxiliary_columns` | `dict[str, int]` | no | empty | Extra named columns to load alongside y, for analyzers that need them (name -> column index). |

### FrameInput

An input the source layer loads (or computes) before the run and binds by name.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `path` | `str (optional)` | no | None | File to load ('{scan_dir}' stands for the device's data directory under the scan). |
| `from_scan` | `ScanStatistic (optional)` | no | None | Compute the frame from a scan's frames (a dark scan's mean, or this scan's median/percentile) instead of loading a file. |
| `fallback_level` | `float (optional)` | no | None | When the file cannot be read, subtract this constant instead of the frame and warn; unset makes a failed read an error. Files only. |
| `table` | `str (optional)` | no | None | Tab-delimited table to read (a calibration survey), relative to the analysis-configs root or absolute; bound for a measure that names it as a table input, never as a frame. |

### ScanStatistic

A frame computed from a scan's own frames of this device, before the run.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `scan` | `int (optional)` | no | None | Scan number of the same day to take the frames from; unset takes them from the scan being analyzed. |
| `statistic` | `'mean' \| 'median' \| 'percentile'` | no | 'mean' | Per-pixel statistic across the scan's frames. |
| `percentile` | `float (optional)` | no | None | The percentile (0-100) when statistic is 'percentile'. |

### StepRef

One processing step by its registered name, parameters as written.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `step` | `str` | yes | — | Registered step name ('roi', 'median', 'background_constant', ...). The remaining keys are that step's parameters. |

### MeasureRef

The measure by its registered kind, parameters as written.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `str` | no | 'none' | Registered measure kind ('beam', 'line', 'none'). The remaining keys are that measure's parameters. |

### RecipeRuntime

How the recipe runs over a scan.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `priority` | `int` | no | 100 | Run order within a group: lower runs first. 100 is the background default. |
| `average_frames_first` | `bool` | no | False | Average each bin's frames before processing and measure once per bin, for metrics that are not linear in the image; default measures every frame. |
| `save` | `bool` | no | True | Write per-shot / per-bin products and the summary figures into the analysis tree. S-file scalar columns are written regardless. |
| `workers` | `int` | no | 1 | Worker processes a scan run may use to read and measure frames in parallel; 1 (the default) runs in the calling process. The host caps it at its own limit and runs small scans serially; the outputs are identical either way. |

### FigureStyle

The per-frame draw: matplotlib keyword groups and overlay styles by id.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `imshow` | `dict[str, Any]` | no | empty | Axes.imshow keywords (cmap, vmin, vmax, ...). |
| `pcolormesh` | `dict[str, Any]` | no | empty | Axes.pcolormesh keywords, used for nonuniform image axes. |
| `plot` | `dict[str, Any]` | no | empty | Axes.plot keywords for traces. |
| `colorbar` | `dict[str, Any]` | no | empty | Figure.colorbar keywords (label, ...); show: false hides it. |
| `axes` | `dict[str, Any]` | no | empty | Axes.set keywords (xlabel, ylabel, title). |
| `fig` | `dict[str, Any]` | no | empty | Figure keywords (figsize, dpi). |
| `overlays` | `dict[str, dict[str, Any]]` | no | empty | Style per overlay id the measure emits (projection_x, projection_y, com): Axes.plot keywords, plus hidden and scale. |

### ImageGridSummary

One panel per bin (the bin-averaged image), one shared colour scale.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'image_grid'` | no | 'image_grid' | One panel per bin, one shared colour scale. |
| `columns` | `int (optional)` | no | None | Panels per row; unset squares the grid. |
| `panel_size` | `tuple[float, float] (optional)` | no | None | Panel (width, height) in inches; unset uses the renderer default. |

### WaterfallSummary

Every bin's (or, on a noscan, every shot's) trace as one row of a heat map.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'waterfall'` | no | 'waterfall' | Every row a bin's (or shot's) trace. |
| `sort_key` | `str (optional)` | no | None | On a noscan, order rows by this s-file column instead of shot number ('Device:Var' or a substring); also skips bin averaging. |
| `sort_sigma` | `float` | no | 3.0 | Drop rows whose sort-key value lies outside mean ± this many standard deviations. |
| `sort_bounds` | `tuple[float, float] (optional)` | no | None | Explicit (low, high) bounds on the sort key; overrides the sigma cut. |
| `even_spacing` | `bool (optional)` | no | None | Draw rows at equal height regardless of position spacing; unset means yes when sorting by a key, no otherwise. |
| `scale` | `'auto' \| 'sequential' \| 'diverging' \| 'custom'` | no | 'auto' | 'sequential' runs 0 to max; 'diverging' is symmetric about zero; 'auto' centres on zero when the data crosses it; 'custom' uses vmin/vmax as given. |
| `cmap` | `str (optional)` | no | None | Matplotlib colormap name. |
| `vmin` | `float (optional)` | no | None | Colour scale minimum. |
| `vmax` | `float (optional)` | no | None | Colour scale maximum. |

### AverageSummary

The scan's averaged frame (a noscan or count scan), drawn with ``figure``.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'average'` | no | 'average' | The scan's averaged frame. |

### ScalarFitSummary

A straight-line fit of measured scalars against the scan position.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'scalar_fit'` | no | 'scalar_fit' | Fit measured scalars against the scan position. |
| `scalars` | `list[str]` | yes | — | Scalar keys to fit, bare as the measure emits them (e.g. kick_1); one fitted line each. |
| `model` | `'linear'` | no | 'linear' | The fitted model: 'linear' (slope and intercept). |

## `analysis_group`

### AnalysisGroup

A named set of diagnostics to run after each scan, in priority order.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `name` | `str` | yes | — | Display name, conventionally <facility>_<purpose>. |
| `description` | `str (optional)` | no | None | Free-text note about when this group is used. |
| `upload_to_scanlog` | `bool` | no | True | Retired upload option; accepted for existing configs and ignored. |
| `analyzers` | `list[AnalyzerRef]` | no | empty | The diagnostics to run; a bare ID means enabled with the diagnostic's own priority. |

Example:

```yaml
schema_version: 1
name: HTU_baseline
analyzers:
  - UC_TopView                   # bare ID: enabled, the diagnostic's own priority
  - {ref: U_FROG, enabled: false}
  - {ref: U_BCaveICT, priority: 1}
```

### AnalyzerRef

One diagnostic in a group, optionally disabled or re-prioritised for this group.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `ref` | `str` | yes | — | The diagnostic's ID (its YAML file stem). |
| `enabled` | `bool` | no | True | False keeps the entry listed but skips it when the group runs. |
| `priority` | `int (optional)` | no | None | Run order within this group; unset uses the diagnostic's own scan.priority. |

## `himg_job`

### HimgJob

One ``.himg`` data-management job over a HASO device folder.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this job file. Leave at 1 — tools update this automatically when the file format changes. |
| `kind` | `'himg_to_stack' \| 'himg_compact' \| 'himg_restore'` | yes | — | Which job: himg_to_stack (folder → capture stack), himg_compact (verify, then delete the .himg files — destructive) or himg_restore (rebuild the .himg files from the stack). |
| `device` | `str` | yes | — | The HASO device whose folder under scans/ScanNNN/ holds the .himg files. |
| `output_name` | `str (optional)` | no | None | Label the portal's Analysis tab lists the job under. Defaults to device; the job writes nothing under analysis/. |
| `description` | `str (optional)` | no | None | Free-text note about this job. |
| `scan` | `HimgJobRuntime` | no | HimgJobRuntime(priority=100) | How the job runs among a group's analyses. |

Example:

```yaml
schema_version: 1
kind: himg_compact               # or himg_to_stack / himg_restore
device: U_HasoLift               # the HASO device folder under scans/ScanNNN/
output_name: HasoLift_compact    # the portal lists the job under this label
scan: {priority: 6}
```

### HimgJobRuntime

How the job runs among a group's analyses.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `priority` | `int` | no | 100 | Run order within a group: lower runs first. 100 is the background default. |

## `log_entry`

### LogEntry

One entry in the scan logbook.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Entry format revision. |
| `entry_id` | `str` | yes | — | Stable id; also the attachment directory. |
| `day` | `str` | yes | — | Run day, as YYYY-MM-DD. |
| `book` | `'scans' \| 'ops'` | no | 'scans' | Campaign record or operations. |
| `scan` | `int (optional)` | no | None | Scan annotated; none for a day-level entry. |
| `after` | `int (optional)` | no | None | For an interscan entry, the scan it follows. |
| `author` | `str` | yes | — | Who wrote it. |
| `kind` | `'note' \| 'agent_analysis' \| 'agent_draft'` | no | 'note' | Human note, or which agent output. |
| `status` | `'kept' \| 'draft'` | no | 'kept' | Drafts await a human. |
| `tags` | `list[str]` | no | empty | #tags parsed from the body at save. |
| `template` | `str` | no | 'blank' | Seed template it started from. |
| `body_md` | `str` | no | '' | The entry. Opaque markdown. |
| `payload` | `AnalysisPayload \| ProblemPayload (optional)` | no | None | Machine-authored structure, never human prose. |
| `attachments` | `list[Attachment]` | no | empty | Files uploaded with the entry. |
| `created_at` | `datetime` | yes | — | First saved. |
| `edited_at` | `datetime (optional)` | no | None | Text last changed. |
| `edited_by` | `str (optional)` | no | None | Who last changed the text. |
| `updated_at` | `datetime` | yes | — | Anything last changed. |
| `deleted_at` | `datetime (optional)` | no | None | Tombstone. |
| `version` | `int` | no | 1 | Optimistic-lock counter. |

Example:

```yaml
schema_version: 1
entry_id: 7f3a9c2b1d04
day: "2026-09-11"
book: scans                      # or ops: the operations log, day-level only
scan: 12                         # omit both scan and after for a day-level entry
author: S. Barber
tags: [jet]                      # parsed from the body's #jet at save
kind: note                       # agent_analysis / agent_draft are born as drafts
status: kept
template: scan_note
body_md: |
  Charge rolloff onset moved to 4.1 mm after the #jet realignment.
  ![top view](attachments/7f3a9c2b1d04/topview.png)
payload:
  kind: analysis                 # optional machine-readable half
  analyzer: Array1DScanAnalyzer
  metrics: {onset_mm: 4.1}
attachments:
  - {id: 7f3a9c2b1d04, filename: topview.png, content_type: image/png,
     size_bytes: 48213, uploaded_at: "2026-09-11T17:42:10+00:00"}
created_at: "2026-09-11T17:40:02+00:00"
updated_at: "2026-09-11T17:42:10+00:00"   # moves on every change; edited_at only on text
version: 2
```

### AnalysisPayload

Structured results an analysis wants to make queryable.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'analysis'` | no | 'analysis' | Payload type tag. |
| `analyzer` | `str` | yes | — | Which analyzer produced this, e.g. Array1DScanAnalyzer. |
| `metrics` | `dict[str, float]` | no | empty | Named scalar results. Free-form by design: an analyzer names its own. |
| `figures` | `list[str]` | no | empty | Paths to figures in the analysis tree, referenced not copied. |

### ProblemPayload

A problem worth finding again later.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'problem'` | no | 'problem' | Payload type tag. |
| `severity` | `'note' \| 'degraded' \| 'blocked'` | no | 'note' | How much it stopped the run. |
| `devices` | `list[str]` | no | empty | Devices implicated, by GEECS name. |
| `issue_url` | `str (optional)` | no | None | Tracker link, when one was filed. |

### Attachment

One file uploaded with an entry — a pasted screenshot, a PDF.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `id` | `str` | yes | — | Opaque id of this upload. |
| `filename` | `str` | yes | — | Name as stored on disk. |
| `content_type` | `str` | yes | — | Media type, e.g. image/png. |
| `size_bytes` | `int` | yes | — | Stored size in bytes. |
| `uploaded_at` | `datetime` | yes | — | When the file was stored. |

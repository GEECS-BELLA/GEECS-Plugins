<!-- GENERATED FILE — do not edit by hand.
     Regenerate with:  python -m geecs_schemas.docgen
     (or GEECS-Schemas/tests/generate_schema_reference.py).
     A no-drift test (tests/test_schema_reference.py) fails CI if this
     file falls out of step with the schema field descriptions. -->

# GEECS config schema reference

## `scan_request`

### ScanRequest

One complete scan, ready to submit: what to do, what to save, how to trigger.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 3 | Format version of this config file. Leave at 3 — tools update this automatically when the file format changes. |
| `mode` | `ScanRequestMode` | yes | — | What kind of scan: 'step' sweeps one or more axes, 'noscan' collects shots without moving anything, 'optimize' lets an algorithm pick the settings. |
| `axes` | `list[ScanAxis]` | no | empty | For step scans: what to sweep. One entry is a simple 1-D scan; several entries form a grid visiting every combination, with the first axis as the outermost (slowest) loop and the last as the innermost (fastest). Leave empty for noscan and optimize. |
| `capture` | `CaptureSettings` | no | CaptureSettings(shots_per_step=1, acquisition=<AcquisitionMode.STRICT: 'strict'>, save_sets=[], background_telemetry=None, native_image_save=None, trigger_profile=None) | How shots are taken and what gets recorded: shots per step, acquisition discipline, save sets, telemetry and native-image toggles, and the trigger profile. Omit for a one-shot strict capture with no named save sets. |
| `actions` | `ActionBindings` | no | ActionBindings(setup=[], per_step=[], closeout=[]) | Named action plans to run before the scan (setup), between steps (per_step), and after it (closeout). |
| `description` | `str` | no | '' | Free-text note about this scan; it ends up in the scan's metadata and the experiment log. |
| `background` | `bool` | no | False | Mark this scan's data as background/calibration shots so analysis can find them later. |
| `optimization` | `OptimizationSpec (optional)` | no | None | The optimization problem definition. Required for (and only allowed with) mode 'optimize'. |

Example:

```yaml
schema_version: 3
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
| `native_image_save` | `bool (optional)` | no | None | Whether capture-eligible cameras (Point Grey — the devicetypes the central PVA capture daemon owns) write their native per-shot image files during this scan. When false, those cameras' images are recorded only by the capture daemon's per-device frame stack (one HDF5 per camera per scan); all other devices — proprietary formats like the HASO, scope traces — keep their native save regardless. Leave unset to inherit the experiment default; set true/false to override for this scan (e.g. force native files back on for one scan while the capture path is being validated). Two engine behaviors to expect when false: the scan is REFUSED before a scan number is claimed if the capture daemon looks absent or is not monitoring every capture camera (fail-closed — start the daemon or drop the override), and the request is silently inert when no capture-eligible cameras resolve (DB unreachable, or none in the save set) — native saving then proceeds unchanged, with a warning in the scan log. |
| `trigger_profile` | `str (optional)` | no | None | Name of the trigger profile that drives the shot trigger. Unset means the scan does not manage the trigger. |

### ActionBindings

Which named action plans run around (and inside) the scan.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `setup` | `list[str]` | no | empty | Plans to run once before the scan starts. |
| `per_step` | `list[str]` | no | empty | Plans to run between scan steps — after each move, before the shots at that position. |
| `closeout` | `list[str]` | no | empty | Plans to run once after the scan finishes (even on abort). |

### OptimizationSpec

The optimization problem: what to vary, within what limits, to improve what.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `variables` | `dict[str, tuple[float, float]]` | yes | — | What the optimizer may move and how far, as 'variable name: [lowest, highest]'. Names may be scan-variable names or 'Device:Variable' strings. |
| `objectives` | `dict[str, str]` | no | empty | What counts as better, as 'objective name: MINIMIZE' or 'objective name: MAXIMIZE'. May be empty for algorithms that only model observables (BAX). |
| `observables` | `list[str]` | no | empty | Extra measured quantities the algorithm should track without optimizing them, e.g. ['x_CoM']. |
| `constraints` | `dict[str, tuple[str, float]]` | no | empty | Hard limits on measured quantities, as 'name: [LESS_THAN or GREATER_THAN, value]'. Usually empty. |
| `evaluator` | `EvaluatorSpec` | yes | — | The analysis code that scores each iteration. |
| `generator` | `GeneratorSpec` | yes | — | The algorithm that proposes the next settings. |
| `max_iterations` | `int (optional)` | no | None | Stop after this many optimization iterations. Leave unset to run until stopped by hand. |
| `seed_dump_files` | `list[str]` | no | empty | Optional earlier results (ECS dump files) used to warm-start the optimizer. Usually empty. |
| `move_to_best_on_finish` | `bool` | no | False | After the optimization ends, drive the variables back to the best settings found. |

### EvaluatorSpec

Which analysis code turns raw shots into the number being optimized.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `module` | `str` | yes | — | Python import path of the evaluator module, e.g. 'geecs_bluesky.optimization.evaluators.beam_sum_counts_evaluator'. |
| `class_name` | `str` | yes | — | Name of the evaluator class inside that module. |
| `kwargs` | `dict` | no | empty | Settings passed to the evaluator when it is created — e.g. which diagnostics/analyzers it should read. Free-form: each evaluator documents its own options. |

### GeneratorSpec

Which optimization algorithm proposes the next settings.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `name` | `str` | yes | — | Name of the optimization algorithm, e.g. 'bayes_default', 'random', or 'multipoint_bax_alignment_l2'. |
| `options` | `dict` | no | empty | Algorithm-specific tuning options. Free-form: each generator documents its own options (legacy 'xopt_config_overrides'). |

## `save_set`

### SaveSet

The devices a scan *requires* — its participation list, not a logging list.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `name` | `str` | yes | — | The name scans use to refer to this save set. |
| `entries` | `list[SaveSetEntry]` | yes | — | The devices to record, one entry per device. |
| `description` | `str` | no | '' | Optional note about what this save set is for. |

Example:

```yaml
schema_version: 1
name: undulator_baseline
# the REQUIRED devices — everything else is still logged in the background
entries:
  - device: UC_Amp4_IR_input
    images: true                     # images are always required-tier
    scalars: [MaxCounts, centroidx]  # extras beyond the DB's standard telemetry
  - device: U_HP_Daq
    db_scalars: false                # record ONLY the listed scalars, not the DB set
    scalars: [AnalogOutput.Channel 1]
    at_scan_start: {Analysis: "on"}  # replace the DB's scan-start value
    at_scan_end: {Analysis: null}    # suppress the DB's scan-end write
  - device: U_BCaveHallProbe
    scalars: [Field, Rawfield]
    role: snapshot
  - device: UC_UndulatorRad2
    images: true
    scalars: [MeanCounts]
    # this device's ritual travels with it: these named plans run once
    # before/after any scan whose save set includes this entry
    setup: [visa1_spectrometer_setup]
    closeout: [visa1_spectrometer_closeout]
```

### SaveSetEntry

One *required* device of a scan and the guarantees it gets.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `device` | `str` | yes | — | GEECS device name exactly as it appears in the GEECS experiment database (MySQL), e.g. 'UC_ALineEbeam1'. Spelling (including case) is checked against the database when the config is loaded. |
| `scalars` | `list[str]` | no | empty | EXTRA scalar readings to record beyond the device's standard telemetry — the variables the GEECS experiment database marks for scan logging (MySQL table expt_device_variable, get='yes'), which 'db_scalars' records by default. E.g. ['MaxCounts', 'centroidx']. Usually empty — list variables here only when you need something the database doesn't mark. |
| `all_scalars` | `bool` | no | False | Record every scalar variable the device publishes instead of naming them one by one. If 'scalars' is also given, the explicit list wins. |
| `images` | `bool` | no | False | Save the device's images / non-scalar files (camera frames, traces) alongside the scalar data. Ignored for an entry with role 'snapshot' (legacy synchronous: false): the snapshot role records scalars only — the scanner neither commands nor suppresses the device's own save flag. |
| `role` | `SaveRole (optional)` | no | None | Override for how this device is synchronized with shots. Leave unset to let the scanner decide; set 'snapshot' for slow readbacks that don't produce one value per shot (scalars only — 'images' is ignored for a snapshot entry). |
| `setup` | `list[str]` | no | empty | Names of action plans that must run before any scan that records this device — its setup ritual (turn analysis on, insert a stage, ...). The plans named by all entries of a save set are collected together, de-duplicated by name, and each runs once before the scan. |
| `closeout` | `list[str]` | no | empty | Names of action plans that run after any scan that records this device — its cleanup ritual. Collected and de-duplicated the same way as 'setup', and run once after the scan (even on abort). |
| `db_scalars` | `bool` | no | True | Record every variable the GEECS experiment database marks for scan logging for this device (MySQL table expt_device_variable, column get='yes') — the MC-style 'standard telemetry', and the default scalar source for a required device. The 'scalars' list adds extras on top. Turn off to record only what 'scalars' lists explicitly (converted legacy elements do this, preserving their exact old behavior). |
| `at_scan_start` | `dict[str, str (optional)]` | no | empty | RESERVED AND NOT APPLIED in this version. The DB set-side scan start/end writes are intentionally disabled: the engine sets up triggering via the trigger profile / shot controller and camera saving via its own save-windowing, so writing the database's set='yes' start values here would race the shot controller. Kept for a possible future re-enable — a config that sets it is not an error but has no effect today (the engine logs a warning). When honored again, it would tweak the database's scan-start writes per variable (unmentioned = database value, a value = replace, null = suppress). |
| `at_scan_end` | `dict[str, str (optional)]` | no | empty | RESERVED AND NOT APPLIED in this version — the scan-end counterpart of 'at_scan_start'. The DB set-side scan start/end writes are intentionally disabled (triggering is owned by the trigger profile / shot controller, camera saving by the scanner's save-windowing), so this has no effect today; it is kept for a possible future re-enable. When honored again, it would tweak the database's scan-end writes per variable (same three cases as 'at_scan_start'). |

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
| `confirm` | `str (optional)` | no | None | Optional 'Device:Variable' that *measures* the result when it differs from the variable being set — e.g. set a supply's current limit but confirm on its measured current. Leave unset when the set variable is also the readback (the common case). Declared but not yet enforced by the engine in v1. |

### PseudoScanVariable

A friendly name that moves several devices together from one number.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'pseudo'` | yes | — | Variable type. 'pseudo' moves several devices from one number. |
| `targets` | `list[PseudoComponent]` | yes | — | The devices this variable moves, each with its own formula. |
| `mode` | `CompositeMode` | yes | — | 'absolute' = each device goes exactly where its formula says. 'relative' = each device is offset from where it was when the scan started. |
| `inverse` | `str (optional)` | no | None | Optional formula recovering the scanned number from the first target's readback. Leave unset if you don't need a readback for this variable. |

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
| `actions` | `DefaultActions` | no | DefaultActions(setup=[], closeout=[]) | Action plans every scan runs by default — setup plans run first (before the scan's own), closeout plans run last (after the scan's own). |
| `apply_db_scan_defaults` | `bool` | no | True | RESERVED AND NOT CURRENTLY HONORED. The DB set-side scan start/end writes (MySQL table expt_device_variable: rows with set='yes', writing their startvalue/endvalue) are disabled in this version — triggering is set up via the trigger profile / shot controller and camera saving via the scanner's own save-windowing, so the database's boundary writes are not applied regardless of this flag. Kept for a possible future re-enable. Note this is the set-side only: the get-side 'db_scalars' (standard telemetry) and 'background_telemetry' are honored as normal. |
| `background_telemetry` | `bool` | no | True | Log every live experiment device that is not in a scan's save set as best-effort snapshot columns — the variables the GEECS experiment database marks for scan logging (MySQL table expt_device_variable, get='yes') — read from the gateway's always-on monitor cache. Safe by construction: read-only and never waited on, so it cannot slow or stall a scan — a dead device is just dropped with a log line. On by default so no data is silently lost; individual scans can override with their own 'background_telemetry' setting. |
| `native_image_save` | `bool` | no | True | Whether capture-eligible cameras (Point Grey — the devicetypes the central PVA capture daemon owns) write their native per-shot image files. On by default: flipping this off is the PNG-deprecation step, taken only after accumulated dual-write evidence that the capture daemon's per-device frame stacks are lossless for this experiment. Devices with proprietary formats (HASO, scope traces) keep their native save regardless of this flag. Individual scans can override with their own 'native_image_save' setting. |
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

## `analysis_diagnostic`

### AnalysisDiagnostic

One device's analysis: which analyzer, how frames are cleaned up, how it runs over a scan.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 2 | Format version of this config file. Leave at 2 — tools update this automatically when the file format changes. |
| `name` | `str` | yes | — | The device whose data folder under scans/ScanNNN/ is analyzed. |
| `output_name` | `str (optional)` | no | None | Label for everything this analyzer writes (s-file column prefix, output folder). Defaults to name; set it to run two analyzers over one device with distinct outputs. |
| `metric_suffix` | `str (optional)` | no | None | Suffix appended to every s-file column name; affects scalars only, never files or folders. |
| `description` | `str (optional)` | no | None | Free-text note about this diagnostic. |
| `analyzer` | `StandardAnalyzerSpec \| LineAnalyzerSpec \| BeamAnalyzerSpec \| MagSpecAnalyzerSpec \| FrogRetrievalSpec \| FrogSpectralPhaseSpec \| IctAnalyzerSpec \| LineStitcherSpec \| HasoAnalyzerSpec \| DownrampPhaseSpec \| HiResMagCamSpec \| BCaveMagSpecStitcherSpec \| BCaveMagOptSpec \| PhaseDownrampSpec` | yes | — | Which analyzer runs and its own parameters; chosen by kind. |
| `image` | `CameraConfig \| Line1DConfig (optional)` | no | None | How raw frames (type: camera) or traces (type: line) are cleaned up before analysis. Omit for analyzers that read their own file formats (kind haso, phase_downramp). |
| `scan` | `ScanRuntime` | no | ScanRuntime(priority=100, mode='per_shot', save=True, gdoc_slot=None, device=None, file_tail=None, data_format=None, renderer=RendererOptions(colormap_mode=None, cmap=None, vmin=None, vmax=None, duration=None, dpi=None, xlabel=None, ylabel=None, colorbar_label=None, mode=None, waterfall_sort_key=None, waterfall_sort_sigma=None, waterfall_sort_bounds=None, waterfall_even_y_spacing=None, figsize=None, figsize_inches=None), background_source=None) | How the analyzer runs over a scan: order, per shot or per bin, saving, files. |

Example:

```yaml
schema_version: 2
name: UC_TopView                 # the device folder under scans/ScanNNN/
output_name: UC_TopView_left     # optional: label outputs differently from the device
analyzer:
  kind: beam                     # picks the analyzer AND the fields below
  compute_slopes: false
  enabled_stats: [image_total, x_CoM, y_CoM, x_fwhm, y_fwhm]
image:
  type: camera
  bit_depth: 16
  roi: {x_min: 0, x_max: 650, y_min: 350, y_max: 650}
  background: {method: constant, constant_level: 5.0}
  filtering: {median_kernel_size: 3}
  pipeline: [background, roi, filtering]   # only listed steps run, in this order
scan:
  priority: 10
  mode: per_shot
  save: true
  gdoc_slot: 0
  renderer: {cmap: plasma}
# v1 documents (image_analyzer class path, image.analysis, kwargs) still
# validate — they are lifted into this shape automatically.
```

### StandardAnalyzerSpec

Run the camera pipeline and report the processed frame — no extra metrics.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'standard'` | no | 'standard' | Processed-frame-only camera analyzer. |

### LineAnalyzerSpec

Run the trace pipeline and report basic trace statistics (peak, centroid, width, area).

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'line'` | no | 'line' | Trace statistics analyzer. |

### BeamAnalyzerSpec

Beam profile metrics: centroid, rms size, FWHM, total counts along x, y and the 45° axes.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'beam'` | no | 'beam' | Beam profile analyzer. |
| `compute_slopes` | `bool` | no | False | Also compute beam slope / straightness metrics from line-by-line fits. Expensive; leave off unless the tilt matters. |
| `enabled_stats` | `list[str] (optional)` | no | None | Emit only these statistics (e.g. ['image_total', 'x_CoM', 'y_fwhm']); unset emits all 18. Names are <axis>_<stat>. |

### MagSpecAnalyzerSpec

Magnetic spectrometer: beam metrics plus an energy-calibrated, resampled spectrum.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'magspec'` | no | 'magspec' | Energy-calibrated magnetic spectrometer analyzer. |
| `calibration` | `PolynomialCalibrationSpec \| ArrayCalibrationSpec \| DnnAxisCalibrationSpec` | yes | — | How image columns map to energy. |
| `energy_range` | `tuple[float, float]` | yes | — | (min, max) of the uniform energy grid the spectrum is resampled onto, MeV. |
| `num_energy_points` | `int` | no | 500 | Number of points on the uniform energy grid. |

### PolynomialCalibrationSpec

Pixel-to-energy map as a polynomial in the column index (E = c0 + c1·x + c2·x² + ...).

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'polynomial'` | no | 'polynomial' | Polynomial pixel-to-energy calibration. |
| `coeffs` | `list[float]` | yes | — | Polynomial coefficients, lowest order first, energy in MeV. |

### ArrayCalibrationSpec

Pixel-to-energy map given explicitly, one energy per image column.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'array'` | no | 'array' | Explicit per-column energy axis. |
| `values` | `list[float] (optional)` | no | None | Inline energy axis, one value per column. |
| `file` | `str (optional)` | no | None | A saved .npy energy axis (alternative to values). |

### DnnAxisCalibrationSpec

The MATLAB-era DNN spectrometer calibration: camera geometry + electron trajectory tables.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'dnn_axis'` | no | 'dnn_axis' | DNN camera + trajectory table calibration. |
| `camera_calibration_file` | `str` | yes | — | Tab-delimited camera geometry table (one row per camera). |
| `trajectory_calibration_file` | `str` | yes | — | Tab-delimited screen-position vs momentum table. |
| `camera_number` | `int` | yes | — | Which camera row of the geometry table to use. |
| `magnetic_field_t` | `float` | no | 1.0 | Dipole field in tesla the trajectory table is scaled to. A live teslameter reading in the shot's auxiliary data overrides it. |
| `lanex_calibration_file` | `str (optional)` | no | None | Lanex counts-to-charge table; when given, spectra are reported in fC. |

### FrogRetrievalSpec

Grenouille / FROG pulse retrieval through the vendor DLL (Windows-only at run time).

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'frog_retrieval'` | no | 'frog_retrieval' | FROG pulse retrieval via the vendor DLL. |
| `delt` | `float` | no | 0.85 | Time-delay step per raw pixel, fs. |
| `dellam` | `float` | no | -0.085 | Wavelength step per raw pixel, nm (negative for Grenouille). |
| `lam0` | `float` | no | 400.0 | Centre wavelength of the trace, nm. |
| `N` | `int` | no | 512 | Retrieval grid size: 512, 256, 128 or 64. |
| `target_error` | `float` | no | 0.005 | FROG error at which the retrieval stops early. |
| `max_time_seconds` | `float` | no | 5.0 | Wall-clock cap on one retrieval, seconds. |
| `max_iterations` | `int` | no | 1000000000 | Iteration cap on one retrieval. |
| `noise_subtype` | `int` | no | 4 | Vendor NoiseSubtraction SUBTYPE parameter. |
| `noise_rad` | `float` | no | 1.0 | Vendor NoiseSubtraction RAD parameter. |

### FrogSpectralPhaseSpec

Fit a polynomial spectral phase to a retrieved FROG spectrum (GDD, TOD, …).

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'frog_spectral_phase'` | no | 'frog_spectral_phase' | Spectral-phase polynomial fit. |
| `fit_order` | `int` | no | 3 | Polynomial order of the phase fit. |
| `mask_threshold` | `float (optional)` | no | 0.5 | Fit only where the spectral intensity exceeds this fraction of its peak. |
| `min_points` | `int (optional)` | no | None | Minimum number of points required for a fit. |
| `fit_num_points` | `int` | no | 300 | Number of points the fitted phase is evaluated on. |
| `reference_wavelength_nm` | `float` | no | 800.0 | Wavelength the phase expansion is taken about, nm. |
| `sign_reference_order` | `int (optional)` | no | None | Polynomial order whose sign is forced to sign_reference; unset leaves the fit as is. |
| `sign_reference` | `float` | no | 1.0 | Sign (+1 / -1) imposed on sign_reference_order. |
| `sign_epsilon` | `float` | no | 0.0 | Dead band around zero within which the sign is not flipped. |

### IctAnalyzerSpec

Integrating current transformer: charge from a scope trace by low-pass filtering and integrating.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'ict'` | no | 'ict' | ICT charge analyzer. |
| `butterworth_order` | `int` | no | 1 | Order of the low-pass Butterworth filter. |
| `butterworth_crit_f` | `float` | no | 0.125 | Normalised critical frequency of the low-pass filter. |
| `calibration_factor` | `float` | no | 0.1 | ICT calibration factor, V·s per C. |
| `dt` | `float (optional)` | no | None | Sample interval in seconds; unset derives it from the trace. |

### LineStitcherSpec

Concatenate this device's trace with its sibling devices' traces into one spectrum.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'line_stitcher'` | no | 'line_stitcher' | Multi-device trace stitcher. |
| `sibling_devices` | `list[str]` | yes | — | The other devices whose traces are appended to this diagnostic's device. Each must have a folder in the scan. |

### HasoAnalyzerSpec

HASO wavefront sensor: slopes to phase and Zernike terms through WaveKit (Windows, licensed).

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'haso'` | no | 'haso' | HASO wavefront analyzer via WaveKit. |
| `wavekit_config_file_path` | `Path` | yes | — | The WaveKit sensor configuration (.dat) for this HASO head. |
| `mask` | `PupilMask` | no | PupilMask(top=1, bottom=-1, left=1, right=-1) | Pupil mask applied to the slopes. |
| `background_path` | `Path (optional)` | no | None | A .has slopes file subtracted as background. |
| `laser_wavelength` | `float` | no | 800.0 | Probe wavelength, nm. |

### PupilMask

Rectangular pupil mask on the HASO slopes grid, inclusive bounds; -1 means the far edge.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `top` | `int` | no | 1 | Top row of the pupil (inclusive). |
| `bottom` | `int` | no | -1 | Bottom row of the pupil (inclusive); -1 = last row. |
| `left` | `int` | no | 1 | Left column of the pupil (inclusive). |
| `right` | `int` | no | -1 | Right column of the pupil (inclusive); -1 = last column. |

### DownrampPhaseSpec

HTU downramp phase-map analyzer over the camera pipeline.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'downramp_phase'` | no | 'downramp_phase' | HTU downramp phase analyzer. |

### HiResMagCamSpec

HTU high-resolution magspec camera: beam metrics plus a bow-tie fit of the trace.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'hi_res_mag_cam'` | no | 'hi_res_mag_cam' | HTU HiResMagCam bow-tie analyzer. |
| `n_beam_size_clearance` | `int` | no | 4 | Bow-tie fit: beam-size clearance in pixels. |
| `min_total_counts` | `float` | no | 2500.0 | Bow-tie fit: skip frames with fewer total counts. |
| `threshold_factor` | `float` | no | 10.0 | Bow-tie fit: threshold factor. |

### BCaveMagSpecStitcherSpec

HTU BCave magspec camera with a Gaussian-weighted vertical lineout for optimization.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'bcave_magspec_stitcher'` | no | 'bcave_magspec_stitcher' | HTU BCave magspec camera analyzer. |
| `gaussian_sigma` | `float` | no | 20.0 | Width of the Gaussian weighting, pixels. |
| `gaussian_center` | `float` | no | 250.0 | Centre of the Gaussian weighting, pixels. |

### BCaveMagOptSpec

HTU BCave stitched-spectrum optimizer metrics over the trace pipeline.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'bcave_mag_opt'` | no | 'bcave_mag_opt' | HTU BCave stitched-spectrum optimizer analyzer. |

### PhaseDownrampSpec

HTU phase-map processor: density from a probe phase map (reads its own TSV/phase files).

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `kind` | `'phase_downramp'` | no | 'phase_downramp' | HTU phase-downramp processor. |
| `pixel_scale` | `float` | yes | — | Spatial calibration, µm per pixel (vertical). |
| `wavelength_nm` | `float` | yes | — | Probe wavelength, nm. |
| `threshold_fraction` | `float` | no | 0.5 | Zero phase values below this fraction of the maximum. |
| `roi` | `tuple[int, int, int, int] (optional)` | no | None | Crop as (x_min, x_max, y_min, y_max); negatives count from the end. |
| `background_path` | `Path (optional)` | no | None | A background phase map to subtract. |

### CameraConfig

How a camera's frames are cleaned up before the analyzer measures them.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `type` | `'camera'` | no | 'camera' | Marks this as a camera (2D image) section. |
| `description` | `str (optional)` | no | None | Free-text note about this camera / view. |
| `metadata` | `dict[str, Any] (optional)` | no | None | Free-form documentation (location, notes, calibration constants). Nothing in the pipeline reads it; keep such notes here so the rest of the schema can stay strict. |
| `bit_depth` | `int` | no | 16 | Camera bit depth: 8, 10, 12, 14, 16 or 32. |
| `roi` | `ROIConfig (optional)` | no | None | Region-of-interest crop. |
| `background` | `BackgroundConfig (optional)` | no | None | Background subtraction. |
| `crosshair_masking` | `CrosshairMaskingConfig (optional)` | no | None | Crosshair masking. |
| `circular_mask` | `CircularMaskConfig (optional)` | no | None | Circular masking. |
| `vignette` | `VignetteConfig (optional)` | no | None | Vignette correction. |
| `thresholding` | `ThresholdingConfig (optional)` | no | None | Thresholding. |
| `filtering` | `FilteringConfig (optional)` | no | None | Smoothing filters. |
| `normalization` | `NormalizationConfig (optional)` | no | None | Intensity normalization. |
| `transforms` | `TransformConfig (optional)` | no | None | Rotation, flips, distortion correction. |
| `pipeline` | `list[ProcessingStepType]` | no | empty | The steps that run, in order. A step runs only if listed here AND its section is present; empty means the raw frame is analyzed. |

### ROIConfig

Crop the frame to a rectangular region of interest, in pixels.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `x_min` | `int` | no | 0 | Left edge (inclusive), pixels. |
| `x_max` | `int` | no | 1024 | Right edge (exclusive), pixels. |
| `y_min` | `int` | no | 0 | Top edge (inclusive), pixels. |
| `y_max` | `int` | no | 1024 | Bottom edge (exclusive), pixels. |

### BackgroundConfig

Subtract a background from every frame before analysis.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `method` | `BackgroundMethod (optional)` | no | None | Primary background: 'constant' subtracts constant_level, 'from_file' subtracts the saved frame at file_path, 'edge' estimates the level from the frame border. Leave unset to apply only additional_constant. |
| `file_path` | `str \| Path (optional)` | no | None | Saved background frame for method 'from_file'. May contain the {scan_dir} placeholder, filled in with the scan folder at run time. |
| `constant_level` | `float` | no | 0.0 | Level subtracted for method 'constant'; also the fallback when a 'from_file' background cannot be read. |
| `additional_constant` | `float` | no | 0.0 | Extra constant subtracted after the primary background. |
| `edge_width` | `int` | no | 1 | Border width in pixels averaged for method 'edge'. |

### CrosshairMaskingConfig

Blank out one or more crosshairs so they do not count as signal.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `crosshairs` | `list[CrosshairConfig]` | no | empty | The crosshairs to mask. |
| `mask_value` | `float` | no | 0.0 | Pixel value written into the masked region. |

### CrosshairConfig

One crosshair to mask out of the frame (a fiducial drawn on a screen, say).

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `center` | `tuple[int, int]` | yes | — | Pixel coordinates (x, y) of the crosshair centre. |
| `width` | `int` | yes | — | Crosshair width in pixels. |
| `height` | `int` | yes | — | Crosshair height in pixels. |
| `thickness` | `int` | yes | — | Thickness of the crosshair lines in pixels. |
| `angle` | `float` | no | 0.0 | Rotation of the crosshair in degrees. |

### CircularMaskConfig

Keep (or discard) only the pixels inside a circle.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `center` | `tuple[int, int]` | no | (512, 512) | Pixel coordinates (x, y) of the circle centre. |
| `radius` | `int` | no | 100 | Circle radius in pixels. |
| `mask_outside` | `bool` | no | True | True masks everything outside the circle; False masks the inside. |
| `mask_value` | `float` | no | 0.0 | Pixel value written into the masked region. |

### VignetteConfig

Undo lens vignetting so the edges of the frame are not artificially dim.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `method` | `VignetteMethod` | no | 'radial_polynomial' | 'radial_polynomial' evaluates vgnt4/vgnt2/vgnt0 radially from the sensor centre; 'map_file' divides by a saved correction map. |
| `full_width` | `int (optional)` | no | None | Full sensor width in pixels (required for radial_polynomial). |
| `full_height` | `int (optional)` | no | None | Full sensor height in pixels (required for radial_polynomial). |
| `x_offset` | `int` | no | 0 | X offset of the saved frame within the full sensor. |
| `y_offset` | `int` | no | 0 | Y offset of the saved frame within the full sensor. |
| `vgnt4` | `float` | no | 0.0 | 4th-order radial coefficient. |
| `vgnt2` | `float` | no | 0.0 | 2nd-order radial coefficient. |
| `vgnt0` | `float` | no | 1.0 | 0th-order (centre) coefficient. |
| `min_model_value` | `float` | no | 1e-09 | Floor on the model value to avoid dividing by ~zero at the corners. |
| `map_file_path` | `str \| Path (optional)` | no | None | Saved .npy correction map for method 'map_file'. |

### ThresholdingConfig

Suppress pixels below (or above) a level — the usual way to kill noise floor.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `method` | `ThresholdMethod` | no | 'constant' | 'constant' uses value as an absolute level; 'percentage_max' uses value as a percentage (0-100) of the frame maximum. |
| `value` | `float` | no | 100.0 | Threshold level: counts for 'constant', percent for 'percentage_max'. |
| `mode` | `ThresholdMode` | no | 'binary' | 'to_zero' zeroes pixels below the level (the common choice); 'binary' makes a 0/1 mask; 'truncate' clips above the level; the _inv variants act on the other side. |
| `invert` | `bool` | no | False | Invert the threshold operation. |

### FilteringConfig

Smooth the frame with a Gaussian and/or a median filter.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `gaussian_sigma` | `float (optional)` | no | None | Gaussian blur width in pixels; unset skips the Gaussian. |
| `median_kernel_size` | `int (optional)` | no | None | Median filter window (odd, in pixels); unset skips the median. |

### NormalizationConfig

Rescale the frame so shots with different exposure or gain compare.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `method` | `NormalizationMethod` | no | 'image_total' | 'image_total' divides by the pixel sum, 'image_max' by the peak, 'constant' by constant_value, 'distribute_value' divides by the sum then multiplies by constant_value. |
| `constant_value` | `float (optional)` | no | None | Divisor for 'constant' or multiplier for 'distribute_value'; required for those. |

### TransformConfig

Rotate, flip, or undistort the frame.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `rotation_angle` | `float` | no | 0.0 | Rotation in degrees, positive = counter-clockwise. |
| `flip_horizontal` | `bool` | no | False | Mirror left-right. |
| `flip_vertical` | `bool` | no | False | Mirror top-bottom. |
| `distortion_correction` | `bool` | no | False | Apply the polynomial distortion correction. |
| `distortion_coeffs` | `list[float] (optional)` | no | None | Distortion coefficients; required when distortion_correction is on. |

### Line1DConfig

How a device's traces are loaded and cleaned up before the analyzer measures them.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `type` | `'line'` | no | 'line' | Marks this as a line (1D trace) section. |
| `description` | `str` | no | '' | Free-text note about this trace. |
| `metadata` | `dict[str, Any] (optional)` | no | None | Free-form documentation (location, notes, calibration constants). Nothing in the pipeline reads it. |
| `data_loading` | `Data1DLoading` | yes | — | How to read one trace file. |
| `label` | `str` | no | 'x vs y' | Human-readable description of what the trace is, e.g. 'charge density vs energy'. Shown on figures; not interpreted. |
| `x_units` | `str (optional)` | no | None | X-axis units (e.g. 'nm', 's'); overrides units read from the file. |
| `y_units` | `str (optional)` | no | None | Y-axis units (e.g. 'V', 'counts'); overrides units read from the file. |
| `x_scale_factor` | `float` | no | 1.0 | Multiplier applied to x before any processing (1e9 turns seconds into nanoseconds). ROI bounds are in the scaled units. |
| `y_scale_factor` | `float` | no | 1.0 | Multiplier applied to y before any processing. Thresholds are in the scaled units. |
| `processing_dtype` | `'float16' \| 'float32' \| 'float64' \| 'int8' \| 'int16' \| 'int32' \| 'int64' \| 'uint8' \| 'uint16' \| 'uint32' \| 'uint64'` | no | 'float64' | NumPy dtype used while processing. |
| `storage_dtype` | `'float16' \| 'float32' \| 'float64' \| 'int8' \| 'int16' \| 'int32' \| 'int64' \| 'uint8' \| 'uint16' \| 'uint32' \| 'uint64'` | no | 'float32' | NumPy dtype used when saving processed traces. |
| `roi` | `LineROIConfig (optional)` | no | None | X-range crop. |
| `interpolation` | `LineInterpolationConfig (optional)` | no | None | Resampling onto a uniform x grid. |
| `background` | `LineBackgroundConfig (optional)` | no | None | Background subtraction. |
| `filtering` | `LineFilteringConfig (optional)` | no | None | Smoothing. |
| `thresholding` | `LineThresholdingConfig (optional)` | no | None | Thresholding. |
| `pipeline` | `list[LinePipelineStepType]` | no | empty | The steps that run, in order. A step runs only if listed here AND its section is present; empty means the raw trace is analyzed. |

### Data1DLoading

How to read one trace file into an x-vs-y array.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `data_type` | `Data1DType` | yes | — | File format: 'tek_scope_hdf5' or 'tdms_scope' for scope captures, 'csv' / 'tsv' for delimited text, 'npy' for a saved array. |
| `trace_index` | `int` | no | 0 | Which trace / channel holds the y values (scope formats). |
| `x_trace_index` | `int (optional)` | no | None | Which trace holds the x values; unset derives x from the waveform properties (scope formats). |
| `delimiter` | `str (optional)` | no | None | Column delimiter for csv/tsv; unset uses the format's default. |
| `x_column` | `int` | no | 0 | Column index of x (text formats). |
| `y_column` | `int` | no | 1 | Column index of y (text formats). |
| `auxiliary_columns` | `dict[str, int]` | no | empty | Extra named columns to load alongside y, for analyzers that need them (name -> column index). |

### LineROIConfig

Keep only the part of the trace between two x values.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `x_min` | `float (optional)` | no | None | Lowest x value kept (inclusive); unset = no lower bound. |
| `x_max` | `float (optional)` | no | None | Highest x value kept (inclusive); unset = no upper bound. |

### LineInterpolationConfig

Resample the trace onto a uniform x grid (so waterfall plots share an axis).

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `num_points` | `int` | no | 1500 | Number of points in the resampled trace. |
| `x_min` | `float (optional)` | no | None | Start of the grid; unset uses the data minimum. |
| `x_max` | `float (optional)` | no | None | End of the grid; unset uses the data maximum. |

### LineBackgroundConfig

Subtract a background from the trace's y values.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `method` | `LineBackgroundMethod` | no | 'none' | 'constant' subtracts constant_level from y; 'from_file' subtracts the trace at file_path (any supported data_type); 'none' skips. |
| `constant_level` | `float (optional)` | no | None | Level subtracted from y for method 'constant'. |
| `file_path` | `Path (optional)` | no | None | Background trace file for method 'from_file'. |

### LineFilteringConfig

Smooth the trace.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `method` | `LineFilterMethod` | no | 'none' | 'gaussian' (uses sigma), 'median' (uses kernel_size), 'bilateral', or 'none'. |
| `kernel_size` | `int (optional)` | no | 3 | Filter window in samples (odd), for the median filter. |
| `sigma` | `float (optional)` | no | 1.0 | Gaussian width in samples, for the Gaussian filter. |

### LineThresholdingConfig

Clip trace values below (or above) a level.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `method` | `LineThresholdMethod` | no | 'none' | 'absolute' uses threshold_value, 'percentile' uses percentile, 'none' skips. |
| `threshold_value` | `float (optional)` | no | None | Level in y units for method 'absolute'. |
| `percentile` | `float (optional)` | no | None | Percentile of y (0-100) for method 'percentile'. |
| `clip_below` | `bool` | no | True | True clips values below the level; False clips values above. |

### ScanRuntime

How the analyzer runs over a scan: order, granularity, what is saved, where files are.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `priority` | `int` | no | 100 | Run order within a group: lower runs first. 100 is the background default. |
| `mode` | `'per_shot' \| 'per_bin'` | no | 'per_shot' | 'per_shot' analyzes every frame; 'per_bin' averages each bin's frames first and analyzes once per bin — for metrics that are not linear in the image. |
| `save` | `bool` | no | True | Write per-shot / per-bin outputs (HDF5, PNG) into the analysis tree. S-file scalar columns are written regardless. |
| `gdoc_slot` | `int (optional)` | no | None | Which cell (0-3) of the scan-log entry's 2x2 figure table gets this analyzer's summary; unset uploads figures as links instead. |
| `device` | `str (optional)` | no | None | Data subfolder under the scan when it differs from the diagnostic name (stitched or post-processed outputs in a sibling folder). |
| `file_tail` | `str (optional)` | no | None | Filename suffix that identifies this device's files ('.png', '.tdms', '_postprocessed.tsv'); unset uses the analyzer's default. |
| `data_format` | `'per_shot_files' \| 'device_hdf5' (optional)` | no | None | 'device_hdf5' reads the capture daemon's per-device frame stack (falls back to per-shot files when absent). Only for analyzers that do not derive output names from the shot file path. |
| `renderer` | `RendererOptions` | no | RendererOptions(colormap_mode=None, cmap=None, vmin=None, vmax=None, duration=None, dpi=None, xlabel=None, ylabel=None, colorbar_label=None, mode=None, waterfall_sort_key=None, waterfall_sort_sigma=None, waterfall_sort_bounds=None, waterfall_even_y_spacing=None, figsize=None, figsize_inches=None) | Summary-figure cosmetics; unset fields keep the renderer defaults. |
| `background_source` | `BackgroundSource (optional)` | no | None | A scan-dependent background (another scan, this scan's own shots, or an autodetected averaged file). Fixed files go on image.background. |

### RendererOptions

Cosmetic overrides for the scan summary figures (colormap, labels, layout).

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `colormap_mode` | `'auto' \| 'sequential' \| 'diverging' \| 'custom' (optional)` | no | None | 'sequential' runs 0 to max; 'diverging' is symmetric about zero; 'auto' picks diverging when the data crosses zero; 'custom' uses vmin/vmax as given. |
| `cmap` | `str (optional)` | no | None | Matplotlib colormap name (e.g. 'plasma', 'RdBu_r'). |
| `vmin` | `float (optional)` | no | None | Colour scale minimum. |
| `vmax` | `float (optional)` | no | None | Colour scale maximum (2D: the old plot_scale). |
| `duration` | `float (optional)` | no | None | Animation frame duration, ms. |
| `dpi` | `int (optional)` | no | None | Figure resolution, dots per inch. |
| `xlabel` | `str (optional)` | no | None | X-axis label. |
| `ylabel` | `str (optional)` | no | None | Y-axis label. |
| `colorbar_label` | `str (optional)` | no | None | Colour bar label. |
| `mode` | `'waterfall' \| 'overlay' \| 'grid' (optional)` | no | None | Trace summary layout: 'waterfall' heat map (x vs scan parameter), 'overlay' of all bins, or a 'grid' of one plot per bin. 1D only. |
| `waterfall_sort_key` | `str (optional)` | no | None | For a noscan waterfall, order rows by this s-file column instead of shot number ('Device:Var' or a substring). 1D only. |
| `waterfall_sort_sigma` | `float (optional)` | no | None | Drop shots whose sort-key value lies outside mean ± this many standard deviations. 1D only. |
| `waterfall_sort_bounds` | `tuple[float, float] (optional)` | no | None | Explicit (low, high) bounds on the sort key; overrides the sigma cut. 1D only. |
| `waterfall_even_y_spacing` | `bool (optional)` | no | None | Draw waterfall rows at equal height regardless of sort-key spacing. 1D only. |
| `figsize` | `tuple[float, float] (optional)` | no | None | Panel (width, height) in inches for grid montages. 2D only. |
| `figsize_inches` | `float (optional)` | no | None | Side of the square animation frames, inches. 2D only. |

### BackgroundSource

Where a scan-dependent background comes from — exactly one of the three.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `scan_number` | `int (optional)` | no | None | Average this earlier scan's frames of the same device and use that. |
| `from_current_scan` | `FromCurrentScanSpec (optional)` | no | None | Collapse this scan's own shots into a background. |
| `autodetect` | `AutodetectBackgroundSpec (optional)` | no | None | Find a precomputed averaged background in the day's analysis folder. |

### FromCurrentScanSpec

Build the background from this scan's own shots, collapsed pixel by pixel.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `method` | `'median' \| 'percentile'` | no | 'median' | How the shot stack is collapsed: per-pixel median or percentile. |
| `percentile` | `float (optional)` | no | None | Percentile (0-100) for method 'percentile'; must be unset for 'median'. |

### AutodetectBackgroundSpec

Use the averaged-background file another analyzer already wrote for this scan.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|

## `analysis_group`

### AnalysisGroup

A named set of diagnostics to run after each scan, in priority order.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `name` | `str` | yes | — | Display name, conventionally <facility>_<purpose>. |
| `description` | `str (optional)` | no | None | Free-text note about when this group is used. |
| `upload_to_scanlog` | `bool` | no | True | Upload the group's summary figures to the experiment scan log. |
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

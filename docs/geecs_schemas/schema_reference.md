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
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `mode` | `ScanRequestMode` | yes | — | What kind of scan: 'step' sweeps one or more axes, 'noscan' collects shots without moving anything, 'optimize' lets an algorithm pick the settings. |
| `axes` | `list[ScanAxis]` | no | empty | For step scans: what to sweep. One entry is a simple 1-D scan; several entries form a grid visiting every combination, with the first axis as the outermost (slowest) loop and the last as the innermost (fastest). Leave empty for noscan and optimize. |
| `shots_per_step` | `int` | no | 1 | How many shots to take at each scan position / grid point (or in total for a noscan). |
| `acquisition` | `AcquisitionMode` | no | 'strict' | 'strict' fires shot by shot and guarantees every device is in every row; 'free_run' lets the trigger run at the machine rate and matches devices up by timestamp. |
| `save_sets` | `list[str]` | no | empty | Names of the save sets — reusable named device groups — recorded for this scan; devices are unioned across them. Each names the devices that get guarantees (completeness, dialogs, images, rituals). A bare string is accepted and stored as a one-element list. Empty means no required devices beyond scan bookkeeping. |
| `background_telemetry` | `bool (optional)` | no | None | Also log every other live experiment device as best-effort snapshot columns — the variables the GEECS experiment database marks for scan logging (MySQL table expt_device_variable, get='yes') — read from the gateway's always-on monitor cache: read-only and never waited on, so it cannot slow or stall the scan; dead devices are dropped with a log line, never a dialog or abort. Leave unset to inherit the experiment default; set true/false to override for this scan. |
| `trigger_profile` | `str (optional)` | no | None | Name of the trigger profile that drives the shot trigger. Unset means the scan does not manage the trigger. |
| `trigger_variant` | `str (optional)` | no | None | Optional variant of the trigger profile to use, e.g. 'laser_off'. Leave unset for the profile's base behaviour. |
| `actions` | `ActionBindings` | no | ActionBindings(setup=[], per_step=[], closeout=[]) | Named action plans to run before the scan (setup), between steps (per_step), and after it (closeout). |
| `description` | `str` | no | '' | Free-text note about this scan; it ends up in the scan's metadata and the experiment log. |
| `background` | `bool` | no | False | Mark this scan's data as background/calibration shots so analysis can find them later. |
| `optimization` | `OptimizationSpec (optional)` | no | None | The optimization problem definition. Required for (and only allowed with) mode 'optimize'. |

Example:

```yaml
schema_version: 1
mode: step
axes:
  - variable: jet_z
    positions: {start: 4.0, end: 6.0, step: 0.5}
  # add more axes to scan a grid — the first axis is the outermost
  # (slowest) loop, the last the innermost (fastest), e.g.:
  # - variable: gas_pressure
  #   positions: {values: [1.5, 2.0, 2.5]}
shots_per_step: 10
acquisition: free_run
save_sets: [undulator_baseline, aux_diagnostics]  # unioned; a bare string also works
trigger_profile: htu_shot_control
actions:
  setup: [pre_scan_ebeam]
  per_step: []
  closeout: []
description: "jet z scan with probe"
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
| `module` | `str` | yes | — | Python import path of the evaluator module, e.g. 'geecs_scanner.optimization.evaluators.beam_sum_counts_evaluator'. |
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
| `images` | `bool` | no | False | Save the device's images / non-scalar files (camera frames, traces) alongside the scalar data. |
| `role` | `SaveRole (optional)` | no | None | Override for how this device is synchronized with shots. Leave unset to let the scanner decide; set 'snapshot' for slow readbacks that don't produce one value per shot. |
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
| `forward` | `str` | yes | — | Formula for this device's value in terms of the scanned number, written with 'composite_var' as the scanned value — e.g. 'composite_var * -2' or '8.5 + (composite_var-10)*2.5'. |

## `trigger_profile`

### TriggerProfile

The device writes that drive the machine through its trigger states.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `schema_version` | `int` | no | 1 | Format version of this config file. Leave at 1 — tools update this automatically when the file format changes. |
| `name` | `str` | yes | — | The name scans use to refer to this trigger profile. |
| `states` | `dict[TriggerState, list[TriggerWrite]]` | no | empty | For each trigger state, the writes that put the machine into it, applied in order from top to bottom. A transition may write several devices. Omit a device variable from a state to leave it untouched. |
| `variants` | `dict[str, TriggerVariant]` | no | empty | Named alternative operating conditions (e.g. 'laser_off'), each listing only the writes that differ from the base states. |
| `description` | `str` | no | '' | Optional note about what setup this profile is for. |

Example:

```yaml
schema_version: 1
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
variants:
  laser_off:
    states:
      OFF:
        - {device: U_DG645_ShotControl, variable: Trigger.Source,
           value: Single shot}
      SCAN:
        - {device: U_DG645_ShotControl, variable: Trigger.Source,
           value: Internal}
      ARMED:
        - {device: U_DG645_ShotControl, variable: Trigger.Source,
           value: Single shot}
```

### TriggerWrite

One device variable set during a state transition.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `device` | `str` | yes | — | The device to write to, e.g. 'U_DG645_ShotControl' or a gas-jet controller — any settable device can take part in a transition. |
| `variable` | `str` | yes | — | Which variable on the device to set, e.g. 'Trigger.Source'. |
| `value` | `str` | yes | — | The value to send, exactly as the device expects it — a number as text ('4.0'), a word ('on'), or a device option name ('External rising edges'). |

### TriggerVariant

A named operating condition that tweaks a few writes of the profile.

| Field | Type | Required | Default | What it does |
|---|---|---|---|---|
| `states` | `dict[TriggerState, list[TriggerWrite]]` | yes | — | Only the writes that differ from the base profile, per state. A write here replaces the base write to the same device variable; writes to new device variables are added after the base ones. Anything not listed keeps its base value. |
| `description` | `str` | no | '' | Optional note about when to use this variant. |

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

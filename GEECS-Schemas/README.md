# GEECS-Schemas

Versioned Pydantic models for every GEECS scanner config — scan requests,
save sets, scan variables, trigger profiles, and action plans — plus
converters from every legacy YAML dialect.

**Configs are schemas; YAML is just serialization.** This package is the
schema layer of the target architecture
(`Planning/vision/00_target_architecture.md` §4). It depends on **pydantic
only**, so the engine, the GUI, scripts, and docs tooling can all import the
same models without dragging in hardware or analysis stacks. Nothing imports
it yet — it lands first, consumers migrate to it converter-first.

## Design principles

- **Pydantic-first.** Every config is a versioned model (`schema_version`);
  loaders validate on read; GUI editors, scripts, and generated docs share
  the same models. Unknown keys are rejected (`extra="forbid"`) so typos
  fail loudly.
- **Declare intent, derive mechanics.** Legacy configs encoded *how*
  (`synchronous` flags, force-appended `acq_timestamp`, per-state write
  matrices, `shots_per_step` derived from rep-rate×wait). The new models
  declare *what*; the engine derives the rest. The derivation rules are
  documented on the models they replace (see `save_set.py`).
- **Device facts live below the configs.** Limits, units, tolerances, enum
  choices, and per-variable scan policy belong to the GEECS experiment
  database (MySQL; the scan policy is the `expt_device_variable` table —
  `get`/`set`/`startvalue`/`endvalue`), surfaced as gateway PV metadata —
  client YAML never repeats them, it only overrides them.
- **Operator-language documentation is part of the schema.** Every field's
  `description` and every model's first docstring paragraph are written for
  operators; `geecs_schemas.docgen` renders them to Markdown, and a test
  fails CI if a field lacks a description. Docs cannot drift from code.
- **Migration over flag day.** Every schema ships with a converter from the
  current YAML (`geecs_schemas.convert`), validated against the full real
  config corpus.

## Model inventory

| Kind (registry key) | Model | Replaces |
|---|---|---|
| `scan_request` | `ScanRequest` | scan presets, `ScanConfig`, GUI submission state |
| `save_set` | `SaveSet` | save elements (`save_devices/*.yaml`) — now tier 1 of the two-tier recording model: the *required* devices with guarantees; everything else is background telemetry (soft, read-only, never waited on) |
| `scan_variables` | `ScanVariables` | `scan_devices.yaml` + `composite_variables.yaml` |
| `trigger_profile` | `TriggerProfile` | shot-control configs (incl. laser-on/off file pairs → variants); states are machine states holding *ordered, multi-device* write lists |
| `action_plan` | `ActionPlan` | one entry of the action library |
| `action_plan_library` | `ActionPlanLibrary` | `action_library/actions.yaml` |
| `experiment_defaults` | `ExperimentDefaults` | (new — legacy kept these choices in GUI state) per-experiment fallbacks where a scan request is silent; defaults run first, then the scan's own |

`SCHEMA_REGISTRY` in `geecs_schemas/__init__.py` maps the kind strings to the
models for generic tooling.

Supporting models: `ScanAxis` (a step scan sweeps one axis or several — a
multi-axis request is an outer-product grid, first axis outermost/slowest;
schema-side only in M1), `PositionRange` / `PositionList`, `ActionBindings`
(setup / **per_step** / closeout slots), `OptimizationSpec` (+
`EvaluatorSpec`, `GeneratorSpec` — covers the legacy Xopt VOCS surface),
`SaveSetEntry` / `SaveRole` (entries carry optional `setup` / `closeout`
action-plan name references — the device's ritual travels with it through
composition — plus the DB scan-default surfaces: `db_scalars`, on by default
for new configs, making the DB's scan-logging telemetry the standard scalar
source with `scalars` as additive extras — converted legacy elements carry
an explicit `db_scalars: false` to preserve their exact old behavior; and
the reserved `at_scan_start` / `at_scan_end` fields, which would override
the DB's set-side start/end writes but are **not honored in this version**
— the set-side is disabled (triggering and camera saving are handled by the
trigger profile / shot controller and the scanner's save-windowing), and the
fields are kept for a possible future re-enable),
`ScanVariable` / `PseudoScanVariable`, `TriggerWrite` /
`TriggerVariant` / `TriggerState`, `DefaultActions`, and the four action
step types.

One non-model module: `geecs_schemas.restricted_expr` — the shared
AST-whitelist core behind both GEECS expression eval sites (the gateway's
derived channels and the engine's pseudo-variable forward formulas).
Stdlib-only; each consumer supplies its whitelist and error wrapping.

## Converter usage

Converters take a parsed dict or a YAML path (path input needs PyYAML, which
is a dev dependency only) and raise `SchemaConversionError` naming exactly
what could not be mapped — nothing is dropped silently.

```python
from geecs_schemas.convert import (
    convert_save_element,
    convert_scan_variables,
    convert_shot_control,
    merge_trigger_variant,
    convert_action_library,
    convert_scan_preset,
    convert_optimizer_config,
)

# Save element → SaveSet (+ extracted setup/closeout ActionPlans + notes)
result = convert_save_element("save_devices/UC_Aline1.yaml")
result.save_set, result.actions, result.notes

# scan_devices.yaml + composite_variables.yaml → one ScanVariables catalog
catalog = convert_scan_variables(
    "scan_devices/scan_devices.yaml", "scan_devices/composite_variables.yaml"
)

# Shot control → TriggerProfile; fold the parallel laser-off file into a variant
base = convert_shot_control("shot_control_configurations/HTU-Normal.yaml")
off = convert_shot_control("shot_control_configurations/HTU-LaserOFF.yaml")
profile = merge_trigger_variant(base, off, "laser_off")
profile.writes_for("SCAN", variant="laser_off")

# Action library → ActionPlanLibrary (nested run-references validated)
library = convert_action_library("action_library/actions.yaml")

# Preset → ScanRequest (compose the referenced elements into one SaveSet)
preset = convert_scan_preset(
    "scan_presets/00_focuscan.yaml",
    save_sets={"LP-FocusDiagnostics": some_save_set},
)
preset.scan_request, preset.composed_save_set

# Optimizer config → OptimizationSpec (+ preserved device_requirements)
opt = convert_optimizer_config("optimizer_configs/hexapod_alignment.yaml")
```

## Generated reference docs

```python
from geecs_schemas.docgen import render_reference
print(render_reference())   # Markdown for every registered schema
```

## Tests

```bash
poetry install
poetry run pytest tests -q
```

The suite is hermetic (fixtures + golden files under `tests/`). The
additional `integration`-marked test walks the sibling
`GEECS-Plugins-Configs` checkout and converts **every** real config file; it
auto-skips when that checkout is absent.

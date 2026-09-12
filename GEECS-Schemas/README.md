# GEECS-Schemas

Versioned Pydantic models for every GEECS scanner config — presets (the
saved scan: device group + plan call), scan requests, scan variables,
trigger profiles, and action plans — plus converters from the legacy YAML
dialects still in use (shot control, optimizer configs).
Scan-variable catalogs, presets and action libraries have no converter:
they are authored new-schema only (the legacy scan-device pair was retired
2026-09, GEECS-Plugins#779; the save elements and scan presets were
regenerated as presets once, #807; the action libraries were regenerated
as `ActionPlanLibrary` documents once, 0.22.0).

**Configs are schemas; YAML is just serialization.** This package is the
schema layer of the target architecture. It depends on **pydantic
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
  declare *what*; the engine derives the rest.
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

## Versioning policy — two dials

Every top-level document carries an integer `schema_version`
(`VersionedSchemaModel` in `_base.py`), and the package has its own
semver in `pyproject.toml`. They answer different questions and move
independently:

| Dial | Answers | Bump when |
|---|---|---|
| `schema_version` (per document kind, integer) | "Does a reader need a migration step to load this file?" | A field moves, is renamed, is removed, or changes meaning. Never for additive changes. |
| Package version (semver, `CHANGELOG.md`) | "What changed in the code and the schemas?" | Every change. Additive field = minor. A `schema_version` bump = minor, because the migration keeps old documents valid. |

- **Bump `schema_version` only with a migration.** A new format generation
  ships a `mode="before"` validator that lifts the previous layout into the
  new one, so every document ever written keeps validating forever. The
  validator normalizes a stale `schema_version` up to the current one and
  never overwrites a newer one. `ScanRequest._lift_v1_layout` is the
  template.
- **Additive changes do not bump `schema_version`.** A new optional field
  with a default leaves old documents valid and needs no migration, so the
  marker stays put. The change is recorded by the package version, the
  changelog, and the regenerated JSON Schema artifact
  (`docs/geecs_schemas/scan_request.schema.json`, kept current by a no-drift
  test) — the artifact's git history is the field-level audit trail.
  The exporter is a registry (`schema_export.EXPORTED_SCHEMAS`, one
  artifact per entry, one no-drift guard iterating it): a new published
  contract is one registry line plus a regenerate.
- **The marker is an integer, not a semver.** Nothing branches on a minor
  schema version: a document is either liftable or already current. The
  finer-grained story lives in the changelog. Do not introduce `1.1`-style
  markers.
- **Each document kind versions on its own.** ScanRequest is at v3 and
  TriggerProfile at v2 (each bumped by its own removed field); Preset and
  the rest stay at v1 until their own layout changes.  The staleness test
  is one helper, `stale_schema_version` in `_base.py`, shared by every
  kind's lifting validator.

## Model inventory

| Kind (registry key) | Model | Replaces |
|---|---|---|
| `preset` | `Preset` | save elements (`save_devices/*.yaml`) + scan presets (`scan_presets/*.yaml`): the device group (`device`, `save_images`) plus the stock plan call — a saved queue item (GEECS-Plugins#807, PR 2) |
| `scan_request` | `ScanRequest` | `ScanConfig`, GUI submission state (the Console / MCP funnel contract until they are rewired onto presets) |
| `scan_variables` | `ScanVariables` | `scan_devices.yaml` + `composite_variables.yaml` — retired 2026-09: catalogs are authored new-schema only, there is no converter |
| `trigger_profile` | `TriggerProfile` | shot-control configs (one profile per operating condition); states are machine states holding *ordered, multi-device* write lists |
| `action_plan` | `ActionPlan` | one entry of the action library |
| `action_plan_library` | `ActionPlanLibrary` | `action_library/actions.yaml` — regenerated once from the legacy `actions:` dialect (0.22.0) and authored new-schema only since; there is no converter |
| `experiment_defaults` | `ExperimentDefaults` | (new — legacy kept these choices in GUI state) per-experiment fallbacks where a scan request is silent; defaults run first, then the scan's own |
| `analysis_diagnostic` | `AnalysisDiagnostic` | the unified analysis diagnostic (`scan_analysis_configs/analyzers/<ns>/<id>.yaml`) — format v2: `analyzer:` is a closed discriminated union on `kind` (one spec model per analyzer the suite ships), `image:` is the camera / line processing section, `scan:` the typed scan-runtime section. pre-v2 files are refused — the corpus was regenerated in v2 once (0.19.0) and is authored v2-only since; there is no converter |
| `analysis_group` | `AnalysisGroup` | analysis groups (`scan_analysis_configs/groups/<ns>/<name>.yaml`) — unchanged shape plus the `schema_version` stamp |

`SCHEMA_REGISTRY` in `geecs_schemas/__init__.py` maps the kind strings to the
models for generic tooling.

Supporting models: `ScanAxis` (a step scan sweeps one axis or several — a
multi-axis request is an outer-product grid, first axis outermost/slowest;
schema-side only in M1), `PositionRange` / `PositionList`, `ActionBindings`
(setup / **per_step** / closeout slots), `OptimizationSpec` (+
`EvaluatorSpec`, `GeneratorSpec` — covers the legacy Xopt VOCS surface),
`PresetDevice` / `PlanCall` (the preset's device group — `device`,
`save_images` — and its stock plan call), `ScanVariable` /
`PseudoScanVariable`, `TriggerWrite` / `TriggerState`, `DefaultActions`, and
the four action step types.

The analysis documents live in the `geecs_schemas.analysis` subpackage:
`processing_2d` (`CameraConfig` + its sections), `processing_1d`
(`Line1DConfig`, the `Line*` sections, `Data1DLoading` — a field-for-field
mirror of GEECS-Data-Utils' `Data1DConfig` so this package stays
pydantic-only), `analyzers` (the `AnalyzerSpec` union and the
`ANALYZER_SPECS` kind → model table), `renderer` (`RendererOptions`, one
typed option set for both summary renderers), `scan_runtime`
(`ScanRuntime`, `BackgroundSource`), `diagnostic` and `group`.  The class
path left the document in v2: ImageAnalysis keeps kind → class in its own
registry, so adding an analyzer means one spec model here and one registry
line there.

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
    convert_shot_control,
    convert_optimizer_config,
)

# Shot control → TriggerProfile (one profile per operating condition)
profile = convert_shot_control("shot_control_configurations/HTU-Normal.yaml")
profile.writes_for("SCAN")

# Optimizer config → OptimizationSpec (+ device_requirements as a device group)
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

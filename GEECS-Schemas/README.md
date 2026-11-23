# GEECS-Schemas

Versioned Pydantic models for every GEECS scanner config — presets (the
saved scan: device group + plan call), scan requests, scan variables,
trigger profiles, and setups. Every kind is authored new-schema only; no
legacy-YAML converter remains. The legacy scan-device pair was retired
2026-09 (GEECS-Plugins#779); the save elements and scan presets were
regenerated as presets once (#807); the action library was converted once
into one setup per plan (`setups/<name>.yaml`); the shot-control files were
regenerated as trigger profiles and their converter deleted (#1361).

**Configs are schemas; YAML is just serialization.** This package is the
schema layer of the target architecture. It depends on **Pydantic and gest-api**, so the engine, the GUI, scripts, and docs tooling can all import the
same models without dragging in hardware or analysis stacks. The runtime VOCS model is GEST’s own model; Xopt and analysis stay worker-side.

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
- **Regenerate, then delete the old reader.** A legacy dialect is migrated
  by regenerating the deployed corpus in the new schema once, then deleting
  its converter; the corpus walk validates every deployed document.

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
  artifact per entry, one no-drift guard iterating it): every file kind in
  `storage.KIND_STORAGE` is published from there, so a new file kind's
  storage row publishes it; another contract is one line in the registry.
  Either way, regenerate.
- **The marker is an integer, not a semver.** Nothing branches on a minor
  schema version. The finer-grained story lives in the changelog. Do not
  introduce `1.1`-style markers.
- **Each document kind versions on its own, in one of two ways.** A kind
  whose old documents must keep loading (ScanRequest, at v4: archived run
  metadata carries every version) lifts them in a before-validator; the
  staleness test is one helper, `stale_schema_version` in `_base.py`. A
  kind with a single format pins `schema_version: Literal[n]` and refuses
  any other version: TriggerProfile (v2, its v1 layout retired once no
  document carried it), the screens, the home layout and the EPICS
  documents (v1). Preset and the rest stay at v1 until their own layout
  changes.

## Model inventory

| Kind (registry key) | Model | Replaces |
|---|---|---|
| `preset` | `Preset` | save elements (`save_devices/*.yaml`) + scan presets (`scan_presets/*.yaml`): the device group (`device`, `save_images`) plus the stock plan call — a saved queue item (GEECS-Plugins#807, PR 2) |
| `scan_request` | `ScanRequest` | `ScanConfig`, GUI submission state (the MCP's funnel contract until it is rewired onto presets; the web scanner submits presets) |
| `scan_variables` | `ScanVariables` | `scan_devices.yaml` + `composite_variables.yaml` — retired 2026-09: catalogs are authored new-schema only, there is no converter |
| `trigger_profile` | `TriggerProfile` | shot-control configs (one profile per operating condition); states are machine states holding *ordered, multi-device* write lists |
| `setup` | `Setup` | `action_library/actions.yaml`, one file per named device state (`setups/<name>.yaml`): `set` / `wait` / `check` / `apply` steps |
| `experiment_defaults` | `ExperimentDefaults` | (new — legacy kept these choices in GUI state) per-experiment fallbacks where a scan is silent |
| `analysis_recipe` | `AnalysisRecipe` | the analysis recipe (`scan_analysis_configs/analyzers/<ns>/<id>.yaml`) — format 3: `input`, ordered `steps` and a `measure` (registry references the analysis core binds), `figure`, `summaries`, `scan`. The retired v2 diagnostic (`analyzer:` + `image:`) is refused |
| `himg_job` | `HimgJob` | a `.himg` data-management job in the same tree (top-level `kind:`) |
| `scalar_job` | `ScalarJob` | new s-file columns computed from existing ones by formula (`kind: scalar_columns`) |
| `analysis_group` | `AnalysisGroup` | analysis groups (`scan_analysis_configs/groups/<ns>/<name>.yaml`) — unchanged shape plus the `schema_version` stamp |

`SCHEMA_REGISTRY` in `geecs_schemas/__init__.py` maps the kind strings to the
models for generic tooling.

Supporting models: `ScanAxis` (a step scan sweeps one axis or several — a
multi-axis request is an outer-product grid, first axis outermost/slowest;
schema-side only in M1), `PositionRange` / `PositionList`, `OptimizerConfig` (GEST VOCS, measurements and derived outputs),
`PresetDevice` / `PlanCall` (the preset's device group — `device`,
`save_images` — and its stock plan call), `ScanVariable` /
`PseudoScanVariable`, `TriggerWrite` / `TriggerState`, and the setup's
step types.

The analysis documents live in the `geecs_schemas.analysis` subpackage:
`recipe` (`AnalysisRecipe` and its parts), `processing_1d`
(`Data1DType`, `Data1DLoading` — a field-for-field mirror of
GEECS-Data-Utils' `Data1DConfig` to keep data/analysis runtime
dependencies outside this vocabulary package), `document`
(`load_analysis_document`), `himg` and `group`. A recipe's steps and
measure are the analysis core's registry vocabulary, so adding an
analyzer happens in GEECS-Analysis alone. The `.himg` data-management jobs (`himg_to_stack`,
`himg_compact`, `himg_restore`: one step over the device folder, no
ImageAnalyzer) are not analyzer kinds but a document of their own,
`himg.HimgJob` (top-level `kind:`, `schema_version: 1`), which
ScanAnalysis's factory maps to its runners. `HimgJob.destructive` marks
the one job whose run deletes data files (`himg_compact`), so a host
that offers one-click runs asks for a typed confirmation first.
`scalar.ScalarJob` (`kind: scalar_columns`, `schema_version: 1`) is the
third document: one formula per new s-file column over existing columns
(the `restricted_expr` whitelist, syntax checked at load), shot by shot,
merged back into the s-file — no device folder, no products.

One non-model module: `geecs_schemas.restricted_expr` — the shared
AST-whitelist core behind both GEECS expression eval sites (the gateway's
derived channels and the engine's pseudo-variable forward formulas).
Stdlib-only; each consumer supplies its whitelist and error wrapping.

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

The suite is hermetic (fixtures under `tests/`). The additional
`integration`-marked test walks the sibling `GEECS-Plugins-Configs` checkout
and validates **every** real config file; it auto-skips when that checkout
is absent.


## Native optimizer documents

`OptimizerConfig` v1 embeds `gest_api.vocs.VOCS` directly. Authors may spell
bounds as `[lo, hi]`; GEST serializes its typed form. The schema package pins
GEST 0.1 to match the worker's validated Xopt release. JSON Schema metadata
covers this field because GEST 0.1's custom mapping types lack schema hooks.
`measurements` selects live signals or camera diagnostics, and `derived`
contains arithmetic expressions (including `camera.image_total`) or an
explicit `python: module:function` callable. Expressions use exact registered
symbols; attribute access is never executed. Python callables are trusted
worker code and receive the reduced measurement mapping.

The six keeper examples are in `tests/fixtures/optimizer_configs/`.
Legacy evaluator/device-requirements documents are refused with a migration
reference. `ScanRequest` no longer accepts optimization; use an `optimize`
preset with an `optimizer_config` ID. The MCP submission migration is #727.

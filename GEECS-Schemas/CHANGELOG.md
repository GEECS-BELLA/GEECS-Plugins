# Changelog

All notable changes to GEECS-Schemas are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-07

### Added

- Initial release: versioned, `extra="forbid"` Pydantic v2 models for every
  scanner config kind (vision doc §4):
  - `ScanRequest` — the one submission object (step / noscan / optimize,
    positions as range or explicit list, `actions.per_step` included, full
    `optimization` block covering the legacy Xopt VOCS surface).
  - `SaveSet` — declarative what-to-record entries; `synchronous` / roles /
    `acq_timestamp` are derived, with the derivation rules documented.
  - `ScanVariables` — friendly-name catalog; `setpoint` / `motor` kinds plus
    `pseudo` composite variables with verbatim numexpr forward expressions.
  - `TriggerProfile` — device-agnostic per-state write tables with explicit
    named **variants** replacing parallel laser-on/off files.
  - `ActionPlan` / `ActionPlanLibrary` — set / wait / check / run steps with
    legacy ActionManager semantics; nested plan references validated.
- `SCHEMA_REGISTRY` mapping config kind → model for generic tooling.
- `geecs_schemas.convert` — loud-failure converters from every legacy YAML
  dialect: save elements, `scan_devices.yaml` + `composite_variables.yaml`,
  shot-control configs (incl. folding parallel files into variants),
  action library, scan presets (→ `ScanRequest`), optimizer configs
  (→ `OptimizationSpec`).
- `geecs_schemas.docgen` — dependency-free Markdown reference generator
  driven by the operator-language field descriptions; a test enforces that
  every field of every model carries one.
- Test suite validated against the full real config corpus in the sibling
  `GEECS-Plugins-Configs` checkout (integration-marked, auto-skipping) with
  hermetic fixtures and golden files for CI.

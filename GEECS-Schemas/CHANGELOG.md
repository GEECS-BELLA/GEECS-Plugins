# Changelog

All notable changes to GEECS-Schemas are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-07-09

### Changed

- `DerivedChannels` now allows cross-device input sets when the derived channel
  declares a positive `stale_after` freshness window. Same-device expressions
  remain frame-coherent and do not require `stale_after`.
- **Reserved the DB set-side scan-write fields — not honored; get-side
  `db_scalars` / telemetry unchanged.**  `SaveSetEntry.at_scan_start` /
  `at_scan_end` and `ExperimentDefaults.apply_db_scan_defaults` are kept in
  the schema but their field descriptions and the surrounding docstrings now
  state clearly that the DB **set-side** (scan start/end writes from the
  `set='yes'` rows) is **not applied in this version**: the engine sets up
  triggering via the TriggerProfile / shot controller and camera saving via
  its own save-windowing, so DB boundary writes would race the shot
  controller (on the DG645 the `set='yes'` rows are the very
  trigger/amplitude variables it drives).  The fields are reserved for a
  possible future re-enable.  No fields were removed, so existing configs
  still validate.  The **get-side** — `SaveSetEntry.db_scalars` /
  `all_scalars` (standard telemetry) and the `background_telemetry` defaults
  — is unchanged and still honored.

## [0.3.0] - 2026-07-08

### Added

- `DerivedChannels` schema for CA gateway derived readbacks: operator-curated
  read-only float PVs computed from one source device's numeric push-frame
  values. Schema version 1 enforces same-source-device inputs so calculations
  such as Convectron pressure from one DAQ analog input are frame-coherent.

## [0.2.0] - 2026-07-07

### Changed

- **`ExperimentDefaults` merge rule: closeout is now mirrored** (ratified
  with the M3b action-execution milestone). Default *setup* plans still run
  before the scan's own, but default *closeout* plans now run **after** the
  scan's own — teardown mirrors setup, so the four setup layers (vision doc
  §4.4b) nest like context managers (defaults → save-set entry rituals →
  the scan's own on the way in; exact reverse on the way out) and an
  experiment-wide "return the machine to standby" always runs last.
  Previously the module documented defaults-first for both slots.
  Documentation-contract change only (the merge itself is implemented by
  resolvers, e.g. `geecs_bluesky.scan_request_runner.apply_experiment_defaults`,
  updated in lockstep); field descriptions and the operator-language test
  pin the new ordering.

## [0.1.0] - 2026-07-07

### Added

- Initial release: versioned, `extra="forbid"` Pydantic v2 models for every
  scanner config kind (vision doc §4):
  - `ScanRequest` — the one submission object (step / noscan / optimize,
    positions as range or explicit list, `actions.per_step` included, full
    `optimization` block covering the legacy Xopt VOCS surface). Step scans
    declare `axes: [ScanAxis]` — one axis is a 1-D scan, several form an
    outer-product grid (first axis outermost/slowest); `grid_shape()` /
    `n_steps()` derive the grid size. Schema-side only in this milestone;
    grid execution lands later.
  - `SaveSet` — tier 1 of the **two-tier recording model**: the devices a
    scan *requires* (guarantees, dialogs, strict completeness, images,
    roles, rituals). Tier 2 — background telemetry — softly records every
    other live experiment device's get='yes' variables from the gateway's
    monitor cache (read-only, never waited on, dead devices dropped with a
    log line; no scan start/end writes for the soft tier). `synchronous` /
    roles / `acq_timestamp` are derived, with the derivation rules
    documented. Entries carry optional `setup` / `closeout` action-plan
    name references so a device's ritual travels with it when entries are
    composed into bigger save sets (references de-duplicate; each plan runs
    once). Entries also surface the GEECS DB's scan defaults:
    `at_scan_start` / `at_scan_end` per-variable overrides of the DB's
    set='yes' start/end writes (a value replaces the DB's, an explicit
    `null` suppresses the write, absent inherits), and `db_scalars` —
    **on by default for new configs**, making the DB's get='yes' telemetry
    the standard scalar source with `scalars` as additive extras; the
    converter emits an explicit `db_scalars: false` on every legacy
    element so converted configs keep their exact old behavior. The DB
    rows themselves get no schema (device facts live below configs); the
    engine applies them at runtime for participating devices only,
    skipping save/localsavingpath, and records everything applied for
    provenance.
  - `ScanVariables` — friendly-name catalog; `setpoint` / `motor` kinds plus
    `pseudo` composite variables with verbatim numexpr forward expressions.
  - `TriggerProfile` — device-agnostic *machine* states, each an **ordered,
    possibly multi-device** write list (`TriggerWrite`; order matters
    within a transition), with explicit named **variants** replacing
    parallel laser-on/off files. The legacy single-device shape was an
    accident of the DG645 carrying everything; the converter emits the
    legacy file's device into every write.
  - `ExperimentDefaults` — per-experiment fallbacks (trigger profile,
    setup/closeout plans) applied where a scan request is silent; defaults
    run first, then the scan's own. Includes `apply_db_scan_defaults`
    (default true = MC parity; false runs the experiment purely
    config-driven, ignoring the DB's start/end writes) and
    `background_telemetry` (default true; the per-scan
    `ScanRequest.background_telemetry` is tri-state — unset inherits this,
    true/false overrides). No legacy dialect behind it; resolvers must
    record applied defaults into the resolved request for provenance.
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

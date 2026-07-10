# Changelog

All notable changes to GEECS-Console are documented here.  Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
semantic.

## [0.1.0] - 2026-07-10

### Added

- Package scaffold: the greenfield PySide6 operator console
  (`geecs_console`), per the cutover-strategy decisions
  (`Planning/cutover_strategy/00_overview.md`) and the approved screen map.
- Main window implementing regions R1–R7 from a hand-authored Qt Designer
  `.ui` file: session bar (experiment / rep rate / trigger profile+variant /
  health chips), save-set lists with union preview, scan form (mode radios,
  axis rows, shots per step, acquisition combo, live shot count with the
  `MAXIMUM_SCAN_SIZE = 1e6` guard, description), presets row (stub), submit
  row, "Now" panel (state pill, progress bar, scan-number label with 10 s
  expiry, log tail), device panel (backend stubbed).
- `request_builder.build_scan_request`: the pure
  `ConsoleFormState → geecs_schemas.ScanRequest` mapping (noscan / 1D /
  grid / background; optimization refused until an editor exists).
- `submission.Submitter` protocol + lazy `make_bluesky_submitter` factory
  (the window opens without the `ca` extra or any network).
- `services.configs.ConsoleConfigs`: offline-safe configs-repo listings and
  resolver-backed union preview / trigger variants.
- `services.health`: `HealthProbe` protocol + all-unknown `StubHealth`.
- `events_adapter.ScanEventsAdapter`: engine `on_event` stream → Qt signals.
- Hermetic test suite (pytest-qt, offscreen) including a guard test that
  `geecs_python_api` is never imported.

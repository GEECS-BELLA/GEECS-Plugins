# Changelog

All notable changes to GEECS-Interlocks are recorded here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package is
bumped when it is deployed or tagged, not per PR (root `CLAUDE.md` §
"Release & Versioning").

## [0.2.0] - 2026-11-20

The interlock server rebuilt on `geecs_core`, replacing the standalone
package that ran on the deleted GEECS-PythonAPI and the Bluesky-suspender
design (both on unmerged branches, never on `master`).

### Added

- The engine: one evaluation loop over `geecs_core.client.GeecsDevice`
  subscriptions, one immutable snapshot per tick that every reader shares;
  fail-safe states (`waiting`, `tripped`, `settling`, `ok`, `off`).
- Configurations as GEECS-Schemas' `InterlockSet`, one file per named
  configuration (`<experiment>/interlocks/<name>.yaml`): OR groups of
  threshold, alignment, stale, external (UDP) and settings-held checks;
  implicit staleness (`stale_after_s`); `settle_s`.
- External signals: engine-owned UDP listeners kept across reloads.
- Settings-held checks armed from the window or `serve --arm-at-start`.
- Master Control's stream in two formats: `summary` (default, one keyword
  per frame) and `legacy` (the original server's).
- The server window (`gui` extra): Monitor with the server log, and Builder.
- The read-only web page on the kit, with a fail-safe stale display.
- The daily log file, 30 days kept.
- `--demo` and `--fake-devices` for running with no lab network.

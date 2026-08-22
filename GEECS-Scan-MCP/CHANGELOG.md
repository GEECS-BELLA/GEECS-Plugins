# Changelog

All notable changes to `geecs-scan-mcp` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-08-22

### Added

- **The package** — the GEECS scan MCP server, homed in GEECS-Plugins by
  owner decision (2026-08-22; the design is the scan-MCP planning
  document, verb surface §1, architecture §3).  FastMCP stdio server
  (`python -m geecs_scan_mcp`), the osprey house pattern: module-level
  `mcp`, self-registering tool modules, a `tool_names` constants leaf
  module for profile permission lists, structured JSON envelopes
  (`{ok, ...}` / `{ok: false, error_kind, message}`) — tools never raise
  to the agent, and engine message text is preserved verbatim.
- **v0 read-only tools** (zero write risk):
  - `scan_status` — manager snapshot + pending queue items (never fails;
    disconnected reads as `connected: false` + detail).
  - `scan_history(limit)` — recent items with exit status and the
    operator-facing error line, field-tolerant against the manager's
    history shape.
  - `get_scan_result(scan_number|uid, day?)` — Tiled lookup: run
    metadata incl. the `submission` provenance record, column names, and
    capped per-column mean/std — never the full event table.
  - `list_scan_configs(kind)` — save_sets / trigger_profiles / presets /
    optimizer_configs / scan_variables (kind/target(s)/confirm — never
    limits: those are hardware truth, not schema data) / actions, via
    the resolver's listing surface (GeecsBluesky 0.60.0, #666).
  - `validate_scan_request(request)` — schema + engine validation + the
    full client-side preflight, nothing submitted.
- Runtime singletons resolve from the standard
  `~/.config/geecs_python_api/config.ini`; every unconfigured piece
  degrades honestly (stub queue client, unconfigured catalog, no
  experiment → `invalid_request` envelopes).

### Review hardening (same release, adversarial review on the PR)

- Non-finite stats serialize as `null` (a one-row run's ddof-1 std and
  all-NaN dead-device columns are routine; bare `NaN` tokens are not
  JSON), and every tool wrapper routes through a guard that turns any
  impl bug into an `internal_error` envelope — the tools-never-raise
  contract is now enforced, not aspirational.
- Scan-variable rows rebuilt from the REAL schema shape
  (target/kind/confirm, targets/mode for pseudo) and pinned with real
  `ScanVariable`/`PseudoScanVariable` models — the earlier fake pinned
  fields the schema deliberately does not carry.
- Unknown run uid reads as `not_found` (the catalog's KeyError
  contract), not `tiled_unreachable`; bad `day` strings are decided
  before any catalog I/O.
- `runtime` singletons build under a lock (the concurrent-first-use zmq
  leak qs_client's #653 lock prevents one level down); the resolver is
  deliberately NOT cached so mid-session config edits appear on the
  next listing call.
- `anyio` declared as a direct dependency.

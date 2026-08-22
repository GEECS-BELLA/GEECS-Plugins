# Changelog

All notable changes to `geecs-scan-mcp` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-08-22

### Added

- **The v1 control tools** (owner decisions 2026-08-22: presets AND
  composed dicts from day one, 1,000-shot cap, approval-gated force):
  - `submit_scan(request|preset, description?, acknowledge_warnings?)` —
    validate → agent shot cap (`[scan_mcp] max_shots`, default 1,000;
    optimize needs an explicit `max_iterations`) → queue etiquette (one
    scan in flight; refuses while anything is queued or running, never
    clears implicitly) → full preflight with the
    **acknowledge-warnings loop** (unacknowledged questions return as
    `needs_acknowledgement`; acknowledgements stamp `continued` into
    the run's `SubmissionRecord`) → stamp with the deployment identity
    (`[scan_mcp] client_identity`, default `geecs-scan-mcp <version>`)
    → queue. Submit-and-poll: returns `item_uid` immediately.
  - `stop_scan(force?)` — graceful stop; refuses another client's scan
    naming its submitting identity unless `force=true` (approval-gated
    osprey-side, logged in the result). Approval-only, never behind the
    kill switch (in-tool doctrine).
  - `clear_queue()` — the one remover; lists exactly what it removed.
  - `scan_progress()` — poll-shaped (read-only): RE state, running item
    + submitting client, queue depth, last outcome.
- Backed by GeecsBluesky 0.61.0's `running_item()`/`clear_queue()` on
  the client protocol and `resolve_preset()` on the resolver.

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

### Codex review hardening (same release, second reviewer per process)

- The envelope serializer itself now owns the strict-JSON contract:
  `make_ok` recursively normalizes non-finite floats to `null`
  (`_json_safe`) with `allow_nan=False` as the raising backstop — no
  future payload field can regress the bare-`NaN` failure (P2).
- `get_scan_result`'s missing-selector check runs before the catalog is
  constructed — pure argument validation no longer depends on archive
  setup, and its test needs no catalog patch (P3).

### v1 review hardening (same release, adversarial review on the PR)

- The shot cap counts via the schema's new non-materializing
  `planned_shots()` (GEECS-Schemas 0.11.0) — a pathological
  agent-composed range is refused arithmetically instead of OOMing the
  server inside its own guard (HIGH finding); the three parallel
  size-counters consolidate to one.
- `acknowledge_warnings` names outside the known check vocabulary are
  refused (typo guard), and the honest residual is documented: a
  stateless server cannot stop a first-call pre-acknowledgement — the
  backstops are OSPREY's approval prompt (which shows the arguments)
  and the provenance record (`continued` stamps only for questions
  actually raised).
- The optimize-without-`max_iterations` refusal is genuinely pinned (the
  old test's spec was schema-invalid and never reached the branch) and
  its message no longer misstates the engine (which defaults to 20).
- `forced` in stop results marks ONLY operator-authorized stops of
  another client's scan — a habitual `force=true` on the MCP's own scan
  no longer pollutes the audit marker.
- One spelling of the must-match identity (`client_identity()` feeds
  both the queue user and the SubmissionRecord), resolved outside the
  runtime lock; unparseable `[scan_mcp] max_shots` warns instead of
  silently running at the default.

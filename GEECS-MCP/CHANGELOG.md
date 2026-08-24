# Changelog

All notable changes to `geecs-mcp` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] - 2026-08-23

The v2 verbs (issue #676) — actions, manual moves, pause/resume, and the
document-stream `scan_progress` upgrade.  (0.4.0, the analysis domain,
landed separately via PR #681.)

### Added

- **`run_action` (Q) + `describe_action` (R)**: run a named ActionPlan
  on demand through the queue (`geecs_run_action_plan`; idle-only —
  submitting mid-scan would silently queue the action to auto-run when
  the scan finishes, so an active RE state refuses like `submit_scan`),
  and preview its resolved steps via the worker's
  `geecs_describe_action` dry-run (read-only by effect, but needs an
  idle manager to answer).
- **`move_scan_variable` (Q)**: one manual scan-variable move via the
  worker's `geecs_move_variable` (`function_execute`) — plain, confirm,
  and pseudo variables resolve exactly as a scan axis would; idle-only
  (manager-enforced), blocking up to the client's ~120 s task budget;
  non-finite / non-numeric values refused before reaching the worker.
- **`pause_scan` (S) + `resume_scan` (Q)**: deferred pause / resume with
  the same ownership etiquette as `stop_scan` (a foreign scan is refused
  by name unless `force=true`; `forced` marks only genuinely foreign
  overrides).  `pause_scan` joins the halt family (`STOP_TOOLS` — never
  behind the headless `write_tools` gate: pausing makes the machine
  strictly quieter); `resume_scan` restarts motion (and retries a failed
  move), so it gates like a submission (`QUEUE_TOOLS`).
- **`scan_progress` stream upgrade**: a process-wide `ProgressCache`
  (`scans/progress_stream.py`) consumes the worker's document stream
  (start-doc totals `num_points × shots_per_step` with the
  `max_iterations` fallback for optimize runs; primary-stream `seq_num`
  → shots done; stop-doc exit status) and the manager's console-output
  stream (the engine's failed-move line → the paused scan's reason,
  surfaced only while actually paused).  Strictly best-effort: the
  result's `stream.available=false` (with the reason) degrades to the
  v1 poll answer; the manager poll stays authoritative.  Threading per
  the console's #653 rules — daemon threads, zmq sockets never touched
  cross-thread, no `stop`.

### Changed

- `tool_names`: `RUN_ACTION`/`MOVE_SCAN_VARIABLE`/`RESUME_SCAN` appended
  to `QUEUE_TOOLS` (headless `write_tools` additions — see
  `deploy/DEPLOYMENT.md`), `PAUSE_SCAN` to `STOP_TOOLS`,
  `DESCRIBE_ACTION` to `READ_TOOLS`.
- Requires geecs-bluesky ≥ 0.62.0 (`FAILED_MOVE_LOG_PREFIX` re-exported
  from `geecs_bluesky.qs_client`, defined in its import-light
  `log_markers` module — the light-import contract holds down to name
  resolution).

### Fixed (in review, pre-merge)

- The failed-move `paused_reason` is cleared when primary-stream
  progress resumes — a second (manual) pause of the same run no longer
  reports the first pause's text as the current why.
- `resume_scan` fails CLOSED on an unreadable running item (a go verb
  must not restart a possibly-foreign scan unforced); the halt family
  stays fail-open by doctrine.  `forced` also marks force past unknown
  ownership.
- The client's ~120 s task-poll timeout on `move_scan_variable` /
  `describe_action` now reports `task_timeout` (the taxonomy kind
  existed but nothing emitted it).
- Action/variable names are submitted stripped, matching how they are
  validated; `move_scan_variable`'s description states the raw
  `Device:Variable` pass-through honestly (a direct setpoint write, no
  catalog semantics).

## [0.4.0] - 2026-08-22

### Added

- **The analysis domain** (#675 — the top post-promotion ask, closing
  "scan → analyze → present"): `get_scan_analysis` (per-analyzer task
  statuses from `analysis_status/*.yaml` — tolerantly parsed, the
  schema is ScanAnalysis-owned — plus the capped analysis output tree)
  and `get_scan_figure` (a rendered summary figure as actual MCP image
  content, ≤1024 px longest edge via pillow; `display_files` routed
  first, then tree images; ambiguous → the candidate list).  Both
  read-only/auto-allow.  **Strictly read-only over the data share**:
  only ScanPaths' pure static path builders (the instance
  `get_analysis_folder()` silently `os.makedirs` and is banned here),
  pinned by a nothing-created-on-miss test.  Requires `[Paths]
  geecs_data` + the mounted share on the serving host; an unconfigured
  host refuses honestly (live-verified) after a one-time
  `reload_paths_config()` init (live-run finding: the class attribute
  starts `None` and raised instead of degrading).
- `pillow` and `pyyaml` declared (direct imports).

### Review hardening (same release, adversarial review on the PR)

- **The status parser reads the REAL `TaskStatus.to_dict()` schema**
  (CRITICAL finding: the first cut was written from ScanAnalysis's
  stale CLAUDE.md prose — `status`/`heartbeat` float — which the writer
  never produces; against production data every task would have read
  null).  Now: `state` (queued/claimed/done/failed/no_data), `error`
  (surfaced — the most useful failed-task field), `claimed_by`,
  `last_heartbeat` as ISO-8601 parsed to an age (UTC-naive tolerated,
  per task_queue's own `_parse_ts`).  The stale ScanAnalysis doc is
  fixed in the same wave with a read-the-writer warning.
- Field coercions sit inside the per-file guard and `display_files`
  entries are type-checked — one odd YAML on the writable share
  degrades that entry to `unreadable`, never the whole tool.
- **Figure candidates are bounded to the scan's own analysis folder**
  (resolve + `is_relative_to`): a `display_files` entry pointing
  anywhere else — outside the share, at another scan's outputs, or into
  the raw `scans/` tree — is dropped with a warning, closing the
  confused-deputy path where share-writers could make the MCP serve
  other host-readable files.  (The first review pass bounded to the
  share root; the codex pass tightened it to the scan's analysis folder,
  which is where the writer puts every legitimate entry.)
- A 64 MP decode cap refuses giant share-resident images before the
  full decode (Pillow's own bomb ceiling is ~178 MP ≈ 700 MB RAM on a
  long-lived server).

## [0.3.0] - 2026-08-22

### Changed

- **Renamed: `GEECS-Scan-MCP` → `GEECS-MCP`** (`geecs_scan_mcp` →
  `geecs_mcp`; owner decision, before anything reached master): one
  general GEECS server with domains as subpackages — scans are the
  first domain (`tools/` → `scans/`), not the identity.  FastMCP server
  name `geecs`; config section `[scan_mcp]` → `[mcp]`; default client
  identity `geecs-mcp <version>`.  Future domains (health, db, logs,
  analysis) register on the same server; Windows-only-SDK analysis
  capabilities become a satellite server on a Windows box rather than
  moving this one (CLAUDE.md records the pattern).

### Fixed

- **Gating docs corrected to VERIFIED osprey semantics** (checked
  against the deployed profile 2026-08-22, replacing the assumed
  story): profile-level custom-server `hooks:` keys are silently
  ignored, and the interactive writes kill switch does not cover
  custom-server tools.  The interactive gate is the native `ask`
  prompt (arguments visible); the headless gate is
  `hook_config.json`'s `write_tools` (from the profile's `config:`),
  listing `submit_scan` + `clear_queue` and deliberately NOT
  `stop_scan` — a halt is never blocked on any path (headless by
  designed omission, interactive because the kill switch does not
  cover custom servers).  Two osprey-side issues to be filed from that
  side: silent unknown-key acceptance, and custom-server exclusion
  from the interactive kill switch.  The `hooks:` key is removed from
  every example.

### Added

- **HTTP transport** (`python -m geecs_mcp --transport http --host
  --port`, default port 8100) — the central-deployment mode: one server
  on the qserver box (everything it talks to is local, and client-side
  validation resolves the SAME configs checkout the worker uses), every
  osprey machine integrating with one `url:` line and zero GEECS
  installs.  `deploy/geecs-mcp.service` (resource-capped systemd unit —
  the box is shared with the production manager) + `deploy/DEPLOYMENT.md`
  (both modes, the shared-drive-clone install rule: non-editable only —
  an editable install on an SMB share reads source off the share at
  runtime, a documented fleet failure class).

## [0.2.0] - 2026-08-22

### Added

- **The v1 control tools** (owner decisions 2026-08-22: presets AND
  composed dicts from day one, 1,000-shot cap, approval-gated force):
  - `submit_scan(request|preset, description?, acknowledge_warnings?)` —
    validate → agent shot cap (`[mcp] max_shots`, default 1,000;
    optimize needs an explicit `max_iterations`) → queue etiquette (one
    scan in flight; refuses while anything is queued or running, never
    clears implicitly) → full preflight with the
    **acknowledge-warnings loop** (unacknowledged questions return as
    `needs_acknowledgement`; acknowledgements stamp `continued` into
    the run's `SubmissionRecord`) → stamp with the deployment identity
    (`[mcp] client_identity`, default `geecs-mcp <version>`)
    → queue. Submit-and-poll: returns `item_uid` immediately.
  - `stop_scan(force?)` — graceful stop; refuses another client's scan
    naming its submitting identity unless `force=true` (approval-gated
    osprey-side, logged in the result). Approval-only; not behind the
    kill switch (since verified: holds because the kill switch does not
    cover custom-server tools — see 0.3.0 Fixed).
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
  (`python -m geecs_mcp`), the osprey house pattern: module-level
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
  runtime lock; unparseable `[mcp] max_shots` warns instead of
  silently running at the default.

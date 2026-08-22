# GEECS-Scan-MCP — Developer Context for Claude

The MCP server exposing the GEECS scan service to AI agents — the OSPREY
HTU assistant is the first consumer.  **The spec is the scan-MCP planning
document** (2026-08-21/22, delivered to the owner; verb surface §1,
safety §2, architecture §3, phasing §4) as amended by the relocation
decision: the server code lives HERE, not in the osprey repo; osprey's
`profile.yml` points a stdio `command:` at `python -m geecs_scan_mcp`.

## Boundaries (load-bearing)

- **A client, never an engine.**  Everything goes through the shared
  seams: `geecs_bluesky.qs_client` (queue verbs + preflight),
  `geecs_bluesky.config_resolver` (config resolution + listings, #666),
  `geecs_data_utils.tiled_catalog` (results).  This package must never
  import engine internals (`scan_request_runner`, `session`, `plans/*`,
  `devices/*`) — when a tool needs something private, promote it into a
  small public module in GeecsBluesky instead (the #668 discipline: the
  engine splits emerge from real seams, not guesses).
- **Write-surface doctrine** (standing, 2026-07-23): agent writes go
  through MCP verbs only; scans stay in the GEECS engine — the MCP
  submits ScanRequests, never drives devices shot-by-shot; raw gateway
  PVs are read-only to the agent.
- **No osprey imports.**  The integration surface is `profile.yml` +
  stdio; `tool_names.py` exists so osprey permission lists import
  symbols, not strings.
- Configuration is ONLY the standard
  `~/.config/geecs_python_api/config.ini` (the fleet contract — no new
  config format): `[Experiment] expt`, `[qserver]`, `[tiled]`, the
  configs-repo path.  Every unconfigured piece degrades honestly and the
  server always starts.

## Layout

```
geecs_scan_mcp/
  server.py       # module-level FastMCP + create_server() (osprey pattern)
  __main__.py     # python -m geecs_scan_mcp → stdio
  tool_names.py   # THE tool-name constants (profile lists import these)
  runtime.py      # lazy cached singletons: experiment, queue client
                  #   (user=CLIENT_IDENTITY — how runs trace back to this
                  #   server), resolver, Tiled catalog.  Tools call
                  #   runtime.get_*() through the module attribute — that
                  #   is the test patch seam; never from-import the getters
  errors.py       # the JSON envelope: make_ok / make_error(error_kind)
                  #   — taxonomy in the module docstring; tools NEVER
                  #   raise to the agent; engine text preserved verbatim
  tools/
    read_tools.py # the five v0 tools: async wrappers (anyio.to_thread)
                  #   over sync _*_impl functions — the impls are the
                  #   tested surface
```

## Conventions

- Every tool: `async def` wrapper → `anyio.to_thread.run_sync(_impl)`;
  the impl returns a JSON string envelope.  No tool blocks on scan
  completion — everything is request/response (submit-and-poll when v1
  lands; the two bounded-blocking exceptions, `move_scan_variable` and
  `stop_scan`, are v2/v1 and cap at the client's own ≤120 s budgets).
- Result payloads are context-sized: `get_scan_result` returns metadata
  + column names + capped stats, never the full event table.
- Field-tolerant reads of the manager's shapes (`.get` everywhere in
  history mapping) — the queueserver's payload fields are not a contract
  we own.

## Verb roadmap (the planning doc's phasing, as amended by owner decisions)

- **v0 (built)**: `scan_status`, `scan_history`, `get_scan_result`,
  `list_scan_configs`, `validate_scan_request` — read-only (R),
  auto-allow.
- **v1 (built — owner decisions 2026-08-22)**: `submit_scan` takes
  presets AND composed dicts from day one (the plan's presets-first
  de-risk was dropped: presets are barely used in practice), agent shot
  cap 1,000 (`[scan_mcp] max_shots`; optimize needs explicit
  `max_iterations`), `stop_scan` with approval-gated `force` for
  foreign scans, `clear_queue` as the one remover, poll
  `scan_progress`.  The standing rules, all enforced in
  `tools/control_tools.py`: acknowledge-warnings loop (no silent
  continue past a preflight question; acknowledgements stamp
  `continued` into `SubmissionRecord.preflight`), `clear_pending=False`
  always, ownership etiquette on stop, stop approval-only and never
  behind the kill switch (in-tool doctrine — per-tool hook matchers
  don't exist for custom servers).  The submitted-as identity is
  `[scan_mcp] client_identity` (deployment-owned, e.g.
  `osprey-htu-assistant`) and MUST match on the queue item and the
  `SubmissionRecord` — the ownership check compares against it.
- **v2**: actions, moves, pause/resume, doc-stream `scan_progress`
  (per-shot counts, paused reasons from the console-text stream).

## Testing

`poetry run pytest` — hermetic: the impls are tested against fakes
patched on `runtime`; no manager, no Tiled, no configs repo.  The
registration test asserts every `tool_names.READ_TOOLS` entry is on the
server.  Live verification rides the phasing checklists in the planning
doc (v0: listings match the console's dropdowns, status agrees with the
console's pill, a known scan number resolves from the archive).

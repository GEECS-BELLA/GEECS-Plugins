# GEECS-MCP — Developer Context for Claude

The **general GEECS MCP server** — AI-agent access to GEECS, one server
process with domains as subpackages.  Renamed from `GEECS-Scan-MCP`
(owner decision 2026-08-22, before anything reached master): scans are
the first domain, not the identity.  **The spec for the scans domain is
the scan-MCP planning document** (2026-08-21/22; verb surface §1, safety
§2, phasing §4) as amended by owner decisions recorded in the CHANGELOG.

Domain roadmap (add as they earn their keep, never speculatively):
`scans/` (built: v0 read + v1 control), then candidates in rough value
order — `health/` (gateway/Tiled/DB probes, read-only), `db/`
(device-variable metadata lookups), `logs/` (the /triage analysis as a
tool), `analysis/` (READ + EXECUTION BUILT — the #675 figure/results verbs
over the ScanAnalysis output tree, and the #686 `run_scan_analysis`
execution slice (0.7.0) with ScanAnalysis-as-is as the backend by owner
decision; deeper analysis-over-Tiled later — the verb surface is
backend-neutral so it slots in behind the same tools;
pure-numpy analysis is cross-platform; capabilities needing **Windows-only acquisition SDKs**
do NOT force this server onto Windows — they become a small *satellite
MCP server* on a Windows box (the PVA-gateway camera-server precedent),
registered as a second `profile.yml` entry with the same envelope
conventions), `archiver/` when that project reactivates.

## Boundaries (load-bearing)

- **A client, never an engine.**  Everything goes through the shared
  seams: `geecs_bluesky.qs_client` (queue verbs + preflight),
  `geecs_bluesky.config_resolver` (config resolution + listings, #666),
  `geecs_data_utils.tiled_catalog` (results).  This package must never
  import engine internals (`scan_request_runner`, `session`, `plans/*`,
  `devices/*`) — when a tool needs something private, promote it into a
  small public module in GeecsBluesky instead (the #668 discipline: the
  engine splits emerge from real seams, not guesses).
- **Write-surface doctrine** (2026-07-23, amended 2026-08-25 to match
  deployed practice — owner correction): GEECS-*semantic* writes (scans,
  actions, manual moves, analysis) go through MCP verbs only; scans stay
  in the GEECS engine — the MCP submits ScanRequests, never drives
  devices shot-by-shot, and does no raw PV I/O of its own.  Channel-level
  setpoint writes (`caput` to `:SP` PVs) are NOT MCP territory: osprey's
  own EPICS write tool performs them, bounded by osprey's limits database
  and its own gating.  That raw path bypasses the GEECS client-side
  hardening (put-failure visibility, wire conventions, confirm/pseudo
  semantics, mid-scan refusals — only device limits + gateway atomicity
  hold server-side), which is why anything with GEECS semantics still
  belongs behind an MCP verb.
- **No osprey imports.**  The integration surface is `profile.yml` +
  stdio or HTTP (`deploy/DEPLOYMENT.md` — central HTTP on the qserver
  box is the multi-machine mode; stdio is the dev loop);
  `tool_names.py` exists so osprey permission lists import symbols,
  not strings.
- **Never duplicate osprey's raw-PV channel tools** — this server
  exposes GEECS-semantic surfaces (scan service, config catalogs,
  archived results, DB metadata) that raw EPICS access cannot provide.
- Configuration is ONLY the standard
  `~/.config/geecs_python_api/config.ini` (the fleet contract — no new
  config format).  **`deploy/DEPLOYMENT.md` is the one full key
  inventory** — don't duplicate it here.  Every unconfigured piece
  degrades honestly and the server always starts.

## Layout

```
geecs_mcp/
  server.py       # module-level FastMCP ("geecs") + create_server()
  __main__.py     # python -m geecs_mcp → stdio (default) or
                  #   --transport http (the central service mode)
  tool_names.py   # THE tool-name constants (profile lists import these)
  runtime.py      # lazy cached singletons: experiment, queue client
                  #   (user=CLIENT_IDENTITY — how runs trace back to this
                  #   server), resolver, Tiled catalog.  Tools call
                  #   runtime.get_*() through the module attribute — that
                  #   is the test patch seam; never from-import the getters
  errors.py       # the JSON envelope: make_ok / make_error(error_kind)
                  #   — taxonomy in the module docstring; tools NEVER
                  #   raise to the agent; engine text preserved verbatim
  scans/          # THE scans domain (future domains = sibling packages)
    read_tools.py # the v0 read tools: async wrappers (anyio.to_thread)
                  #   over sync _*_impl functions — the impls are the
                  #   tested surface
    control_tools.py # the v1 verbs: submit (cap + etiquette + the
                  #   acknowledge-warnings loop), stop (ownership),
                  #   clear_queue, scan_progress — plus the v2 verbs:
                  #   run_action/describe_action, move_scan_variable,
                  #   pause_scan/resume_scan (ownership like stop)
    progress_stream.py # ProgressCache — the best-effort document-stream
                  #   + console-text-stream picture behind scan_progress
                  #   (daemon threads, zmq never touched cross-thread,
                  #   no stop — the console's #653 rules)
  analysis/       # the analysis domain (#675)
    read_tools.py # get_scan_analysis (task statuses from
                  #   analysis_status/ + the output tree, payload-
                  #   budgeted with explicit *_truncated flags) and
                  #   get_scan_figure — a figure REFERENCE by default
                  #   (label, dims, bytes, share-relative path,
                  #   server-relative figure_url), 0.6.0 payload
                  #   doctrine: a 247 KB inline PNG blew a haiku-tier
                  #   agent context in the first web-UI integration, so
                  #   image bytes ride model context ONLY via the
                  #   opt-in thumbnail=true (≤768 px JPEG).  The
                  #   /figures/{day}/{scan}/{label} custom route on the
                  #   same server streams the ORIGINAL bytes (bounded
                  #   by the tool's own candidate set + a byte cap) for
                  #   clients to fetch-and-save as artifacts.
                  #   STRICTLY read-only over the data share: pure
                  #   ScanPaths static builders ONLY — the instance
                  #   get_analysis_folder() silently mkdirs and must
                  #   never be used here; pinned by a nothing-created
                  #   test.  Needs [Paths] geecs_data + the share
                  #   mounted on the serving host (degrades honestly)
    run_tools.py  # the execution slice (#686, 0.7.0): run_scan_analysis
                  #   (Q — validate-then-refuse BEFORE side effects:
                  #   exactly-one selector, configs root, scan folder
                  #   EXISTS (never created), analyzers construct on
                  #   this host so Windows-SDK diagnostics refuse up
                  #   front; statuses initialized server-side; then a
                  #   detached run_worker subprocess drives ScanAnalysis's
                  #   own task queue) + list_analyzers /
                  #   list_analysis_groups (R).  ScanAnalysis-as-is is
                  #   the backend BY OWNER DECISION 2026-08-24 — verb
                  #   surface backend-neutral, analysis_status/ = the
                  #   progress contract, so the Tiled stack can slot in
                  #   later.  Rides the optional analysis-run extra;
                  #   gdoc upload hard-off (an outward publish needs its
                  #   own gated verb)
    run_worker.py # the detached worker: one JSON argv payload ->
                  #   build_worklist + run_worklist for one scan; stdio
                  #   dropped — the status YAMLs are the observable
                  #   surface
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
  cap 1,000 (`[mcp] max_shots`; optimize needs explicit
  `max_iterations`), `stop_scan` with approval-gated `force` for
  foreign scans, `clear_queue` as the one remover, poll
  `scan_progress`.  The standing rules, all enforced in
  `geecs_mcp/scans/control_tools.py`: acknowledge-warnings loop (no silent
  continue past a preflight question; acknowledgements stamp
  `continued` into `SubmissionRecord.preflight`), `clear_pending=False`
  always, ownership etiquette on stop, stop approval-only.
  **Gating semantics (VERIFIED against osprey 2026-08-22, replacing the
  earlier assumed story)**: profile-level custom-server `hooks:` keys
  are silently ignored, and the interactive writes kill switch does not
  cover custom-server tools (deny augmentation walks the framework's
  own servers only).  The interactive gate is the native `ask` prompt
  on every control verb (arguments visible — also the backstop for the
  acknowledge-warnings residual); the headless gate is
  `hook_config.json`'s `write_tools` (from the profile's `config:`) —
  listing `submit_scan` + `clear_queue` and deliberately NOT
  `stop_scan`, so a halt is never blocked on any path (headless by
  designed omission; interactively because the kill switch does not
  cover custom servers — upstream gap).  Two osprey-side issues to be
  filed from that side: silent unknown-key acceptance, and custom
  servers excluded from the interactive kill switch.  See
  `deploy/DEPLOYMENT.md`.  The submitted-as identity is
  `[mcp] client_identity` (deployment-owned, e.g.
  `osprey-htu-assistant`) and MUST match on the queue item and the
  `SubmissionRecord` — the ownership check compares against it.
- **v2 (built — issue #676)**: `run_action` (Q; idle-only — an active
  RE state refuses, because a mid-scan submission would silently queue
  the action to auto-run when the scan finishes) with `describe_action`
  (R; worker dry-run, needs an idle manager), `move_scan_variable` (Q;
  the worker's `geecs_move_variable`, idle-only + blocking ≤ ~120 s,
  non-finite values refused), `pause_scan` (S — the halt family, never
  in `write_tools`) and `resume_scan` (Q — it restarts motion and
  retries a failed move, so it gates like a submission), both with
  stop's ownership etiquette (`force` only for genuinely foreign scans,
  `forced` marks only those).  `scan_progress` gains the best-effort
  `stream` picture from `scans/progress_stream.py`: start-doc totals
  (`num_points × shots_per_step`, `max_iterations` fallback),
  primary-stream `seq_num` → shots done, stop-doc exit status, and the
  console-text stream's failed-move line as the paused reason
  (surfaced only while actually paused — sticky otherwise).  The
  manager poll stays authoritative; `stream.available=false` names why.
  The consumer threads start lazily on the first `scan_progress` call
  (stdio) but the HTTP entry point warms them at startup (#685) — a
  long-lived service must be consuming before its first start document
  passes, or that run shows no counts.

## Testing

`poetry run pytest` — hermetic: the impls are tested against fakes
patched on `runtime`; no manager, no Tiled, no configs repo.  The
registration test asserts every `tool_names.READ_TOOLS` entry is on the
server.  Live verification rides the phasing checklists in the planning
doc (v0: listings match the console's dropdowns, status agrees with the
console's pill, a known scan number resolves from the archive).

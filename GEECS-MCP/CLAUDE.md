# GEECS-MCP — Developer Context for Claude

The **general GEECS MCP server** — AI-agent access to GEECS, one server
process with domains as subpackages.  Renamed from `GEECS-Scan-MCP`
(owner decision 2026-08-22, before anything reached master): scans are
the first domain, not the identity.  **The spec for the scans domain is
the scan-MCP planning document** (2026-08-21/22; verb surface §1, safety
§2, phasing §4) as amended by owner decisions recorded in the CHANGELOG.

Domain roadmap (add as they earn their keep, never speculatively):
`scans/` (built: read + halt; the write verbs were deleted in 0.9.0),
then candidates in rough value order — `health/` (gateway/Tiled/DB probes, read-only), `db/`
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
- **This server has no write path** (2026-09-16, owner ruling).  The
  native-Bluesky rebuild retired the `qs_client` calls behind
  `submit_scan`, `run_action`, `describe_action` and
  `move_scan_variable` (`submit_scan`, `submit_action`,
  `describe_action`, `move_variable` are all gone; the submission
  surface is `submit_plan`/`submit_preset` over `count`/`sweep`/
  `optimize`), and `run_submit_preflight` now takes a preset rather
  than a `ScanRequest`, which broke `validate_scan_request` too.  All
  five verbs were **deleted** in 0.9.0 rather than rewired: **this
  server is an experiment, not an operator surface**, so it never gates
  a client-seam change — scans are submitted from `GeecsScanner`.  If a
  write path is ever wanted back (#727), the reference implementation
  to copy is `GeecsScanner/geecs_scanner/service/scanner.py` — the
  `build_submission_record` → `md={"geecs": {"submission": ...}}` →
  `submit_plan` shape.  What survives is read + observe + halt.
- **Write-surface doctrine, for whenever that day comes** (2026-07-23,
  amended 2026-08-25 to match deployed practice — owner correction):
  GEECS-*semantic* writes (scans, actions, manual moves, analysis) go
  through MCP verbs only; scans stay in the GEECS engine — the MCP
  submits plans, never drives devices shot-by-shot, and does no raw PV
  I/O of its own.  Channel-level
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
    control_tools.py # stop (ownership), clear_queue, scan_progress,
                  #   pause_scan/resume_scan (ownership like stop).
                  #   NO SUBMIT VERB since 0.9.0 — see the write-surface
                  #   note below
    progress_stream.py # ProgressCache — the best-effort document-stream
                  #   + console-text-stream picture behind scan_progress
                  #   (daemon threads, zmq never touched cross-thread,
                  #   no stop — the console's #653 rules)
  analysis/       # the analysis domain (#675)
    read_tools.py # get_scan_analysis (task statuses from
                  #   analysis_status/ — read through the SHARED
                  #   geecs_data_utils.analysis_status reader (#682),
                  #   never a local parser; ScanAnalysis's suite pins
                  #   its TaskStatus.to_dict() writer against that
                  #   reader — + the output tree, payload-budgeted
                  #   with explicit *_truncated flags) and
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
  completion — everything is request/response; `stop_scan` is the one
  bounded-blocking verb and caps at the client's own ≤120 s budget.
- Result payloads are context-sized: `get_scan_result` returns metadata
  + column names + capped stats, never the full event table.
- Field-tolerant reads of the manager's shapes (`.get` everywhere in
  history mapping) — the queueserver's payload fields are not a contract
  we own.

## Verb surface

**Built and live: read + observe + halt.**  `scan_status`,
`scan_history`, `get_scan_result`, `list_scan_configs`, `scan_progress`
(R, auto-allow); `clear_queue` and `resume_scan` (Q); `stop_scan` and
`pause_scan` (S — the halt family); the analysis domain's
`get_scan_analysis` / `get_scan_figure` / `list_analyzers` /
`list_analysis_groups` (R) and `run_scan_analysis` (Q).

**Deleted in 0.9.0** (see the boundaries section): `submit_scan`,
`run_action`, `describe_action`, `move_scan_variable`,
`validate_scan_request`.  The standing doctrine they carried — the
acknowledge-warnings loop, `clear_pending=False` always, the agent shot
cap — went with them; re-derive it from the planning document if a
submit verb is ever rebuilt, and do not assume the old code was right
about the new seam.

**Ownership etiquette still binds what is left**: `stop_scan`,
`pause_scan` and `resume_scan` compare the running item's submitted-as
identity against `[mcp] client_identity` and refuse a foreign scan
without `force=true`.  `resume_scan` is Q (it restarts motion);
`pause_scan` joins `stop_scan` in the halt family.

**Gating semantics (VERIFIED against osprey 2026-08-22)**:
profile-level custom-server `hooks:` keys are silently ignored, and the
interactive writes kill switch does not cover custom-server tools (deny
augmentation walks the framework's own servers only).  The interactive
gate is the native `ask` prompt on every control verb (arguments
visible); the headless gate is `hook_config.json`'s `write_tools` (from
the profile's `config:`) — list `QUEUE_TOOLS` and deliberately NOT the
halt family, so a halt is never blocked on any path.  Two osprey-side
issues to be filed from that side: silent unknown-key acceptance, and
custom servers excluded from the interactive kill switch.  See
`deploy/DEPLOYMENT.md`.

`scan_progress` carries the best-effort `stream` picture from
`scans/progress_stream.py`: start-doc totals, primary-stream `seq_num`
→ shots done, stop-doc exit status, and the console-text stream's
failed-move line as the paused reason (surfaced only while actually
paused).  The manager poll stays authoritative; `stream.available=false`
names why.  The HTTP entry point warms the consumer threads at startup
(#685) — a long-lived service must be consuming before its first start
document passes.  Stdio starts them lazily on the first `scan_progress`
call (owner scope on #685); dropping the transport gate in
`__main__.main` is the one-line remedy if that bites.

## Testing

`poetry run pytest` — hermetic: the impls are tested against fakes
patched on `runtime`; no manager, no Tiled, no configs repo.  The
registration test asserts every `tool_names.READ_TOOLS` entry is on the
server.  Live verification rides the phasing checklists in the planning
doc (v0: listings match the scanner page's dropdowns, status agrees with
its Now chip, a known scan number resolves from the archive).

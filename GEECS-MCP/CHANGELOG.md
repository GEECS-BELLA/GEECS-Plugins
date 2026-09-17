# Changelog

All notable changes to `geecs-mcp` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.9.0] - 2026-09-16

### Removed

- **Every write verb** — `submit_scan`, `run_action`, `describe_action`,
  `move_scan_variable` and `validate_scan_request`.  The native-Bluesky
  rebuild removed the `geecs_bluesky.qs_client` calls all five stood on
  (`submit_scan`, `submit_action`, `describe_action`, `move_variable` are
  gone; the submission surface is now `submit_plan` / `submit_preset`
  over the `count` / `sweep` / `optimize` plans), and
  `run_submit_preflight` takes a preset rather than a `ScanRequest`.
  The four control verbs would have raised `AttributeError` on their
  first call and `validate_scan_request` would have run the preflight
  against the wrong document.
  They are **deleted rather than rewired** by owner ruling
  (2026-09-16): this server was spun up as an experiment, not an
  operator surface, so it gates no client-seam change.  Scans are
  submitted from the web scanner (`GeecsScanner`).  See #727 if an
  agent-facing write path is ever wanted back — the shape to copy is
  `GeecsScanner/geecs_scanner/service/scanner.py`.
- The doctrine those verbs carried (the acknowledge-warnings loop, the
  1,000-shot agent cap, `clear_pending=False`, idle-only writes) went
  with them, along with `_task_error_kind` and the `GOOD_REQUEST`
  fixture.  `[mcp] max_shots` is now unread.

### Fixed (adversarial review of this PR)

- **A sixth tool was broken by the same rebuild, and this PR had
  re-advertised it.** `list_scan_configs(kind="save_sets")` routed to
  `ConfigsRepoResolver.list_save_sets`, removed alongside the client
  verbs (presets carry the device group). It answered
  `not_found: listing save_sets failed: 'ConfigsRepoResolver' object has
  no attribute 'list_save_sets'` — which an agent reads as *this
  experiment has no save sets*, not *this tool is broken*. `save_sets` is
  dropped from `_CONFIG_KINDS`, from `_FakeResolver`, and from the five
  places that advertised it — including the FastMCP `instructions`
  string, which is the catalog description every connecting agent reads
  first (second review round).
- `test_list_scan_configs_without_experiment` was left **vacuous** by
  that same fix: it passed `"save_sets"`, which now trips the
  kind guard before reaching the `resolver is None` branch it exists to
  test. Proven by deleting the guard and watching it still pass; it now
  passes a valid kind and fails when the guard goes (second review
  round — the fix for one mislabelled refusal had disarmed the test
  protecting against another).
- Two more `docs/sites/data_flow/` bullets in the list already corrected
  still described a submit cap and acknowledge loop, and that map's
  GEECS-Schemas panel still named the MCP among the packages importing
  the models. `overview.md`'s new cross-reference pointed at
  `#the-safety-model` while the bullet it names lives under
  `#where-it-sits-in-the-architecture` — the anchor resolved, so the
  build stayed quiet and the reader landed in the wrong section.
- The error taxonomy's prose no longer describes the deleted submit path
  (`policy_refusal` is ownership/RE-state, not cap/acknowledgement; the
  `needs_acknowledgement` `extra` is gone). `task_timeout` keeps its slot
  in `ERROR_KINDS` with a note that it currently has no producer — it is
  a published envelope value an agent may branch on.
- Root `CLAUDE.md` called `resume_scan` and `clear_queue` "halt verbs"
  and the server "READ-ONLY", contradicting `tool_names.py`, which
  classifies resume as **Q, not S** because it restarts motion. It now
  reads "the halt family (stop/pause) + three gated go verbs".
- Four published surfaces outside `docs/geecs_mcp/` still sold the submit
  path: `docs/agentic/index.md`, both `docs/geecs_schemas/` admonitions
  (which said the MCP "is rewired onto" presets — it is not, it was
  deleted), and the `docs/sites/data_flow/` map, which drew an
  OSPREY→queue submission arrow this arc removes.
- `pyproject.toml`'s published description said "scan submission"; the
  `geecs-schemas` dependency comment claimed a runtime import that does
  not exist (nothing under `geecs_mcp/` imports it — the listing tools
  duck-type catalog rows; it is test-only, now labelled as such).
- The note added to `tool_names.py` had been spliced into the middle of
  the `QUEUE_TOOLS` comment's sentence.

### Added (adversarial review of this PR)

- **`tests/test_seam_pins.py`** — the fake-vs-real assertions, in one
  module instead of beside each double, because the review found the
  drift a *second* time (`_FakeResolver.list_save_sets`) one file over
  from the first. It pins all three doubles' public methods against
  `QueueClient` / `ConfigsRepoResolver`, both fabricated `status()`
  objects against `QueueStatus`'s real fields (the unpinned half: a
  renamed field left the suite green and every reader raising
  `AttributeError` in production), and — closing a pre-existing gap —
  that the safety groups *partition* the registered tools, so a tool can
  no longer ship in neither `allow` nor `ask`/`write_tools`.
  Each assertion was verified to fail against its own drift.
- `test_every_config_kind_maps_to_a_real_resolver_capability` — every
  advertised `list_scan_configs` kind must name a capability the **real**
  resolver has, checked against the class rather than the fake.

**Known gap, waived:** the pins check *names*, not signatures or return
shapes — a `request_pause()` returning a `SubmitResult` instead of
`(ok, message)` would pass and still break the tuple unpack. Pinning
shapes wants a typed conformance helper beside the protocol in
GeecsBluesky, so the scanner's `DemoQueueClient` gets it too; that is a
cross-package change.

### What survives

Read + observe + halt: `scan_status`, `scan_history`, `get_scan_result`,
`list_scan_configs`, `scan_progress`, `stop_scan`, `pause_scan`,
`resume_scan`, `clear_queue`, and the whole analysis domain
(`get_scan_analysis`, `get_scan_figure`, `list_analyzers`,
`list_analysis_groups`, `run_scan_analysis`).  Ownership etiquette on
stop/pause/resume is unchanged.

### Added

- `test_the_fake_client_only_promises_verbs_the_real_client_has` — the
  suite stayed green through this breakage because `_FakeClient` still
  defined all four removed methods, so the tools were tested against a
  seam that no longer existed.  The new test asserts every public method
  on the fake exists on the `QueueClient` protocol; verified to fail
  when a removed verb is put back on the fake.

### Changed

- `deploy/DEPLOYMENT.md` permission lists, the README verb inventory and
  the server's `instructions` string drop the removed tools.  A profile
  deployed before 0.9.0 names tools this server no longer registers —
  inert, but drop those entries.

## [0.8.10] - 2026-09-16

### Fixed

- Document and enforce retirement of legacy optimize ScanRequests while preserving plain scans with a null optimization field. Remove obsolete unbounded-optimization cap guidance. Refreshed locks include the existing CA extra’s PVA dependencies.

## [0.8.9] - 2026-09-14

### Changed

- The shot-cap refusal tells the agent to have an operator run the scan
  "from the scanner page" — GEECS-Console is deleted (web scanner arc PR 6).

## [0.8.8] - 2026-09-10

### Changed

- `validate_scan_request`'s tool description says what the client-side
  preflight checks now that GeecsBluesky 0.79.0 (#807 phase 1) deleted the
  worker-side resolver: save sets resolve, worker ready, device liveness,
  trigger staleness — no engine validation, no unserved-variables check
  until the plan layer.  Text only (Codex review of #816).

## [0.8.7] - 2026-09-05

### Changed

- `poetry.lock` refreshed for the `analysis-run` extra: ScanAnalysis 1.19.0
  / ImageAnalysis 2.0.0 bring `geecs-schemas` as a path dependency (the
  analysis-config schema overhaul). No code change — the runner still goes
  through `load_diagnostic` + `create_scan_analyzer`; the Windows-only
  test stand-in is now the real `haso` analyzer kind.

## [0.8.6] - 2026-09-04

### Changed

- `submit_scan`'s acknowledgeable-check vocabulary
  (`_ACKNOWLEDGEABLE_CHECKS`) gains **`snapshot_images`**, the new
  pre-submit warning GeecsBluesky 0.73.3 raises when a snapshot-role
  save-set entry has `images: true` (#754). Without this the tuple's
  fail-closed drift guard would refuse to acknowledge the new question and
  such a request could never be submitted through the MCP. Pinned by a
  control-tools test.

## [0.8.5] - 2026-09-03

### Fixed

- `poetry.lock` refreshed for ImageAnalysis 1.14.1 (#752): the boto3
  chain (boto3 / botocore / jmespath / s3transfer) leaves the lock.
  matplotlib's resolution is unchanged here — the `analysis-run` extra
  already carried it through ScanAnalysis's main declaration. Lock-file
  refresh only — no code change.

## [0.8.4] - 2026-09-03

### Changed

- `deploy/geecs-mcp.service` is now a **template** rendered from the host's
  `site.env` by `deploy/render_units.sh`; the baked venv lives at
  `<checkout root>/geecs-mcp-venv` (no `/opt`, no sudo) and is installed from
  the worker's `qs-checkout` with the `analysis-run` extra. `DEPLOYMENT.md`
  updated. No runtime change.

## [0.8.3] - 2026-09-01

### Changed

- `get_scan_analysis` reads the `analysis_status/*.yaml` task files
  through the shared `geecs_data_utils.analysis_status` reader
  (GEECS-Data-Utils 0.26.0, #682) instead of its local
  `_read_task_statuses` parser + `_heartbeat_age_s`; the tool's own
  presentation (heartbeat age, the display_files payload cap +
  `display_files_truncated` flag, `unreadable` entries) and its tests
  are unchanged. One deliberate delta: a `.yml` file in `analysis_status/`
  no longer surfaces as a task — the queue's own readers glob `*.yaml`
  only, so it was a phantom the queue would never run (#750 review). The writer/reader contract is now pinned in
  ScanAnalysis's suite, so a `TaskStatus.to_dict()` change fails a test
  there instead of silently drifting from this tool.
## [0.8.2] - 2026-09-01

### Fixed

- `poetry.lock` refreshed for ImageAnalysis 1.13.1 (#739): the
  `analysis-run` extra no longer drags pytest, pluggy and iniconfig
  into the **main** group — this is the one deploy path that actually
  changes, since `deploy/DEPLOYMENT.md` installs the package with a
  non-editable `pip install` (main deps only). Lock-file refresh only
  — no code change.

## [0.8.1] - 2026-09-01

### Fixed

- **First scan after an HTTP-service restart streamed no counts**
  (#685): the `ProgressCache` consumer threads started lazily on the
  first `scan_progress` call, so a long-lived `--transport http` service
  whose first poll came after the run's start document showed
  `stream: {available: true}` with no scan number / shot counts. The
  HTTP entry point now warms the cache at startup
  (`__main__.warm_progress_stream`) from the queue client's own
  `doc_addr` / `info_addr` — the same resolution `scan_progress` uses,
  now shared as `progress_stream.start_for_client`. Best-effort: a
  client that will not build or a missing address is logged and the
  server still comes up. Stdio sessions keep the lazy start.

## [0.8.0] - 2026-09-01

### Changed

- **ScanRequest v2 / request-record split adoption** (geecs-schemas
  0.14.0, GeecsBluesky 0.70.0): `submit_scan` builds the provenance
  record with `build_submission_record` and passes it beside the request
  (`client.submit_scan(request, submission=...)`) instead of stamping it
  into the document. Agent-composed flat v1 request dicts keep
  validating via the schema's lifting validator; the worker must be at
  GeecsBluesky ≥ 0.70.0 for the `submission` kwarg.

## [0.7.2] - 2026-08-25

### Changed

- Docs-only: the write-surface doctrine statement in `CLAUDE.md` amended
  to match deployed practice (owner correction on docs PR #692): osprey's
  own EPICS write tool sets gateway `:SP` PVs directly, bounded by its
  limits database — "raw gateway PVs are read-only to the agent" was
  stale.  The surviving rule: GEECS-semantic writes are MCP verbs only,
  and the MCP does no raw PV I/O of its own.

## [0.7.1] - 2026-08-25

### Fixed

- `poetry.lock` regenerated from the main checkout: the geecs-bluesky
  extras lines referenced a `.claude/worktrees/` session worktree (the
  lock had been generated from inside one — 0.7.0's relock included),
  leaving absolute paths that dangle once the worktree is removed.
  Lock-file refresh only — no code change.

## [0.7.0] - 2026-08-24

### Added

- **The analysis-domain execution slice** (issue #686, owner request —
  restores the post-migration-dormant analysis capability through the
  new architecture; ScanAnalysis as-is is the backend by owner decision
  2026-08-24, with the verb surface kept backend-neutral and
  `analysis_status/` as the progress contract so the Tiled-based stack
  can slot in behind the same tools later):
  - `run_scan_analysis(scan_number, day?, analyzer|group, rerun_failed,
    rerun_completed)` — Q-class (native `ask` + `write_tools`),
    submit-and-poll: validates everything refusable *before* side
    effects (exactly-one selector, configs root, the scan folder EXISTS
    — the cross-package invariant, pinned by a nothing-created test —
    and the analyzers construct on this host, so a Windows-only-SDK
    diagnostic is refused up front instead of half-running), initializes
    the status YAMLs server-side (a dead worker leaves visible queued
    rows, never silence), then spawns a detached worker subprocess
    (`analysis/run_worker.py`) driving ScanAnalysis's own task queue
    (claim/heartbeat/stale-reclaim).  Poll with the existing
    `get_scan_analysis`; figures via `get_scan_figure`.  Google-Doc
    upload stays hard-off.
  - `list_analyzers` / `list_analysis_groups` — R-class listings over
    the configs repo's `analyzers/`/`groups/` trees
    (`discover_analyzers`/`discover_groups`), so agents name
    diagnostics instead of guessing (payload-capped, truncation
    flagged).
  - New optional `analysis-run` extra (ScanAnalysis path dep — heavy
    via ImageAnalysis, hence an extra mirroring GeecsBluesky's
    pattern); without it the three tools refuse naming the extra and
    the server starts normally.  New consumed config:
    `SCAN_ANALYSIS_CONFIG_DIR` / `[Paths] scan_analysis_configs_path`
    (see `deploy/DEPLOYMENT.md`).
  - **The pre-claim double-start window is closed** (Codex P1): the
    realistic vector — a second call into this server while the first
    worker is still importing its stack — refuses via an in-process
    dispatch ledger (side-effect-free, before any status write; entries
    expire when the pid dies, the tasks leave `queued`, or after the
    queue's own 180 s staleness bound), and the cross-process backstop
    is ScanAnalysis 1.16.0's atomic claim gate in `run_worklist` (a
    losing worker skips the task instead of double-running it).
  - Review-hardened status semantics (adversarial findings 1–2): the
    rerun flags reset `failed`/`done` (and stale-claimed) rows to
    queued *server-side* (`reset_status_for_scan`), so the dead-worker
    visibility contract holds on every path; a task another runner is
    actively working (fresh heartbeat — ScanAnalysis 1.16.0's
    `claim_is_active`) refuses the call (`policy_refusal`) instead of
    double-running into the same output files; the envelope reports
    `tasks` (what this call runs) vs `skipped` (done/failed without
    their rerun flag), and an all-skipped call is an honest
    `started: false` no-op with no worker spawned.  Task ids come from
    ScanAnalysis's exported `analyzer_task_id` (no mirrored
    derivation), and the server-side validation builds analyzers
    through the same `run_worker.build_analyzers` the worker executes.

## [0.6.0] - 2026-08-24

Payload discipline for figures (osprey-side integration finding: an
inline 247 KB PNG forced context compaction on the deployment's
haiku-tier agent — image bytes never ride model context by default
again).

### Changed

- **`get_scan_figure` returns a REFERENCE by default**: figure label,
  pixel dimensions (header read, no decode), byte size, a
  share-relative path (POSIX, relative to the GEECS data root — never a
  `Z:\` or `/mnt/` primary handle), and `figure_url`.  Candidate and
  ambiguity listings carry the same reference entries (label, bytes,
  URL) instead of bare names.  `thumbnail=true` opts into bounded inline
  image content: ≤768 px longest edge, JPEG q80 — the only path that
  decodes, still behind the 64 MP decode cap.

### Added

- **`GET /figures/{day}/{scan_number}/{label}`** on the same HTTP
  server: streams the ORIGINAL figure bytes (bypasses model context, so
  no downscale) for clients to fetch and save as local artifacts.
  Bounded by exactly the tool's candidate set (the scan's own analysis
  folder, exact label match) plus a 50 MB byte cap; never raises (plain
  4xx/5xx text).  The tool's `figure_url` is SERVER-RELATIVE by design —
  clients resolve it against the MCP base URL they already hold, so no
  advertised host is baked into results and service re-homing stays a
  client-config-only change.  Unreachable over stdio (no HTTP app), by
  construction.
- **Payload budgets on `get_scan_analysis`** (the audit ask): one
  task's `display_files` listing caps at 20 and the outputs listing at
  40 device dirs, each with an explicit `*_truncated` flag — truncation
  is never silent.

## [0.5.1] - 2026-08-24

First-deployment live findings (qserver-box HTTP service, real netapp
tree — both invisible to the hermetic fixtures):

### Fixed

- **Windows-written `display_files` entries localize instead of
  crashing**: production statuses come from the Windows analysis
  machines (`Z:\data\...\analysis\Scan<NNN>\...`); on the Linux
  service host such an entry is not absolute, was joined onto the
  analysis folder as one giant backslash component, and the stat's
  `OSError` killed the whole figure tool — including the healthy
  tree-scan fallback.  Now: a Windows-style entry re-roots by its tail
  after `analysis\Scan<NNN>\` onto the local analysis folder (served
  like any candidate), an unmatchable one is skipped with a warning,
  and every per-candidate filesystem touch is guarded so one bad entry
  can never take down the tool.
- **`get_scan_analysis` outputs walk the nested analyzer tree**: the
  production layout is `Scan<NNN>/<device>/<Analyzer>/files`, and the
  one-level listing read every device as `n_files: 0`.  Files are
  counted through the whole device subtree (names relative, listing
  still capped).

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

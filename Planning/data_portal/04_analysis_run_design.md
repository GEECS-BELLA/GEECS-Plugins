# Analysis runs from the portal — design

*The design note for the `feat/analysis-tab` arc (owner discussion,
2026-09-01; adversarial review of the first draft folded in — see
"What the review changed"). It amends the portal's read-only charter:
the portal may trigger a ScanAnalysis run on the scan the user is
looking at, and serve what that run produced. Everything below is
grounded in a code-level recon of the current implementations;
statuses follow the `03_analysis_tabs_design.md` vocabulary (exists /
extend / new).*

## What the owner ruled (2026-09-01)

1. **Single-shot view stays ephemeral ImageAnalysis** (exists — W2a's
   `run_diagnostic_ephemeral`). Scan-wide views use the **real
   ScanAnalysis pipeline**: the wrapper analyzers, their figures, their
   s-file columns.
2. **"Just run it."** The portal instantiates the scan analyzer and
   calls `run_analysis` directly, without taking part in the file-based
   task queue (`scan_analysis.task_queue`) — the owner's stated reason:
   "no worry about multiple runners", and the queue + live runner
   "deserve a complete audit" (the file-based protocol was a
   placeholder for a proper service + queue, to be designed later).
   **Re-affirmed after review (same day)** with the MCP runner on the
   table — see "The runner decision" below.
3. **No Google Doc uploads** from the portal.
4. **s-file collisions: warn and overwrite** — ScanAnalysis's existing
   protocol (`ScanAnalyzer.append_to_sfile`, "columns already exist in
   s-file … (will overwrite)"). The owner regards s-files as
   regenerable derived copies, not sacrosanct records: "people write to
   sfiles constantly, delete them and regenerate."
5. **Rendered-view toggle** on the Images tab: a third display mode
   (raw / processed / *rendered*) that draws the analyzer's own
   matplotlib figure (`render_image` + `render_data` overlays) as the
   served PNG. Still static; a stop-gap for the missing
   matplotlib-style interactivity, judged higher value per cost than a
   plotly heatmap.
6. **Deferred, not lost**: failure-log propagation for the ephemeral
   Images path (the run record subsumes it); a web config editor over
   ScanAnalysis's `ConfigFileGUI` load/save layer (its own arc); the
   live-runner/task-queue audit; the owner's further Images-tab
   comments ("save other comments for later").

## What the review changed (2026-09-01, PR #760)

The adversarial review of the first draft surfaced facts the owner
discussion did not have on the table. Recorded here because they bound
the design, not to relitigate the rulings:

- **A second on-demand runner already exists on the same host.**
  `GEECS-MCP/geecs_mcp/analysis/run_tools.py` (`run_scan_analysis`,
  #686, MCP 0.7.0) runs one diagnostic or group on one scan from the
  worker host: it validates, calls `init_status_for_scan`, spawns a
  detached `run_worker` that builds the worklist and runs it through
  `run_worklist` (claims, heartbeats, `done`/`failed`/`no_data`
  narration into `analysis_status/`, gdoc hard-off), and its read
  tools poll those files and serve figures with an analysis-folder
  containment check. That was built on the owner's **2026-08-24
  decision** that ScanAnalysis-as-is is the backend and the
  `analysis_status/` YAMLs are the backend-neutral progress contract
  every runner reports into (`GEECS-MCP/CLAUDE.md`,
  `ScanAnalysis/CLAUDE.md` "analysis status contract"). Its shared
  builder is `run_worker.build_analyzers` (= `load_diagnostic` +
  `create_scan_analyzer`, exactly the two calls below).
- **Runs write inside `scans/ScanNNN/` too.** The scan pipeline passes
  `auxiliary_data["file_path"]` (the gate the ephemeral seam refuses),
  so MagSpec / LineStitcher / HASO / Grenouille write their derived
  per-shot outputs into `<scan_dir>/<output_name>-interp/`-style
  subfolders (`ImageAnalysis/CLAUDE.md` "Filesystem invariants for
  analyzers that write inside `scans/ScanNNN/`"). The scan folder
  itself is never created (`ScanPaths(read_mode=True)`) — the
  invariant holds — but "the portal's only writes are the analysis
  folder and the s-file" was false. A run writes wherever ScanAnalysis
  writes.
- **The deployed share is documented read-only.** `DEPLOYMENT.md`
  ("mount the share read-only") and `01_data_portal_scope.md` ruling 2
  both prescribe a read-only mount. Analysis runs need a **read-write
  mount on the portal host**, or every run fails — or worse, partially
  succeeds: MagSpec's writer logs "Failed to save calibrated outputs"
  and continues, so a run over a read-only share can finish `done`
  with silently missing outputs. Deployment step, owed at promotion.
- **Read-only was the no-auth argument.** Scope ruling 2: read-only
  "is what makes 'anyone on the network' safe with no auth story". A
  run verb is an unauthenticated POST that overwrites s-file columns
  and writes into the share, reachable from any lab-network browser
  and through the OSPREY reverse-proxy mount. Flagged for the owner;
  the working assumption (same standing as the MCP verb, which is
  likewise unauthenticated on the lab network) is that this is
  accepted — the outputs are regenerable and the share is internal.
- **Skips are a state.** `run_analysis` returns `None` (a warning, not
  an error) when the s-file / ini is missing or the scan parameter is
  absent, and the wrapper raises `DataUnavailableWarning` when the
  device folder is missing or empty — `run_worklist` maps that to
  `no_data`. A running / done / failed vocabulary would turn a missing
  s-file into `done` with zero artifacts and a device-less click into
  `failed`. The record vocabulary is **queued / running / done /
  failed / no_data**, matching the status contract either way.
- **pyplot on the portal's threads.** ScanAnalysis's renderers and
  `StandardAnalyzer.render_image` (via `base_render_image`) use
  `plt.subplots`; the portal rule is "`Figure`, never pyplot" because
  pyplot's global registry is not thread-safe. Two consequences: the
  process pins `matplotlib.use("Agg")` before any ScanAnalysis import
  (in `__main__`), and pyplot users are serialised — one analysis
  worker thread, and the rendered view either goes through a
  `Figure`-object renderer or takes the same lock.
- **Factual fixes.** `render_image` is a `@staticmethod` on
  `StandardAnalyzer` (overridden by Beam / MagSpec / stitcher /
  HiResMagCam) and an instance method on `Standard1DAnalyzer`, not a
  base-class hook; the class is reachable from
  `diag.image_analyzer.class_path`, so reaching the renderer from the
  portal is a *choice* of seam, not a necessity. Applicability is
  `scan.device or diag.name` present among the scan's device folders
  (`data_device_name or device_name` in the wrapper; the image config's
  camera name is not consulted). `load_diagnostic` caches nothing —
  its discovery rglobs the tree per call; the mtime+size fingerprint
  is the portal's own selector cache. `_merge_auxiliary_data` uses
  `combine_first`, so a re-run that yields NaN for a shot keeps the
  old value — "re-run overwrites cleanly" must be checked per cell.
  Root `CLAUDE.md`'s dependency graph must gain ScanAnalysis for the
  portal; `DEPLOYMENT.md` must name the extra; ScanAnalysis's pyproject
  makes LogMaker4GoogleDocs a hard install dep (optional at runtime
  only), so the install closure pulls the Google libs.

## Recon findings that shape the plan

- **`ScanAnalyzer.run_analysis` is queue-free.** It resolves paths
  (`ScanPaths(read_mode=True)`), reads the ini, loads the s-file, runs
  the subclass core, and returns the artifact list. Status files,
  claim locks, heartbeats, `cleanup()` orchestration and the gdoc
  upload all live in `task_queue.run_worklist`. `cleanup()` is the
  caller's duty when calling it directly.
- **One analyzer by ID is two existing calls**:
  `image_analysis.config.load_diagnostic(id, config_dir=…)` →
  `scan_analysis.config.create_scan_analyzer(diag)`. MCP's
  `run_worker.build_analyzers` is this plus the group form. **Two
  clients now want it: it belongs in ScanAnalysis next to
  `create_scan_analyzer`** (a `build_analyzers(analyzer | group,
  config_dir)` helper both import), not copied a third time.
- **The config tree is already wired.** `--processing-configs` is
  "the parent of `analyzers/`" — the same unified tree ScanAnalysis
  and MCP read (`image:` + `scan:` sections in one YAML). YAML edits
  are live on the next run (nothing caches parsed content); analyzer
  *code* edits need a restart. **No new flag.**
- **Outputs land where the portal already reads.** Figures go under
  `analysis/ScanNNN/<output_name>/<WrapperClass>/`; columns go to
  `analysis/s{N}.txt` via `append_to_sfile` (lock + merge on
  Shotnumber). The Plot tab's frame is rebuilt per request by
  `scan_frame` with the s-file unioned in under provenance `"sfile"`,
  and union responses are never immutable-cached (0.15.3). New columns
  therefore appear on the next Plot-tab load with **no portal change**.
- **Nothing derived survives a request today.** The ephemeral path
  runs the analyzer on every image request (no processed-result cache;
  the pixel cache holds raw frames only) and discards `scalars`. That
  is why config edits show up immediately, and why the binned view
  re-processes every member on each display change. Scan-wide derived
  scalars are the ScanAnalysis path's job, not a second cache.
- **Status reading is already a portal dependency.**
  `geecs_data_utils.analysis_status.read_analysis_statuses(scan_folder)`
  is the tolerant, read-only reader of `analysis_status/` (the #682
  envelope MCP's read tools use); `AnalysisStatus` carries `state`,
  `error`, `display_files`, timestamps. The portal can render run
  state from disk with no new dependency — including runs the MCP
  verb started, and runs from before a portal restart.
- **Dependency.** ScanAnalysis depends on ImageAnalysis, Data-Utils and
  LogMaker4GoogleDocs. Adding it to the portal's `analysis` extra
  pulls PyQt5 only on win32 (marker-gated), plus watchdog / h5py /
  requests and the Google client libs — acceptable for the worker-host
  deployment, where MCP's `analysis-run` extra already installs the
  same closure.

## The runner decision (RULED 2026-09-01: A)

Two runners on one host for one scan is the actual situation, not a
hypothetical. The options put to the owner:

- **A. Portal-private (the 2026-09-01 ruling as spoken).** A thread
  calls `build → run_analysis → cleanup`; the job record (state,
  artifacts, error, captured log) lives in portal memory. Simplest to
  read; no coupling to `task_queue`. Costs: no coordination with the
  MCP runner — an OSPREY agent's `run_scan_analysis` and a browser
  click on the same scan write the same figure tree and HDF5
  concurrently (only the s-file has a lock); MCP's `get_scan_analysis`
  never sees portal runs; records vanish on restart (artifacts on disk
  survive, errors do not).
- **B. Share the MCP shape (the 2026-08-24 contract).** The portal
  thread runs `init_status_for_scan` → `reset_status_for_scan` (for a
  re-run) → `build_worklist` → `run_worklist(gdoc_enabled=False)` for
  the one analyzer — MCP's `run_worker.main` in-process — and the tab
  reads state from `read_analysis_statuses`. Claims coordinate the two
  runners (a losing runner skips, never double-runs); persistence,
  error text, `no_data` and `display_files` come free; the portal
  becomes a second *consumer* of the seam the queue audit will replace,
  and migrates with MCP when it does. Costs: the status protocol's
  warts (stale-claim windows, a share write per state change) reach
  portal users; the portal imports `task_queue`; the captured log
  needs its own home (the status record has `error` only).
- **C. A + the claim gate only** (`try_acquire_claim` /
  `release_claim`): coordination without narration. Removes the
  double-run, keeps MCP blind to portal runs and keeps the private
  record. Half of B for most of B's coupling.

The agent recommended B (most reuse, honours the 08-24 contract, one
seam to migrate at the queue audit). **The owner ruled A**: "geecs-mcp
is fairly experimental. We probably over-specified this on 8/24.
Someday we might try to use MCP to run scan analysis, but I don't know
when. We will use the 'just run it' option in the data portal close to
day 1." So: the portal is the day-1 consumer, the status-file contract
is not extended to it, and the MCP runner's coordination is a future
concern to revisit when MCP runs are real. Consequences accepted with
the ruling: no coordination between portal and MCP runs on the same
scan; MCP's `get_scan_analysis` does not see portal runs; portal
records live in memory. The shared-builder placement finding is waived
for now (MCP experimental; the portal's factory is two calls) — it
becomes real the day MCP's runner is adopted. The write stance
(read-write mount at promotion, unauthenticated run verb on the lab
network, regenerable outputs) was accepted in the same ruling.

## The run model

- **Endpoints.** `GET /api/run/{uid}/analysis` lists the loadable
  diagnostics with applicability, the in-memory job record, and the
  files under each one's output directory (so a page loaded after a
  portal restart still shows what an earlier run produced).
  `POST /api/run/{uid}/analysis?analyzer=<id>` starts a run — 202 with
  the record; 404 when the feature is off or the extra missing (the
  existing ladder), the diagnostic unknown, or the folder unresolvable;
  409 when a job is already running for the scan (one per scan). `GET /run/{uid}/artifact?path=…` serves one file from
  the scan's analysis folder — the portal has no generic file endpoint
  (images are decoded and re-encoded), so this one carries its own
  containment check: the resolved path must stay inside the resolved
  analysis folder (MCP's `_gather_figure_candidates` is the precedent).
- **Execution.** One worker thread (pyplot serialisation, above);
  build + run happen on it, so config and instantiation failures land
  in the record as `failed`, never as a 500. `cleanup()` in `finally`.
  `run_analysis` returning `None` and `DataUnavailableWarning` both
  map to `no_data` (the worklist runner's own mapping). The record
  keeps the run's log lines, captured from the root logger filtered
  by the worker thread's id.
- **Applicability.** `scan.device or diag.name` among the scan's
  device folders. Every loadable diagnostic is listed; the tab
  collapses the inapplicable ones so a device-less diagnostic is still
  reachable.
- **Re-run** is the same POST; on disk it overwrites (figures by name,
  columns per the warn-and-overwrite protocol, subject to the
  `combine_first` NaN caveat).

## The Analysis tab (new)

The scan page grows an **Analysis** tab: the applicable analyzers,
each with state, a run / re-run button, error text when failed, and
the artifacts rendered inline when done (images inline, other files as
links). The page polls the list endpoint while any run is active. The
Plot tab needs nothing (see recon).

## The rendered-view toggle (extend)

- Seam: either the ephemeral seam gains a render form (results +
  figures, one analyzer instantiation per batch, same write-free
  contract and denylist) or the portal reaches the analyzer class's
  `render_image` directly. Decide at build time; the pyplot constraint
  above applies either way — prefer a `Figure`-object path.
- Portal: `display.mode = "rendered"` on the image endpoints; the PNG
  is the rendered figure. Per-bin view renders the *averaged processed
  image* through the base renderer with per-shot overlays dropped
  (they do not average meaningfully).

## Charter amendment

`GEECS-DataPortal/CLAUDE.md`'s "Read-only by doctrine" becomes
"Read-only, **except explicit analysis runs**": the portal never
creates `scans/ScanNNN/` (invariant unchanged), and its only writes are
what a ScanAnalysis run writes on the user's click — figures and
HDF5 under the analysis folder, s-file columns, and (for the analyzers
that do so) derived subfolders inside the scan folder. No annotations,
no config writes (the config editor arc will revisit that line).
`01_data_portal_scope.md` ruling 2 and `DEPLOYMENT.md`'s read-only
mount guidance get the same amendment in the PR that lands the run
backend: the mount must be read-write where runs are enabled, and the
no-auth stance is restated with the run verb in view.

## Wave plan → PRs (each into `feat/analysis-tab`, /land ritual)

1. **`portal/analysis-run-design`** — this document.
2. **`portal/analysis-run-backend`** — the endpoints, worker thread,
   job record, artifact serving, Agg pin; ScanAnalysis added to the
   `analysis` extra; charter amendments (CLAUDE.md, scope doc,
   DEPLOYMENT.md, root dependency graph). Tests against a fake
   analyzer (no share, no hardware).
3. **`portal/analysis-tab`** — the tab: listing, run, poll, artifacts,
   errors.
4. **`portal/rendered-view`** — the seam + the third display mode
   (#766: `render_diagnostic_ephemeral` composing
   `tools.rendering.render_result_figure`; `display.mode`).
5. **Promotion PR → master** (maintainer merges), then deploy (extra +
   read-write mount) and the live check below.

## Hardware / live verification (owed at promotion)

From a browser against a real completed scan: run one 2D analyzer →
its figures render in the Analysis tab; the Plot tab shows the new
`"sfile"`-provenance columns; re-run overwrites (warn logged, no
duplicate columns, spot-check a cell); a deliberately broken YAML shows
as `failed` with the exception text; a device-less diagnostic shows
`no_data`; run a 2D analyzer *while* browsing images in another tab
(the process-pool fork from a thread-rich process — #763 review
finding 3); `systemctl restart` during a run waits, then serves; the
rendered toggle shows the analyzer's overlays on the per-shot image.

## Open questions (carried, not blocking)

- Whether the job record should persist (a JSON sidecar in the
  analysis folder) so a reload after a portal restart still shows the
  last error and log. Today: files persist, records do not.
- Portal ⇄ MCP run coordination, the day MCP's runner is adopted for
  real (the claim gate, or the status contract — see B/C above).
- Per-bin rendering with overlays that *do* average (projections):
  needs a per-analyzer "aggregate render" hook; not designed.
- ~~Where the rendered toggle's figure size / dpi live in the display
  state vocabulary~~ — answered by PR 4 (#766): they are the seam's
  defaults (5.0×4.2 in @ 110 dpi), not display state; add knobs only
  when someone asks.

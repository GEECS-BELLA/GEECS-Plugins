# Analysis runs from the portal — design

*The design note for the `feat/analysis-tab` arc (owner discussion,
2026-09-01). It amends the portal's read-only charter: the portal may
now trigger a ScanAnalysis run on a scan the user is looking at, and
serve what that run produced. Everything below is grounded in a
code-level recon of the current implementations; statuses follow the
`03_analysis_tabs_design.md` vocabulary (exists / extend / new).*

## What the owner ruled

1. **Single-shot view stays ephemeral ImageAnalysis** (exists — W2a's
   `run_diagnostic_ephemeral`). Scan-wide views use the **real
   ScanAnalysis pipeline**: the wrapper analyzers, their figures, their
   s-file columns.
2. **"Just run it."** The portal instantiates the scan analyzer and
   calls `run_analysis` directly. It does **not** take part in the
   file-based task queue (`scan_analysis.task_queue`): no status
   records, no claim locks, no heartbeats, no worklist. The queue and
   the live runner are a separate concern that "deserves a complete
   audit" — the file-based protocol was a placeholder for a proper
   service + queue, to be designed later. Out of scope here.
3. **No Google Doc uploads** from the portal. (They live in the
   worklist runner, which we bypass — nothing to switch off.)
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
   Images path (the analysis run's job record subsumes it); a web
   config editor over ScanAnalysis's `ConfigFileGUI` load/save layer
   (its own arc); the live-runner/task-queue audit; the owner's further
   Images-tab comments ("save other comments for later").

## Recon findings that shape the plan

- **`ScanAnalyzer.run_analysis` is queue-free.** It resolves paths
  (`ScanPaths(read_mode=True)` — the read-only constructor, so the
  scan-folder invariant holds), reads the ini, loads the s-file, runs
  the subclass core, and returns the artifact list. Status files,
  claim locks, heartbeats, `cleanup()` orchestration and the gdoc
  upload all live in `task_queue.run_worklist`. Calling the analyzer
  directly touches none of them. `cleanup()` is the caller's duty.
- **One analyzer by ID is two existing calls**:
  `image_analysis.config.load_diagnostic(id, config_dir=…)` (the
  unified diagnostic YAML) → `scan_analysis.config.diagnostic_factory
  .create_scan_analyzer(diag)`. The group loader is exactly this in a
  loop. **No new ScanAnalysis helper is needed.**
- **The config tree is already wired.** `--processing-configs` is
  "the parent of `analyzers/`" — the same unified tree ScanAnalysis
  reads (`image:` + `scan:` sections in one YAML). The Images tab's
  selector fingerprints that tree (mtime+size) and re-validates on
  change; `load_diagnostic` caches resolved *paths* only, never parsed
  content. So YAML edits are live on the next run with no extra work;
  analyzer *code* edits still need a restart. **No new flag.**
- **Outputs land where the portal already reads.** Figures go to the
  analysis folder (`ScanPaths.get_analysis_folder()`); columns go to
  `analysis/s{N}.txt` via `append_to_sfile` (lock + merge on
  Shotnumber). The Plot tab's frame is rebuilt per request by
  `scan_frame` with the s-file unioned in under provenance `"sfile"`,
  and union responses are already never immutable-cached (0.15.3). New
  columns therefore appear on the next Plot-tab load with **no portal
  change**.
- **Nothing derived survives a request today.** The ephemeral path
  runs the analyzer on every image request (no processed-result cache;
  the pixel cache holds raw frames only) and discards `scalars`. That
  is why config edits show up immediately, and also why the binned
  view re-processes every member on each display change. Scan-wide
  derived scalars are the ScanAnalysis path's job, not a second cache.
- **`render_image` is the rendered-view hook.** `ImageAnalyzer
  .render_image(result, vmin, vmax, cmap, figsize, dpi, ax)` returns a
  matplotlib figure; subclasses (Standard, Standard1D, MagSpec) draw
  their overlays there from `result.render_data`. The ephemeral seam
  returns result objects only, so the portal cannot reach the renderer
  — that is the one ImageAnalysis change in this arc. (Closes 03's
  open question "`render_function`-style rich rendering … revisit if a
  tab ever wants analyzer-drawn overlays" — it does.)
- **Applicability.** A diagnostic names its device in
  `ScanRuntimeConfig.device` (optional; the image config carries the
  camera name otherwise). The scan page knows the devices present in
  the scan (the rail's device list). Offer the analyzers whose device
  is present; list the rest collapsed as "other analyzers" so a
  device-less diagnostic is still reachable.
- **Dependency.** ScanAnalysis depends on ImageAnalysis, Data-Utils and
  LogMaker4GoogleDocs (optional everywhere — missing it is a silent
  skip, and the portal never reaches the upload path anyway). Adding
  ScanAnalysis to the portal's `analysis` extra pulls PyQt5 only on
  win32 (marker-gated) and watchdog/h5py/requests generally —
  acceptable for the worker-host deployment.

## The run model (new)

- **Endpoint** `POST /api/run/{uid}/analysis?analyzer=<id>` starts a
  job; `GET /api/run/{uid}/analysis` lists the applicable analyzers
  with each one's job state. Both 404 when `--processing-configs` is
  unset or the `analysis` extra is missing (the existing ladder).
- **Job record**: portal-private, in-memory, keyed `(uid, analyzer_id)`
  — `state` (running / done / failed), `started`, `finished`,
  `artifacts` (paths returned by `run_analysis`, relative to the
  analysis folder), `error` (exception text), `log` (the logging
  records emitted during the run, captured by a handler attached for
  the job's duration). Lost on restart; the outputs on disk are the
  durable part, and a fresh page load lists artifacts from disk.
- **Execution**: a bounded background thread (one worker); **one job
  per scan at a time** — a second POST while one is running returns
  409 with the running job. The thread runs `run_analysis`, then
  `cleanup()` in `finally`. Analyzer exceptions become `failed` +
  text, never a 500.
- **Serving artifacts**: a new `GET /run/{uid}/artifact?path=…`
  serves a file from the scan's analysis folder. The portal has no
  generic file-serving endpoint today (images are decoded and
  re-encoded as PNG), so this one carries its own containment check:
  the resolved path must stay inside the analysis folder (the
  device-name allowlist in `resources.load_shot_array` is the
  precedent for refusing traversal at the boundary).
- **Re-run** is the same POST; on disk it overwrites (figures by name,
  columns per the warn-and-overwrite protocol).

## The Analysis tab (new)

The scan page grows an **Analysis** tab: the applicable analyzers,
each with state, a run / re-run button, error text + captured log when
failed, and the artifacts rendered inline when done (images inline,
other files as links). The page polls the list endpoint while any job
is running. The Plot tab needs nothing (see recon).

## The rendered-view toggle (extend)

- ImageAnalysis: the ephemeral seam gains a way to render — either
  `run_diagnostic_ephemeral(…, render=True)` returning figures
  alongside results, or a sibling `render_diagnostic_ephemeral`. Same
  write-free contract and denylist. Decide at build time; keep one
  analyzer instantiation per batch.
- Portal: `display.mode = "rendered"` on the image endpoints; the PNG
  is the matplotlib figure (Agg backend, `dpi` from the display
  state). Per-bin view renders the *averaged processed image* through
  the base renderer with per-shot overlays dropped (they do not
  average meaningfully).

## Charter amendment

`GEECS-DataPortal/CLAUDE.md`'s "Read-only by doctrine" becomes
"Read-only, **except explicit analysis runs**": the portal never
creates anything under `scans/ScanNNN/` (invariant unchanged — the
analyzer's own `ScanPaths(read_mode=True)` enforces it), and its only
writes are what a ScanAnalysis run writes on the user's click: figures
in the analysis folder and s-file columns. No annotations, no config
writes (the config editor arc will revisit that line again).
`01_data_portal_scope.md`'s read-only line gets the same footnote in
the PR that lands the run backend.

## Wave plan → PRs (each into `feat/analysis-tab`, /land ritual)

1. **`portal/analysis-run-design`** — this document.
2. **`portal/analysis-run-backend`** — the run endpoints, job record,
   worker thread, log capture, artifact serving; ScanAnalysis added to
   the `analysis` extra; charter amendment in CLAUDE.md + scope doc.
   Tests against a fake analyzer (no share, no hardware).
3. **`portal/analysis-tab`** — the tab: listing, run, poll, artifacts,
   errors.
4. **`portal/rendered-view`** — the ImageAnalysis seam extension + the
   third display mode (one concern, two packages, one PR).
5. **Promotion PR → master** (maintainer merges), then deploy to the
   worker-host portal checkout and the live check below.

## Hardware / live verification (owed at promotion)

From a browser against a real completed scan: run one 2D analyzer →
its figures render in the Analysis tab; the Plot tab shows the new
`"sfile"`-provenance columns; re-run overwrites cleanly (warn logged,
no duplicate columns); a deliberately broken YAML shows as `failed`
with the exception text and log; the rendered toggle shows the
analyzer's overlays on the per-shot image.

## Open questions (carried, not blocking)

- Whether the job record should persist (e.g. a JSON sidecar in the
  analysis folder) so a page reload after a portal restart still shows
  the last error. Today: artifacts persist, errors do not.
- Per-bin rendering with overlays that *do* average (projections):
  needs a per-analyzer "aggregate render" hook; not designed.
- Where the rendered toggle's figure size / dpi live in the display
  state vocabulary (`analysis.py::_DISPLAY_FIELDS`).

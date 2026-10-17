# ScanAnalysis — Developer Context for Claude

Post-scan analysis framework. Explicit portal and Python runs execute configurable
image/1D analyzers. Automatic watching and Google Docs uploads are retired.

## Package Layout

```
scan_analysis/
  base.py                          # ScanAnalyzer abstract base class (the contract CoreScanAnalyzer and the .himg jobs honour)
  core_inputs.py                   # core compilation + loaded file-background and table bindings + services
  core_services.py                 # the services a measure names (the FROG retriever, the WaveKit engine), built from config.ini + the scan's stack
  core_backgrounds.py              # scan backgrounds: exact strip-wise per-pixel statistics over a scan's frames, cached
  core_source.py                   # completed-scan native/stack input mapping and reads
  core_scan.py                     # write-free scan preparation, grouping and execution
  core_products.py                 # write-free average/bin and summary product planning (ProductCollector streams it)
  core_workers.py                  # the worker count a run gets: recipe request, host cap (config.ini), small-run floor
  core_sink.py                     # legacy-named HDF5/PNG product writes under analysis/ScanNNN (+ parse_output_filename, their inverse); draw_product / draw_summary; per-shot sidecar tables beside the raw file; sibling outputs (magspec -interp/-interpSpec) beside the device folder; ShotStore (a measure's per-scan HDF5 of every shot's frame + extras)
  core_preview.py                  # the editor's previews through the run's own calls (frame; a summary's layout over a few shots)
  core_analyzer.py                 # CoreScanAnalyzer: the one analysis route behind the ScanAnalyzer contract
  route_compare.py                 # snapshot + compare two analysis trees (one equality rule); goldens on disk
  task_queue.py                    # Task claiming, heartbeat, YAML status system
  config/
    loader.py                      # load_diagnostic(stem | path): one document, the config.ini root as the default tree
    diagnostic_factory.py          # create_scan_analyzer(AnalysisRecipe | HimgJob)
    analysis_group_loader.py       # discover_analyzers/groups + load_analysis_group,
                                   #   ResolvedDiagnosticConfig (the models: geecs_schemas.analysis)
  services/                        # the vendor launchers core_services builds (moved from the retired ImageAnalysis):
                                   #   frog_dll_retrieval.py + _frog_dll_worker.py (FrogDllRetrieval, Kane's 32-bit FROG.dll),
                                   #   haso_wavekit.py + _wavekit_worker.py (HasoWaveKit, WaveKit), wavekit_doctor.py (geecs-wavekit-doctor)
  analyzers/
    common/
      himg_kinds.py                    # the .himg jobs (HimgJob documents), run out of process: HimgToStackAnalyzer (folder → capture stack),
                                       #   HimgCompactAnalyzer (verify, then DELETE the .himg — destructive), HimgRestoreAnalyzer (rebuild them)
```

The legacy route — `SingleDeviceScanAnalyzer` and the `Array1D` /
`Array2D` wrappers around an ImageAnalysis analyzer, their renderers,
`ScatterPlotterAnalysis` — was retired in #1039 (2026-10-16); its last
state is the tag `legacy-analysis-final`. Never adopt it in new code: the
core route serves every kind, and `tests/data/golden/` pins the trees the
wrappers wrote (see "Goldens" below).

## Config System (YAML → Pydantic → Factory → Instances)

`core_inputs.prepare_v2` prepares a document for the numerical core. It is
shared with portal processing/unsaved previews. Compile capabilities before any reads, then load backgrounds once via
data-utils and bind immutable Frames. Failed file loads retain the constant
fallback; loaded shape errors propagate. `data_dir` means the device directory
when resolving `{scan_dir}`. Context-free previews leave that placeholder
literal. This adapter never mutates the caller's config. A scan background
(`scan.background_source` `scan_number` / `from_current_scan`, a recipe's
`from_scan` input) is computed by `core_backgrounds.resolve_scan_background`
before any shot is read: the mean of a dark scan's frames, or the median /
a percentile of this scan's, exact — a float64 running sum for the mean, and
for order statistics one pass into a native-dtype scratch copy (in memory
when small, on disk otherwise) reduced a strip of rows at a time, so memory
is frames × one strip. Pinned bit for bit against the legacy whole-stack
aggregation. It is cached in the analysis tree of the scan it comes from
(`analysis/ScanNNN/<device>/<device>_background_{avg|median|pNN}.npy` — a
dark scan's mean is the legacy wrapper's own file); a missing source scan
raises, never creates. `compute_scan_backgrounds=False` (every preview)
only reads that cache and refuses otherwise (`ScanContextRequired`, an
`UnsupportedRecipe`): a per-request view never reads a whole scan or writes.
A measure that names a service (`frog`: the FROG.dll retrieval; `haso`:
the WaveKit engine — external programs per frame) gets it from
`core_services.services_for(measure, data_dir=...)`, built from
this host's `config.ini` when the run is prepared — a host without the DLL
fails there, before any shot is read. The WaveKit engine is also built from
the scan: it needs one `.himg` header of the sensor, taken from the device's
capture stack (`core_source.stack_required` — an unconverted scan is
refused as `StackMappingUnavailable`, i.e. `no_data`, naming the
`himg_to_stack` converter). The same rule sits in `core_source.prepare_source`
and `core_backgrounds`: `.himg` frames (`file_tail: .himg`) are read through
the stack only, never per shot. `prepare_v2(services=False)` refuses
such a recipe with `ServicesNotRequested` (an `UnsupportedRecipe`):
`core_preview.prepare_document` defaults to that, so the portal's shot
browser keeps its old route (which refuses FROG) and never starts the DLL
per view; `preview_frame` / `preview_summary` — the editor's explicit
previews — and scan runs ask for services.

`core_source.prepare_source` maps a completed scan's scalar rows to native
files or capture-stack frames through data-utils. The returned `V2ShotSource`
snapshots references and reader settings, retains `ShotRef` indices, and loads
native arrays without dtype conversion or caching. Diagnostic identity selects
timestamp columns; `scan.device` selects the folder/native filename stem.
Camera stack preference may fall back to native files; stack-only traces cannot.
The source never creates folders or writes files. Hosts must wait for scan
completion before discovering HDF5 stacks over SMB. Reader metadata is not
returned by this raw-array adapter; recipes supply configured labels/units.
The source is the run's loader (callable per shot) **and a context manager**:
`PreparedScan.run` enters it once, and a pooled run enters it once in every
worker, so one capture stack is opened once per run/worker and read through
`scan_stack.read_frame`, not once per shot (each open is several SMB round
trips). Outside the context a read opens and closes by itself, as before.
Handles never pickle (`__getstate__` drops them; the source travels closed)
and are closed on completion, error or an early `close()` of the run's
iterator — `open_stacks` shows what is held, and `tests/test_core_streaming.py`
pins one open per run. Trace stacks (`pva_stack`) still open per read inside
the shared 1-D reader; a one-handle path for them is a data-utils change.
A stitched line input (the legacy `line_stitcher`; a recipe's
`input.siblings`, the document's `sibling_folders`) maps every sibling
folder the way it maps its own — each sibling's device is its folder less
the input folder's suffix, so its own timestamp column joins its own files
(the legacy stitcher's filename swap finds nothing on native timestamp
names) — and `load` returns the input's trace joined with each sibling's
same-shot trace, sorted by x with the legacy sort. A shot or folder a
sibling lacks is stitched without it, with a warning. The stitched TSVs
the legacy analyzer wrote into the scan folder are dropped on this route
(owner ruling 2026-09-27); the processed trace is saved under `analysis/`.
A joined trace is unevenly spaced, so its `rms`/`fwhm` are measured over x
on this route (GEECS-Analysis 0.26.0, #1029: the rms Δx-weighted, so it
does not depend on how densely each camera samples) where the legacy
analyzer counted samples times one spacing; `route_compare.compare_snapshots(nonuniform_axis=True)`
is the one allowance for that (the harness's `--nonuniform-axis`), and
`tests/test_core_stitch.py` compares the two routes through it, everything
else exact.

`core_scan.prepare_scan` snapshots config, rows, source, recipe and scalar naming
for one explicit run. A measure that reads a recorded column per shot
(`pipeline.shot_scalar_columns`: the magspec field) gets its finite cells on
each shot's `ShotMeta.scalars` (`shot_metadata`; a NaN or absent cell is
left out, so the measure's fallback applies), passed to `run_units` as
`shot_metadata`. Table inputs (`recipe.tables`) are read in
`core_inputs.prepare_v2`, a relative path against
`config_roots.scan_analysis_config`; an unreadable table fails the run. `PreparedScan.run()` streams core `UnitResult` outcomes;
`scalar_records()` projects bare scalars to `{output_name}_{key}{metric_suffix}`
without mutating the result or caller's rows. Per-bin updates include all bin
members even when some inputs fail. Groups follow scalar-row order; empty bins
are omitted and missing bin values form no group. Duplicate/nonpositive shot
numbers and fractional bin ids are explicit preparation errors. No sinks or
legacy factory route change are included in this adapter. `run(workers=N)`
hands the count to the core's ordered pool (`geecs_analysis.engine.v2_run`);
the outcomes arrive in group order whatever the count.

`core_products.plan_products` chooses saved average/bin measurements and ordered
summary panels without rendering or writing. Preserve the old figure gate
(more than two successful execution units), separately from scalar persistence.
Noscan means are unweighted over successful units; scanned per-shot summaries
use NaN-aware post-analysis bin averages, while raw-bin mode reuses its outcomes.
Scan positions average all scalar rows in each bin, including missing inputs.
Line sort requests bypass scanned-bin rendering; finite/bounds/sigma filtering
changes waterfall rows only, not the all-unit average. Omitted products carry
notes; the sink owns logging and file naming.

`ProductCollector` is that planning **as the run streams** (#1003), and
`plan_products` is built on it — one definition. It is built before the run
from what is known then (noscan or scanned, raw-bin or per-shot, a line
sort request, the shot→bin membership from the rows): a noscan (or a sorted
line waterfall) keeps one `RunningAverage` over every unit plus, for a line
recipe, every unit's trace for the waterfall rows **as the waterfall draws
it** (`geecs_analysis.summaries.waterfall.display_trace`: block means above
4096 samples, so a scope recording is never kept per row; rows of unequal
recorded length with any one reduced skip the waterfall with a note); a scanned per-shot run
keeps one `RunningAverage` per bin, folded in row order — the order the old
`group_shots`/`average_results` pair reduced in, so the numbers are
identical; a raw-bin run keeps each bin's measurement. A camera frame is
folded and dropped, so a camera scan holds one running frame per product
however long it is (memory target: bins × one frame or trace + the
waterfall's display rows + the in-flight window; `test_core_streaming.py`
tracks the frames a 300-shot run keeps alive,
`test_a_long_line_noscan_holds_display_rows_not_recordings` the rows).
Still unbounded in bins: a scanned run holds one full-resolution average
per bin (saved data, never reduced) until `save_products`; `plan` releases
each bin's accumulator as it makes that bin's average, so it plans once. `plan(rows, ...)` runs after the scalar merge because a waterfall
sort column may be one of this run's own outputs. The figure gate
(`MIN_SUCCESSFUL_UNITS`) is a counter. Anything that must see every
outcome again does not belong in the analyzer — it belongs in the
collector as running state.

`core_sink.save_products` is the only core-route writer of scan products. It
resolves the sibling `analysis/ScanNNN/<output_name>/Array{1,2}DScanAnalyzer/`
directory from an existing raw scan folder, refuses path components and
symlinks that escape it, and never creates `scans/ScanNNN/`. File names keep
the legacy shapes (`{device}_{id}_processed.h5`, `_processed_visual.png`,
`_averaged_image_grid.png`, `_summary_waterfall.png`) so `parse_output_filename`
and MCP's display-file contract are unchanged. A summary kind that computes
scan-level numbers (`scalar_fit`) also gets `<stem>.json` beside its PNG
(nonfinite as `null`; never a display file; classified `summary` by the
`_summary_` marker). HDF5 retains the legacy dataset
name, storage dtype and gzip level. `scan.save: false` writes nothing. A
rendering failure omits only its figure and is returned as a note; data and
write errors propagate. Scalar persistence is independent of this sink.

`core_analyzer.CoreScanAnalyzer` is the core route behind the contract the task
queue, the portal and MCP call. It inherits scan-tag handling, s-file reading
and scalar persistence from `ScanAnalyzer` and runs `prepare_scan` → `run`
(streamed into a `ProductCollector`, scalar records queued per outcome) →
sidecar + s-file merge → `collector.plan` → `save_products`. The worker
count is `core_workers.effective_workers(spec.workers, len(groups),
cap=self.worker_cap)`: serial when the recipe asks for one or the run has
fewer than `MIN_UNITS_FOR_POOL` (50) units, else the recipe's `scan.workers`
capped by the host — `CoreScanAnalyzer.worker_cap` when a host process set
it, else `config.ini [analysis] worker_cap` (a facility value: never in a
recipe), else one core fewer than the machine has — and logged as
`<device>: <n> units, <k> workers`. The count is a throughput knob only:
`test_pooled_run_writes_the_identical_tree` holds the pooled tree to the
serial one at 0 ulp. Hosts that run analyses (the portal's thread pool, the
task queue, MCP's detached worker) are threaded processes; the pool is
`spawn`, so their entry points must keep their `__main__` guards.
A missing or empty device folder, or stack-only input without a stack, raises
`DataUnavailableWarning`. Scalars are persisted before products, so a product
write failure never loses them, and the waterfall sort column resolves against
the refreshed rows as the legacy wrapper did. Summary figures are labelled
with the cleaned ScanInfo parameter, not the s-file column. Construction
compiles the document (no reads): one that does not bind to the registry
raises `RecipeError`, one asking for what the core does not do
`UnsupportedRecipe`, before any scan is touched — there is no other route.

**Goldens.** `route_compare.snapshot_analysis_tree` / `compare_snapshots`
are the one definition of "the same outputs" (HDF5 payloads and tables
exact, PNGs by presence, scan-folder products byte for byte); a snapshot
is stored with `write_golden` (a `manifest.json` + one `arrays.npz` of the
payloads and numeric table columns) and read back with `read_golden`. The
differential tests (`tests/test_core_{analyzer,backgrounds,downramp_phase,
frog_spectral_phase,hi_res_mag_cam,ict,magspec,stitch}.py`) run the core on
synthetic scans and compare file lists, HDF5 payloads, s-file columns,
sidecars, decoded sibling spectra and images and the display-file list against
`tests/data/golden/<case>/` to `route_compare.GOLDEN_RTOL` (1e-6, relative
to each array's or column's scale — cross-platform float drift between the
macOS where they were generated and Linux CI; a ×1.0001 change still
fails, `tests/test_route_compare.py` pins it) — the trees the legacy wrappers wrote, captured
from the core in the golden-first commit of #1039 while the two were
compared and matched (every stated allowance of that comparison is recorded
in the test's docstring). Regenerate deliberately, never casually:
`GEECS_UPDATE_GOLDENS=1 pytest ScanAnalysis/tests/test_core_*.py` rewrites
every golden from the current core, so a numeric change to the core must be
argued in the PR that re-strikes them. The pooled-vs-serial pins use the
same comparison at 0 ulp.

Scan analysis is driven by YAML config files stored in the
**GEECS-Plugins-configs** repository (not this repo). The documents are
**GEECS-Schemas'** (`geecs_schemas.analysis`): one YAML per diagnostic
under `analyzers/<namespace>/<id>.yaml`, read by `load_analysis_document`
(beside the `.himg` jobs' `HimgJob`, below) as the `AnalysisRecipe`
(format 3, the analysis core's native shape: `input:`, ordered `steps:`,
`measure:`, the per-frame `figure:`, the `summaries:` kinds; the whole
corpus since 2026-10-16). Diagnostics are assembled into
`AnalysisGroup` files under `groups/<namespace>/<group>.yaml`, which
explicit runners and the task queue consume directly. A file of the
retired v2 format (`analyzer:` + `image:`) is refused at load (#1039).
`core_recipe.ScanRecipe` is the one host-side view of a recipe; the
source, runner, planner and sink read it.

### Public loader / factory API

```python
from scan_analysis.config import (
    load_analysis_group, create_scan_analyzer,
    discover_analyzers, discover_groups,
)

group = load_analysis_group("baseline", config_dir=...)
analyzers = [
    create_scan_analyzer(r.diagnostic, id=r.id, priority=r.priority)
    for r in group.analyzers
]
for a in analyzers:
    a.run_analysis(scan_tag)
```

`create_scan_analyzer(document, *, id=, priority=, allow_destructive=)` has
one analysis route: every analysis document becomes a `CoreScanAnalyzer`
on `geecs_analysis`, compiled at construction. Runtime attributes assigned
after construction (`background_source`, `file_tail`) are inert on it: its
behaviour comes from its own copy of the document, so override the
document (`load_diagnostic(..., overrides=)`). `load_diagnostic` (in
`config/loader.py`) is the one single-document loader — data-utils'
`read_diagnostic` over the tree + the schemas' `load_analysis_document` +
the client `config.ini` default root, which neither of those packages may
read — used by the group loader, the portal's Analysis tab, MCP's worker
and notebooks alike.

The other route: a `.himg` job — the second document of the
analyzers tree, `geecs_schemas.analysis.HimgJob` (top-level `kind:`
`himg_to_stack` / `himg_compact` / `himg_restore`, `device`,
`output_name`, `scan.priority`; not an analysis, so not a recipe) — is one step over the device folder with no
ImageAnalyzer, no per-frame results and no products under `analysis/`.
`create_scan_analyzer` maps it through `himg_kinds.HIMG_JOB_CLASSES`
(kind → ScanAnalysis class) before anything reaches the core, and every
core entry point (`compile_document`, so `CoreScanAnalyzer` and the
previews) refuses it as `UnsupportedRecipe` (`tests/test_himg_kinds.py`
pins the coverage). The three live in
`analyzers/common/himg_kinds.py` and run their work **out of process**
through `geecs_data_utils.io.himg_worker.run_himg_job` (a child
interpreter streams frames done/total back; the analyzer relays them
through `ScanAnalyzer.report_progress`, the base-class hook a host such
as the portal's run record sets — a 44 GB scan streamed through the
portal's own process sat at its memory limit for half an hour with
nothing on the page): `HimgToStackAnalyzer` (`himg_to_stack`) converts a
HASO device's `.himg` files into its capture stack (the same function as
`geecs-himg convert`), verified after writing; the `.himg` files stay, a
second run verifies the existing stack. `HimgCompactAnalyzer`
(`himg_compact`, the one job with `destructive` true) verifies every
frame against the stack and the file on disk, then deletes the `.himg`
files and leaves `himg_manifest.json`; every guard (a `.himg` younger
than a minute, no `ScanData` table or s-file, a file the stack lacks, a
mismatch, another writer's `.part`) is data-utils' and comes back as a
failed run with nothing deleted. `HimgRestoreAnalyzer` (`himg_restore`)
rebuilds the files byte-identically. For all three a missing or empty
device folder is `no_data` and the run returns one label — files and GB
before and after — because everything they touch lives in the raw scan
folder. The compaction kinds are the one deliberate exception to
"analysis only adds files" (root CLAUDE.md), and the gate is the
factory's, not one host's: `create_scan_analyzer(...,
allow_destructive=False)` raises `DestructiveKindRefused` for a
`destructive` document unless the caller says it confirmed the run —
the portal passes `True` only past its typed-scan-number check, the
task queue's group loader (`load_analyzers_from_config`) skips such a
job with a logged reason, and MCP's single-analyzer path is refused.

`discover_analyzers` delegates to `geecs_data_utils.analysis_configs`; group
lookup remains here because group aliases have different rules.

`task_queue.load_analyzers_from_config(group_name, config_dir=...)` is a
thin wrapper around the same two calls.

### The documents (`geecs_schemas.analysis`)

```
AnalysisRecipe                    # One YAML per diagnostic (schema_version 3)
  device: str                     # Device name; stems the products
  output_name: Optional[str]      # Output stem override (defaults to device)
  scalar_suffix: Optional[str]    # Scalar-key-only suffix (no dir/file effect)
  input: CameraInput | LineInput  # kind: camera | line; folder (data-subfolder
                                  #   override), file_tail, format
                                  #   ("device_hdf5" opts in to the capture
                                  #   frame stack; a camera falls back to
                                  #   per-shot files, a "pva_stack" line does
                                  #   not and records no_data), a line's
                                  #   loading / scales / units / siblings
  inputs: {name: FrameInput}      # frames/tables the host loads and binds:
                                  #   path | from_scan | table
  steps: [StepRef]                # ordered registry references
  measure: MeasureRef             # the registry's measure kind + parameters
  scan: RecipeRuntime             # priority (100 default; lower runs first),
                                  #   average_frames_first, save, workers
  figure: FigureStyle             # the per-frame draw
  summaries: [Summary]            # image_grid | waterfall | average | scalar_fit

AnalysisGroup                     # One YAML per group under groups/
  analyzers: List[AnalyzerRef]    # bare stems or {ref, enabled, priority}

ResolvedDiagnosticConfig          # What the loader hands the factory (this package)
  id: str                         # Diagnostic filename stem (task-queue ID)
  enabled: bool                   # Refs with enabled: false are excluded
  priority: int                   # Group override, else the diagnostic's own
  diagnostic: AnalysisRecipe | HimgJob   # whichever the file declares
```

`scan_analysis.config` exports only its own things — `load_diagnostic`,
the group loader, `ResolvedDiagnosticConfig` and `create_scan_analyzer`;
the models are imported from `geecs_schemas.analysis`.

**Output-naming contract (#412)** — measures emit **bare** scalar keys
(`x_fwhm`, not `UC_TopView_x_fwhm`); the scan run (`PreparedScan.scalar_records`)
applies the document's `output_name` prefix (defaults to the device) and
`scalar_suffix` / `metric_suffix` when storing per-shot results.
`output_name` also names the per-analyzer output directory under
`analysis/Scan<NNN>/`; override it to run two variants over the same
camera with distinct output trees and s-file columns
(`output_name: UC_TopView_left` / `UC_TopView_right`). The suffix affects
scalar keys only, never directory or file names.

### The config editor (`config_store.py` + `config_editor/`)

The config editor (the Qt `ConfigFileGUI` it replaced was deleted in
1.21.0). Three layers, so the data portal is one host and not the only one:

- **`scan_analysis.config_store.ConfigStore(root)`** — plain Python, no
  web, no Qt: `list(kind)` (validity + summary per file), `read(kind, id)`
  (raw document + **etag**), `validate(kind, document)` (pydantic locations
  + canonical YAML via `geecs_schemas.analysis.canonical_document`),
  `save(kind, namespace, id, document, etag=)` (atomic temp+rename;
  stale etag / duplicate stem across namespaces → `ConflictError`; an
  invalid document is never written; `etag=None` = create, refuses to
  overwrite), `delete`, `pending_changes()` (git status of the tree),
  `schema(kind)` (the JSON Schema the form renders: for `analyzer` the
  recipe's, with `steps` and `measure` bound to the analysis core's
  registry by `geecs_analysis.recipe.recipe_schema`). A recipe validates
  as its schema AND binds to the registry (`compile_recipe`): an unknown
  step or parameter, a step or measure for the wrong frame shape, a frame
  input no step or measure uses — reported at the form's field path, listed as
  invalid, never written. `list()` runs the same cross-checks as
  `validate()` (a group naming an unknown document lists as invalid too;
  the analyzers tree is walked once per listing). Writes touch only the configs tree — the repo's
  scan-folder invariant is irrelevant by construction, and pinned
  portal-side.
- **`scan_analysis.config_editor.create_editor_router(store, preview=,
  read_only=)`** — a FastAPI router (the `editor` extra): `/api/list`,
  `/api/schema/{kind}`, `GET/PUT/DELETE /api/{analyzers|groups}/…`,
  `POST /api/validate/{kind}`, `POST /api/preview` (404 unless the host
  passes a `preview(document, params) -> PNG bytes`), the editor page and
  `static/editor.js` + `editor.css`. Every URL is relative to the mount.
  `editor.js` is a hand-written schema-driven form over pydantic's JSON
  Schema (objects, `anyOf [T, null]` optionals as toggled sections,
  discriminated unions as a kind select that swaps the variant's fields,
  enums, tuples, keyed mappings as named cards, keyword mappings as
  key/value rows), live YAML preview, server-side error placement by
  pydantic location, and the optional preview pane (a `preview` button
  renders the edited document on the host's shot; `auto` re-renders per
  edit, remembered in localStorage). **The form is the recipe's** (format
  3), laid out as the document reads, each section a `<details>` closed
  on open with a one-line gist in its header (`sectionGist`; a server
  error opens the section and card holding it): Source (`device`, the
  resolved-names line — reads `scans/ScanNNN/<folder>/`, writes
  `<label>_<metric><suffix>` and `analysis/ScanNNN/<label>/`, from
  `namesOf` — then a nested **naming overrides** group, `output_name` +
  `scalar_suffix`, closed unless the document sets one; `input`, frame
  `inputs`) → Steps (each step / summary card folds to its gist; an ROI's
  `bounds` are fixed `y`/`x` rows, or one `x` row for a line, by the
  document's input kind; a schema `maxItems` hides a list's adder at
  the cap; ordered cards; the add-select lists the registry with
  `(images)` / `(traces)` hints from `x-ndim`; a name the registry does not
  know is kept as written and flagged, never swapped) → Measure → Figure
  (matplotlib keyword rows: numbers, true/false, [lists] typed, other text
  a string; overlay styles by id) → Summaries → Scan. Reorder and remove
  rebuild the list from its current values so every field path
  (`steps.2.bounds`) stays true. A `.himg` job, or a file that is not a
  recipe (the retired v2 format lists as invalid), opens read-only — a
  note, the file as saved, Delete. The preview is **the run's own draw**: the host renders the
  document through `scan_analysis.core_preview` — `preview_frame`, the
  sink's per-frame call with the document's `figure` block, frame inputs
  resolved from the document's device folder under the scan — cropped
  tight like the product PNG, so what the pane shows is the product file
  the run would write. The **summaries** block (`POST /api/preview/summary`,
  host hook `summary_preview=`, cap `summary_shots_max=` carried in
  `/api/list`) lays out each summary kind over a few of the host's shots
  through `preview_summary` — `core_sink.draw_summary`, the sink's own
  call — one panel (row) per shot at its shot number under the run's
  noscan label (`core_products.NOSCAN_POSITION_LABEL`), or the shots'
  average. That is the kind's *layout* on real frames (a run's grid
  panels are per-bin averages, and a camera run on a noscan writes no
  grid), while the frame preview is the product file itself. Both draws
  are the sink's functions (`draw_product` / `draw_summary`), pinned
  byte-for-byte against `save_products`' files
  (`tests/test_core_preview.py`); a host never re-derives the pairing. The form is pinned under node on a fake DOM
  (`test_recipe_form_round_trips_and_reorders`: a corpus recipe reads back
  canonical-equal, move-up swaps steps and renumbers paths). No build
  chain, no library — the portal's doctrine.
- **Host.** The data portal mounts it at `/configs` with the preview of
  the **unsaved** document on the scan page's current shot
  (`GEECS-DataPortal/CLAUDE.md`); its **edit configs** link is the
  full-page form. It is the one host: the standalone
  `scan-config-editor` launcher was removed (owner ruling 2026-09-13 —
  the editor stays beside the analysis it configures, and a laptop edits
  the configs repo in an editor and commits).

Adding a field to a schema model is all an editor change needs: the form
is generated. Adding a step, measure or summary kind in GEECS-Analysis
(its spec's `description=` on every field — the form shows it as help)
shows up in the add-select automatically.

## Analyzer Class Hierarchy

```
ScanAnalyzer  (base.py)
  ├── CoreScanAnalyzer  (core_analyzer.py) — every analysis document, on geecs_analysis
  └── HimgToStackAnalyzer / HimgCompactAnalyzer / HimgRestoreAnalyzer  (common/himg_kinds.py)
```

### `ScanAnalyzer.run_analysis(scan_tag) -> Optional[list[Path | str]]`

The main entry point. Returns a list of **display files** (paths to summary
figures) that the task queue records, or
`None` when there was nothing to analyze.

## Task Queue System (`task_queue.py`)

Retained for explicit MCP runs and status compatibility. Atomic claims prevent
concurrent runners from owning the same task; this is not a new automatic
post-scan service. That service remains a separate design decision.

### How It Works

1. A **status YAML** is created per scan per analyzer:
   `<scan_folder>/analysis_status/<analyzer_id>.yaml`
2. States: `queued → claimed → done / failed`
3. When a runner picks up a task it writes a **heartbeat** (timestamp) every 30s
4. A claimed task is considered **stale** after 180s without a heartbeat update
5. Other runners can re-claim stale tasks — safe parallelism without a central
   coordinator
6. **Claims are atomic since 1.16.0** (`try_acquire_claim`/`release_claim`,
   an `O_CREAT|O_EXCL` lock file at `analysis_status/<id>.claim` —
   invisible to status readers, which glob `*.yaml`): two runners whose
   worklists both saw `queued` can no longer double-run one task
   (previously a pure check-then-claim race; surfaced when GEECS-MCP's
   `run_scan_analysis` began spawning workers at call time, #690).
   Liveness stays in the status heartbeat (`claim_is_active`, public);
   a dead holder's lock is broken automatically

### `TaskStatus` fields

**`TaskStatus.to_dict()` in `task_queue.py` is the authoritative shape**
(this list was stale once and a downstream consumer shipped a dead
parser from it — GEECS-MCP #675 review; read the writer when consuming):

- `analyzer_id: str`, `priority: int`
- `state: str` — queued / claimed / done / failed / no_data
- `error: Optional[str]` — why a failed task failed
- `claimed_by: Optional[str]` — runner identifier
- `claimed_at`, `last_heartbeat`: Optional[str] — **ISO-8601 strings**
  (UTC; `_parse_ts` assumes UTC when naive)
- `display_files: Optional[List[str]]` — absolute paths under the day's
  `analysis/` tree, populated when the analyzer completes.  Consumed by
  GEECS-MCP's `get_scan_figure`.

Any package other than this one consumes these files through
`geecs_data_utils.analysis_status.read_analysis_statuses` (the one
tolerant, schema-light, read-only reader — #682), never a local parser;
the exception is code that must drive the queue with the typed
`TaskStatus` (`claim_is_active`), which uses `read_statuses` here
(GEECS-MCP's `run_tools`, behind its `analysis-run` extra).
`tests/test_analysis_status_contract.py` pins the writer against that
reader: a new `to_dict()` key fails there until `STATUS_FIELDS` /
`AnalysisStatus` in GEECS-Data-Utils learn it — extend both in the same
PR as the writer change.

## Scalar persistence

Scalar persistence in `base.py` delegates to `geecs_data_utils.scalar_files`:
one lock/merge implementation coordinates legacy and new runners. ScanAnalysis
still owns sidecar destination names, sibling analysis-directory creation and
refreshing `auxiliary_data`. Preserve the difference between s-file merging
(missing updates retain old cells) and sidecars (generated NaNs remain NaN).

## Product output names are a consumed contract

The sink's product names (`core_sink.save_products`: `{device}_{bin}_processed*`,
`_average_processed*`, `_averaged_image_grid`, `_summary_{kind}` + its
JSON, the noscan `.gif` — the legacy renderers' shapes, kept) are parsed
back by `core_sink.parse_output_filename` in the same module — the data
portal's Analysis tab classifies outputs through it (summary vs per-bin).
Change a name and its parser together;
`tests/test_renderer_output_names.py` pins the pairing.

## Retired watching and uploads

LiveWatchGUI, LiveTaskRunner and Google Docs uploads were removed in 1.26.0.
The config editor, ConfigStore, group loader and explicit task queue remain.
Existing `gdoc_slot` and `upload_to_scanlog` fields validate but have no effect;
the editor hides these deprecated fields and preserves authored values.
ScanAnalysis has no LogMaker, watchdog or Qt dependency.

## Key Design Decisions

- **`priority`** — Lower number runs first. Default 100; use low numbers
  for fast diagnostics.
- **`enabled: false`** on a group ref — Disable an analyzer without
  removing it from the group config.
- **`measure.kind` / `steps`** — A recipe names the core's registered
  measure and steps; a new kind is one spec + function in GEECS-Analysis
  (its `CLAUDE.md`), never a class path in a YAML.
- **Everything in the document** — The per-device processing (ROI,
  background, pipeline) lives inside the diagnostic YAML itself; there is
  no separate camera-config lookup.

## Filesystem invariants

**ScanAnalysis never creates `scans/ScanNNN/`.** It is a consumer of scan
folders that the scanner already wrote. This rule is load-bearing — see
[Cross-package invariants](../CLAUDE.md#cross-package-invariants) in the root
for the full background and the production incident that motivated it.

In practice:

- All `ScanPaths(...)` calls in this package use the default `read_mode=True`
  (which raises on a missing folder). Never pass `read_mode=False` from
  analysis code.
- `task_queue.init_status_for_scan` and `task_queue.update_status` verify
  `scan_folder.is_dir()` and bail with an `ERROR` log if it's missing — they
  do **not** auto-create. The queue keeps running other work; if the scan
  folder later reappears, discovery can pick it up on a later processing pass
  or after relaunch.
- `analysis_status/` is the only directory ever auto-created by this package,
  and only via `mkdir(exist_ok=True)` — no `parents=True`.
- Analyzers write their outputs to `<date>/analysis/Scan<NNN>/...`, the
  *sibling* of `scans/Scan<NNN>/`. Never write back into the scans tree.
  One deliberate exception (owner ruling 2026-09-26): a measure's per-shot
  sidecar table — the FROG `*_retrieved_lineouts.tsv` — is written beside
  the shot's raw file (`core_sink.write_shot_table`), where the legacy
  analyzer wrote it and follow-on analyzers read it through a recipe
  input's `folder`. It creates no directory. Moving derived artifacts
  under `analysis/` and letting inputs read from there is a separate,
  undecided design question. The second exception (the HASO arc brief the
  owner approved 2026-09-28): the `himg_to_stack` job writes
  the device's capture stack `<device>/<device>.h5` beside the `.himg`
  sources — the one place every stack reader looks, by the file-plugin
  contract — through `geecs_data_utils.io.himg_stack` (temp file, verify,
  rename; no directory; the sources untouched). The HASO measure's own
  per-shot products take the other road: a measure registered with
  `shot_store="wavefront"` gets one HDF5 per scan and recipe **under the
  analysis tree** (`core_sink.ShotStore`, `<device>_wavefront.h5` in the
  analyzer directory: `shots`, `frame` (N, …) float32, `extras/<key>`;
  written as `.part`, renamed at the end of a clean run, discarded on a
  store error without losing the run's scalars; honours `save`; single-shot
  units only). Nothing is written beside the `.himg` files.  The
  compaction kinds (`himg_compact` / `himg_restore`, 1.45.0) are the one
  place analysis-side code DELETES or REWRITES files — the `.himg` sources
  of a converted device folder, verify-first, guarded in data-utils, behind
  the portal's typed confirmation — and they still create no directory and
  touch nothing outside that one device folder (pinned in
  `GEECS-Data-Utils/tests/test_himg_compact.py` and `tests/test_himg_kinds.py`).
  The third exception (#1039 rulings): a measure's **sibling outputs**
  (`@measure(sibling_outputs=...)`; magspec's `-interp` 16-bit PNG, the
  frame × 1000, and `-interpSpec` TSV) are written per shot of a per-shot run
  into `scans/ScanNNN/<output_name><suffix>/<raw stem>.png|.tsv` by
  `core_sink.write_sibling_outputs`, byte for byte the legacy analyzer's
  per-shot files (a capture-stack frame gets the shot-number stem — the
  announced shot in scan mode, `core_scan.file_numbers`, as data-utils maps
  it back; the FROG sidecar too — where legacy overwrote one file named by
  the stack), whatever `save` says: the scan folder must exist and the suffix
  folder is the one directory created (`mkdir(exist_ok=True)`). A failed
  write is logged and the run keeps its scalars. A per-bin run writes
  none, even for a bin of one loaded shot (the legacy per-bin path had no
  file to name them by). The raw
  stem keeps old and new data readable by one stitcher; on a
  timestamp-named scan the join rests on data-utils' stem-agnostic
  listing fallback (`core_source._sibling_references`).

Do not treat a missing entire scan folder as `no_data`. `no_data` means the
scan exists but a specific device/analyzer has no usable data. If the scan
folder itself is absent, `analysis_status/` is unavailable because it lives
inside that folder; logging and skipping is the safe behavior.

When writing a new analyzer, **do not** use `Path.mkdir(parents=True, ...)` on
any path that could traverse through a `scans/` folder. The invariant is
pinned by tests in `tests/test_task_queue.py::TestScanFolderCreationInvariant`.

## Adding a New Scan Analyzer

1. (Optional) Add or reuse a step, measure or summary kind in
   `GEECS-Analysis` (its `CLAUDE.md`); the editor's add-select lists the
   registry automatically.
2. Create the recipe YAML in the configs repo under
   `analyzers/<namespace>/<id>.yaml` — or in the portal's config editor,
   which previews it on a real shot (the shape below is illustrative; the
   editor's schema is the authority on step names and fields):

   ```yaml
   schema_version: 3
   device: MyDevice                    # the folder under scans/ScanNNN/
   input: {kind: camera}
   steps:                              # ordered, from the core's registry
     - {step: background_constant, level: 5.0}
     - {step: roi, bounds: {y: [350, 650], x: [0, 650]}}
   measure: {kind: beam}
   summaries: [{kind: average}]
   scan:
     priority: 50
     # workers: 4                      # a process pool; the host caps it
   ```

3. Add the diagnostic to one or more groups under
   `groups/<namespace>/<group>.yaml` — bare filename stem, or a dict
   with per-group overrides:

   ```yaml
   analyzers:
     - MyDevice
     - {ref: OtherDevice, priority: 5}
   ```

4. No Python changes needed in ScanAnalysis itself: `load_diagnostic`
   reads it, `create_scan_analyzer` compiles it against the core's
   registry and runs it as a `CoreScanAnalyzer`.

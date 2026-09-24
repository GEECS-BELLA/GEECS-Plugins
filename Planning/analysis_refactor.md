# Analysis refactor: ImageAnalysis + ScanAnalysis core → `geecs_analysis`

*Planning note (see `Planning/README.md`). Delete this file in the PR that
deletes the old analysis cores (roadmap keystone A3's last route flip);
anything still load-bearing moves to `GEECS-Analysis/CLAUDE.md` first.*

Drafted 2026-09-06 from four parallel code audits of the #803 tree plus a
field-by-field census of the 61-file analysis-config corpus; discussed over
2026-09-06..08; refreshed 2026-09-18 against master (native-Bluesky rebuild,
console deletion, capture-daemon retirement, logbook). Scope amended
2026-09-22: retire LiveWatch and Google Docs uploads; preserve the DataPortal
config editor. Status: **baseline, frames, initial processing/measures, v2
compatibility and rendering implemented; consumer migration in progress.**
Owner: Sam.

---

## Where we stand

**Replace the analysis core and retire the legacy watcher and uploads.** ImageAnalysis and the
core of ScanAnalysis (`base.py`, `analyzers/common/`, `analyzers/renderers/`)
are replaced by one new package. Retire `LiveTaskRunner`, the LiveWatch Qt
GUI and Google Docs uploads. Keep the task queue and status YAML contract
used by MCP, the group loader, `ConfigStore` and the web config editor.
The DataPortal mounts the editor at `/configs` and embeds its form in the
Analysis tab's editor drawer, including previews of unsaved diagnostics.

**In-repo, on an integration branch.** The new package lands as
`GEECS-Analysis/`, developed on `codex/analysis-refactor` through focused
child branches and reviewed PRs targeting that integration branch. Every retained consumer
(the portal, MCP, the optimizer) reaches it through the interface
it already calls; the old cores are deleted when the last recipe's route
flips.

**Automatic analysis is deferred.** Retiring LiveWatch does not introduce a
replacement watcher; explicit runs through the portal and MCP remain. The data
*architecture* (folder layout, file formats, the s-file) is a separate
conversation and is not touched here; how code finds and reads it is in scope.

**Timing is agent-days plus external gates** (a fixture scan on the share, a
live optimization run, the Windows machine for the vendor SDKs), not calendar
weeks.

## Minimum success agreed 2026-09-22

- **Operational endpoint: the DataPortal.** An operator can edit and preview
  a diagnostic, run it on a scan, and inspect saved results. Automatic
  post-scan triggering is deferred; a future service needs its own design.
  Preserving the existing MCP queue/status contract during migration is
  compatibility work, not a decision to reuse YAML status files for that
  future service.
- **2D: `BeamAnalyzer` (`beam`).** Preserve processing, scalar names and
  values, overlays and scan outputs for representative beam diagnostics.
- **1D: MagSpec spectra and waterfall plots.** The existing canonical test
  uses `LineAnalyzer` (`line`) over `U_BCaveMagSpec-interpSpec`, wrapped by
  `Array1DScanAnalyzer`; the wrapper produces the waterfall. This is distinct
  from `MagSpecManualCalibAnalyzer` (`magspec`), which calibrates 2D camera
  images. Cover per-shot and per-bin waterfall data, axes, ordering and output
  names, not just successful per-shot line statistics. The local configs
  checkout confirms `BcaveMagSpecStitcherSpec` uses `kind: line` and selects
  `U_BCaveMagSpec-interpSpec` as its scan device.
- **GeecsBluesky optimization adopts `geecs_analysis` in this effort.** Its
  measurement evaluator currently calls
  `image_analysis.ephemeral.run_document_ephemeral` on timestamp-matched live
  frames. Update diagnostic loading, config validation, declared-output
  discovery, frame/source adaptation, evaluation and package dependencies as
  needed to use the new core directly. A compatibility adapter is an interim
  route, not the optimizer's final integration. Preserve selected scalar names and
  `{measurement}.{scalar}` outputs, per-shot versus average-before-analysis
  semantics, reductions, minimum-valid-shot handling and the no-writes
  contract. Replace the old denylist with explicit input/capability checks;
  only enable new inputs when the live source supplies their axes, identity
  and required context. The current live path accepts camera diagnostics only;
  broader trace optimization is not implied by deleting that guard.
- **Vendor-dependent diagnostics have separate validation gates.** FROG
  retrieval requires the 32-bit Windows `frog.dll` execution path; HASO needs
  its vendor SDK/runtime. Isolate these dependencies so core, portal and
  optimizer tests run without them. Contract tests with doubles verify
  integration only; scientific parity requires real data on a capable Windows
  host. Neither blocks the beam/line milestone, and neither is retired merely
  because local validation is unavailable.
- **Retire `bcave_magspec_stitcher` (owner approved).** This is the legacy
  camera analyzer, not `line_stitcher`, `bcave_mag_opt`, or the `line` recipe
  that reads an already-stitched spectrum for waterfall plots.
- **Additional analyzer families await owner selection.** Beam and line are
  the minimum release gate, not permission to delete every other analyzer.
  Keep unported routes until their required coverage or retirement is decided.

### Existing reference data and gaps

`tests/conftest.py` already names canonical Undulator scans:

| Fixture | Scan | Existing coverage |
|---|---|---|
| `undulator_2d` | 2025-02-20, Scan014 | Beam image loading and finite scalars; 2D scan-analysis integration |
| `undulator_magspec` | 2025-11-18, Scan002 | `LineAnalyzer` over text `interpSpec` data; scan execution and centroid scalar |
| `undulator_ict` | 2025-11-13, Scan001 | TDMS trace loading and 1D scan analysis |
| `undulator_bluesky_1d` | 2026-08-29, Scan001 | Bluesky-era scalar union-frame integration |

These are references to data on the share, not bundled scan fixtures. Configs
for the analysis integration tests also come from the separate configs repo;
their correspondence to each fixture must be verified. Legacy-format data
remains useful for numerical parity and reader compatibility. It does not
alone establish parity for current acquisition formats.

The repo also has synthetic beam/line accuracy tests, HDF5 `ShotRef` and stack
join/loading tests (including 1D traces), and
`GeecsBluesky/tests/optimization/test_live_measurements.py`, which exercises
real beam analysis through the optimizer's evaluator. Build on these, then
add the differential scalar/output checks and waterfall-content checks the
current smoke tests do not provide. No new reference-data selection is needed
from the owner unless these scans prove unavailable or unrepresentative.

Initial access check after merging master through #952–#954: both canonical
beam and MagSpec scan folders are reachable, with PNG and text inputs
respectively. The separate configs checkout contains `Amp4Input.yaml` and
`BcaveMagSpecStitcherSpec.yaml`. The current beam config requests
`data_format: device_hdf5`, while its canonical input is PNG; a fixture-specific
reader override is needed for scan-level baseline runs. Its configured device
also ends in `_Input`, whereas the archived folder ends in `_input`; preserve
the archived identity explicitly rather than relying on case-insensitive
filesystem behavior. This establishes availability, not numerical parity or
successful execution.

The first implementation slice, `scripts/analysis_baseline.py`, captures
processed arrays and named scalars through the existing write-free analysis
API, then compares backend-independent NPZ archives. Archives record the
resolved recipe and ordered input hashes, load without pickle, and cannot
overwrite a reference or be written inside raw scan folders. Exact comparison
is the default; tolerances must be explicit. This slice deliberately excludes
scan discovery, binning, figures and s-file writes. Recipes with external
processing dependencies are refused until those dependencies are fingerprinted.

Baseline execution on the canonical scans (first three shots each): beam
produces 18 scalars per shot and 599×599 processed arrays; repeated per-shot
and averaged-frame runs match exactly. MagSpec produces six scalars and
2000×2 arrays. Its second shot is zero throughout the processed ROI, and the
current stack emits NaN centroid/RMS/FWHM. The comparator flags these instead
of silently treating matching NaNs as scientific parity; preserve this case
for an explicit invalid-measurement policy in the new core. No source scan
files or configs were modified by these captures.

The worktree's Python 3.11 Poetry environment is now installed. The existing
beam/line, ephemeral-document and renderer-name baseline selection passed
46 tests. These results establish a starting reference, not acceptance of the
future backend, waterfall renderer or optimizer integration.

### First core slices (2026-09-23)

Baseline harness #955 and Frame #956 are merged to the integration branch;
pure steps #957 and beam/line measures #958 follow. The v2 adapter now compares
the first three canonical raw beam inputs against the saved baseline exactly
(per shot and average-before-analysis). All three MagSpec processed arrays and
all finite scalars match; the zero shot's three undefined metrics remain
explicitly flagged. Inputs/configs are unchanged. This is numerical acceptance
for these recipes only: waterfall figures, scan products, portal routes,
optimizer integration and newer acquisition formats still need their gates.

The compatibility adapter preserves legacy precision and output quirks without
sharing mutable frames. General core pipelines use the new coordinate model.
Unsupported active steps still take the old portal/scan route until ported.

Rendering #960 adds object-API single-frame and same-grid waterfall figures;
the canonical beam/waterfall previews were inspected. Shared diagnostic reading
#962 removes discovery copies from both legacy packages. Optimizer adoption is
the first consumer migration: direct compiled v2 recipes, scalar discovery from
the compiled measure, live timestamp metadata, preserved reductions and no
config reads during evaluation. Active optimizer beam recipes fit this subset
(including explicitly disabled transforms). Live optimization acceptance remains
owed; portal routing, scan runners/sinks and retirement are still outstanding.

### Consumer and input progress (2026-09-23)

Portal processing and unsaved previews use the core for supported recipes;
ConfigStore/editor and explicit scan execution remain intact. LiveWatch and
Google Docs analysis integration are retired (#966). Circular masks and trace
interpolation (#967), shared shot-file mapping (#968), and the approved BCave
camera retirement (#969) are merged. The distinct `line` spectrum recipe and
waterfall path remain supported. Pure loaded-frame bindings follow in #970.

File-background source preparation preserves legacy constant fallback, offset
order and device-directory placeholder semantics without mutating configs or
writing files. The updated corpus check validates all 50 documents and compiles
30 of 37 beam/line recipes with this opt-in; the seven remaining VISA recipes
also need crosshair masking, and three need image rotation. A per-step census
found these behind the initial background refusal. The two newly compiled
recipes match all pixels and 36 finite scalars
on one canonical archived beam image using fallback backgrounds. This tests
recipe behavior, not acquisition-specific physics or actual saved dark files.
Synthetic saved-background tests separately cover successful loads and failures.

The following camera-geometry slice ports crosshair rasterization and fixed-
canvas rotation without changing their algorithms. All 37 beam/line recipes
now compile with file-background opt-in. All nine background-dependent recipes
match every processed pixel and 162 finite scalars on the same archived image
using fallback backgrounds. Differential synthetic tests cover masks at edges,
rotation, cropped local centers, repeated steps and nonfinite classification.
These broaden recipe coverage; they do not replace per-diagnostic physics or
scan-output acceptance. Flips and distortion correction remain unported.

The streaming v2 unit runner separates source loading from compiled evaluation
and yields explicit outcomes for per-shot or average-before-analysis groups.
Native dtype and full bin membership are retained; raw buffers are bounded to
one group and source failures are recorded separately from contributors. The
declared member order replaces the old loader's nondeterministic completion
order. On the first three canonical beam and MagSpec inputs, both modes match
every processed array and 93 finite scalars exactly; three undefined MagSpec
shot scalars remain flagged separately. Hosts still need to supply grouping,
readers and sinks before this changes explicit scan execution.

Completed-scan source preparation now retains native reader precision, stack
frame indices and diagnostic identity separately from folder overrides. The
archived beam and MagSpec source maps match all 218 and 101 references, and
the first three loaded arrays/dtypes from each match the old wrapper exactly.
Write-free scan orchestration connects those sources to per-shot and per-bin
execution, preserving full-bin scalar propagation and bare core measurements.
It snapshots grouping and names before execution; output sinks and the factory
route remain to be connected.

### Scan route adoption (2026-09-23)

Products (#979), the sink with legacy names (#980), the `CoreScanAnalyzer`
adapter behind the unchanged `ScanAnalyzer` contract (#981) and the factory
route flip are on the integration branch. Every beam/line/standard/trace recipe
the core compiles now runs on `geecs_analysis` for the portal's Analysis tab,
MCP and the queue; scan-context backgrounds (the four HTT MagCam recipes),
unported kinds and steps keep the legacy wrappers. A differential test runs
both routes on synthetic scans and compares file lists, HDF5 payloads, s-file
columns, sidecars and display files exactly (noscan averages within a few ulps:
the legacy wrapper sums shots in directory-listing order).

`scripts/analysis_scan_compare.py` runs that comparison on a real scan from
private copies of its inputs (`--set scan.data_format=per_shot_files` for the
beam recipe, whose canonical input is PNG); recipes with scan-context
backgrounds are refused because the legacy wrapper would write beside the
archived reference scan. Both use `scan_analysis.route_compare`, the one
definition of equal outputs.

### Archived-scan comparison (2026-09-23)

The harness ran on the two canonical scans from the mounted share, on the
integration branch at the #983 merge (4f95ba02), with `MPLBACKEND=Agg`:

| Scan | Recipe | Shots | Files per route | legacy / core | Result |
|---|---|---|---|---|---|
| Undulator 25_0220 Scan014 | `HTU/Amp4Input.yaml`, `--set scan.data_format=per_shot_files` | 218 | 4: s-file, sidecar, average HDF5, average figure | 5.8 s / 4.7 s | MATCH |
| Undulator 25_1118 Scan002 | `HTU/BcaveMagSpecStitcherSpec.yaml` | 101 | 5: the same plus the waterfall | 1.2 s / 1.3 s | MATCH |

MATCH means `route_compare` found no difference in the file lists, the
average HDF5 payloads, the s-file and sidecar tables (decoded, value by
value: 18 beam scalars over 218 rows; 6 line scalars over 101 rows) or the
display-file names. The beam average (599×599 float64) is bit-identical; the
MagSpec average (2000×2 float32) differs by at most 1.1e-11 absolute,
1.8e-7 relative, 3 ulps of the stored float32 by value spacing, inside the
4-ulp noscan tolerance above: the one place the tolerance was exercised. The
comparison bites: adding 1e-3 to one element of the core's average HDF5, or
changing one value in one s-file row, each produced a `DIFF` line against
the legacy tree. Tables are compared decoded, so an edit that leaves the
value unchanged is invisible, as is a figure's pixel content: display files
are compared by name only.

Two parity facts the MATCH does not show. 50 MagSpec shots are beamless
(integrated intensity 0) and yield non-finite CoM/rms/fwhm on both routes:
the sidecars match NaN for NaN; the core logs each (150 `Nonfinite scalar`
lines), the legacy wrapper logs nothing because its per-shot work runs in
process pools. And `merge_updates` writes s-file cells through
`combine_first`, so those 150 NaN cells left the archived 2025 values in
place on both routes: the s-file comparison there compares the share's
values with themselves, and the sidecars are the evidence that both routes
computed the same scalars. On the beam scan every s-file device cell equals
its sidecar cell. The legacy `append_to_sfile` "columns already exist (will
overwrite)" notice fires on both routes and is not literally true for NaN
cells: pre-existing behaviour, part of the cleanup already promised, not a
route difference.

Figure content is outside the harness (two renderers). Side-by-side
inspection of the three figures (beam average, MagSpec average trace,
MagSpec waterfall; the waterfall is the harness's only MagSpec display
file): identical image, trace, waterfall rows and colour scales; the labels
and the canvas/tick layout differ. Legacy: `X Pixels`/`Y Pixels` with an
unlabeled colourbar; `X`/`Y` on the average trace; the TSV column headers on
the waterfall (`Momentum_GeV/c (MeV)`, `ChargeDen_pC/GeV`). Core: `x (px)`/
`y (px)` with an `Intensity` colourbar; `x (MeV)` on the trace and the
waterfall, with the recipe's `label` (`Charge density vs Energy`) as the
trace's y label and the waterfall's colourbar label; both core canvases are
larger (beam 616×617 vs 586×475 px, waterfall 1514×1217 vs 1460×1184) and
the beam ticks sit at 200/400 rather than from the origin. Whether the
core's labels stand is the operator figure review; nothing in the data
differs.

`./scripts/check.sh` on this branch ran all ten suites: OK, 3,408 passed.

### Operator figure review (2026-09-24)

The maintainer reviewed the three side-by-side figures. Verdict: the core's
axis labels are accepted as drawn. One defect: on the beam average the
colourbar is oddly sized against the image. Cause: `single_v2` draws a square
figure with constrained layout and `draw_frame` attaches the colourbar to the
axes, while `imshow` keeps the image aspect equal, so the axes shrink to fit
and the colourbar keeps the full allocated height. Fix: size the colourbar
from the image axes for 2D frames, in both places that attach one: `single`
(through `draw_frame`) and `image_grid`'s shared colourbar. The maintainer
also asked that cosmetic controls (labels, colourbar, sizes) and overlays be
exposed to users. Labels and sizes already are on v2: `scan.renderer` carries
`xlabel`, `ylabel`, `colorbar_label`, `figsize`, `figsize_inches` and `dpi`,
and the editor form is schema-driven. The colourbar is row F1; overlays and
references are the ruling under § 4 and row F2.

Still required for the first milestone: the operator check of a line preview
on an archived scan (#984 merged 2026-09-24), the colourbar fix (F1, half a
day, operator-visible on every beam figure; a judgment call to land it before
promotion rather than ship it as a known defect), live optimizer acceptance,
then the promotion PR. This file is deleted at prune, so the promotion PR's body
carries the numbers above and the review verdict as its verification section.

## Decisions

| Topic | Status | Where we landed |
|---|---|---|
| One package for image + scan analysis | decided | The split was organisational; the seam between them (`render_function` on the result, `output_name` threading, in-place mutation of the camera config for scan backgrounds, `getattr` attribute injection from the factory) is where the complexity lives. |
| Hard fork vs in-repo | decided | In-repo, developed on `codex/analysis-refactor`; focused PRs target that integration branch. Promote complete, validated milestones to master, not unfinished layers. Adopt per recipe behind the factory and merge master forward periodically. |
| LiveWatch and Google Docs uploads | decided | Retire the Qt GUI, `LiveTaskRunner`, upload hooks and analysis-side LogMaker dependency. No replacement watcher or Google Docs export in this effort. |
| Task queue, status YAML, group loader | decided | Keep for MCP's explicit analysis runs; decouple from upload code. Portal runs already call the analyzer directly without queue participation. |
| ConfigStore and web config editor | decided | Keep: the DataPortal uses `/configs`, its Analysis-tab editor drawer and unsaved-document previews. Preserve these through the schema migration. |
| Qt ConfigFileGUI | done in #803 | Deleted with the re-export shims, the model aliases and the v1 converter. The web editor is the one editor. |
| Pipeline = ordered, repeatable list of typed steps | decided | No canonical order. Any order, duplicates allowed (two medians; a clip before and after a filter). Each step declares `ndim`; the loader validates the list. |
| Steps: one file, spec + pure function; variants are separate steps | decided | No method enums with conditional fields: `background_constant`, `background_frame`, `clip_below`, `clip_above` instead of `method × value × mode × invert`. |
| 1D and 2D share one container | decided | `Frame` = ndarray + typed axes + provenance. Six steps shared, one 1D-only, five 2D-only. ROI slices the axes, so positional stats stay in global coordinates without an offset. |
| Rendering: matplotlib pass-through + overlays by id | decided | 90% of wants are matplotlib kwargs: named dicts (`imshow`, `colorbar`, `axes`, `fig`) validated by the editor's preview. Measures declare overlays by id; recipes style or hide them. `@figure` hook for bespoke panels. Rendering stays in the package because the logbook needs figures. |
| Naming | decided | `id` (file stem; scalar prefix and output dir), `device` or `devices` (finds files), optional `scalar_suffix`. Replaces `output_name`, `metric_suffix`, `scan.device`, `output_label`. |
| Per-bin mode | decided | Kept as `average_frames_first: true`; the optimizer's `frames: per_bin` is the same thing. Implemented once in `run`. |
| Side effects | decided | Measures return derived products in memory; whether they are written is the runner's decision. Deletes the `file_path` gate and `EPHEMERAL_DENYLIST`, which the optimizer still has to check today. |
| Optimizer adoption | decided | Direct `geecs_analysis` integration is required, including necessary loader/schema/dependency and evaluator changes. Preserve existing optimization semantics; validate supported live-source capabilities before enabling additional input kinds. |
| BCave legacy camera stitcher | decided | Retire `bcave_magspec_stitcher`; retain the separate line stitcher, BCave optimization measure and spectrum-waterfall path unless separately retired. |
| Image analysis stays data-agnostic | decided | Steps and measures see Frames only. A scan-N background or a calibration file is resolved by data-utils before execution. A test forbids filesystem/reader imports below the sinks; `geecs_data_utils.frames` is the allowed shared value model. |
| The resolver is GEECS-Data-Utils | decided | Shot→file resolution, `ShotSource`, `Frame`, background-from-scan live in data-utils, additively. The analysis package has no file reading. |
| Data-utils read-side convergence | tentative | Data-utils already grew `shot_join` (pure nearest-with-window join), `scan_grid` and shared shot-identity resolution since 09-09. ScanAnalysis's three-strategy file ladder converges on those rather than being relocated verbatim. `ScanLayout` / `ScanScalars` beside `ScanPaths` / `ScanData`; facade until the last caller moves. Not on the critical path. |
| Threads, not processes | tentative | Default thread pool; `ProcessPoolExecutor` opt-in per recipe. The portal and the worker both forbid fork. |
| Own `Frame` vs xarray | tentative | Own dataclass first (~150 lines); `to_xarray()` bridge if it ever pays off. |
| Where spec models live | tentative | `geecs_analysis.specs`, numpy-free, owns the unions; portal, MCP and the optimizer config import it for validation. Back into GEECS-Schemas only if a numpy-free consumer appears. |
| Heavy analyzers as plugins | tentative | HASO (wavekit), FROG (32-bit DLL), MagSpec DNN in their own small packages, registered by entry point; specs in the core so every recipe validates everywhere. They are Windows-only; the services box is Linux. |
| Zero-config analyzer kinds | open | `standard` (2D) and `trace` (1D): preprocessing only, both map to `measure: none`. `downramp_phase` / `phase_downramp`: work in progress, out of parity scope until one is chosen. `hi_res_mag_cam`: its optimizer config references a diagnostic that does not exist. `frog_spectral_phase`: cheap to port if GDD/TOD numbers are wanted. |
| Pure-Python FROG (`grenouille.py`) as the Linux path | open | Dead today, but the only Linux-capable FROG retrieval in the repo. Physics judgement is Sam's. |
| Automatic post-scan analysis, worker service, second machine | deferred | Decide with the non-scalar-over-PVA rollout and the new box. Compute is not the constraint (1–5 cores for 30 cameras at 1 Hz); network fan-out is, and each camera server's PVA gateway is the subscription point now that the central capture daemon is retired. |
| Live scalars as PVs (#744) | deferred | The optimizer already consumes live PVA frames in memory; a central publisher node is the same core plus ~200 lines of p4p. Engine-side telemetry identity is the open question. |
| Web vs desktop operator surface | done | The PySide6 console was deleted 2026-09-14; GeecsScanner is the web console. Not part of this effort. |
| Overlay kinds are a registry | decided (2026-09-24) | One file per kind: data class + draw function, registered like steps and measures; `draw_overlays` stops being an `isinstance` chain. Overlays are data in frame coordinates with stable ids, never artists. |
| Second overlay source = static references | decided (2026-09-24) | `scan.renderer.references` on v2 (`figure.references` on v3): marker, circle, box, segment, text with explicit frame-unit coordinates, producing the same primitives measures do. No data-derived decoration kinds for now; a target whose offset must be a scalar is a measure parameter. |
| No arbitrary matplotlib in YAML | decided (2026-09-24) | Recipes never name arbitrary matplotlib calls or carry Python: unvalidatable, unlistable by the editor, couples the corpus to matplotlib's API. The fixed kwarg groups (`imshow`, `plot`, `colorbar`, …) and named overlay kinds with free kwargs are the vocabulary, validated by rendering the preview. |

## The grade (09-06 audits, #803 tree)

Overall **C+**. The stack works in production, and its one good idea, a typed
document loaded through one factory into one result type consumed identically
by six packages, is genuinely good and survives. But the analysis core is
mostly not analysis.

| Package | Grade | Size (source, 2026-09-18) | Justification |
|---|---|---|---|
| ImageAnalysis | C+ | 15.0k | Good config spine and pure processing functions. Soft result contract, two parallel frameworks, side effects inside `analyze_image`, ~2.7k dead including a registered analyzer that cannot run. |
| ScanAnalysis | C | 9.5k | Queue, group loader, ConfigStore and web editor are solid. The 2.6k core is 60% I/O; renderer options travel through three models and a dict. |
| GEECS-Data-Utils (as foundation) | C+ | 13.2k | The right primitives (`ScanTag`, `ScanPaths`, `native_files`, `scan_stack`, `io`, now `shot_join`) but not the operations: shot→file resolution exists several times across the repo, s-file reading three, binning three. |
| GEECS-Schemas `analysis` | B- | 2.1k | Typed and docgen'd, but ~92 of ~181 leaves are never set, and pipeline-as-name-list plus separate sections is double bookkeeping that 8 of 50 files already violate. |

### Evidence

- **The config vocabulary is twice its use.** 50 diagnostics set 89 of ~181
  leaf paths. `renderer:` has 16 options; 2 files set 2, identically. Camera
  thresholding in practice is one boolean: 27 of 28 files use
  `constant / 0.0 / to_zero / invert: false`. `spatial_calibration` sits in a
  free-form metadata dict in 33 files and is read by nothing. 34 of 50 are
  `beam`; 6 of 15 kinds have zero configs; 17 distinct `pipeline:` sequences,
  all drawn from 10 steps.
- **Two sources of truth for one fact.** A step runs iff it is listed in
  `pipeline` and its section exists. Seven files carry sections not in
  `pipeline`; `UC_TopView` lists four steps with no section, silently skipped.
- **Rendering has five paths and three configs.** pyplot `base_render_image`;
  per-class `render_image` as a 2D staticmethod vs a 1D instance method; a
  legacy staticmethod with zero callers; object-API wrappers for the portal;
  ScanAnalysis stuffing `getattr(analyzer, "render_image")` into
  `result.render_function`. Options travel `RendererOptions → as_kwargs() →
  renderer_kwargs → Image2DRendererConfig(**kwargs)` rebuilt inside
  `try/except → defaults`. `xlabel`, `ylabel`, `colorbar_label` are read by
  nothing; `mode` is force-overwritten to `waterfall`; `cmap` is not passed to
  grid panels; `BeamAnalyzer.render_image` hard-codes a debug marker at
  (100, 100).
- **The result type is a bag.** `metadata: Dict[str, Any]`, `render_data:
  Dict[str, Union]`, `render_function: Callable`, `extra="allow"`.
  `scalars: Dict[str, float]` holds bools and `None`; `average()` type-sniffs
  every `render_data` value. Five analyzers write files from `analyze_image`
  gated on a string key; four keep per-shot state on the instance.
- **Data-layer work lives in analyzer classes.** `ScanAnalyzer` base: 408 of
  653 LOC are s-file locking/merging, ini parsing, column resolution.
  `SingleDeviceScanAnalyzer`: 310 LOC of shot→file joining, 137 of
  background-config mutation, 131 of its own binning. `ScanData` is
  constructed and only `.paths` is read. `if len(self.results) > 2` gates all
  post-processing. `LiveTaskRunner` reuses analyzer instances across scans, so
  a `{scan_dir}` placeholder resolved on scan N sticks for scan N+1.
- **Dead after #803's deletion pass, ~2.7k LOC**: `BCaveMagSpecStitcher` 282
  (calls a method that exists nowhere); `HIMGWithAveraging`; `grenouille.py`
  751 and `qwlsi.py` 718 imported only by their tests;
  `density_from_phase_analysis` duplicating `downramp_phase_analyzer`; the
  `use_injected_data` mode of `ScanAnalyzer` (its only consumer, the old
  evaluator, was replaced by the native optimizer); four heavyweight
  dependencies carried by dead code.

## What each client needs

| Client | Input | Call | Output | Constraints |
|---|---|---|---|---|
| Post-scan run (portal Analysis tab, MCP `run_scan_analysis`) | `ScanTag` + a diagnostic id | run over every shot of one device | per-bin figures + one summary in `analysis/ScanNNN/<id>/`, s-file columns, display-file list | never create `scans/ScanNNN/`; parallel over shots; `no_data` vs `failed` |
| Optimizer (`geecs_bluesky.optimization.measurements`) | per bin: timestamp-matched `ndarray` frames from bounded PVA monitors + the event rows | one document over a list of frames, `frames: per_bin \| per_shot`, reduce, `min_shots` | `dict[str, float]` per bin, `{measurement}.{scalar}` keys | no writes, no fork, worker thread; camera-only; must not be on `EPHEMERAL_DENYLIST` |
| Data portal | already-loaded frames (per shot or bin average); a validated document, possibly unsaved | processing selector and editor preview over the ephemeral seam; `run_analysis` for the Analysis tab | processed frame, scalars, overlays; object-API `Figure`; artefact list | guaranteed no I/O on the ephemeral path; thread-safe; Agg-safe; no pyplot state |

Every client wants three verbs: **measure** a frame, **draw** a result, **run**
a scan. Only the third writes. The optimizer and the portal already stop at
the ephemeral seam, which is exactly what `run(recipe, ArraySource)` replaces.

## The design

One package, four layers, one rule: **nothing below the sinks touches the
filesystem or pyplot.** Reading the world is GEECS-Data-Utils' job.

```
data-utils sources ─▶ steps ─▶ measures ─▶ Measurement ─▶ render ─▶ Figure
 (Frame out)       Frame→Frame  Frame→Measurement    │
                                                     ▼
                                                  run.py ──▶ sinks (the only layer that writes:
                                       aggregate · bin ·        analysis tree · scalars · s-file)
                                       average-frames-first
```

The portal's ephemeral path and the optimizer stop at `run.py` and `render`.
Post-scan runs attach sinks. The adapter behind `create_scan_analyzer` is the
only place the old world's contracts are honoured.

### 1. `Frame`: one container for traces and images

```python
@dataclass(frozen=True)
class Axis:   values: np.ndarray; unit: str = "px"; label: str = ""

@dataclass(frozen=True)
class Frame:
    data: np.ndarray          # (N,) or (H, W); float64 during processing
    axes: tuple[Axis, ...]    # len == data.ndim; sliced by roi, transformed by rotate
    shot: ShotMeta | None     # shot id, acq_timestamp, device; read-only provenance
```

Lives in data-utils, because the portal and the DAQ readback need the same
container. ROI slices the axis values along with the data, so a centroid
against `axis.values` is in global pixel coordinates without an offset, or in
millimetres if the axis is calibrated. This retires `roi_offset`,
`left_ROI`/`top_ROI` in metadata, and the unread `spatial_calibration` key.
Rotation transforms the coordinate arrays. The 45° diagonal projections have no
global frame today and still won't.

### 2. Steps: reorderable, repeatable, one file each

```python
# geecs_analysis/steps/median.py
class MedianSpec(StepSpec):
    step: Literal["median"] = "median"
    kernel: int = Field(5, ge=1, description="Kernel size in samples.")

@step(MedianSpec, ndim={1, 2})
def median(frame: Frame, spec: MedianSpec) -> Frame:
    return frame.replace(data=scipy.ndimage.median_filter(frame.data, spec.kernel))
```

The pipeline *is* the ordered list of typed step objects: any order,
duplicates allowed, no sections, no enum, no dispatcher. The union is
assembled from the registry, so a new step appears in validation, the editor
form and docgen on its own. Steps are pure `Frame → Frame` and act on axes
too. Composition is a fold.

| Step | ndim | Spec fields |
|---|---|---|
| `background_constant` | 1, 2 | `level` |
| `background_frame` | 1, 2 | `source` (file, or `from_scan` + `method`, resolved by data-utils before execution), `extra` |
| `roi` | 1, 2 | one `[lo, hi]` per axis, `units: index \| axis` |
| `clip_below` / `clip_above` / `zero_below` | 1, 2 | `level` |
| `median` / `gaussian` | 1, 2 | `kernel` / `sigma` |
| `interpolate` | 1 | `n`, optional `[lo, hi]` |
| `crosshairs` | 2 | list of `{center, size, thickness, angle}` |
| `circle_mask` | 2 | `center`, `radius`, `outside` |
| `vignette_radial` | 2 | `full_size`, `offset`, `coeffs` |
| `rotate` / `flip` | 2 | `degrees` / `axis` |

Twelve steps cover every sequence in the corpus, on the order of 700 lines
replacing ~3,000. Each ships invariant tests (axes match shape; a constant
background shifts the mean by exactly the level) and a golden comparison
against the old function.

### 3. Measures and `Measurement`

```python
@measure(BeamSpec, ndim=2)
def beam(frame: Frame, spec: BeamSpec, ctx: ShotContext) -> Measurement:
    stats = beam_profile_stats(frame.data, frame.axes)     # algorithms/, ported as-is
    return Measurement(scalars=stats.as_dict(), frame=frame,
                       overlays=[Projection.of(frame, 0), Projection.of(frame, 1),
                                 Marker("com", x=stats.x_com, y=stats.y_com)])

@dataclass
class Measurement:
    scalars: dict[str, float]      # numeric only, bare keys
    frame: Frame | None
    overlays: list[Overlay]        # Projection | Marker | VLine | HLine | Curve | Box | Text, each with an id
    products: list[Product]        # derived files as in-memory payloads; the runner decides whether to write
    notes: list[str]
```

One file per measure. `ctx` carries the aux row read-only. No
`metadata: Any`, no `render_function`, no `extra="allow"`. Composite loaders
(the stitcher across sibling devices, HASO's vendor reader) are *sources*, not
measures. Experiment-specific measures are a directory
(`measures/undulator/`), not a subclass.

### 4. Rendering

```yaml
figure:
  imshow:   {cmap: plasma, vmin: 0, vmax: 4000, interpolation: nearest}
  colorbar: {label: counts}
  axes:     {xlabel: x (mm), ylabel: y (mm), title: "{device} bin {bin}"}
  fig:      {figsize: [4, 4], dpi: 150}
  overlays:
    projection_x: {color: white, lw: 1}
    projection_y: {hidden: true}
    com:          {marker: "+", ms: 12, color: cyan}
```

Named kwargs dicts go straight to matplotlib and are validated by rendering
the editor's preview. Overlays are styled or hidden by the id the measure gave
them. `draw_frame` + `draw_overlays` handle every measure in the corpus; the
`@figure` hook covers bespoke panels (FROG's trace and phase). `figure:` is
the **per-frame draw**: it draws every single product (a shot, a bin) and
every panel inside a summary. Scan-level layouts are not chosen by
`frame.ndim`; they are listed, see the ruling below.

**Figure and summaries (ruling 2026-09-24, slice 1 of the surface arc).**
The maintainer's verdict from portal testing was that the backend became
elegant while the surface stayed the old one: the editor is the v2 form and
its renderer fields behave inconsistently (`figsize` sizes grid panels,
`figsize_inches` the square single canvas; the preview honours only
cmap/vmin/vmax; labels reach saved figures alone). The gap under it was
structural: rendering ONE frame and rendering the SUMMARY of a scan were one
option set. The ruling separates them:

- `figure:` = the per-frame draw (`imshow`/`pcolormesh`/`plot`/`colorbar`/
  `axes`/`fig` keyword groups + `overlays` by id, `references` when F2
  lands). One block, reused identically by the editor's preview, the
  per-shot and per-bin products, and every summary panel.
- `summaries:` = a **list of frozen kinds**, a discriminated union with a
  registry like `@step`/`@measure` (one file per kind: its option model in
  GEECS-Schemas, its layout function, what it consumes, its filename marker
  in `geecs_analysis.summaries`). A kind's options are its own; the frame
  dimensionality it draws is **validation, not selection** (the document
  refuses a waterfall on a camera recipe). The initial kinds, from the
  maintainer's answer on what a summary *is* (usually the bin-averaged
  representation, a grid for images or a waterfall for traces; one averaged
  frame for a noscan): `image_grid` (columns, panel size), `waterfall`
  (sort key/sigma/bounds, even spacing, colour scale rule, cmap, limits),
  `average`. No animation (gifs retired) and no line overlay until a corpus
  file asks — none sets `renderer.mode` today. The plan of products is
  kind-agnostic (per-unit singles plus ordered panels); the sink resolves
  each listed kind against the registry, draws it from the products it
  consumes and skips it silently when the run produced none (a grid on a
  noscan). File names keep today's markers, so the portal's parser and
  MCP's display-file contract are untouched.

The v2 document has neither block; `scan.renderer` is translated into both
(`compat.v2_render.figure_v2` / `summaries_v2`) so a v2 diagnostic draws
through the same kinds. The diverging image palette becomes a centred norm;
the trace waterfall's data-dependent rules live on the waterfall kind.

**Overlays (ruling 2026-09-24).** Custom rendering that is intuitive for
users was a stated reason for this refactor, and overlays are the common
customization. The structure: an overlay is *data in frame coordinates with a
stable id*, never a matplotlib artist; `FigureSpec.overlays` (the `overlays:`
block above, in the v3 shape) styles or hides it by id, kwargs pass straight
to matplotlib and the editor's preview validates them; and `draw_frame(ax, …)`
/ `draw_overlays(ax, …)` take the caller's axes, so a notebook composes
natively and adds its own matplotlib calls on top. What exists: the beam
measure emits `projection_x`, `projection_y` and `com`; bin grids average and
draw them; the noscan average omits shot overlays by the recorded legacy
convention (GEECS-Analysis `CLAUDE.md` § v2 compatibility, `v2_average`),
which is why the reviewed beam-average figure carried none. Whether the noscan
average should carry averaged overlays is an open question for the
maintainer: a deliberate departure from legacy figure parity, a small change
in `v2_average` plus that `CLAUDE.md` sentence. On v2 nothing exposes per-id
styling yet: `single_v2` builds the `FigureSpec` without `overlays`.

Two structural fixes are owed before more measures accumulate (only the beam
and line measures exist today): overlay kinds become a registry like `@step`
/ `@measure` (one file per kind holding the data class and its draw function;
today `draw_overlays` is an `isinstance` chain over `Marker` and `Projection`,
so every new primitive edits the renderer), and a second producer besides
measures. Almost every overlay derives from a measure; the remaining case is
**static reference geometry the user inputs**, such as a target position or
tolerance circle on a beam image. On the v2 document both live under
`scan.renderer` (`RendererOptions` gains `overlays` and `references`, mapped
into the `FigureSpec` by `single_v2` / `image_grid_v2` / `waterfall_v2`); the
`figure:` block is the v3 shape of the same thing:

```yaml
scan:
  renderer:
    references:
      - {kind: marker, id: target,    x: 312, y: 205, marker: x, color: white}
      - {kind: circle, id: tolerance, x: 312, y: 205, r: 15, fill: false}
    overlays:
      com: {marker: "+", color: cyan}
```

Reference kinds are the primitive names (`marker`, `circle`, `box`,
`segment`, `text`). References produce the same primitives the measures
produce, styled by the same ids and drawn by the same registry; frame axes
are origin-aware, so a reference in frame units survives the ROI. When the
*number* matters (offset from the target), the target is a measure parameter
and the measure emits the scalars and the overlay together, so they cannot
disagree. No data-derived decoration kinds (contours, masks) for now.
Refused: YAML that names arbitrary matplotlib calls or carries Python; the
fixed kwarg groups (`imshow`, `plot`, `colorbar`, …) plus named overlay kinds
with free kwargs give the same reach with a vocabulary the editor can list.

### 5. The document

Format 3, `geecs_schemas.analysis.AnalysisRecipe` (shipped 2026-09-24,
GEECS-Schemas 0.34.0), as the converter writes it for a corpus recipe:

```yaml
schema_version: 3
device: UC_VisaEBeam1                # finds the files; stems the products
output_name: UC_VisaEBeam1-left      # optional: s-file column prefix + output folder
scalar_suffix: _left                 # optional
description: ...                     # the human notes, kept
input: {kind: camera}                # or kind: line + loading/x_scale/x_unit/label/storage_dtype
                                     # + folder / file_tail / format when the files need it
inputs:                              # frames the source layer loads and binds by name
  camera_background: {path: "{scan_dir}/computed_background.npy", fallback_level: 0}
steps:
  - {step: background_frame, source: camera_background, alignment: samples}
  - {step: crosshair_mask, center: [285, 722], width: 108, height: 108, thickness: 10}
  - {step: roi, bounds: [[290, 623], [558, 891]]}
measure: {kind: beam}
scan: {priority: 1}                  # + average_frames_first, save
figure: {imshow: {vmin: 0}}
summaries:
  - {kind: image_grid}
  - {kind: average}
```

Two vocabularies meet in it, and the dependency direction decides where
each lives. The document's *frame* (naming, `input`, `inputs`, `scan`,
`figure`, the summary kinds' option models) is schema vocabulary, fully
typed in GEECS-Schemas. The *numerical* vocabulary (which steps and
measures exist, their parameters) is the core's registry and stays there
(the "where spec models live" decision); the schema carries a step or
measure as its registered name plus its parameters as written (`StepRef`,
`MeasureRef`), and `geecs_analysis.recipe.compile_recipe` binds them to the
registry, refusing unknown names, unknown parameters, undeclared or unused
frame bindings, and steps or a measure that do not process the input's
frames. The core's `FigureSpec` is the schema's `FigureStyle` made
immutable, so the field list exists once. Consumers call
`compile_document` / `figure_of` / `summaries_of` and never ask which
format they hold; `load_analysis_document` dispatches on `schema_version`
in every loader (the group loader, the config store, the portal, the
optimizer's resolver).

**Naming (corrected from the earlier "id = file stem" decision).** The
corpus has 20 of 37 core-served recipes whose file stem differs from the
device (`HTU/Amp2Input` reads `UC_Amp2_IR_Input`) and two recipes over one
device (`HTT-D-EBeam_Profiler`, one per namespace), so the output label
cannot be the file stem without renaming files, breaking s-file columns
and colliding. `device` finds the files and stems the product files;
`output_name` (default `device`) labels the columns and the output folder;
`scalar_suffix` ends the columns; the file stem remains the document ID
groups reference. `scan.device` became `input.folder`, `file_tail` and
`data_format` (as `format`) moved onto `input` with it: they say how the
files are found and read.

**Conversion is built on the compile output.** `compat.convert.to_v3`
compiles the v2 document with the adapter, writes the compiled steps and
measure back as registry references, carries the naming, input, runtime
and renderer facts, recompiles the result and compares it with the
source's compilation before accepting it. Both formats compile to the one
in-memory recipe and run through the one evaluator, so a converted recipe
is identical by construction; the in-suite comparison of a v2 diagnostic
against its converted recipe on the core route pins identical analysis
trees. What the v3 shape does not carry is reported, never dropped
silently: `bit_depth` (unused by the core), `gdoc_slot` (retired), and one
corpus quirk — `UNCLASSIFIED/HTT-C-ASSERTHighR` defines an ROI it never
runs, and the legacy beam analyzer (reproduced by the v2 adapter) still
offset its coordinates by that ROI's origin; the recipe has no inactive
sections, so its coordinates start at (0, 0) and the note says so. The
image palette's zero floor, which the v2 renderer imposed on every image,
is written out as `figure.imshow.vmin: 0` so the pure draw keeps it.

`scripts/analysis_convert_corpus.py` converts a configs tree in place
(37 of 50 on 2026-09-24: 34 beam, 3 line), leaving the 13 the core does
not serve as v2 (4 `magspec` with scan backgrounds, 2 `line_stitcher`,
2 `ict`, 2 `frog_retrieval`, `bcave_mag_opt`, `haso`, one `trace` with a
preprocessing-only ROI). The converted corpus lives on the configs branch
`analysis-recipe-v3` for the maintainer to test against; it merges to the
configs `main` when he is happy. The editor shows a recipe read-only until
slice 2 (the recipe form, with the figure preview through the same draw as
the run); slice 3 previews the summaries over N shots.

### 6. Algorithms

| Pile | Modules | LOC | Change |
|---|---|---|---|
| Port as-is | `basic_beam_stats`, `basic_line_stats`, `beam_slopes`, `polynomial_fit`, `bowtie_fit`, `axis_interpolation` | 1,438 | take coordinate arrays instead of `roi_offset`; otherwise untouched, bit-for-bit under the harness |
| De-leak, then port | `ict_algorithms`, `frog_dll_retrieval` + worker | 1,298 | drop the schema-spec and processing imports; the DLL path becomes a parameter the FROG plugin passes, not `GeecsPathsConfig` |
| Drop unless promoted | `grenouille`, `qwlsi` | 1,469 | test-only today; `grenouille` is the only Linux-capable FROG path (open) |

### 7. Touch lists, before and after

| Task | Today | After |
|---|---|---|
| New preprocessing step | 8+ files across 3 packages | 1 file + 1 test |
| New analyzer | 3 files + an `ax=`-honouring renderer + a denylist decision | 1 file + 1 test; optional `@figure` |
| New summary figure | 6 source files across 2 packages + 2 test files | 1 layout function + its filename marker |
| Analyzer in the optimizer | `run_document_ephemeral` + denylist check, camera-only | `run(recipe, ArraySource)`; explicit source/recipe capability validation replaces legacy kind guards |
| Analyzer in the portal | ephemeral module + denylist + signature sniffing | `run(recipe, ArraySource(frames))` + `draw` |

## Adoption without a flag day

The retained scan-analysis clients go through `create_scan_analyzer(diag)`.
The adapter preserves attributes `id`, `priority`; `run_analysis(scan_tag) ->
list[Path] | None`, raising `DataUnavailableWarning` for no data; `cleanup()`.
`gdoc_slot` is no longer part of the required runtime contract after the
upload hooks are removed.
An ~80-line adapter wraps `run_scan(recipe, scan_tag, sinks)` behind that
contract. Inside the factory, a recipe is served by the new stack if its
measure exists there, otherwise by the old wrapper. The queue,
the portal's Analysis tab and the MCP worker flip per recipe without
knowing. The ephemeral consumers (portal processing selector and editor
preview, the optimizer's `run_document_ephemeral`) get the same routing inside
`image_analysis.ephemeral` until they are pointed at `run(ArraySource)`
directly.

| Consumer | Changes on day one | Changes eventually |
|---|---|---|
| LiveWatch and Google Docs uploads | retire in a separate removal PR before core adoption | no replacement in this effort |
| Task queue and status YAML | remove upload imports, options and hooks; preserve execution and status semantics | retained for MCP |
| ConfigStore and web config editor | preserve the portal mount, drawer, saving and preview | support the new document schema before corpus conversion |
| Portal | none | optional: `run(ArraySource)` + `draw` directly; delete its own file ladder once data-utils has `ShotSource` |
| MCP | none | none in this effort |
| Optimizer | preserve existing execution via interim routing | required direct `run(ArraySource)` integration: loading, validation, scalar discovery, frame adaptation, evaluator and dependencies; capability checks replace legacy guards |
| Configs repo | none (v2 read through the adapter) | one-shot conversion to v3 when nothing reads v2 |

Rules:

1. **Switch and delete are never the same PR.** A route flips, runs a week in
   production, then a deletion PR removes the old path. This applies to
   replaced analysis routes; LiveWatch and uploads are intentionally retired
   without a replacement and have their own removal PR.
2. **Delete when the last consumer is gone, not when the replacement exists.**
   Checked with grep, not judgement.
3. **No new features in a code path that has a replacement in production.**
   Bug fixes go wherever the bug is.
4. **The vision is frozen between keystones.** New ideas go into the parking
   lot; the roadmap changes only when a keystone lands.

The one real cost of keeping the queue: one task per recipe per scan, so two
recipes on the same camera load every frame twice. Rare in the corpus; if it
starts to matter, that is the fact that justifies replacing the queue.

## Branch and release workflow

`codex/analysis-refactor` is the integration branch for this effort. Focused
child branches start from it and return through reviewed PRs; they do not
target master while the first milestone is incomplete. Keep the integration
branch alive and periodically merge master forward into it. Existing master
behavior is unaffected until a milestone is ready for promotion.

The first promotion PR delivers the complete beam/line milestone: core and
comparison harness (A1), DataPortal scan execution, editing, previews and
saved outputs (A2), and direct optimizer adoption (O1). LiveWatch/upload
retirement (R0) is a separately reviewed piece of that integration. Required
gates are reference-data comparisons, retained-consumer tests, operator review
of representative beam and waterfall outputs, and a live optimization check.
Unported diagnostics retain working routes. FROG/HASO validation must not hold
the first promotion hostage; their migrations land as later complete pieces.

The maintainer merges promotion PRs into master. Each constituent PR receives
the repository's normal review and checks; the promotion records those reviews
and the combined validation, following the bulk-integration review exception
in `CONTRIBUTING.md`. Do not delete replaced analysis paths in that promotion:
observe the new routes in production for a week, then remove obsolete paths
in a follow-up PR. Intentional retirement of LiveWatch/uploads and the broken
BCave camera stitcher does not require a replacement observation period.

## Roadmap

A core-adoption keystone is done when it is in production behind the factory,
not when the code exists. The retirement step is done when its entry points
and dependencies are removed and retained consumers still pass their checks.
Rows below are work packages on the integration branch; A1/A2/O1 are promoted
together as the first complete milestone. Their original estimates are
historical rough sizes. The current first-milestone budget is 7–12 focused
agent-days, plus data/physics review, a live optimization check and the
production observation period; re-estimate after the differential baseline.

| # | Keystone | Unlocks | Can delete afterwards | Agent work | External gate |
|---|---|---|---|---|---|
| R0 | Retire LiveWatch GUI, `LiveTaskRunner` and Google Docs upload integration; remove launchers, upload hooks/options and the analysis-side LogMaker dependency; accept old v2 upload fields inertly until corpus conversion | explicit portal and MCP analysis without watcher/upload dependencies | watcher/upload-only tests, configuration and dependencies after checking all callers; retain queue, status readers, group loader and editor | size after caller audit | portal editor/preview and explicit-run smoke checks; MCP queue/status tests |
| 0 | Dead-code chore: BCave stitcher, `grenouille`/`qwlsi` (unless promoted), HIMG, orphaned data files, `use_injected_data` | clarity; ~3k lines gone | itself | ½ day | none |
| A1 | Core: `Frame` (in data-utils), 12 steps, `beam` + `line`, `Measurement`, `draw` + `single`, v2 adapter, differential harness | portal processing selector, editor preview and the optimizer on the new stack | nothing yet | 1–2 days | one archived scan as golden fixture; review of `Frame`, step names, overlay set |
| O1 | GeecsBluesky direct adoption: load/validate recipes, discover scalar outputs, adapt live frames and call `run(ArraySource)`; update dependencies and input capability checks | optimizer uses the new core directly; required for the beam/line milestone | old optimizer imports and ephemeral routing once no longer used there | size after integration audit | evaluator parity tests plus a live optimization run; preserve timing/shot association, reductions, minimum counts and no-writes behavior |
| A2 | `run_scan`, `grid`/`waterfall`/`animation`, analysis-tree + scalar + s-file sinks, the factory adapter and routing | Analysis tab and MCP on the new stack for every `beam`/`line` recipe (37 of 50) | after a week: the 2D/1D wrappers' beam path is unreachable | 2 days | one archived day end-to-end; s-file columns byte-identical for beam recipes |
| F1 | Colourbar sized from the image axes in `single` (`draw_frame`) and `image_grid`'s shared colourbar; rendered before/after in the PR | every core beam figure looks right; lands before promotion (see "Still required") | nothing | ½ day | operator look at the before/after |
| F2 | Overlay-kind registry (`Marker`, `Projection`, `Circle`, `Box`, `Segment`, `Text`); `overlays` (per-id style) and `references` on `RendererOptions` under `scan.renderer`, mapped into `FigureSpec` by the v2 renderers; one editor section (overlays + references) with the live preview as validator | operators add reference geometry without code; developers add a kind in one file | the `isinstance` chain in `draw_overlays` | 1–2 days, two PRs (core + schema, then editor) | preview renders every kind; the maintainer confirmed the set (measure overlays + static references) covers the cases in mind |
| A3 | Long tail: `ict`, `magspec`, `frog_retrieval`, stitcher as a source, HASO as a source, `bcave_mag_opt`; plugins for the Windows-only ones | each recipe flips as it passes the harness | old ImageAnalysis and ScanAnalysis cores when the last route flips; **this file** | ½ day each | Windows machine, vendor SDK, real data per diagnostic; physics sign-off |
| A4 | Data-utils read side, additive: `ScanLayout`, `ScanScalars`, `ShotSource` converging on `shot_join` / `scan_stack` | portal deletes its ladder; `ScanData` goes | `ScanPaths` facade after the last caller moves | 2 days | none; spread across small PRs |
| A5 | Optional: corpus conversion to v3; delete the v2 adapter | one config vocabulary | v2 adapter | 1 day | none |

## Deferred, and the parking lot

- **Automatic post-scan analysis and a worker service.** LiveWatch is being
  retired; automatic triggering is not an acceptance requirement of this
  refactor. A future service could subscribe to the engine's run-stop
  documents, or use Redis lists per capability with one worker per machine
  (a Windows box runs a user-session worker, no admin needed). Decide
  separately; explicit portal and MCP runs remain supported.
- **Live scalars as PVs (#744).** The optimizer already measures live PVA
  frames in memory; a publisher node is the same core plus ~200 lines of p4p.
  Live scalars and post-scan figures are different products of the same
  recipe; the post-scan run recomputes rather than reads them, which is the
  online/offline drift test. Open question is engine-side: how the node's PVs
  get a device identity in the save set.
- **Client-side live analysis off PVA.** Fifty lines with the new core; right
  for ad-hoc viewing, wrong as a source of truth.
- **Data-utils beyond the read side.** `scans_database`, `modeling/ml`,
  `plotting_utils` out of the package; the producer side collected into one
  named module. Audit `doc_id_lookup` and the standalone LogMaker package
  during R0: remove anything used only by the retired integration, and record
  any surviving external consumers before deleting shared code. The existing
  GeecsLogbook service and the portal's send-to-logbook feature stay.
- **Parking lot.** Load-once-measure-many for multiple recipes on one camera;
  a typed run manifest replacing the status YAML; a `TiledSource`; xarray.

## Acceptance

The safety net is **differential testing against the live old stack over the
real corpus**: every recipe runs on both stacks against the same fixture
scans, and scalars (exact for identical steps, tolerance-tagged where an
implementation deliberately changes), figure filenames and s-file columns are
compared. It runs continuously while both stacks are live.

Kept green throughout: the task-queue execution tests and the
`analysis_status` contract; `test_renderer_output_names`; the
scan-folder-creation invariant tests; the ConfigStore and editor tests; the
shot→file join tests, moved with their code; the processing tests, ported as
step goldens; the beam and line centroid-accuracy tests; ImageAnalysis 2.3.0's
`test_emitted_scalars` (declared scalars pinned against synthetic executions),
which becomes the measure-side scalar contract. New: step invariants, a
nothing-is-written test for every measure kind including the plugins, the
`ax=` contract test, and an import test that nothing below the sinks touches
the filesystem.

R0 also verifies that portal editor mounting, drawer saves and unsaved-document
previews still work; portal and MCP explicit runs still produce artifacts;
and retained execution paths no longer import LiveWatch or Google Docs upload
code. Remove upload-only tests with the retired feature; preserve queue
claims, heartbeat and status semantics. Do not delete the ScanAnalysis package
wholesale while it still owns the queue, group loader or editor.

## Honest costs

- **Two backends behind one factory for a while.** Cheap, but the harness
  must run on every PR that touches either.
- **The join ladder and the s-file merge hide behaviour.** Converge on
  data-utils' primitives with the tests; do not improve them in the same PR.
- **The overlay set will be wrong the first time.** `@figure` is the escape
  hatch so a new panel never forces a schema change.
- **The long tail is graded B until real data validates it.** HASO, FROG and
  MagSpec can be clean in structure and still not be checkable without the
  SDK, the DLL and a day of data.
- **The differential harness is the hard part, not the rewrite.** Spend A1's
  first hours on the fixture scan and the comparison before porting a step.

### R0 implementation — 2026-09-23

LiveWatchGUI, LiveTaskRunner, upload hooks and notebooks, the Qt screenshot
generator, and the LiveWatch-only DocIDLookup helper are removed. ScanAnalysis
no longer depends on LogMaker, Qt or watchdog. ConfigStore/editor, unsaved
portal previews, group loading, task claims/status files and explicit MCP
runs remain. Legacy v2 upload fields are accepted inertly and hidden in the
editor without losing authored values. The standalone LogMaker package is
retained; only its analysis integration is retired. Automatic post-scan
execution remains a future service-design decision.

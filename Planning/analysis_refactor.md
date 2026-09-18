# Analysis refactor: ImageAnalysis + ScanAnalysis core → `geecs_analysis`

*Planning note (see `Planning/README.md`). Delete this file in the PR that
deletes the old analysis cores (roadmap keystone A3's last route flip);
anything still load-bearing moves to `GEECS-Analysis/CLAUDE.md` first.*

Drafted 2026-09-06 from four parallel code audits of the #803 tree plus a
field-by-field census of the 61-file analysis-config corpus; discussed over
2026-09-06..08; refreshed 2026-09-18 against master (native-Bluesky rebuild,
console deletion, capture-daemon retirement, logbook). Status: **direction
settled, no code written.** Owner: Sam.

---

## Where we stand

**Refactor the analysis core, not the orchestration.** ImageAnalysis and the
core of ScanAnalysis (`base.py`, `analyzers/common/`, `analyzers/renderers/`)
are replaced by one new package. The task queue, `LiveTaskRunner`, the
LiveWatch Qt GUI, gdoc upload, the group loader, `ConfigStore` and the web
config editor stay as they are, bug fixes only.

**In-repo, not a fork.** The new package lands as `GEECS-Analysis/`, on normal
feature branches through the usual PR ritual. Every existing consumer
(LiveWatch, the portal, MCP, the optimizer) reaches it through the interface
it already calls; the old cores are deleted when the last recipe's route
flips.

**Automatic analysis is deferred.** LiveWatch keeps doing that job. The data
*architecture* (folder layout, file formats, the s-file) is a separate
conversation and is not touched here; how code finds and reads it is in scope.

**Timing is agent-days plus external gates** (a fixture scan on the share, a
live optimization run, the Windows machine for the vendor SDKs), not calendar
weeks.

## Decisions

| Topic | Status | Where we landed |
|---|---|---|
| One package for image + scan analysis | decided | The split was organisational; the seam between them (`render_function` on the result, `output_name` threading, in-place mutation of the camera config for scan backgrounds, `getattr` attribute injection from the factory) is where the complexity lives. |
| Hard fork vs in-repo | decided | In-repo, adopted per recipe behind the factory. The whole-repo fork was dropped: master does not stand still, and every consumer is in-repo. |
| LiveWatch, task queue, status YAML, gdoc | decided | Kept, untouched, served through the adapter. Revisit with automatic analysis. (Reversed from the 09-06 draft, which dropped them.) |
| Qt ConfigFileGUI | done in #803 | Deleted with the re-export shims, the model aliases and the v1 converter. The web editor is the one editor. |
| Pipeline = ordered, repeatable list of typed steps | decided | No canonical order. Any order, duplicates allowed (two medians; a clip before and after a filter). Each step declares `ndim`; the loader validates the list. |
| Steps: one file, spec + pure function; variants are separate steps | decided | No method enums with conditional fields: `background_constant`, `background_frame`, `clip_below`, `clip_above` instead of `method × value × mode × invert`. |
| 1D and 2D share one container | decided | `Frame` = ndarray + typed axes + provenance. Six steps shared, one 1D-only, five 2D-only. ROI slices the axes, so positional stats stay in global coordinates without an offset. |
| Rendering: matplotlib pass-through + overlays by id | decided | 90% of wants are matplotlib kwargs: named dicts (`imshow`, `colorbar`, `axes`, `fig`) validated by the editor's preview. Measures declare overlays by id; recipes style or hide them. `@figure` hook for bespoke panels. Rendering stays in the package because the logbook needs figures. |
| Naming | decided | `id` (file stem; scalar prefix and output dir), `device` or `devices` (finds files), optional `scalar_suffix`. Replaces `output_name`, `metric_suffix`, `scan.device`, `output_label`. |
| Per-bin mode | decided | Kept as `average_frames_first: true`; the optimizer's `frames: per_bin` is the same thing. Implemented once in `run`. |
| Side effects | decided | Measures return derived products in memory; whether they are written is the runner's decision. Deletes the `file_path` gate and `EPHEMERAL_DENYLIST`, which the optimizer still has to check today. |
| Image analysis stays data-agnostic | decided | Steps and measures see Frames only. A scan-N background or a calibration file is resolved by data-utils before execution. A test asserts no `pathlib` / `geecs_data_utils` import below the sinks. |
| The resolver is GEECS-Data-Utils | decided | Shot→file resolution, `ShotSource`, `Frame`, background-from-scan live in data-utils, additively. The analysis package has no file reading. |
| Data-utils read-side convergence | tentative | Data-utils already grew `shot_join` (pure nearest-with-window join), `scan_grid` and shared shot-identity resolution since 09-09. ScanAnalysis's three-strategy file ladder converges on those rather than being relocated verbatim. `ScanLayout` / `ScanScalars` beside `ScanPaths` / `ScanData`; facade until the last caller moves. Not on the critical path. |
| Threads, not processes | tentative | Default thread pool; `ProcessPoolExecutor` opt-in per recipe. The portal and the worker both forbid fork. |
| Own `Frame` vs xarray | tentative | Own dataclass first (~150 lines); `to_xarray()` bridge if it ever pays off. |
| Where spec models live | tentative | `geecs_analysis.specs`, numpy-free, owns the unions; portal, MCP and the optimizer config import it for validation. Back into GEECS-Schemas only if a numpy-free consumer appears. |
| Heavy analyzers as plugins | tentative | HASO (wavekit), FROG (32-bit DLL), MagSpec DNN in their own small packages, registered by entry point; specs in the core so every recipe validates everywhere. They are Windows-only; the services box is Linux. |
| Zero-config analyzer kinds | open | BCave stitcher: drop (cannot run). `standard`: free (`measure: none`). `downramp_phase` / `phase_downramp`: work in progress, out of parity scope until one is chosen. `hi_res_mag_cam`: its optimizer config references a diagnostic that does not exist. `frog_spectral_phase`: cheap to port if GDD/TOD numbers are wanted. |
| Pure-Python FROG (`grenouille.py`) as the Linux path | open | Dead today, but the only Linux-capable FROG retrieval in the repo. Physics judgement is Sam's. |
| Automatic post-scan analysis, worker service, second machine | deferred | Decide with the non-scalar-over-PVA rollout and the new box. Compute is not the constraint (1–5 cores for 30 cameras at 1 Hz); network fan-out is, and each camera server's PVA gateway is the subscription point now that the central capture daemon is retired. |
| Live scalars as PVs (#744) | deferred | The optimizer already consumes live PVA frames in memory; a central publisher node is the same core plus ~200 lines of p4p. Engine-side telemetry identity is the open question. |
| Web vs desktop operator surface | done | The PySide6 console was deleted 2026-09-14; GeecsScanner is the web console. Not part of this effort. |

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
| Post-scan run (LiveWatch queue, portal Analysis tab, MCP `run_scan_analysis`) | `ScanTag` + a diagnostic id | run over every shot of one device | per-bin figures + one summary in `analysis/ScanNNN/<id>/`, s-file columns, display-file list | never create `scans/ScanNNN/`; parallel over shots; `no_data` vs `failed` |
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
| `clip_below` / `clip_above` | 1, 2 | `level` |
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
    projection: {color: white, lw: 1}
    com:        {marker: "+", ms: 12, color: cyan}
    beam_box:   {hidden: true}
```

Named kwargs dicts go straight to matplotlib and are validated by rendering
the editor's preview. Overlays are styled or hidden by the id the measure gave
them. `draw_frame` + `draw_overlays` handle every measure in the corpus; the
`@figure` hook covers bespoke panels (FROG's trace and phase). Layouts:
`single`, `grid`, `waterfall`, `animation`, chosen by `frame.ndim`. Output
filenames keep today's shapes so the portal's parser and LiveWatch's
display-file list are untouched.

### 5. The document

```yaml
schema_version: 3
# id is the file stem: scalar prefix + output dir
device: UC_TopView                   # or devices: [A, B, C] for a stitched source
scalar_suffix: _left                 # optional
input: {kind: camera}                # camera | trace (+ loader options for traces)
steps:
  - {step: background_constant, level: 5.0}
  - {step: roi, x: [0, 650], y: [350, 650]}
  - {step: clip_below, level: 0}
  - {step: median, kernel: 3}
  - {step: clip_below, level: 20}    # repeats are fine
measure: {kind: beam, compute_slopes: false}
scan: {priority: 10, average_frames_first: false, save: true, gdoc_slot: 0}
figure: {imshow: {cmap: plasma}}
```

Roughly 70–75 leaf fields cover the corpus, against ~181. `priority` and
`gdoc_slot` stay because the queue reads them. During the transition the new
package reads today's v2 files through an in-memory adapter; the corpus
converts once, at the end.

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
| Analyzer in the optimizer | `run_document_ephemeral` + denylist check, camera-only | `run(recipe, ArraySource)`; the denylist and the camera-only restriction go |
| Analyzer in the portal | ephemeral module + denylist + signature sniffing | `run(recipe, ArraySource(frames))` + `draw` |

## Adoption without a flag day

Everything that calls scan analysis today goes through
`create_scan_analyzer(diag)` and then uses six things on what it gets back:
attributes `id`, `priority`, `gdoc_slot`; `run_analysis(scan_tag) ->
list[Path] | None`, raising `DataUnavailableWarning` for no data; `cleanup()`.
An ~80-line adapter wraps `run_scan(recipe, scan_tag, sinks)` behind that
contract. Inside the factory, a recipe is served by the new stack if its
measure exists there, otherwise by the old wrapper. LiveWatch, the queue,
gdoc, the portal's Analysis tab and the MCP worker flip per recipe without
knowing. The ephemeral consumers (portal processing selector and editor
preview, the optimizer's `run_document_ephemeral`) get the same routing inside
`image_analysis.ephemeral` until they are pointed at `run(ArraySource)`
directly.

| Consumer | Changes on day one | Changes eventually |
|---|---|---|
| LiveWatch, task queue, gdoc | none | none in this effort |
| Portal | none | optional: `run(ArraySource)` + `draw` directly; delete its own file ladder once data-utils has `ShotSource` |
| MCP | none | none in this effort |
| Optimizer | none | `run(ArraySource)` in `evaluate_bin`; delete the denylist and camera-only checks |
| Configs repo | none (v2 read through the adapter) | one-shot conversion to v3 when nothing reads v2 |

Rules:

1. **Switch and delete are never the same PR.** A route flips, runs a week in
   production, then a deletion PR removes the old path.
2. **Delete when the last consumer is gone, not when the replacement exists.**
   Checked with grep, not judgement.
3. **No new features in a code path that has a replacement in production.**
   Bug fixes go wherever the bug is.
4. **The vision is frozen between keystones.** New ideas go into the parking
   lot; the roadmap changes only when a keystone lands.

The one real cost of keeping the queue: one task per recipe per scan, so two
recipes on the same camera load every frame twice. Rare in the corpus; if it
starts to matter, that is the fact that justifies replacing the queue.

## Roadmap

A keystone is done when it is in production behind the factory, not when the
code exists.

| # | Keystone | Unlocks | Can delete afterwards | Agent work | External gate |
|---|---|---|---|---|---|
| 0 | Dead-code chore: BCave stitcher, `grenouille`/`qwlsi` (unless promoted), HIMG, orphaned data files, `use_injected_data` | clarity; ~3k lines gone | itself | ½ day | none |
| A1 | Core: `Frame` (in data-utils), 12 steps, `beam` + `line`, `Measurement`, `draw` + `single`, v2 adapter, differential harness | portal processing selector, editor preview and the optimizer on the new stack | nothing yet | 1–2 days | one archived scan as golden fixture; review of `Frame`, step names, overlay set |
| A2 | `run_scan`, `grid`/`waterfall`/`animation`, analysis-tree + scalar + s-file sinks, the factory adapter and routing | LiveWatch, Analysis tab, MCP on the new stack for every `beam`/`line` recipe (37 of 50) | after a week: the 2D/1D wrappers' beam path is unreachable | 2 days | one archived day end-to-end; s-file columns byte-identical for beam recipes |
| A3 | Long tail: `ict`, `magspec`, `frog_retrieval`, stitcher as a source, HASO as a source, `bcave_mag_opt`; plugins for the Windows-only ones | each recipe flips as it passes the harness | old ImageAnalysis and ScanAnalysis cores when the last route flips; **this file** | ½ day each | Windows machine, vendor SDK, real data per diagnostic; physics sign-off |
| A4 | Data-utils read side, additive: `ScanLayout`, `ScanScalars`, `ShotSource` converging on `shot_join` / `scan_stack` | portal deletes its ladder; `ScanData` goes | `ScanPaths` facade after the last caller moves | 2 days | none; spread across small PRs |
| A5 | Optional: corpus conversion to v3; delete the v2 adapter | one config vocabulary | v2 adapter | 1 day | none |

## Deferred, and the parking lot

- **Automatic post-scan analysis and a worker service.** LiveWatch does this
  today. The replacement is a small service subscribing to the engine's
  run-stop documents, or Redis lists per capability with one worker per
  machine (a Windows box runs a user-session worker, no admin needed). Decide
  with the non-scalar-over-PVA rollout and the new box.
- **Live scalars as PVs (#744).** The optimizer already measures live PVA
  frames in memory; a publisher node is the same core plus ~200 lines of p4p.
  Live scalars and post-scan figures are different products of the same
  recipe; the post-scan run recomputes rather than reads them, which is the
  online/offline drift test. Open question is engine-side: how the node's PVs
  get a device identity in the save set.
- **Client-side live analysis off PVA.** Fifty lines with the new core; right
  for ad-hoc viewing, wrong as a source of truth.
- **Data-utils beyond the read side.** `scans_database`, `modeling/ml`,
  `plotting_utils` out of the package; `doc_id_lookup` with the logbook's
  export phase; the producer side collected into one named module.
- **Parking lot.** Load-once-measure-many for multiple recipes on one camera;
  a typed run manifest replacing the status YAML; a `TiledSource`; xarray.

## Acceptance

The safety net is **differential testing against the live old stack over the
real corpus**: every recipe runs on both stacks against the same fixture
scans, and scalars (exact for identical steps, tolerance-tagged where an
implementation deliberately changes), figure filenames and s-file columns are
compared. It runs continuously while both stacks are live.

Kept green throughout, unchanged: the task-queue tests and the
`analysis_status` contract; `test_renderer_output_names`; the
scan-folder-creation invariant tests; the ConfigStore and editor tests; the
shot→file join tests, moved with their code; the processing tests, ported as
step goldens; the beam and line centroid-accuracy tests; ImageAnalysis 2.3.0's
`test_emitted_scalars` (declared scalars pinned against synthetic executions),
which becomes the measure-side scalar contract. New: step invariants, a
nothing-is-written test for every measure kind including the plugins, the
`ax=` contract test, and an import test that nothing below the sinks touches
the filesystem.

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

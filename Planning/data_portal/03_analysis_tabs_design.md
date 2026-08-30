# Analysis tabs — backend decomposition

*The scouting/design doc for the `feat/analysis-tabs` arc: every surface
in the accepted mockup (the "GEECS Analysis Tabs" artifact, v4,
2026-08-30) mapped to its backend primitive with a status, grounded in
a code-level recon of the actual implementations (not the docs — two of
which turned out to be wrong, see Findings). Statuses:*

- ***exists*** — *use as-is.*
- ***extend*** — *right implementation, needs additive capability.*
- ***improve*** — *right concept, implementation predates current
  standards (owner's own diagnosis): rewrite or extract, keeping the
  vocabulary.*
- ***new*** — *doesn't exist anywhere.*

*Architecture recap (settled): three layers — pure primitives
(data-utils / ImageAnalysis) → JSON endpoints (portal one-liners) →
vendored-Plotly tabs. The rail holds the shared selection (scan
identity, named filters); popups edit, the rail summarizes; every view
is reproducible in a notebook by calling the same primitive.*

## Recon findings that shape the plan

1. **`ScanData.binned_scalars` has almost no consumers.** Grep: its
   definition, two error strings in `plotting_utils`, one
   share-dependent integration test, and one docs example notebook
   (`docs/geecs_data_utils/examples/basic_usage.ipynb` —
   `set_binning_config` → `binned_scalars` → `plot_binned`).
   `GEECS-Data-Utils/CLAUDE.md`'s claims that ScanAnalysis renderers
   consume it and that a `sd.bin(config)` method exists are both
   **false** (ScanAnalysis bins independently; the method was never
   written — this PR fixes both). Owner confirms (2026-08-30): the
   binning that is actually exercised in production is the *image-side*
   per-bin path in ScanAnalysis, and most of this layer is in light use
   generally — so the scalar-binning implementation can be rewritten
   freely; the `ScanData` delegate keeps the notebook working and W1c
   refreshes it to the new API.
2. **The s-file is read in three independent places** with copies of
   the `s{number}.txt` path convention: `scan_data.py` (`read_csv` via
   ScanPaths), `ScanAnalysis/base.py:296` (its own `read_csv`, ignoring
   the ScanData it constructed 90 lines earlier), and the append/merge
   path's re-read. There is no standalone `read_sfile(path)`. (The path
   *convention* additionally appears in two writer-side sites —
   `copy_fresh_sfile_to_analysis` and `tiled_export`'s s-file writer —
   which a read-side consolidation deliberately leaves alone.)
3. **`ScanData.from_date` defaults are hostile to a request path**:
   `append_paths=True` costs N_devices+1 SMB directory listings before
   any data; the constructor requires config.ini + mounted share +
   existing folder. Web callers must pass `append_paths=False`, and the
   union-frame primitive should not require ScanPaths at all when it
   already holds paths from `resolve_scan_folder`.
4. **The ImageAnalysis purity boundary is real but conventional**: the
   factory (`create_image_analyzer`) and both processing pipelines
   (`apply_camera_processing_pipeline`, `apply_line_processing_pipeline`)
   are pure and tested. Six analyzers write from `analyze_image`/
   `load_image`; five gate on `auxiliary_data["file_path"]` or
   constructor state (omit them → pure), **one — the HASO processor —
   writes five sidecars unconditionally from `load_image`** and cannot
   be made ephemeral without editing it. The gate is an undocumented
   convention; the ephemeral runner must make it a contract.
5. **`ImageAnalyzerResult.render_function` is a live callable** — it
   cannot cross a JSON boundary. The web result envelope keeps
   `scalars`/`line_data`/`processed_image` and drops render machinery.
6. **The event↔s-file bridge half-exists**:
   `tiled_export.build_legacy_scalar_dataframe` (pure, tested) maps an
   event frame into legacy s-file shape (`Bin #`, `Shotnumber`, legacy
   headers). No join exists anywhere; the two column namespaces share
   zero names — which is exactly why union-with-provenance (no
   reconciliation) is the right model.
7. **Diagnostic-config resolution has two competing root-resolution
   paths** (`load_diagnostic` reads `ScanPaths.paths_config` directly;
   `load_camera_config` goes through the `ConfigDirManager` with env
   overrides). Passing `config_dir=` explicitly bypasses both — the
   portal should do that from its own config.
8. **Per-bin frame averaging exists three times**:
   `SingleDeviceScanAnalyzer.average_data` (`np.mean` — one NaN pixel
   poisons a bin; `_postprocess_noscan` delegates to it), the
   hand-rolled `np.mean` in
   `analyzers/Undulator/HIMG_with_average_saving.py:70`, and
   `ImageAnalyzerResult.average` (`nanmean` — the correct one).

## Decomposition — mockup surface → primitives

### The union frame (rail: provider badges; Plot tab: provenance chips)

| Primitive | Status | Notes |
|---|---|---|
| `read_sfile(path) -> DataFrame` | **new** (tiny) | The standalone s-file reader all three duplicate sites converge on: one `read_csv(sep="\t")` + the `s{number}.txt` convention + dtype tolerance. Home: `geecs_data_utils/data/` (or `io/`). ScanAnalysis `base.py` and `ScanData.load_scalars` delegate to it (their PRs can trail). |
| `scan_frame(detail, scan_folder) -> ProvenancedFrame` | **new** | THE union-with-provenance primitive: event columns (from `RunDetail.data`) ∪ s-file columns (via `read_sfile`), joined on shot identity — **`Shotnumber == scan_event_index`, both
  1-based** (the plans increment before emitting; the legacy exporter
  writes `Shotnumber = 1..N` over the same rows, and every existing
  consumer maps them as equal — verified against
  `plans/step_scan.py`, `tiled_export.py:98`, `analysis/camera.py`) —
  each column tagged `run`/`sfile` (later `computed`). No name reconciliation, duplicates allowed. Provenance rides as a parallel `dict[str, str]` (a DataFrame `attrs` entry is fragile across copies — carry explicitly). Home: `geecs_data_utils`. |
| `RunDetail.data` | **exists** | The event side, already served by the portal's `CachingScanCatalog`. |
| `display_name` / `geecs_scalar_headers` | **exists** | Pretty names for the pick list (the M5 "aliases" deferral) when wanted. |

### Filters (rail chips + popup editor)

| Primitive | Status | Notes |
|---|---|---|
| `apply_row_filters` + `RowFilterSpec` | **extend** | Clean, tested, AND-only, 6 comparison ops. Extend to the mockup's model: named groups of AND conditions, OR across groups, per-condition `within/outside` (bounds pair), group enable flags. Shape: a small Pydantic model (`RowFilterGroup`/`RowFilters`) that *lowers* to the existing tuple specs per condition — the proven kernel stays. NaN policy must become explicit — today it is inconsistent per operator (comparisons drop NaN rows, but `!=` keeps them). |
| Live pass-count | **new** (trivial) | `mask.sum()` endpoint over the same primitive. |
| top-N-per-bin | **deferred** | Rail-evicted per ruling; returns later as a filters-popup option (`top_n_per_bin(frame, bin_col, value_col, n, desc)` — trivial when wanted). |

### Plot tab (per-shot ⇄ binned, multi-Y, gear settings)

| Primitive | Status | Notes |
|---|---|---|
| `BinningConfig` | **exists** | The vocabulary is right (center/err_low/err_high; std/stderr/mad/iqr/percentile; `Bin #` vs numeric edges/width/quantile binning). Keep it. |
| `bin_frame(frame, cfg) -> DataFrame` | **improve** (rewrite) | Replace the 145-line stateful `binned_scalars` property with a pure function honoring the same config and output schema, minus the warts: no `id(df)` cache key, no `("count","center")` shape special-case (make count a proper 3-col level or a separate series), label-aligned error assignment, vectorized `mad`, `value_cols` default that excludes `Shotnumber`/bin col (as the config docstring already promises), unit tests for every err mode. `ScanData.binned_scalars` becomes a thin delegate (and `sd.bin(cfg)` finally exists, making the CLAUDE.md true). Freedom granted by finding 1. |
| Per-shot scatter data | **exists** | The filtered frame's columns, serialized — no computation. |
| `plottable_columns` / `numeric_series` | **exists** | The shared pick-list rule (portal + console B4 both on it since #715). Needs a provenance-aware wrapper over the union frame. |
| Plotly view | **new** | The vendored `plotly.min.js` (version-pinned, committed — the approved doctrine amendment lands in the portal CLAUDE.md in that PR) + the tab JS. Per-series y-axes, `editable: true`, PNG/SVG export config. |
| "show the code" | **new** (trivial) | A template string per endpoint rendering the exact primitive call. |

### Images tab (per-shot ⇄ per-bin averages, processing selector)

| Primitive | Status | Notes |
|---|---|---|
| Per-shot serving | **exists** | The portal's resource layer + prefetch cache (0.5.0). |
| `average_frames(frames) -> ndarray \| None` | **improve** (extract) | One pure `nanmean` frame-averager with the homogeneity guard, extracted so the three divergent copies (`average_data`'s `np.mean` — which `_postprocess_noscan` delegates to — the hand-rolled `np.mean` in `HIMG_with_average_saving.py:70`, and `ImageAnalyzerResult.average`'s `nanmean`) converge on it later. Home: ImageAnalysis (or data-utils io). |
| Bin membership | **exists** | `Bin #`/`bin_number` from the union frame + the filter mask. |
| Ephemeral processing (`processing:` selector) | **new** (seam over exists) | `run_diagnostic_ephemeral(diag_name, frames/paths, aux) -> result`: `load_diagnostic(name, config_dir=<portal's own>)` → `create_image_analyzer` → `apply_*_processing_pipeline` / `analyze_image` with **writes forbidden by contract**: never pass `auxiliary_data["file_path"]`, never set writer instance state, and **refuse analyzers on a small denylist until they grow an ephemeral mode (HASO — finding 4)**. Document the gate in ImageAnalysis CLAUDE.md; consider a follow-up `allow_writes: bool` to make it structural. |
| `computed` provenance columns | **new** | Ephemeral scalars merged into the union frame as a third provenance tag, feeding the Plot tab. |

### Traces tab (raster ⇄ line stack, processing selector)

| Primitive | Status | Notes |
|---|---|---|
| `read_1d_data` + `Data1DConfig` | **exists** | 5 formats (tek HDF5, TDMS scope, csv/tsv/npy), Nx2 contract, validated aux columns, already shared with GeecsBluesky's Tiled readback. Extend with formats only as they appear. |
| Per-shot trace resolution | **new** (thin) | (scan, device, shot) → trace file via the existing native-file join, then `read_1d_data` — the resource layer's pattern applied to 1D. Note magspec lineouts are *round-trip* files (written by ImageAnalysis analyzers), so the "processing" selector and the reader meet naturally here. |
| Raster/waterfall serving | **new** (thin) | Stack the per-shot Nx2 traces into a (shots × samples) array server-side; Plotly heatmap renders. Data propagation question stays open per ruling — this tab ships last. |

### Rail, landing, configs

| Primitive | Status | Notes |
|---|---|---|
| Scan/day steppers | **exists** | Catalog day listings + uid; in-place identity swap is pure URL mechanics. |
| Analysis configs (landing card) | **scaffold only** | Per ruling: surface reserved, storage/schema deliberately unresolved. When picked up, note a config ≈ a named URL-state; the schema question is what else it should carry. |
| Multi-user | **exists** (free) | Statelessness: all view state in the URL; shared caches are a feature. |

## Endpoint sketch (all JSON, all one-liners over the above)

```
GET /api/run/{uid}/columns        → [{name, provenance, pretty}]
GET /api/run/{uid}/frame?cols=…&filters=…            → per-shot series
GET /api/run/{uid}/binned?cols=…&filters=…&bincfg=…  → center/err_low/err_high per bin
GET /api/run/{uid}/filter-count?filters=…            → {pass, total}
GET /api/run/{uid}/bin-images?device=…&filters=…&processing=…  → per-bin PNG grid refs
GET /api/run/{uid}/traces?device=…&filters=…&processing=…      → (shots × samples) + axes
```

`filters` / `bincfg` are the URL-serialized forms of the Pydantic
models — the same objects the notebook snippet shows. Every endpoint's
"show the code" is generated from its own parameters.

## Wave plan → PRs (each into `feat/analysis-tabs`, /land ritual)

1. **W1a `data-utils/sfile-and-union`** — `read_sfile`, `scan_frame`
   (union-with-provenance), consolidation of the three s-file reads
   (ScanAnalysis delegation can trail as its own patch). Hermetic tests
   on synthetic s-files + fake RunDetails.
2. **W1b `data-utils/row-filter-groups`** — the OR-of-AND filter model
   lowering onto `apply_row_filters`; explicit NaN policy; pass-count.
3. **W1c `data-utils/bin-frame`** — the pure `bin_frame` rewrite +
   `sd.bin(cfg)` delegate + err-mode unit tests + the
   `basic_usage.ipynb` docs-notebook refresh onto the new API
   (notebook hygiene per docs/CLAUDE.md).
4. **W1d `portal/plotly-plot-tab`** — vendored Plotly (doctrine
   amendment in CLAUDE.md), the rail (steppers, filter chips + popup),
   the Plot tab (per-shot ⇄ binned, multi-Y ≤ 4, gear popup,
   show-the-code), the endpoints above. **The architecture checkpoint
   applies at the end of W1: if adding the Images tab (W2) isn't
   dramatically cheaper than W1d, stop and fix the seams.**
5. **W2a `portal/images-tab`** — promotion of the gallery + per-bin
   averages (`average_frames`) + the ephemeral-processing seam
   (`run_diagnostic_ephemeral`, the write-gate contract, the HASO
   denylist).
6. **W2b `portal/traces-tab`** — per-shot trace resolution +
   raster/line serving (pending the propagation ruling).
7. **Parallel** — the historic-MC ingester (separate sizing, per the
   inventory); the landing page's analysis-configs (blocked on its own
   schema discussion).

## Open questions (carried, not blocking W1)

- Union join key for MC-era s-files once the ingester lands (shot
  number is synthesized there — provenance markers must say so).
- The `computed`-columns lifetime: per-request only (recompute) vs
  cached alongside the pixel cache (probably the latter, same
  completed-run invalidation).
- `render_function`-style rich rendering for analyzers that have it:
  out of scope for JSON; revisit if a tab ever wants analyzer-drawn
  overlays.
- Whether `bin_frame`'s numeric-binning label change
  (`"{src} (binned)"`) survives the rewrite or gets a stable index name.

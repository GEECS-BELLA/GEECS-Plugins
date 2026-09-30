# PR-E — Loader API Refactor (Mode 1 / Mode 2 Symmetry)

Planning artifact for the post-cutover refactor that consolidates the
config-loading surface across ImageAnalysis and ScanAnalysis. Stacks on
top of `feat/unified-diagnostic-configs-cutover` (PR-D).

---

## Goal

Establish two clean, parallel API patterns for using image and scan
analyzers:

- **Mode 1 — direct construction.** Build configs in code, hand to
  class constructors. No YAMLs.
- **Mode 2 — config-driven.** Load → (optional mutate) → build. One
  source of truth on disk; same code path the task queue uses.

After PR-E, every notebook cell and every production call site falls
cleanly into one mode or the other. No hybrid patterns, no polymorphic
constructors, no alias registry, no back-compat shims.

---

## Final API surface

Four public functions, each doing exactly one thing.

### Loading (I/O + validation)

```python
# image_analysis.config
load_diagnostic(name_or_path) -> DiagnosticAnalysisConfig
```

Resolves a stem against the configs tree (or accepts an absolute
`Path`), `yaml.safe_load`s it, validates against
`DiagnosticAnalysisConfig`. Pydantic recursively validates the nested
`image:` (against `CameraConfig` or `Line1DConfig`) and all sub-models.
The `scan:` field is weakly typed at the ImageAnalysis level
(`Optional[Dict[str, Any]]`) so ImageAnalysis can own this loader
without a circular import. ScanAnalysis re-validates the scan dict
against `ScanRuntimeConfig` when it builds a scan analyzer.

### Building (config → live instance)

```python
# image_analysis.config
create_image_analyzer(diag: DiagnosticAnalysisConfig) -> ImageAnalyzer

# scan_analysis.config
create_scan_analyzer(diag: DiagnosticAnalysisConfig) -> ScanAnalyzer
```

`create_image_analyzer` resolves the `image_analyzer:` class path via
`importlib`, instantiates the class with the validated `image:` config.
`create_scan_analyzer` calls `create_image_analyzer` internally to
build the inner analyzer, parses `diag.scan` into `ScanRuntimeConfig`,
picks `Array1DScanAnalyzer` vs `Array2DScanAnalyzer` from `scan.type`,
wraps, attaches `id` / `priority` / `gdoc_slot` / `background_source`.

### Group loading (production: task queue, LiveWatch)

```python
# scan_analysis.config
load_analysis_group(name_or_path) -> LoadedAnalysisGroup
```

Unchanged from PR-D. Internally uses `load_diagnostic` to load each
referenced diagnostic.

---

## Mode 1 and Mode 2, side by side

```python
# ----- Mode 2: load from disk -----
from image_analysis.config import load_diagnostic, create_image_analyzer

diag = load_diagnostic("UC_Amp4_IR_Input")
analyzer = create_image_analyzer(diag)
result = analyzer.analyze_image_file(some_path)

# Mode 2 with tweak — mutate between load and build
diag = load_diagnostic("UC_Amp4_IR_Input")
diag.image.roi.x_max = 200
analyzer = create_image_analyzer(diag)

# Mode 2 for scan analyzer (same diag, different factory)
from scan_analysis.config import create_scan_analyzer
scan_analyzer = create_scan_analyzer(diag)
scan_analyzer.run_analysis(scan_tag)
```

```python
# ----- Mode 1: construct in Python, no YAML -----
from image_analysis.config import DiagnosticAnalysisConfig, CameraConfig, ROIConfig
from image_analysis.config import create_image_analyzer

diag = DiagnosticAnalysisConfig(
    name="UC_MyCam",
    image_analyzer="image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer",
    image=CameraConfig(name="UC_MyCam", roi=ROIConfig(x_min=0, x_max=600)),
)
analyzer = create_image_analyzer(diag)

# Mode 1 truly raw — skip DiagnosticAnalysisConfig entirely
from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer
cfg = CameraConfig(name="UC_MyCam", roi=ROIConfig(x_min=0, x_max=600))
analyzer = BeamAnalyzer(camera_config=cfg)
```

Note the unification: `create_image_analyzer` is the same function
whether the config was loaded or constructed in code. Mode 1 and Mode 2
differ only in where the config came from.

---

## YAML schema (unchanged from PR-D, except `image_analyzer:` form)

```yaml
name: UC_Amp4_IR_Input
image_analyzer: image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer
# OR verbose:
# image_analyzer:
#   class_path: image_analysis.offline_analyzers.line_stitcher.LineStitcher
#   kwargs:
#     stitch_axis: 0

image:
  bit_depth: 16
  description: ...
  background:
    method: constant
    constant_level: 90.0
  roi: {x_min: 1, x_max: 600, y_min: 1, y_max: 600}
  thresholding: {enabled: true, method: constant, value: 0.0, mode: to_zero}

scan:
  priority: 30
  mode: per_shot
  save: true
  type: array2d   # explicit; drives Array2DScanAnalyzer vs Array1DScanAnalyzer dispatch
```

No aliases. Full class paths. `scan.type` is the explicit
scan-wrapper discriminator.

---

## Delete inventory

| Symbol | Reason |
|---|---|
| `scan_analysis.config.aliases` (entire module) | No more aliases. Full class paths in YAML. |
| `scan_analysis.config.diagnostic_factory._instantiate_image_analyzer` | Folded into ImageAnalysis's `create_image_analyzer`. |
| `scan_analysis.config.diagnostic_factory.create_diagnostic_analyzer` | Renamed to `create_scan_analyzer` and relocated to `scan_analysis.config`. |
| `scan_analysis.config.analysis_group_loader.load_diagnostic` | Renamed/relocated to `image_analysis.config.load_diagnostic`. (The helper that landed on PR-D.) |
| `image_analysis.config_loader.load_camera_config` | Folded into `load_diagnostic` (get the `.image` field). |
| `image_analysis.config_loader.load_line_config` | Same. |
| `image_analysis.config_loader._unwrap_diagnostic_image_section` | Back-compat shim no longer needed. |
| `image_analysis.config_loader._apply_nested_overrides` + `__`-syntax | Unused. Mutate Pydantic models directly. |
| `image_analysis.config_loader._load_camera_config_dict` | Internal helper of the deleted functions. |
| `image_analysis.config_loader._load_config_via_search` | Internal helper of the deleted functions. |
| Polymorphic `camera_config_name=` kwarg on all 9 ImageAnalyzer constructors | Constructors take a typed `CameraConfig` / `Line1DConfig` model only. String→file→model lives at the factory layer. |
| `geecs_scanner.optimization.evaluators.multi_device_scan_evaluator._create_scan_analyzer` (hand-rolled importlib path) | Replaced by `create_image_analyzer` + `Array{1,2}DScanAnalyzer.from_scan_config`. |
| `_migration/migrate_to_unified.py::_alias_for_class` + `ALIAS_MAP` + `NO_IMAGE_KIND_ALIASES` (configs repo) | No more aliases to emit. |

---

## Refactor inventory (moves, no behavior change)

| Symbol | From | To |
|---|---|---|
| `DiagnosticAnalysisConfig` | `scan_analysis.config.diagnostic_models` | `image_analysis.config.models` |
| `image_analyzer:` field validator | currently coupled to alias resolution | reduced to parsing a class-path string or `{class_path, kwargs}` dict |
| `_load_diagnostic_file` (private helper) | `scan_analysis.config.analysis_group_loader` | `image_analysis.config` (becomes public `load_diagnostic`) |
| `ScanRuntimeConfig`, `BackgroundSource`, `FromCurrentScanSpec`, `AnalysisGroupConfig`, `AnalyzerRef`, `ResolvedDiagnosticConfig` | stay in `scan_analysis.config.diagnostic_models` | unchanged |
| The `scan:` field on `DiagnosticAnalysisConfig` | typed as `Optional[ScanRuntimeConfig]` | typed as `Optional[Dict[str, Any]]` to break the circular dep. ScanAnalysis validates via `ScanRuntimeConfig.model_validate(diag.scan or {})` at build time. |

---

## Add inventory

| Symbol | Module |
|---|---|
| `load_diagnostic(name_or_path)` | `image_analysis.config` (public) |
| `create_image_analyzer(diag)` | `image_analysis.config` (public) |
| `create_scan_analyzer(diag)` | `scan_analysis.config` (public) |
| `Array1DScanAnalyzer.from_scan_config(...)` | `scan_analysis.analyzers.common.array1d_scan_analysis` (classmethod) |
| `Array2DScanAnalyzer.from_scan_config(...)` | `scan_analysis.analyzers.common.array2D_scan_analysis` (classmethod) |

---

## ImageAnalyzer constructor simplification

Today, every offline analyzer (`BeamAnalyzer`, `LineAnalyzer`,
`StandardAnalyzer`, `Standard1DAnalyzer`, `ICT1DAnalyzer`,
`LineStitcher`, `GrenouilleAnalyzer`, `MagSpecManualCalibAnalyzer`,
`HASOHimgHasProcessor`, `FrogSpectralPhaseAnalyzer`) accepts a
polymorphic `camera_config_name` / `line_config_name` kwarg that can be
a string (file lookup), `Path`, dict, or already-built model.

After PR-E, each constructor takes only the typed model:

```python
# Before
BeamAnalyzer(camera_config_name="UC_VisaEBeam1")          # gone
BeamAnalyzer(camera_config_name=Path("..."))               # gone
BeamAnalyzer(camera_config_name={...})                     # gone

# After
BeamAnalyzer(camera_config=cfg_model)                      # only path
```

The string→file→model load happens in `load_diagnostic` /
`create_image_analyzer` — never in the constructor.

---

## Companion configs-repo PR (`feat/loader-api-refactor` in
`GEECS-Plugins-Configs`)

1. **Patch `_migration/migrate_to_unified.py`**: drop `_alias_for_class`
   lookup, emit verbose `class_path:` forms unconditionally. Delete
   `ALIAS_MAP` and `NO_IMAGE_KIND_ALIASES`.
2. **Re-run migration**: regenerates 37 unified YAMLs with full class
   paths instead of aliases.
3. **Hand-edit `analyzers/HTU/U_BCaveICT.yaml`** (added on PR-C) to use
   the class path instead of `image_analyzer: ict_1d`.
4. **Migrate UNCLASSIFIED orphans to unified shape**. Each of the 37
   bare configs becomes a unified diagnostic with `image_analyzer:`
   set to a sensible default class (`StandardAnalyzer` for generic 2D
   cameras, `Standard1DAnalyzer` for generic 1D, etc.). This removes
   the last reason `load_camera_config` / `load_line_config` need to
   exist as fallbacks.

---

## Test plan

### New tests

- `ImageAnalysis/tests/config/test_load_diagnostic.py` — happy path,
  validation errors, file-not-found, path vs name forms.
- `ImageAnalysis/tests/config/test_create_image_analyzer.py` —
  instantiation via class_path, kwargs propagation, dispatch by
  `image:` model type (CameraConfig → 2D, Line1DConfig → 1D).
- `ScanAnalysis/tests/test_create_scan_analyzer.py` — dispatch on
  `scan.type`, kwargs propagation to the wrapper, `id` / `priority` /
  `gdoc_slot` / `background_source` attachment.
- `ScanAnalysis/tests/test_from_scan_config_classmethods.py` — Mode 1
  ergonomics: build `ScanRuntimeConfig` in code, classmethod
  constructor produces correctly-wired analyzer.

### Migrated/refactored tests

- `ScanAnalysis/tests/test_analysis_group_loader.py` — group loader now
  delegates to `image_analysis.config.load_diagnostic`; tests stay
  end-to-end (verify production behavior unchanged).
- `ImageAnalysis/tests/test_config_loader.py` (if it exists) —
  delete the parts that tested the deleted polymorphic paths.

### Deleted tests

- Anything testing the alias registry directly.
- Anything testing `_unwrap_diagnostic_image_section` (the back-compat
  shim is gone).

---

## New example notebooks

Two new files under `docs/api_patterns/` (new directory; cross-cuts
both packages). No edits to existing notebooks.

### `mode1_direct_construction.ipynb`

Sections:
1. Build a `CameraConfig` in Python (`from image_analysis.config import
   CameraConfig, ROIConfig, BackgroundConfig`). Hand to
   `BeamAnalyzer(camera_config=cfg)`. Call `analyze_image_file`.
2. Same for 1D with `Line1DConfig` + `LineAnalyzer`.
3. Wrap a runtime-built ImageAnalyzer in `Array2DScanAnalyzer`. Two
   flavors: explicit kwargs vs new `from_scan_config` classmethod with
   a `ScanRuntimeConfig` model.
4. Mutate `analyzer.camera_config.roi.x_max = 800` and re-run — confirm
   the change takes effect on subsequent calls.

Punchline: "You own the Python objects, no YAML coupling, ideal for
exploration."

### `mode2_config_driven.ipynb`

Sections:
1. `load_diagnostic("UC_VisaEBeam1")` → `DiagnosticAnalysisConfig`.
   Inspect the loaded model.
2. `create_image_analyzer(diag)` → live `ImageAnalyzer`. Call
   `analyze_image_file`.
3. Mutate `diag.image.roi.x_max = 200` between load and build; observe
   the new analyzer reflects the change.
4. `create_scan_analyzer(diag)` from the same diag → live
   `ScanAnalyzer`. Call `run_analysis(scan_tag)`. Note that
   `scan_analyzer.image_analyzer` is functionally equivalent to step 2.
5. `load_analysis_group("baseline")` → production path. Iterate the
   resolved diagnostics, show that `create_scan_analyzer(r.diagnostic)`
   produces what the task queue would.

Punchline: "YAML is the source of truth, same code path as the task
queue, share configs across runs/people/notebooks."

Both notebooks run end-to-end against a small test scan (or skip-marked
when data unavailable).

---

## Sequencing

```
master
  ├── feat/shot-by-shot-scan-analyzer        (PR-1)
  ├── feat/unified-diagnostic-configs        (PR-2)
  ├── feat/unified-diagnostic-configs-cutover (PR-3 / PR-D)
  └── feat/loader-api-refactor                (PR-4 / PR-E)  ← this PR
       └── companion configs PR

GEECS-Plugins-Configs:
  master
  ├── feat/unified-diagnostic-configs        (PR-C, force-pushed)
  └── feat/loader-api-refactor                (companion to PR-E)
```

PR-E doesn't need to wait for PR-D to merge to start; the changes don't
conflict. But PRs should land in order.

---

## Open design questions to settle in implementation

1. **`from_scan_config` classmethods — yes or no?** Adds Mode 1
   ergonomics on the scan side at the cost of ~30 lines per class.
   Probably yes, but worth confirming the call-site improvement is real
   before committing.

2. **Validation ordering on `scan:` field.** When ImageAnalysis loads a
   diagnostic, the `scan:` dict isn't validated against
   `ScanRuntimeConfig` — that happens at scan-side build time.
   Acceptable trade-off, but means a malformed `scan:` block isn't
   caught at load time. Alternative: a Pydantic-discriminated-base
   pattern where ImageAnalysis defines a `BaseScanField` and
   ScanAnalysis registers `ScanRuntimeConfig` as the concrete
   implementation. More machinery, but type-safe end-to-end.

3. **`ScanRuntimeConfig.type` field naming.** Currently the literal is
   `"array1d"` / `"array2d"`. After Mode 1/Mode 2 split is in place,
   worth asking if there's a cleaner name. Defer to a rename PR.

---

## Size estimate

| Bucket | Effort |
|---|---|
| Delete alias registry + simplify field validators | 1–2 hr |
| Delete `image_analysis.config_loader` modules | 1 hr |
| New `image_analysis.config.factory` (load_diagnostic + create_image_analyzer) | half-day |
| Simplify 9 ImageAnalyzer constructors | half-day |
| Update all internal callers (scan_analysis factory, optimizer, group loader) | half-day |
| Slim down `scan_analysis.config.diagnostic_factory` to just `create_scan_analyzer` + `_wrap_in_scan_analyzer` | 1–2 hr |
| `Array{1,2}DScanAnalyzer.from_scan_config` classmethods | 1 hr |
| Re-run configs migration + UNCLASSIFIED → unified shape | half-day |
| Two new notebooks | 1 day |
| Test updates + pre-commit + CHANGELOG + version bumps | half-day |

**~3 days** of focused work. Net code count: meaningfully negative
(deletes outweigh adds, models stay put).

---

## What this PR is NOT doing (explicit non-goals)

- **Not renaming `ScanAnalyzer`** — the discomfort with `ScanAnalyzer`
  being "an orchestration layer, not a diagnostic" is real but is a
  separate repo-wide rename PR.
- **Not editing existing example notebooks** — only two new ones under
  `docs/api_patterns/`.
- **Not touching ConfigFileGUI** — that's PR-F, lands against this PR-E
  surface as the stable target.
- **Not changing the on-disk YAML schema** beyond the
  `image_analyzer:` field format (alias → class path). The structure
  is the same.
- **No performance work.**

---

# Outcome (post-implementation, written after the final commit)

## What landed

12 commits on `feat/loader-api-refactor` (GEECS-Plugins) plus 3 on
`feat/loader-api-refactor` (GEECS-Plugins-Configs). The PR grew
beyond the original scope as the review pass surfaced more cleanup
opportunities — every dead-code audit ended up paying for itself.

### GEECS-Plugins commit log (in order)

| Commit | Summary |
|---|---|
| `12b14bf6` | Planning doc (this file, original) |
| `ca044e40` | Relocate `DiagnosticAnalysisConfig` to `image_analysis.config` |
| `47534315` | Add public `load_diagnostic` + `create_image_analyzer` (ImageAnalysis-side) |
| `4f38eb95` | Rename `create_diagnostic_analyzer` → `create_scan_analyzer`, slim factory |
| `b68af19b` | Kill the `image_analyzer` alias registry |
| `7df36856` | Drop polymorphic `camera_config_name=` / `line_config_name=` kwargs on 12 analyzers |
| `66453911` | Optimizer: load typed image config before instantiation |
| `35b531f6` | Replace `BackgroundManager` (310 LoC class) with `apply_background()` function |
| `1a1f2500` | Consolidate all config under `image_analysis.config/` (move array1d/2d models, loader, diagnostic) |
| `42a62756` | Rename `offline_analyzers/` → `analyzers/` (historical holdover) |
| `7e99c1fd` | Drop dead-code utilities from `config.loader` (8 functions, ~170 LoC) |
| `277fc4dc` | Drop `ScatterAnalyzerConfig` + `create_analyzer` (dead config plumbing) |

### GEECS-Plugins-Configs commit log

| Commit | Summary |
|---|---|
| `33ffbda` | Emit `image_analyzer` as `class_path` (no aliases) — 38 YAMLs |
| `5943572` | Drop alias-emission in migration script |
| `f59df5f` | `image_analysis.offline_analyzers` → `image_analysis.analyzers` (49 files) |

### LoC delta (vs. cutover branch tip)

|  | Insertions | Deletions | Net |
|---|---:|---:|---:|
| All code + tests | 1745 | 2197 | **−452** |
| Code only (excluding tests) | 1346 | 1855 | **−509** |
| Configs repo | ~140 | ~58 | +82 |

Net negative across the board, despite adding ~300 LoC of new public
factory module + ~370 LoC of new test coverage. The deletion pile is
real.

## Final repo layout

### ImageAnalysis

```
image_analysis/
  config/                          ← single home for ALL config concerns
    __init__.py                    (public re-exports)
    aliases.py                     (ImageAnalyzerSpec + ImageKind/ScanType enums)
    array2d_processing.py          (CameraConfig + 2D sub-models — moved from processing/)
    array1d_processing.py          (Line1DConfig + 1D sub-models — moved from processing/)
    diagnostic.py                  (DiagnosticAnalysisConfig — moved from scan_analysis)
    factory.py                     (load_diagnostic, create_image_analyzer — NEW)
    loader.py                      (load_camera_config, load_line_config — moved + slimmed)
  analyzers/                       (was offline_analyzers/)
    beam_analyzer.py
    grenouille_analyzer.py
    ict_1d_analyzer.py
    line_analyzer.py
    line_stitcher.py
    standard_analyzer.py
    standard_1d_analyzer.py
    Undulator/
      BCaveMagSpecStitcher.py
      BCaveMagSpecStitcherOpt.py
      hi_res_mag_cam_analyzer.py
  processing/                      (processing functions only — no models)
    array1d/
      background.py, filtering.py, interpolation.py, pipeline.py, roi.py, thresholding.py
    array2d/
      background.py (now has apply_background fn — was BackgroundManager class)
      filtering.py, masking.py, normalization.py, pipeline.py, thresholding.py,
      transforms.py, vignette.py
  types.py                         (Array1D, Array2D, ImageAnalyzerResult — left alone)
  base.py
```

### ScanAnalysis

```
scan_analysis/
  config/
    __init__.py
    analysis_group_loader.py       (discover_analyzers, discover_groups, load_analysis_group)
    diagnostic_factory.py          (create_scan_analyzer)
    diagnostic_models.py           (ScanRuntimeConfig, AnalyzerRef, BackgroundSource, …)
  analyzers/
    common/
      single_device_scan_analyzer.py
      array1d_scan_analysis.py
      array2D_scan_analysis.py
      scatter_plotter_analysis.py  (kept as standalone utility)
    Undulator/                     (yaml-only items; audit deferred per user)
```

## Final public API surface

Four functions plus the models. This is what notebooks should import.

```python
# ----- ImageAnalysis side -----
from image_analysis.config import (
    # Loading
    load_diagnostic,            # name | path → DiagnosticAnalysisConfig
    load_camera_config,         # name | path | dict → CameraConfig (Mode 1 disk-backed)
    load_line_config,           # name | path | dict → Line1DConfig (Mode 1 disk-backed)

    # Factory
    create_image_analyzer,      # DiagnosticAnalysisConfig → ImageAnalyzer

    # Models
    DiagnosticAnalysisConfig,
    CameraConfig, Line1DConfig,
    ROIConfig, BackgroundConfig, CrosshairMaskingConfig, VignetteConfig, …
    ImageAnalyzerSpec, ImageKind, ScanType,
)

# ----- ScanAnalysis side -----
from scan_analysis.config import (
    # Loading
    load_analysis_group,        # group name → LoadedAnalysisGroup (production path)

    # Factory
    create_scan_analyzer,       # DiagnosticAnalysisConfig → ScanAnalyzer

    # Models
    ScanRuntimeConfig, AnalyzerRef, AnalysisGroupConfig,
    ResolvedDiagnosticConfig, BackgroundSource, FromCurrentScanSpec,
)
```

### Two usage modes, side by side

```python
# ----- Mode 1: direct construction (no YAML) -----
from image_analysis.config import CameraConfig, ROIConfig
from image_analysis.analyzers.beam_analyzer import BeamAnalyzer

cfg = CameraConfig(name="UC_MyCam", roi=ROIConfig(x_min=0, x_max=600))
analyzer = BeamAnalyzer(camera_config=cfg)
result = analyzer.analyze_image_file(some_path)

# ----- Mode 2: config-driven (load from YAML) -----
from image_analysis.config import load_diagnostic, create_image_analyzer
from scan_analysis.config import create_scan_analyzer

diag = load_diagnostic("UC_VisaEBeam1")
diag.image.roi.x_max = 200          # optional tweak

image_analyzer = create_image_analyzer(diag)        # just the ImageAnalyzer
scan_analyzer  = create_scan_analyzer(diag)         # full ScanAnalyzer wrapper
```

## Side audits that paid off (not in the original plan)

These came out of the consolidation pass at the end:

1. **`BackgroundManager` (310 LoC class) → `apply_background()` function (~30 LoC).**
   11 of 12 public methods were dead code from the deleted dynamic-background
   era. The path-keyed cache moved from manager-instance state to an
   analyzer-instance dict. Commit `35b531f6`. Net −275 LoC.

2. **`config.loader` dead-code purge.**
   `_apply_nested_overrides`, `create_processing_configs`,
   `save_config_to_yaml`, `load_config_from_yaml`, `validate_config_file`,
   `get_config_schema`, `convert_from_processing_dtype`,
   `list_available_configs` — all zero external callers. Commit `7e99c1fd`.
   Net −170 LoC.

3. **Scatter config plumbing.**
   `ScatterAnalyzerConfig`, `PlotParameterConfig`, `create_analyzer`
   factory, dedicated test file — none referenced by production YAMLs;
   only the test plumbed them. The `ScatterPlotterAnalysis` class itself
   stays (real working code, used by `ICTPlotAnalysis`). Commit `277fc4dc`.
   Net −395 LoC.

## TODO before opening PRs

1. **CHANGELOGs + version bumps** — three packages touched:
   - ImageAnalysis: 1.4.0 → 1.5.0 (`config/` reorg, `analyzers/` rename, factory addition, BackgroundManager removal, polymorphic-kwarg removal)
   - ScanAnalysis: 1.6.0 → 1.7.0 (`create_diagnostic_analyzer` → `create_scan_analyzer` rename, ScatterAnalyzerConfig removal)
   - Scanner-GUI: 0.21.0 → 0.22.0 (optimizer load-then-pass fix)

2. **Two new notebooks under `docs/api_patterns/`** — Mode 1 + Mode 2
   demos. *User flagged earlier they want to do these hands-on for
   themselves; defer to user.*

3. **Open PRs** — none yet on this stack. Decision still pending on
   whether to open as one mega-PR off master or stack four PRs
   (shot-by-shot → unified-configs → cutover → loader-api-refactor).
   Companion configs-repo PRs need to land atomically with PR-C and
   PR-E.

## Explicitly deferred items

- **Array1D/Array2D processing pipeline simplification.** `pipeline.py`
  has 8 three-line wrapper functions + a `STEP_REGISTRY` dict that
  could collapse to an if/elif chain in the main loop (~100 LoC → ~30).
  User acknowledged it's hokey but explicitly deferred to a separate
  PR. Worth doing eventually.
- **`scan_analysis/analyzers/Undulator/` audit.** User said "almost
  100% delete-able" but doing it themselves.
- **ConfigFileGUI rework.** Was PR-F all along. Still deferred. Will
  land against this PR-E's final API surface as the stable target.
- **`ScanAnalyzer` → `ScanOrchestrator` rename.** User noted the
  naming feels off (it's an orchestration layer, not analysis), but
  deferred to a dedicated rename PR with full repo sweep.
- **Notebook updates beyond the two new Mode 1 / Mode 2 demos.** User
  takes the rest.
- **Scatter config integration into unified schema.** Explicitly
  decided against. Scatter stays a standalone utility class —
  occasional notebook use only.

## Branch state

```
master (GEECS-Plugins)
  └── feat/shot-by-shot-scan-analyzer        (5 commits, no PR)
       └── feat/unified-diagnostic-configs   (+11, no PR)
            └── feat/unified-diagnostic-configs-cutover  (+7, no PR)
                 └── feat/loader-api-refactor (+12, no PR)  ← this branch

master (GEECS-Plugins-Configs)
  └── feat/unified-diagnostic-configs        (5 commits, no PR — PR-C content)
       └── feat/loader-api-refactor          (+3 commits, no PR)
```

All branches pushed to origin. No PRs opened yet.

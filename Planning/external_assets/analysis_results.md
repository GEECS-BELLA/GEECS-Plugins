# Bluesky Post-Run Analysis Results

## Purpose

This document defines the first GeecsBluesky post-run analysis contract. The
goal is to preserve the useful existing GEECS workflow:

```text
raw scan -> per-shot feature extraction -> full-scan analysis
```

while replacing scan-folder/file-name inference with Bluesky/Tiled run, event,
resource, and datum identity.

This is not intended to make analysis part of the live acquisition plan. The
first implementation is a post-run consumer of completed raw Bluesky runs.

## Current Workflow Analogy

The legacy analysis path is:

```text
s-file appears
  -> ScanAnalysis/ImageAnalysis extract features
  -> new s-file columns are written
  -> auxiliary files may be written
  -> scan-level analysis consumes the enriched s-file
```

The Bluesky-era equivalent should be:

```text
raw Bluesky run appears in Tiled
  -> post-run analysis fills external assets locally
  -> ImageAnalysis or another extractor computes per-shot features
  -> features table is written under the day-level analysis directory
  -> derived files are written under the same analysis invocation
  -> scan-level analysis consumes the feature table
```

In this model, the feature table replaces the "enriched s-file" behavior for
Bluesky runs. The raw run and raw external assets remain the protected
acquisition record.

## Directory Layout

Analysis products should live under the existing day-level `analysis/`
directory, not under `scans/`.

```text
{day_folder}/
  scans/
    Scan006/
      ... raw acquisition files ...
  analysis/
    Scan006/
      beam_centroid_v1/
        20260628T190012Z/
          analysis_metadata.json
          features.jsonl
          features.parquet        # preferred when parquet support is available
          assets/
            shot_000003_overlay.png
```

The invocation timestamp level prevents accidental overwrites when the same
analyzer is rerun with different code or configuration.

## Analysis Invocation

An analysis invocation describes one execution of one analyzer against one raw
run.

It records provenance needed to reproduce or audit the results:

- raw run UID
- experiment and scan number
- analyzer identity and version
- input event/datum/resource references
- output feature table and derived asset references
- canonical analysis/input storage roots and POSIX paths below those roots
- analyzer configuration
- git/code version
- package/runtime environment
- user, timestamp, notes, and optional status

This information is written to `analysis_metadata.json`.
The metadata must not embed machine-local data mounts such as `/Volumes/...`;
local readers map canonical roots, for example `Z:/data`, to local storage at
runtime.

## Feature Table

The feature table has one row per analyzed shot/input asset. Required columns
are intentionally boring and queryable:

```text
raw_run_uid
event_uid
scan_number
scan_event_index
shot_number
device
data_key
datum_id
asset_spec
analyzer_id
status
error_message
elapsed_s
```

Feature columns are analyzer-defined scalar columns. Derived asset references
may also be included as columns when they are convenient to query, but the
authoritative derived asset inventory lives in `analysis_metadata.json`.

## Execution Boundary

GeecsBluesky owns orchestration:

```text
Tiled lookup
event selection
Resource/Datum/local fill
analysis output layout
feature table writing
provenance capture
```

ImageAnalysis or another algorithm package owns scientific extraction:

```text
array/file in
features/derived files out
algorithm configuration
algorithm version
```

The first implementation should use a small generic analyzer protocol. Existing
`ImageAnalyzer` classes can be wrapped later without making ImageAnalysis
import Tiled, Bluesky documents, datum IDs, or scan-folder layout rules.

Analyzers can require one of two execution scopes:

```text
event
  one input event/asset is sufficient
  suitable for standalone per-shot work and distributed live workers

scan
  the analyzer first prepares scan-level context from multiple events/assets
  then still emits per-event FeatureRows
  suitable for dynamic background subtraction, image binning, and other
  ScanAnalysis-like workflows
```

Both scopes write the same feature table and metadata contract.

## Relationship to Bluesky Standards

This contract is intentionally compatible with a future derived Bluesky run or
Tiled catalog entry, but it does not require that step immediately.

The raw run remains the source of truth. The sidecar invocation records the raw
run UID and input datum/resource IDs so that a future catalog entry can point at
the same products without changing the file layout.

## Out of Scope for the First PR

- Running analysis automatically after every scan
- Server-side Tiled handlers
- Derived Bluesky run emission
- Full ImageAnalysis or ScanAnalysis refactors
- FROG SDK execution
- Optimization loops

The first PR should define the contract, writer, and runner seam. A later PR can
wire one real camera feature extractor end-to-end.

## First End-to-End Helper

The exploratory implementation includes a sidecar-first camera convenience path:

```python
from geecs_bluesky.analysis import run_tiled_camera_image_analysis
from image_analysis.analyzers.beam_analyzer import BeamAnalyzer

metadata = run_tiled_camera_image_analysis(
    year=2026,
    month=6,
    day=24,
    scan_number=6,
    experiment="Undulator",
    device_name="UC_Amp2_IR_input",
    image_analyzer=BeamAnalyzer(camera_config),
    analyzer_id="beam_centroid_v1",
    analyzer_config=camera_config.model_dump(mode="json"),
    retry_intervals=[],
)
```

This performs:

```text
Tiled run lookup
  -> primary event iteration
  -> camera Resource/Datum reconstruction
  -> local file fill through GEECS_CAMERA_IMAGE handler
  -> ImageAnalyzerAdapter
  -> AnalysisRunner
  -> analysis_metadata.json + features.jsonl/parquet
```

The generic readback helpers can also reconstruct registered non-camera assets
such as MagSpec text arrays via `load_asset_from_tiled(...)` /
`load_asset_from_tiled_run(...)`. TDMS event assets should remain file-backed until
the analysis request supplies the `Data1DConfig` trace/channel selection used by
`geecs_data_utils.io.array1d.read_1d_data`; they are not the scan-level TDMS
table files read by `ScanData`.

The device-type asset registry is the right place for stable file facts:
payload kind, loader family, default 1D data type, file suffixes, and whether a
loader config or SDK-capable worker is required. Per-diagnostic choices such as
TDMS trace/channel indices should stay in the analysis configuration that
instantiates the analyzer.

By default, the helper writes only sidecar artifacts. When called with
`emit_derived_run=True`, it also builds a small derived Bluesky document stream
that points back to the raw run and forward to the sidecar artifacts. Passing a
document callback captures or publishes those docs; passing `publish_to_tiled=True`
sends them to the configured Tiled catalog.

The derived run is a searchable pointer, not the storage container. The
feature table and derived assets remain under the day-level `analysis/`
directory.

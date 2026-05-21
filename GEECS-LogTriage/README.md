# GEECS-LogTriage

Harvest, group, and classify error/warning entries from GEECS scan execution
logs (`scan.log` files written by `geecs_scanner.logging_setup`).

The package is the **Stage 1** floor of an auto-debugger pipeline: a
deterministic Python utility that walks a date's worth of scans, parses each
`scan.log`, normalizes errors into stable fingerprints, classifies them by
exception family (bug candidate vs. config issue vs. hardware issue vs.
operator error), and emits a structured `TriageReport` JSON document.

Stage 2 (LLM triage that opens dedup'd GitHub issues) and Stage 3 (autonomous
fix loop) layer on top of this output but are not part of this package.

## Install

```bash
cd GEECS-LogTriage
poetry install
```

## Quick start

```bash
poetry run geecs-log-triage --date 2026-05-07 --experiment Undulator \
    --level error --output triage.json
```

Date range:

```bash
poetry run geecs-log-triage --date-range 2026-05-01:2026-05-07 \
    --experiment Undulator --level warning --output week.json
```

Inspect a single scan directly:

```bash
poetry run geecs-log-triage --scan-folder /path/to/Scan037 --output one.json
```

## Output shape

`TriageReport` (Pydantic v2) — see `geecs_log_triage.schemas`. Top-level fields:

- `date_range`: `(start, end)`
- `total_scans_examined`
- `total_log_entries`
- `total_errors`
- `grouped`: `dict[fingerprint_hash, list[ErrorOccurrence]]`

Each `ErrorOccurrence` carries the originating `LogEntry`, the `ScanTag` /
folder path, and the `ErrorFingerprint` (with classification).

## How it composes

- Reads logs via `geecs_data_utils.load_scan_log(scan_path)` — owned by
  Data-Utils so anyone can use it (notebooks, plotting, etc.).
- Walks scans via `geecs_data_utils.ScanPaths` — already knows the
  `Y{YYYY}/{MM-Month}/{YY_MMDD}/scans/Scan{NNN}/` convention.
- All output models are Pydantic v2.

## Status

v0.1.0 — scaffolding + parser + fingerprint + classifier + harvester + CLI.
Tested against synthetic fixtures only; needs validation against real
production `scan.log` files.

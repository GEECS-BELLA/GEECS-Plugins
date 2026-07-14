---
name: scan-audit
description: >
  Scan timing & cadence audit for the Bluesky path. Use for "why was the
  scan slow", "did every shot land", "shots are skipping / landing on every
  other trigger", start-latency questions, or any per-shot acq_timestamp
  cadence analysis of a scan folder. Reads scan.log, the s-file, ScanInfo
  ini, and optionally Tiled + the console log; needs no hardware. For error
  triage ("what went wrong / what errored") use /triage instead.
---

# /scan-audit — scan timing & cadence audit (Bluesky path)

`/triage` answers "what errored"; this answers "**why was the scan slow /
did every shot land**". Built from the 2026-07-13 investigation that found
the unstaged-reads bug (#540) — the diagnostic signatures below are the
distilled version of that session.

## Arguments

`$ARGUMENTS` — a scan folder path, or `--date YYYY-MM-DD --experiment NAME
[--scan N]`. Resolve folders via the GEECS data convention
(`{base}/{expt}/Y{YYYY}/{MM-Mon}/{YY_MMDD}/scans/ScanNNN/`); base path from
`~/.config/geecs_python_api/config.ini`.

## Inputs per scan (all plain text; no hardware needed)

- `scan.log` — parse with `geecs_data_utils.scan_log_loader.parse_scan_log`
  (handles both legacy `shot=` and Bluesky `scan=` formats as of
  geecs-data-utils 0.13.3).
- `ScanDataScanNNN.txt` (tab-separated s-file) — per-shot device
  `acq_timestamp` columns.
- `ScanInfoScanNNN.ini` — mode, shots per step.
- Optional (on-network): the Tiled run via
  `geecs_data_utils.tiled_catalog.TiledScanCatalog` for column counts and
  telemetry shape.
- The console log (`~/.config/geecs_console/logs/console.log`) for
  submission-time context (start latency lives *before* scan.log starts).

## Analysis

1. **Phase timeline from scan.log**: bracket setup (start → first
   SCAN/ARMED), acquisition window, teardown (STANDBY/OFF → finished).
   In free-run also note t0-sync spread (logged) and any
   "t0 seed check" WARNING (seeding suspect). At DEBUG the engine logs
   per-row and fire/frame-wait durations — use them when present.
2. **Per-shot cadence from the s-file**: for each `<device> acq_timestamp`
   column compute consecutive deltas. Interpret against the trigger
   period T = 1/rep_rate:
   - deltas ≈ exact integer multiples of T (e.g. 2.000, 2.000 at T=1) —
     the trigger and devices are HEALTHY; **software row cycle exceeds
     one period** and rows land on every k-th tick. This is the #540
     signature. Suspects: unstaged reads, oversized telemetry, VPN RTTs.
   - deltas non-quantized ~constant (e.g. 1.6–2.0 s, strict mode) — the
     cadence *is* the software cycle: fire put + frame wait + reads.
   - first delta larger than the rest — normal warmup (first monitor
     delivery, save-on, arm settle); only flag if > ~2 periods.
   - deltas at T with occasional 2T — marginal cycle riding the cliff;
     report the headroom, not just the median.
3. **Start latency** (console log): submission timestamp → "Claimed scan
   number". > ~5 s warm is anomalous post-0.33.0 (telemetry connects are
   batched); a cold first-scan-after-launch pays one-time CA channel
   creation and can be tens of seconds over VPN — say which case it is.
4. **Column shape** (Tiled, optional): total vs `telemetry_*` columns and
   device count; compare to the framework cost model
   `ms/event ≈ 0.5 + 0.30·devices + 0.013·columns` (mock-benchmarked;
   see `GeecsBluesky/scripts/bench_plan_overhead.py` to re-derive).
5. **Verdict**: total time vs ideal (`shots × T + setup + teardown`),
   where the loss is (setup / cadence / teardown), and the top suspect
   with evidence. Reference budgets: 5 Hz is the system limit → 200 ms
   row budget; staged reads ≈ 0 network cost; strict mode floor ≈ 0.4 s
   fire+frame (structurally ≲2 Hz).

Write the findings as a short report (per scan: phase table, cadence
verdict, anomalies). Only propose code changes when a signature clearly
matches; otherwise report the measurements and the ambiguity.

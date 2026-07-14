# Scan Browser

The scan browser is the quick-look client for recorded scans: pick a day,
pick a scan, see what happened — without writing a line of analysis code.
It reads the Tiled catalog directly and is fully independent of the
operator console (analysts can install and run just this).

```bash
cd GEECS-Console
poetry run geecs-scan-browser        # or: python -m geecs_console.browser
```

## What it shows

- **Day list** — the scans recorded on a chosen date for the selected
  experiment (newest first), with mode and shot counts, filterable as you
  type.
- **Run detail** — for the selected scan:
  - **Plot** — any recorded column vs shot (or vs the scan variable),
    for eyeballing trends and step structure;
  - **Table** — the raw per-shot event rows;
  - **Drift** — the telemetry drift report: which background-telemetry
    channels *moved* during the scan (|last − first| beyond 3σ of the
    in-scan spread), sorted by significance. The fastest way to answer
    "did anything else change while I was scanning?"
- **Open scan folder** — jumps to the classic `ScanNNN/` directory for
  the raw files (strictly read-only — the browser never creates or
  modifies anything on the data share).

## Notes

- Every catalog call runs off the GUI thread — a slow network (VPN) makes
  the browser *wait*, never freeze; superseded requests are dropped.
- Offline it opens with an empty catalog and says so; nothing breaks.
- Aborted or legacy runs without event rows display metadata only.
- The column semantics behind the plot/table/drift views are the shared
  `geecs_data_utils.tiled_schema` module — the browser interprets no
  column names on its own, so it stays automatically consistent with the
  data contract in [Scan Data](scan_data.md).

# The analysis domain

The analysis domain closes the loop the scan service opens: after a scan
runs, an agent can see what the
[ScanAnalysis](../scan_analysis/overview.md) pipeline produced, fetch the
summary figures, and — since GEECS-MCP 0.7.0 — trigger the analysis
itself. The backend is the existing ScanAnalysis framework, used as-is:
the same diagnostic YAMLs, analyzers, and task-queue bookkeeping that the
LiveWatch tooling drives, now callable on demand.

## Reading results (R)

| Tool | What it returns |
|---|---|
| `get_scan_analysis` | Per-scan task statuses (from the scan's `analysis_status/` files: queued / claimed / done / failed / no_data, with errors and heartbeat ages) plus a capped map of the analysis output tree |
| `get_scan_figure` | A figure **reference**: label, dimensions, byte size, and a fetch URL served by this same server. `thumbnail=true` opts into a small bounded inline preview |
| `list_analyzers` | Every diagnostic ID in the configs repo's `analyzers/` tree — the names `run_scan_analysis` accepts |
| `list_analysis_groups` | Every analysis-group name (`groups/` tree); both bare stems and namespace-qualified forms resolve |

Figures follow the payload doctrine strictly: image bytes never ride the
agent's context by default (a single full-size PNG can swamp a model's
context window — a real integration finding). The reference's URL points
at a `/figures/{day}/{scan}/{label}` route on the server, so a client
fetches the original bytes out-of-band and saves them as an artifact.

## Running analysis (Q)

`run_scan_analysis(scan_number, day?, analyzer|group, rerun_failed,
rerun_completed)` executes one diagnostic or a whole group for one scan,
submit-and-poll style: the call returns as soon as the tasks are enqueued
and a detached worker process is started; `get_scan_analysis` is the
poll. The tool validates *everything refusable before any side effect*:

- exactly one of `analyzer` / `group`, with names from the listing tools
  (unknown names are refused with the known-names list);
- the scan folder **must already exist** — analysis is a consumer of
  scan folders, never a producer (the repo-wide invariant: a missing
  folder is a refusal naming the path, never an auto-create);
- the analyzers must actually construct **on this host** — a diagnostic
  whose image-analyzer class needs a Windows-only SDK (the HASO
  wavefront and Grenouille/FROG diagnostics) is refused up front with a
  message naming the future Windows satellite server, instead of
  half-running.

The execution itself is the ScanAnalysis task queue's own machinery:
statuses are initialized before the worker spawns (so a worker that dies
is *visible* as a stuck row, never a silent nothing), the worker claims
each task through an atomic cross-process claim gate (two near-
simultaneous requests can never double-run one analyzer into the same
output files), heartbeats tick every 30 s, and a dead worker's claim goes
stale after 180 s so a repeat call re-runs it.

Rerun semantics are explicit: by default only never-run tasks execute —
a repeat call is a cheap, honest no-op that reports the already-`done`
tasks as `skipped`. `rerun_failed` / `rerun_completed` re-queue those
states server-side before the worker starts. Google-Doc upload stays
hard-off: publishing figures to the experiment log is an outward-facing
action that would need its own explicitly gated verb.

## A worked example

A real first-contact run, on archived data (Undulator, 2026-05-01,
Scan 1, the `Amp2Input` diagnostic over the `UC_Amp2_IR_input` camera):

```text
list_analyzers                             → 82 diagnostics, incl. Amp2Input
run_scan_analysis(1, "2026-05-01",
                  analyzer="Amp2Input")    → ok, started: false —
                                             skipped: {Amp2Input: done}
                                             (this scan was analyzed in May)
run_scan_analysis(..., rerun_completed=True)
                                           → ok, started: true, worker pid
get_scan_analysis(1, "2026-05-01")         → Amp2Input: claimed → done,
                                             display_files: [...visual.png]
get_scan_figure(1, day="2026-05-01")       → figure reference with
                                             /figures/2026-05-01/1/... URL
run_scan_analysis(999, "2026-05-01", ...)  → not_found: scan folder does
                                             not exist (nothing created)
```

## The longer arc

Using ScanAnalysis as-is is a deliberate owner decision, not an endpoint.
The verb surface is backend-neutral (it names a diagnostic, a scan, a
day) and the `analysis_status/` files are the progress contract — so the
Tiled-based analysis stack in GeecsBluesky (provenance-tracked results
published back into the archive) can slot in behind the same tools later
without the agent-facing surface changing.

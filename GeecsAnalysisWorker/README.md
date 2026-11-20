# GeecsAnalysisWorker

The analysis worker's HTTP service: `scan-analysis-worker`, default port
8900.

It batch-runs ScanAnalysis groups over a day's scans. A **job** is one
analysis group over one day (or some of its scans); it runs task by task
through ScanAnalysis' task queue, so a task that already finished is
skipped unless the job asks to redo it. The Data Portal's day page
("Analyze this day…") is its client, over HTTP alone.

The job runner itself is ScanAnalysis' (`scan_analysis.worker`). This
package adds the HTTP API over it and the command, and nothing else.

Run it:

```bash
scan-analysis-worker --configs <scan_analysis_configs> --experiment <name>
```

It binds 127.0.0.1 unless `--host` (or `GEECS_ANALYSIS_WORKER_BIND`) says
otherwise: the API is unauthenticated. The data root is the client
`config.ini`'s, as for every analysis run. On a services host it runs as
its own unit, `deploy/geecs-analysis-worker.service`, rendered from
`site.env` (deployed only where `GEECS_ANALYSIS_WORKER_PORT` is set); the
portal is pointed at it with `--analysis-worker-url` and never starts one.

The worker starts idle. It can also **watch** a day: every few seconds it
looks for scans that have closed and runs the ticked analyzers on each
once. A watch runs only after an operator starts one (the day page's
"Watch for new scans", or `POST /api/watch`); it is held in memory, so a
restart ends it.

This package is a **consumer of scan folders** and never a producer. See
the repository `CLAUDE.md` for the invariant.

# GeecsAnalysisWorker: Developer Context for Claude

The analysis worker's HTTP service. `scan-analysis-worker` serves an API
over ScanAnalysis' job runner (`scan_analysis.worker.Worker`): batch
analysis of a day's scans, for the Data Portal's day page.

## Layout

```
geecs_analysis_worker/
  app.py        # create_worker_app(worker): the routes; JobRequest, WatchRequest (the bodies)
  __main__.py   # scan-analysis-worker: argparse, the Agg pin, uvicorn
deploy/
  geecs-analysis-worker.service   # the unit template (deploy/render_units.sh)
tests/
  test_api.py   # the routes over a real Worker (fake analyzers through the real queue); the command
```

## The split

- **The job logic is ScanAnalysis'.** What a job is, how it expands into
  the task queue's worklist, when a scan counts as closed, the watch's
  loop: all in `scan_analysis/worker/jobs.py`, documented in
  `ScanAnalysis/CLAUDE.md` § "The analysis worker's job runner". Nothing
  of it is restated or re-implemented here. A new behaviour of a job goes
  there; this package gains a route or a field for it.
- **The web side is this package's.** ScanAnalysis has no web code (the
  `scan-analysis-no-web` import contract): fastapi and uvicorn are this
  package's dependencies, not an extra of ScanAnalysis.
- **Clients reach it over HTTP alone.** The portal's `geecs_portal/batch.py`
  is an httpx client; the `portal-edges` contract forbids it this import,
  and the layers contract forbids every other surface the same. Nothing in
  the repo imports `geecs_analysis_worker`.
- This package imports `scan_analysis` only (the `analysis-worker-edges`
  contract).

## The API

```
GET  /health                 {ok, service, experiment, config_dir, active}
GET  /api/groups             the configs tree's loadable groups
POST /api/plan               {day, group, scans?, redo?, analyzers?, from_scan?, experiment?} -> what a job would run
POST /api/jobs               same body -> 202 + the queued job
GET  /api/jobs               every job, newest first (no task lists)
GET  /api/jobs/{id}          one job with its tasks
POST /api/jobs/{id}/cancel   stop it before its next task
GET  /api/watch              {watch}: the running watch, or null
POST /api/watch              {day, group, analyzers?, from_scan?, experiment?} -> start watching (replaces any watch)
POST /api/watch/stop         stop watching (its queued jobs still run)
```

An unknown group or a day with no scans is a 404, a malformed body a 422.
The portal's `tests/test_batch.py` answers these routes from a
hand-written fake transport: no test ties the two, so a route or shape
changed here must be changed there by hand, in the same PR.

## Rules

- **Unauthenticated.** It binds 127.0.0.1 by default; a wider bind is a
  site's decision (`GEECS_ANALYSIS_WORKER_BIND`), for a portal on another
  host on the lab network.
- **The watch is operator-started only.** The command has no flag that
  starts one, the app's lifespan starts the job thread and nothing else,
  and a restart forgets a running watch. Pinned by
  `tests/test_api.py::TestCommand` here and
  `ScanAnalysis/tests/test_worker.py::test_a_worker_starts_with_no_watch`.
  Do not add a start-up watch: automatic analysis with no operator behind
  it is a separate decision (a trigger from the engine's run-stop
  document would submit jobs like any client).
- **Nothing starts the worker but its unit or a person.** The portal's
  autostart (a detached child process no unit owned) was removed.
- **No facility value in code.** The experiment and the configs tree are
  arguments; the bind and port come from flags or `site.env`.
- **A consumer of scan folders, never a producer** (root `CLAUDE.md`
  § "Cross-package invariants"): the task queue it drives writes status
  records inside existing scan folders only.
- **pyplot runs on one thread.** `__main__` selects the Agg backend
  before anything is served, so before any figure is drawn (pyplot itself
  is already imported by then, with ScanAnalysis); the job thread is the
  only pyplot user.
- **A stop waits `STOP_SECONDS` (45) for the running task**, then the
  process exits and a longer task is abandoned mid-run (the job thread is
  a daemon); its claim goes stale after the queue's 180 s and a later job
  runs it again. The unit's `TimeoutStopSec` (60) sits above it: change
  the two together.

## Tests

`poetry install --with dev && poetry run pytest tests` from this
directory (its own env; a CI leg of its own).

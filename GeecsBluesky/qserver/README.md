# GEECS Queueserver Launch Assets

This directory contains the user-level launch mechanics for a local
bluesky-queueserver RE Manager, plus the startup profile itself
(`startup/startup.py`) that turns the launched manager into a runnable GEECS
worker: it builds the `GeecsSession`, defines the module-level `RE` the
manager keeps alive across queue items, subscribes the Tiled/s-file/scan-log
callbacks, registers the optimization loader when the `optimize` extra is
installed, and exposes `geecs_scan_request_plan` — the one plan every
`ScanRequest` (step, noscan, optimize) runs through. See
`startup/startup.py`'s module docstring for the import-order and
experiment-resolution contracts.

## Launch

From this directory:

```bash
./launch_re_manager.sh
```

The launcher expects `start-re-manager` on `PATH`. It checks
`127.0.0.1:6379` and starts a local Redis server when that port is not
answering. The Redis binary can be overridden with `QS_REDIS_SERVER`; the
default is `redis-server`.

```bash
QS_REDIS_SERVER=/path/to/redis-server ./launch_re_manager.sh
```

The startup directory defaults to `./startup` and can be overridden with
`QS_STARTUP_DIR`.

```bash
QS_STARTUP_DIR=/path/to/startup ./launch_re_manager.sh
```

The launcher passes `user_group_permissions.yaml` explicitly. Keep that
argument: without a permissions file, RE Manager may accept startup but reject
queue submissions with `success: False`, and the command-line failure can be
silent.

## Verify

In another terminal:

```bash
qserver status
qserver environment open
```

Once the environment is open, `geecs_scan_request_plan` is registered and
callable; submit a `ScanRequest` dict as its sole argument, for example:

```bash
qserver queue add plan '{"name": "geecs_scan_request_plan", "args": [{"mode": "noscan", "shots_per_step": 2, "acquisition": "free_run", "save_sets": ["UC_Test"]}], "item_type": "plan"}'
qserver queue start
```

`QS_EXPERIMENT` (or `config.ini`'s `[Experiment] expt`) must resolve before
the manager starts — the profile fails loud at import time otherwise (see
`startup/startup.py`).

## Troubleshooting

- **`queue add` returns `success: False` with no reason at the CLI** — the
  manager was launched without a permissions file, or the file lacks the
  group the client submits as (the `qserver` CLI uses `primary` by
  default). See `user_group_permissions.yaml`.
- **`queue start` succeeds but the item bounces back and nothing runs** —
  the startup profile does not define `RE = RunEngine({})` (paired with
  the launcher's `--keep-re`). Only the manager log shows the cause:
  `Run Engine is not found in the RE Worker environment`.
- **`qserver function execute` fails with `RE Manager must be in idle
  state`** — function execution requires a fully idle manager; it is not
  available while a plan is running *or paused*. Nothing in the GEECS
  design may rely on it mid-run.
- **Optimize-mode requests refused with "without an optimization loader
  registered" even though the `optimize` extra is installed** — the
  manager process predates the install. The worker inherits the manager's
  import state, so `environment close`/`open` is *not* enough: restart
  the manager process itself after any `poetry install` change
  (empirical, 2026-08-21 live checkpoint).
- **`queue start` re-runs an old failed item, or the queue keeps
  growing** — on plan failure the manager returns the failed item to the
  *front* of the queue (default `ignore_failures: false`). Clear the
  queue (or remove the item) before resubmitting a corrected request;
  clients that blindly add-and-start will re-execute the failed item
  first.
- **`qserver history get` shows a literal `'...'` entry** — the CLI
  truncates long histories for display; the newest items may not be
  shown. Read history through the API (`bluesky-queueserver-api`) for
  anything programmatic.

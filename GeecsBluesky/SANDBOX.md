# GeecsBluesky Sandbox

This package includes a local fake-hardware sandbox for external developers who
do not have access to the BELLA/GEECS network.  It runs the real Bluesky
`RunEngine`, GeecsBluesky device classes, UDP/TCP transport layer, and
`geecs_step_scan` plan against an in-process `FakeGeecsServer` bound to
localhost.

The sandbox does not require:

- GEECS hardware
- GEECS MySQL database access
- `~/.config/geecs_python_api/config.ini`
- Tiled
- BELLA network or VPN access

## Quick Start

Install the package from the monorepo checkout:

```bash
cd GeecsBluesky
poetry install
```

Run the sandbox scan:

```bash
poetry run python examples/sandbox_run_engine_scan.py
```

The script prints the run UID, plan name, number of event documents, exit status,
and the first event's data payload.  A normal run produces six events: three
motor positions times two shots per position.

## Programmatic Use

```python
from geecs_bluesky.testing import run_fake_step_scan

result = run_fake_step_scan(positions=(0.0, 0.5, 1.0), shots_per_step=2)
print(result.start_doc)
print(result.event_docs[0]["data"])
print(result.stop_doc)
```

This returns the emitted Bluesky documents in memory as `(name, doc)` pairs.
Consumers can attach their own RunEngine callback if they want to prototype a
databroker, Osprey, or MCP-facing ingestion layer.

## What It Exercises

- `FakeGeecsServer` speaks the same UDP/TCP wire protocol shape used by GEECS
  device servers.
- `GeecsMotor` moves a fake `Position (mm)` variable.
- `GeecsGenericDetector` waits on `acq_timestamp` advances and reads a fake
  `Signal` variable.
- `geecs_step_scan` emits ordinary Bluesky start, descriptor, event, and stop
  documents.
- Shot companion columns are included when the detector is configured for shot
  IDs.

## What It Does Not Exercise

- `GeecsDevice.from_db(...)` and `GeecsMotor.from_db_axis(...)`, because those
  require the GEECS database.
- `BlueskyScanner` construction from GUI scan configs, because that production
  path currently resolves devices through the GEECS database.
- Tiled persistence.  The sandbox keeps documents in memory by default.  A local
  Tiled server can be added later by subscribing a `TiledWriter` to the
  `RunEngine`.
- Native camera/image file saving.  The fake device only simulates scalar
  variables.

## Tests

The focused sandbox test is:

```bash
poetry run pytest tests/test_sandbox.py -v
```

The broader offline fake-server suite is:

```bash
poetry run pytest tests -m "fake_server and not integration" -v
```

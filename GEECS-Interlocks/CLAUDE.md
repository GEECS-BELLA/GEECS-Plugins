# GEECS-Interlocks

The software interlock server. It watches device values and external
status packets against one named configuration, and **reports** the
verdict to Master Control on a TCP stream. It never pauses a scan or
writes to a device: what a trip stops is Master Control's decision. The
operator's view is `docs/web_services/interlocks.md`; this file holds the
rules for changing the code.

## Layout

| Module | What it owns |
|---|---|
| `conditions.py` | The pure verdict: `evaluate(condition, Inputs, tripped=)` → `Verdict`. No I/O, no clock |
| `monitor.py` | The loop: one tick (`TICK_S`), every interlock's state, one immutable `Snapshot` per tick, events, per-check trends, arming |
| `sources.py` | `DeviceSource`: one `GeecsDevice` subscription per device, first connection retried in the background |
| `external.py` | UDP listeners (`ExternalSignals`), owned by the engine |
| `engine.py` | The unit: file, source, monitor, stream, listeners; `reload(path)` |
| `wire.py`, `status_server.py` | Master Control's frames and the stream that sends them |
| `config.py` | `config.ini [interlocks]` settings and the configuration paths |
| `builder.py`, `catalog.py` | The Builder's `Draft` (every edit) and what it lists (DB or file) |
| `present.py` | The words both readers use: the verdict line, values, the kit's state words |
| `logs.py` | Terminal, the window's `LogBuffer`, the daily file |
| `gui/` | The PyQt6 window (`gui` extra) |
| `web/` | The read-only page on the kit |
| `demo.py` | `--demo`, `--fake-devices`, the fake UDP sender, the demo catalog |

## Rules

- **Fail-safe, everywhere.** Not heard from yet, disconnected, silent past
  `stale_after_s`, not a finite number, an external signal before its first
  packet, a settings check not armed: all unsafe. A new condition kind or
  input path states its unsafe cases first and tests them.
- **Report only.** The `interlocks-report-only` import contract keeps scan
  execution and analysis out. No route, button or file write acts on the
  experiment.
- **One snapshot, many readers.** The window, the stream and the page draw
  the monitor's snapshot and compute no verdict of their own, so they never
  disagree. A new display reads `Engine.snapshot()`.
- **Master Control's frame holds one keyword.** It reads a frame only when
  it holds a single `SAFE` or `WARNING`. `summary` (the default) keeps
  that: its one keyword is the leading word, and "safe" / "warn" inside a
  name, reason or packet is broken up by `wire.clean`. The server's own
  reasons are worded so they never need cleaning (`clear again`, `alarm`).
  `legacy` repeats the keyword per interlock, so it suits Master Control
  only while one interlock is armed. Pinned over every mix of states in
  `tests/test_wire.py`.
- **UDP listeners outlive a reload.** Senders transmit only on a change, so
  a rebuilt listener would forget the last state and read unsafe until the
  next one. A reload that names a port it cannot bind is refused whole.
- **A packet holding `unsafe` is unsafe.** It contains "safe"; the decode
  checks `warning` and `unsafe` before `safe`.
- **The window must never take the server down.** PyQt6 ends the process
  on an exception that escapes a slot; `gui.log_uncaught` logs it instead,
  and `run_window` installs it. No DB or network call on the window's
  thread: the catalog loads on a worker and falls back to the file's
  devices.
- **The page never keeps a verdict it stopped hearing.** Past 2 s without
  a snapshot it says "No signal", shows the last verdict with its time and
  greys every chip (`web/static/interlocks.js`).
- **Logging has one owner.** `configure_logging` installs the handlers;
  uvicorn runs with `log_config=None`, because its own config closes every
  handler already installed. The daily file reopens if closed anyway.
- **Every Builder edit goes through `Draft`.** It validates the whole
  document the way the server loads it, and `save` writes atomically
  (temp file, then rename). The tab only translates clicks into calls.
- **Facility values have one home:** ports, bind hosts, the wire format and
  the log folder come from `config.ini [interlocks]`; without them every
  server listens on `127.0.0.1` only.

## Running and testing without the lab

`serve --demo` scripts every device; `--fake-devices` runs
`geecs_core.testing.FakeGeecsServer` devices through the real subscribe
path; both start a fake UDP sender. `listen HOST:PORT` prints what Master
Control receives. Tests need `poetry install --extras gui`: the window's
tests run offscreen (`QT_QPA_PLATFORM`, set by `conftest.py`), and CI and
`scripts/check.sh` fail on a PyQt6 import skip.

## Owed on hardware

- Whether a real device's subscription delivers frames between changes, so
  `stale_after_s` times receipt rather than activity (`acq_timestamp`).
- Master Control reading the `summary` frame on the lab PC.
- The window on Windows; the Builder's DB device list on the lab network.

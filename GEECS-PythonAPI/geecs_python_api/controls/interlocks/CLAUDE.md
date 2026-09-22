# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A software-interlock library for GEECS experiment control: it polls live device state
(via the GEECS control system) on background threads, evaluates safety conditions against
that state, and broadcasts a pass/fail flag per named monitor over a small TCP protocol so
other processes (e.g. a trigger/PLC bridge) can gate hardware on it. It ships both a
scriptable API and a Tkinter operator GUI.

This package (`geecs_python_api/controls/interlocks/`) is the reusable library.
`InterlockScripts/` (repo root, sibling of `GEECS-PythonAPI/`) is *not* part of this
package — it holds real per-experiment deployment scripts and ad-hoc notebooks that import
from here (e.g. `InterlockScripts/PW_interlocks/windmill_interlock.py`,
`InterlockScripts/interlock_client.py` as a debug TCP client, and
`InterlockScripts/general_interlock_gui.py` as a thin launcher). When changing behavior
here, check whether it affects those consumer scripts.

## Architecture

Layered, bottom-up:

1. **`monitor_conditions.py`** — condition builders. Each is a callable class:
   `__call__()` returns `True` = unsafe, `False` = safe, and each exposes
   `get_diagnostic_info()` for human-readable failure text.
   - `ThresholdCheck` — value vs. threshold with an operator (`<`, `>`, `<=`, `>=`, `==`, `!=`).
   - `AlignmentCheck` — one or more value/target/tolerance triples (accepts scalars or
     equal-length lists for multi-axis alignment).
   - `MultiCheck` — OR-combines other checks/conditions (including nested `MultiCheck`s);
     short-circuits on the first unsafe condition and surfaces its diagnostic.
   - `CustomCheck` — wraps an arbitrary `predicate(device_state_dict) -> bool`.
   - `ConfigurationChangeCheck` — not a plain safety check but an arm/disarm gate: captures
     a baseline of variable values via `arm_interlock()`, then reports unsafe if any of them
     drift from that baseline (used to force re-arming after e.g. exposure/gain changes).
   - `ExternalStatusCheck` — the one check that does *not* go through `DeviceMonitorGroup` at
     all. Wraps an `ExternalStatusListener` (see below) for signals that aren't a GEECS device
     variable. Unsafe until the listener's first packet arrives, then mirrors its last known
     state with no staleness timeout (see fail-safe section below for why).
   - Every other check funnels device reads through `DeviceMonitorGroup`, so all of them
     automatically inherit its staleness fail-safe — there is no separate staleness wrapper.

   **`external_status_listener.py`** (`ExternalStatusListener`, `default_decode`) — a
   background UDP receiver for non-GEECS, one-shot status signals (e.g. a motion controller
   on another computer that unicasts a single "safe"/"warning" packet on state change, not a
   continuous stream). Caches the last decoded state under a lock; `decode` is a pluggable
   `bytes -> bool | None` function (default: case-insensitive "safe"/"warning" text match).
   Create these via `InterlockConstructor.add_external_listener()` rather than directly, so
   `start()`/`stop()` are tied to the constructor's lifecycle like devices are.

2. **`device_monitor_group.py`** (`DeviceMonitorGroup`) — owns `GeecsDevice` instances
   keyed by short alias, manages `subscribe_var_values()`/`unsubscribe_var_values()`, and is
   the single read path (`get_value`, `get_timestamp`) condition builders use. It tracks each
   device's `acq_timestamp` and, if it hasn't advanced within `staleness_timeout_ms`,
   `get_value()` returns `None` for that device — which every condition builder above treats
   as unsafe. Initializes `GeecsDevice.exp_info` via `GeecsDatabase.collect_exp_info(exp_name)`
   in `__init__`, so constructing one talks to the GEECS database immediately.

3. **`geecs_interlock_server.py`** (`InterlockServer`) — low-level TCP server plus a generic
   `register_monitor(name, check_func, interval, diagnostic_func, safe_release_checks)`
   polling loop. Knows nothing about GEECS devices; `check_func` can be any zero-arg
   callable. Each registered monitor runs on its own daemon thread and calls
   `set_interlock()`, which is the thread-safe state store consumed by the TCP handler.
   - Safety-state latching: an unsafe (`True`) result is applied immediately, but clearing
     back to safe requires `safe_release_checks` consecutive safe polls in a row (default
     varies by caller — 5 here, 3 in `InterlockConstructor.add_monitor`). This debounces
     flapping conditions from a false "safe" pulse.
   - Wire protocol: one TCP server, N clients. Per client, every 0.1s it sends a 4-byte
     big-endian length prefix followed by a UTF-8 message, one `" | "`-joined
     `SAFE: <name>` / `WARNING! <name> - <diagnostic>` segment per registered monitor. See
     `InterlockScripts/interlock_client.py` for the reference client-side framing/parsing.

4. **`interlock_constructor.py`** (`InterlockConstructor`) — the facade scripts/GUI use.
   Combines a `DeviceMonitorGroup` with an `InterlockServer`, exposes `add_device` /
   `add_monitor` / `start` / `stop` / context-manager support, and queues monitor
   registrations (`monitor_registrations: {name: (check_func, interval, safe_release_checks)}`)
   so they can be declared before the server exists and are (re-)registered on `start()`.
   `add_monitor` auto-wires `check_func.get_diagnostic_info` as the server's
   `diagnostic_func` when present, so diagnostics "just work" for any condition builder that
   implements that method.
   - **Naming note:** the docs (`README.md`, `INTERLOCK_ABSTRACTION_GUIDE.md`) still refer to
     this class as `InterlockBuilder` in code samples — that name doesn't exist in the code,
     the real class is `InterlockConstructor`. Don't reintroduce an `InterlockBuilder` alias
     without checking whether the docs were meant to be updated instead.

5. **`interlock_config.py`** — JSON (de)serialization of an `InterlockConstructor`'s devices
   and monitor tree (`save_configuration` / `load_configuration`), format tag
   `"geecs-interlock"`, versioned (`version: 1`). `_condition_to_dict`/`_condition_from_dict`
   are an explicit `isinstance` switch over the condition builder types above — adding a new
   condition builder that should be persistable means extending both functions.
   `load_configuration` refuses to run against a live (`interlock.server is not None`)
   constructor and validates the experiment name matches.
   - **`ExternalStatusCheck` is not persistable here** — it wraps a live listener object
     (bound socket, decode callback), not plain data, so it isn't in the `isinstance` switch.
     Scripts that use it re-create the listener/check in code on every run rather than
     round-tripping it through a saved JSON config.

6. **`interlock_gui.py`** (`InterlockGUI`, `launch_interlock_gui`,
   `launch_general_interlock_gui`) — Tkinter operator console over an `InterlockConstructor`.
   It is a thin view/controller: it never evaluates safety logic itself, only starts/stops
   the constructor, arms `ConfigurationChangeCheck`s, and edits condition parameters
   (threshold/tolerance) live via `_condition_rows`/`apply_device_settings`. Two entry points:
   `InterlockGUI(interlock, armable_checks=[...])` wraps a constructor you built in code (see
   the windmill example referenced in `README.md`); `launch_general_interlock_gui()` prompts
   for a GEECS experiment name against the database and builds an empty
   `InterlockConstructor` interactively — this is what `__main__.py` and
   `InterlockScripts/general_interlock_gui.py` invoke.
   - The Interlocks tab's "Add external signal" button creates an `ExternalStatusListener` +
     `ExternalStatusCheck` pair directly (`_open_add_external_signal_dialog`) — the only
     condition type addable from that tab rather than the device-scoped Devices tab, since it
     has no GEECS device behind it. "Edit selected interlock" on it
     (`_open_external_status_edit_dialog`) can change its label/host/port; both require the
     interlock to be stopped first, same as every other edit dialog.
   - `_condition_description`/`_leaf_current_value_text` have an `isinstance` case per
     condition builder type (same pattern as `interlock_config.py`'s serializer) — a new
     condition type needs a case there too or it silently falls back to a generic label.

`__init__.py` re-exports the public surface; prefer importing from
`geecs_python_api.controls.interlocks` rather than reaching into submodules.

## Fail-safe design (don't weaken without a safety review)

Everything defaults toward "unsafe" on ambiguity:
- Missing/`None` device data → unsafe (`ThresholdCheck`, `AlignmentCheck`, `CustomCheck`).
- Stale device data (no new `acq_timestamp` within `staleness_timeout_ms`) → `DeviceMonitorGroup.get_value()` returns `None` → unsafe, automatically, for every check reading that device.
- Any exception inside a check → caught and treated as unsafe, never propagated to silently disable a monitor.
- An unarmed `ConfigurationChangeCheck` → unsafe until `arm_interlock()` is called.
- No packet received yet on an `ExternalStatusListener` → unsafe (`ExternalStatusCheck`).
  Deliberately *not* extended with a staleness timeout after that: the sender only transmits
  on change (per its own design), so silence is expected and doesn't by itself mean it died —
  a naive "hasn't updated recently" timeout would misfire and flag it unsafe during any long
  period where nothing changed. If a sender is ever changed to also heartbeat periodically,
  layer staleness on top via `ExternalStatusListener.get_last_received_at()` in a custom check
  rather than assuming silence-since-last-message is always safe to trust indefinitely.
- State only clears to safe after `safe_release_checks` consecutive safe polls; an unsafe result always latches on the very next poll.

## Running things

There's no dedicated test suite for this package. Development commands run from the
monorepo root (`GEECS-Plugins/`, which has the Poetry workspace and Ruff config) or from
`GEECS-PythonAPI/` (its own Poetry project):

```bash
# install (from GEECS-Plugins/, pulls in geecs-pythonapi as a path dependency)
poetry install

# lint / format, matching what pre-commit runs
poetry run ruff check .
poetry run ruff format .

# launch the standalone experiment-picker GUI
poetry run python -m geecs_python_api.controls.interlocks
# equivalent launcher used at the experiment site:
poetry run python InterlockScripts/general_interlock_gui.py

# run a real deployment script (registers devices/checks in code, then starts the server)
poetry run python InterlockScripts/PW_interlocks/windmill_interlock.py

# debug client: connect to a running server and print live status
poetry run python InterlockScripts/interlock_client.py
```

Deployment scripts under `InterlockScripts/` read `SERVER_IP`/`SERVER_PORT` from a
`.env` file (via `python-dotenv`) in that directory rather than hardcoding them.

## Dev/deploy split

There's a second git worktree for freely experimenting on this code without touching
what's running at the experiment — see `InterlockScripts/WORKFLOW.md` for how the
deploy checkout and the `dev/interlocks` worktree relate and how to promote tested
changes between them.

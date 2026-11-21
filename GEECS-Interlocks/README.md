# GEECS-Interlocks

Software interlocks for GEECS experiments. The server watches device values,
and status packets from programs outside GEECS, against one of the
experiment's named configurations (`interlocks/<name>.yaml`), and reports the
verdict to Master Control on a TCP stream. It only reports: what a trip stops
is Master Control's decision.

An interlock holds one or more conditions and trips when any of them is
unsafe, so several checks can be reported under one name. The condition
kinds: `threshold`, `alignment`, `stale`, `external` (UDP packets) and
`config_change` (settings held at the values recorded when an operator arms
them). The schema and its reference live in GEECS-Schemas
(`geecs_schemas.interlocks`).

## Try it locally

No lab network, DB or Master Control needed:

```bash
poetry install -E gui
poetry run geecs-interlocks serve --demo --arm-at-start --interlocks examples/demo_interlocks.yaml
```

In a second terminal, see exactly what Master Control would receive:

```bash
poetry run geecs-interlocks listen 127.0.0.1:5001
```

The server window opens with it (the `gui` extra; `--no-window` runs
headless):

- **Monitor** — the verdict strip, Master Control's clients, Arm settings,
  Reload file, Always on top, the interlocks worst first (a group expands
  into its checks), and the server log under them.
- **Builder** — pick a device, pick a variable, set the check, and choose
  the interlock it joins ("Add to" an existing one groups it). Edits are a
  draft until **Save and reload**, which validates, shows the diff, writes
  the file and reloads without dropping Master Control.

The demo scripts every device, sends the external signal's UDP packets
itself and moves the windmill's settings now and then, so every state shows
up within a couple of minutes. `--fake-devices` in place of `--demo` starts
fake GEECS devices on localhost and reads them through `geecs_core`, the same
path the lab uses.

## In the lab

```bash
poetry run geecs-interlocks serve --config <name>
```

`--experiment <experiment>` when `config.ini` names none; `--arm-at-start`
arms the `config_change` interlocks once their settings have arrived.

## The web client

`serve` also serves a read-only page for any browser on the network,
`http://<server>:8600/` (`config.ini [interlocks] web_host`, `web_port`;
`--no-web` to leave it out): the verdict, the interlocks worst first, and an
inspector with each check's recent trend against its limit and its history.
The selection is in the address (`?interlock=<name>`), so a link opens it.
If the page stops hearing from the server for 2 s it says **No signal**,
shows the last verdict with its time and greys every state: it never keeps
showing a verdict it is no longer receiving. Nothing on the page writes.

## The stream

Two message formats, chosen with `config.ini [interlocks] wire_format` or
`--wire-format` (details in `geecs_interlocks/wire.py`):

- `summary` (default): the verdict first: `SAFE | OK: <name>` or
  `WARNING! TRIPPED | TRIPPED: <name> - <reason>`.
- `legacy`: the original server's format:
  `SAFE: <name> | WARNING! <name> - <reason>`.

Master Control reads a frame only when it holds a single `SAFE` or
`WARNING`. `summary` always does; `legacy` repeats the keyword per
interlock, so it works with Master Control only while one interlock is
armed. `listen` reads either.

Ports and bind addresses (the stream, the web client, the external signals)
come from `config.ini [interlocks]`; see `geecs_interlocks/config.py`. With
none set, everything listens on `127.0.0.1` only.

## The log

Every line goes to the terminal, the window's log pane and a daily file,
`interlocks-YYYY-MM-DD.log`, kept 30 days, in `config.ini [interlocks]
log_dir` (default `~/.config/geecs_python_api/interlock_logs/`).

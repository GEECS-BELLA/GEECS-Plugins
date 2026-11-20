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

## The stream

Two message formats, chosen with `config.ini [interlocks] wire_format` or
`--wire-format` (details in `geecs_interlocks/wire.py`):

- `legacy` (default): the original server's format, which Master Control
  reads today: `SAFE: <name> | WARNING! <name> - <reason>`.
- `summary`: the verdict first: `SAFE | OK: <name>` or
  `WARNING! TRIPPED | TRIPPED: <name> - <reason>`.

In both, a frame holds `WARNING` exactly when it is not an all-clear.
`listen` reads either.

Ports and bind addresses (the stream, the web client, the external signals)
come from `config.ini [interlocks]`; see `geecs_interlocks/config.py`. With
none set, everything listens on `127.0.0.1` only.

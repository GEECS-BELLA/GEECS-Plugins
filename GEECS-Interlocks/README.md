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
(`geecs_schemas.interlocks`); configurations are edited in the configurator.

## Try it locally

No lab network, DB or Master Control needed:

```bash
poetry install
poetry run geecs-interlocks serve --fake-devices --arm-at-start --interlocks examples/demo_interlocks.yaml
```

In a second terminal, see exactly what Master Control would receive:

```bash
poetry run geecs-interlocks listen 127.0.0.1:5001
```

and open the read-only page at `http://127.0.0.1:8600/`. The fake devices
are real GEECS device servers on localhost, read through `geecs_core`; the
demo also sends the external signal's UDP packets itself and moves the
windmill's settings now and then, so every state shows up within a couple
of minutes.

## In the lab

```bash
poetry run geecs-interlocks serve --config <name>
```

`--experiment <experiment>` when `config.ini` names none; `--arm-at-start`
arms the `config_change` interlocks once their settings have arrived. The
server runs until SIGTERM or Ctrl-C and logs to stderr.

## The stream

Two message formats, chosen with `config.ini [interlocks] wire_format` or
`--wire-format` (details in `geecs_interlocks/wire.py`):

- `summary` (default): the verdict first: `SAFE | OK: <name>` or
  `WARNING! TRIPPED | TRIPPED: <name> - <reason>`.
- `legacy`: the original server's format:
  `SAFE: <name> | WARNING! <name> - <reason>`.

Master Control reads a frame only when it holds a single `SAFE` or
`WARNING`. `summary` always does; `legacy` repeats the keyword per
interlock. `listen` reads either.

Ports and bind addresses (the stream, the web page, the external signals)
come from `config.ini [interlocks]`; see `geecs_interlocks/config.py`. With
none set, everything listens on `127.0.0.1` only.

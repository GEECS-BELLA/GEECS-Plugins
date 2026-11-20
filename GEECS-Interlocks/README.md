# GEECS-Interlocks

Software interlocks for GEECS experiments. The server watches device values
against one of the experiment's named configurations (`interlocks/<name>.yaml`) and reports the verdict to Master
Control on a TCP stream. It only reports: what a trip stops is Master
Control's decision.

## Try it locally

No lab network, DB or Master Control needed:

```bash
poetry install -E gui
poetry run geecs-interlocks serve --demo --interlocks examples/demo_interlocks.yaml
```

In a second terminal, see exactly what Master Control would receive:

```bash
poetry run geecs-interlocks listen 127.0.0.1:5001
```

`--fake-devices` in place of `--demo` starts fake GEECS devices on localhost
and reads them through `geecs_core`, the same path the lab uses.

## In the lab

```bash
poetry run geecs-interlocks serve --config <name>          # --experiment <experiment> if config.ini has none
```

Ports and bind addresses come from `config.ini [interlocks]`; see
`geecs_interlocks/config.py`.

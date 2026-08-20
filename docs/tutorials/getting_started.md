# Getting started

This page takes you from nothing to a working GEECS-Plugins development
setup: the toolchain, the repo, the one config file everything reads,
and a smoke test that proves it all works. It is also the reference the
repo's AI tooling uses when it helps a new developer get set up.

!!! tip "You don't have to do this alone"
    This repo is developed heavily with AI coding agents. If you have
    [Claude Code](https://claude.com/claude-code) installed, you can
    clone the repo, launch `claude` from the repo root, and type
    `/get-started` — the agent will walk you through everything on this
    page interactively, fix problems as they come up, and help you plan
    your first project. And if you're ever confused or stuck, contact
    Sam — questions are welcome.

## What you need first

- **git** — on macOS it ships with Xcode command-line tools; on Windows
  use [Git for Windows](https://git-scm.com/download/win).
- **Python 3.11** — specifically 3.11 (`>=3.11,<3.12`); a newer or older
  Python will fail with a version error at install time. Install from
  [python.org](https://www.python.org/downloads/) alongside any other
  Pythons you have.
- **Poetry** — the package manager used throughout:
  [installation instructions](https://python-poetry.org/docs/#installation).

## Get the code

```bash
git clone https://github.com/GEECS-BELLA/GEECS-Plugins.git
cd GEECS-Plugins
git checkout dev
```

!!! note "Why `dev`?"
    Until the planned branch cutover, `dev` is the active development
    line and the default base for new work; `master` carries the legacy
    scanner. See `CONTRIBUTING.md` at the repo root for the branch
    topology.

If you'll work on analyzer configurations, also clone the sister config
repo **as a sibling directory** (several tools expect it at
`../GEECS-Plugins-Configs/`):

```bash
cd ..
git clone https://github.com/GEECS-BELLA/GEECS-Plugins-Configs.git
cd GEECS-Plugins
```

## Install

From the repo root:

```bash
poetry install
poetry run pre-commit install
```

The first command builds the main development environment (ImageAnalysis,
ScanAnalysis, GEECS-Data-Utils, and friends run from it). The second
installs the git hooks that auto-format code on commit.

Some packages keep their own environment instead — `GeecsBluesky`,
`GeecsCAGateway`, `GEECS-Console` and others are installed by running
`poetry install` inside that package's directory. You only need those
when you work on them; `scripts/check.sh` knows which is which.

!!! warning "The most common setup failure"
    If any poetry command prints *"Current Python version (3.x) is not
    allowed"*, your default `python` is not 3.11. Point poetry at the
    right interpreter and reinstall:

    ```bash
    poetry env use /path/to/python3.11
    poetry install
    ```

## The config file

Everything in the suite reads one small INI file:

```
~/.config/geecs_python_api/config.ini
```

Create the directory and file if they don't exist. A representative
file (ask a teammate or Sam for the right values for your machine —
the data path in particular depends on how the data share is mounted):

```ini
[Paths]
geecs_data = Z:/data/
scan_analysis_configs_path = C:/path/to/GEECS-Plugins-Configs/scan_analysis_configs
image_analysis_configs_path = C:/path/to/GEECS-Plugins-Configs/image_analysis_configs

[Experiment]
expt = Undulator
rep_rate_hz = 1
```

What each section is for, and who reads it:

| Section / key | Purpose | Read by |
|---|---|---|
| `[Paths] geecs_data` | Root of the experiment data share; also where `Configurations.INI` (database credentials) lives | Everything that touches scan data or the GEECS database |
| `[Paths] scan_analysis_configs_path` | Analyzer/diagnostic YAMLs in the configs repo | ScanAnalysis, LiveWatch, ConfigFileGUI |
| `[Paths] image_analysis_configs_path` | Camera/1D analyzer configs in the configs repo | ImageAnalysis |
| `[Paths] scanner_config_root_path` | Scanner save-element configs (optional) | GEECS Console |
| `[Experiment] expt` | Your experiment's GEECS name (e.g. `Undulator`) | Nearly everything |
| `[Experiment] rep_rate_hz` | Machine rep rate, for shot-count estimates | Scanner/Console |
| `[tiled] uri`, `[tiled] api_key` | Tiled data-server access (optional) | GeecsBluesky, Scan Browser |
| `[epics] ca_addr_list` | EPICS client addressing (optional, gateway clients) | GeecsBluesky and other CA clients |

Database credentials are **not** stored here: tools follow
`[Paths] geecs_data` to the `Configurations.INI` file on the data share
and read the `[Database]` section there. If the data share is mounted
and `geecs_data` is right, database access follows automatically.

## Prove it works

**Offline** (no lab network needed) — run the gateway's self-checking
demo against an in-process fake device:

```bash
cd GeecsCAGateway
poetry install
poetry run python -m geecs_ca_gateway.demo
```

Success looks like two `[self-check]` lines confirming the round trip;
the demo then keeps serving its fake PVs until you stop it with
`Ctrl-C` — it is a server, so it won't exit on its own. If the
self-check lines appeared, your toolchain is healthy end to end.

**On the lab network (or VPN)** — prove the config → database chain with
a three-line query:

```python
from geecs_core.db.geecs_db import GeecsDb

devices = GeecsDb.get_all_experiment_variables("Undulator")
print(f"{len(devices)} devices tracked")
```

Run it from the `GeecsCAGateway` directory with
`poetry run python your_script.py`. A device count means config file,
data share, and database all line up.

!!! note "Off the lab network?"
    Database and device calls block for over a minute before timing out
    when the lab isn't reachable. Everything offline — installs, tests,
    the fake-server demo, working with already-copied data — works
    fine from anywhere.

## Contributing your work

The full workflow contract is `CONTRIBUTING.md` at the repo root; the
short version for a new developer:

- Work on a branch, never directly on `dev` or `master`. New developers
  get a personal branch (`users/<yourname>`) that feature work merges
  into; moving work from there into the mainline is a reviewed pull
  request.
- Commit with `./scripts/commit.sh -m "..."` after `git add` — a plain
  `git commit` is aborted by the auto-formatting hooks by design.
- Run `./scripts/check.sh` before opening a pull request; it runs the
  same lint and tests the CI does.

If you work with an AI agent, it knows all of this — the repo ships
agent instructions (`CLAUDE.md` files) and skills (`/land`, `/check`,
`/env-doctor`) that encode the workflow, so you can let it drive the
mechanics while you watch.

## Where to next

- [Analysis tutorial](analysis.md) — the end-to-end walkthrough:
  configure an analyzer, run it on a real scan, get summary figures.
- [Home page](../index.md) — the map of what each package does and how
  they fit together.
- [Camera images over PVA](../geecs_gateway/image_pvs.md) — how to
  consume live camera images in three lines of `p4p`.

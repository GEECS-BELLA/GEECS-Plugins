# GEECS-Console

The greenfield PySide6 operator console for GEECS — the front-end of the
Bluesky/gateway target architecture.  It builds one
`geecs_schemas.ScanRequest` per scan and submits it to the `geecs-bluesky`
engine (`BlueskyScanner`).  Successor to GEECS-Scanner-GUI, built clean
against the new architecture instead of retrofitting the PyQt5 GUI.

## Launch

```bash
cd GEECS-Console
poetry install
poetry run python main.py
```

The window opens with zero network and zero configs: empty listings, health
chips `unknown`, Start disabled until a save set is selected.  With a configs
repo available (`GEECS_SCANNER_CONFIG_DIR` or config.ini
`[Paths] scanner_config_root_path`), the combos and lists populate from it.

## Architecture rules (binding)

- **`geecs_python_api` is NEVER imported** — not at day 1, not ever
  (decided 2026-07-10, `Planning/cutover_strategy/00_overview.md`).  Manual
  set/readback goes through gateway PVs (CA monitor on the readback, put to
  `:SP`); DB autocompletes go through `GeecsDb` from `geecs_core`
  (a declared direct dependency).  A test pins this.
- Core dependencies: **PySide6 + geecs-bluesky + geecs-schemas +
  geecs-core** (plus geecs-data-utils for the scan browser).
- PySide6, never PyQt (LGPL; this repo is public).
- The GUI submits `ScanRequest` objects — there is no other submission shape.

## Tests

```bash
QT_QPA_PLATFORM=offscreen poetry run pytest -q
```

All tests are hermetic (no network, no configs repo, fake submitter/engine).

See `CLAUDE.md` for the screen map, seams, and developer context.

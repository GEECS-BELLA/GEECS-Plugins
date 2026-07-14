# /env-doctor — diagnose and fix a package's Poetry environment

Every package in this monorepo is an independent Poetry project, and the
recurring failure modes are all environment-shaped. Run this when a
package's tests/imports fail for reasons that smell like setup rather
than code.

## Arguments

`$ARGUMENTS` — the package directory (e.g. `GeecsBluesky`). If empty,
diagnose the package the current task is about.

## Checks, in order

1. **Python version.** All packages need `>=3.11,<3.12` (except
   `LogMaker4GoogleDocs`, `>=3.9`). Symptom: any poetry command prints
   "Current Python version (3.10.x) is not allowed". Fix — a stale env
   must be *removed*, not just re-pointed:

   ```bash
   cd <Package>
   poetry env list                      # find the stale one
   poetry env remove <stale-env-name>
   poetry env use /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
   poetry install
   ```

2. **Extras.** Plain `poetry install` does NOT install extras, and the
   hardware/Tiled test layers skip or fail without them:

   | Package | Install for full test suite |
   |---|---|
   | `GeecsBluesky` | `poetry install --extras "ca tiled"` |
   | `GeecsCAGateway` | `poetry install` (self-contained) |
   | `GEECS-Console` | `poetry install` (+ `--extras optimization` for the Xopt path) |
   | `GEECS-LogTriage` | `poetry install` |

   Symptom of missing `ca`: the whole GeecsBluesky mock-device test layer
   collects as "1 skipped" (`pytest.importorskip("aioca")`).

3. **Which env runs which tests** (match CI, `.github/workflows/unit-tests.yml`):
   - **Root env** (`poetry install` at repo root): root `tests/`,
     `ImageAnalysis`, `ScanAnalysis`, `GEECS-Data-Utils`, `GEECS-Schemas`
     — e.g. `poetry run pytest GEECS-Data-Utils/tests -m "not integration and not gui" -q`
     from the repo root. These packages' own dirs have no dev env — running
     `pytest` inside them fails with "Command not found".
   - **Package env**: `GeecsBluesky`, `GeecsCAGateway`, `GEECS-Console`,
     `GEECS-LogTriage` run their own suites from inside the package dir.
   - `GEECS-Console` additionally needs `QT_QPA_PLATFORM=offscreen`.

4. **Worktrees have their own envs.** A fresh worktree needs its own
   `poetry install` per package you touch there; envs do not follow
   branches. Watch for poetry creating a *root-level* venv when invoked
   from the worktree root by accident — always `cd` into the package.

5. **Network-dependent tests.** `integration` markers (lab DB at
   192.168.6.14) are deselected by default — leave them that way off-site;
   a MySQL attempt off-network blocks ~75 s before timing out. Never run
   `test_bluesky_scanner.py` (top-level hardware script) without lab
   access and operator awareness.

6. **Pre-commit.** `pre-commit` runs from the root env. Committing with
   plain `git commit` gets aborted by the auto-fixers ("files were
   modified by this hook") — use `./scripts/commit.sh -m "..."`.

Report what was wrong and what you fixed; if everything checks out, say
so and move the diagnosis back to the code.

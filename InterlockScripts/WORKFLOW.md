# Interlock dev/deploy workflow

The interlock code (this folder plus
`GEECS-PythonAPI/geecs_python_api/controls/interlocks/`) stays inside the
`GEECS-Plugins` monorepo. To get a safe sandbox for experimenting without
touching what's actually running at the experiment, we use a second **git
worktree** — a separate folder on disk sharing this repo's history, checked
out on its own branch, with a sparse-checkout so it only pulls in the two
trees the interlock code needs.

## The two folders

- **Deploy** — `C:\GEECS\Developers Version\source\GEECS-Plugins` (this
  checkout). Whatever branch this is on is what's actually running at the
  experiment. Only bring in changes here once they're tested.
- **Dev** — `C:\GEECS\Developers Version\source\GEECS-Plugins\_dev-interlocks`,
  on branch `dev/interlocks`. Edit, break, and commit freely here. It's a
  git worktree nested inside the deploy folder (ignored by git — see
  `.gitignore`), sparse-checked-out to only the trees the interlock code
  needs, not the rest of the monorepo's plugins.

Both folders share one `.git`, so `git branch -a`, `git log`, and
`git worktree list` from either folder show the full picture.

## Setting up the dev worktree (one-time)

From the deploy folder:

```powershell
git worktree add -b dev/interlocks "_dev-interlocks"
cd "_dev-interlocks"
git sparse-checkout init --cone
git sparse-checkout set GEECS-PythonAPI InterlockScripts ImageAnalysis GEECS-Data-Utils
cd GEECS-PythonAPI
poetry config virtualenvs.in-project true
poetry install
```

Notes on the sparse-checkout set:

- It includes **all of `GEECS-PythonAPI/`**, not just the `interlocks/`
  subfolder — the interlock library imports other parts of
  `geecs_python_api` (e.g. `GeecsDevice`, the database interface), so it
  needs the whole package to run.
- It also includes `ImageAnalysis/` and `GEECS-Data-Utils/` — `poetry install`
  fails without them, since `GEECS-PythonAPI/pyproject.toml` depends on both
  as local sibling path dependencies (`{ path = "../ImageAnalysis" }`, etc.),
  not something specific to interlocks.
- The size win versus a full monorepo checkout comes from skipping the
  *other* plugins (ScanAnalysis, GEECS-Scanner-GUI, etc.) that the interlock
  code doesn't touch.

## Day to day

- Do interlock development in the dev worktree, on `dev/interlocks`. Commit
  as you go.
- When a change is tested and you trust it, promote it to deploy:

  ```powershell
  cd "C:\GEECS\Developers Version\source\GEECS-Plugins"
  git merge dev/interlocks          # promote everything tested so far
  # or, for a single change:
  git cherry-pick <commit-hash>
  ```

- No file copying, no separate repo, no dependency pins to juggle — it's the
  same repo, just two working folders.

## Removing the dev worktree

If you ever want to tear it down:

```powershell
git worktree remove "..\GEECS-Plugins-interlocks-dev"
```

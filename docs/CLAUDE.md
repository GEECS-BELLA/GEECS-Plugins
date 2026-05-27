# docs/ — Developer Context for Claude

This directory holds the **mkdocs-material** site published at the project's
docs URL. Site config is `mkdocs.yml` at the repo root; everything served is
authored as Markdown (`.md`) or Jupyter notebooks (`.ipynb`) under `docs/`.

The site is organised into top-level tabs in `mkdocs.yml`'s `nav:`. The
canonical ordering is:

```
Home → Apps → <per-package tabs> → Skills
```

The Apps tab is the cross-package, user-facing entry point (ConfigFileGUI,
LiveWatch, the end-to-end tutorial). Each per-package tab is owned by the
package it documents and follows roughly Diátaxis: Overview (explanation),
Tutorial (when applicable), How-To pages, Examples (notebooks), API
Reference (mkdocstrings-generated). Skills sits last because it's
experimental tooling rather than core suite.

## When to put what where

| Kind of content | Lives in |
|---|---|
| Cross-package user workflow (uses ≥2 packages) | `docs/apps/` |
| Single-package overview, how-to, or reference | `docs/<package>/` |
| API reference auto-generated from docstrings | `docs/<package>/api/` (uses mkdocstrings) |
| Hands-on example using real data | `docs/<package>/examples/*.ipynb` |
| Hero landing surface | `docs/index.md` |

If you find yourself wanting to document a workflow that spans two packages
(say, "configure analysis in ConfigFileGUI then run via LiveWatch"), it
belongs under `apps/`, not under either constituent package.

## Building & previewing locally

From the repo root:

```bash
poetry run mkdocs serve     # live-reload preview at http://127.0.0.1:8000
poetry run mkdocs build     # static site → ./site/, clean output expected
```

`mkdocs build` should exit 0. Treat any new `ERROR` in the output as a
regression to fix before merging, even when the exit code is 0 (some
errors are non-fatal but indicate broken pages).

The build pulls in path-installed dev packages (Scanner GUI, ImageAnalysis,
ScanAnalysis, etc.) because mkdocstrings imports them to read docstrings.
Missing imports surface as `griffe:` or `mkdocstrings:` warnings.

## Screenshots — author them, don't placeholder them

When documenting a PyQt5 GUI, **generate real screenshots** rather than
shipping `![TODO: …]` placeholders. The technique is headless and
reproducible:

1. The script `scripts/generate_docs_screenshots.py` drives each app under
   `QT_QPA_PLATFORM=offscreen`, walks it through representative states
   (loading a sample config, toggling panels, picking a group), and
   captures each state via `widget.grab().save(path, "PNG")`.
2. Output PNGs live under `docs/<tab>/assets/` — e.g.
   `docs/apps/assets/configgui_02_analyzer_camera.png`.
3. Naming convention: `<app>_NN_<state>.png` so screenshots are
   self-sorting and the state is greppable from the filename.
4. Re-run the script whenever the GUIs change in a user-visible way:

   ```bash
   poetry run python scripts/generate_docs_screenshots.py
   ```

   It reads sample configs from the sister `GEECS-Plugins-Configs` repo
   (expected at `../GEECS-Plugins-Configs/scan_analysis_configs/`); if
   that path moves, edit the `SAMPLE_CONFIGS` constant.

The screenshots **don't** match macOS/Windows native chrome — offscreen QPA
renders a generic Qt look. That's a feature for documentation: consistent
across platforms, free of distracting OS-specific elements.

Extending the script for a new app: add a `shoot_<app>(app)` function that
imports the window class, instantiates it, drives it through interesting
states with `app.processEvents()` between steps, and calls `_save(w,
"<app>_NN_<state>.png")`. The `_settle()` helper and `_save()` helper are
already there.

## Conventions for new pages

- Headings use sentence case for body sections; the page title is the only
  H1.
- Internal links use relative paths (`../image_analysis/overview.md`), not
  absolute. mkdocs validates these at build time.
- Code samples that exercise the API should be runnable as-is. If a sample
  has external dependencies, mark them explicitly in prose, don't hide them.
- For "Where to start" / "Where to next" sections, prefer linking to
  concrete next pages, not back to the index.
- Material grid cards (`<div class="grid cards" markdown>`) work; they
  require `md_in_html` in `markdown_extensions` (already enabled).
- Admonitions: `!!! note` / `!!! warning` / `!!! tip` for callouts;
  `??? note` for collapsible details.

## Notebook hygiene

`mkdocs-jupyter` is strict about cell metadata:

- Every cell needs a non-empty `id` matching `^[a-zA-Z0-9-_]+$`. Jupyter
  4.5+ assigns these automatically; older notebooks may have empty IDs.
- Notebooks declaring `nbformat_minor < 5` reject the `id` field; either
  bump `nbformat_minor` to `5` or remove the IDs.
- If you add an `.ipynb` to `docs/` and the build errors with
  `NotebookValidationError: '' does not match …`, the cell-ID/nbformat
  combination is the cause. A small fixer is in `scripts/` heritage
  (currently inline in the relevant commit) — it bumps `nbformat_minor`
  to 5 and patches missing IDs with short uuids.
- Notebooks under `docs/` are rendered by mkdocs-jupyter even if they're
  not listed in `nav:`. Move broken-but-archived notebooks out of `docs/`
  or fix them; don't leave them stranded.

## What `docs/` is *not* for

- **Internal architecture notes for a single package** belong in that
  package's `CLAUDE.md`, not here.
- **Release notes per package** belong in each package's `CHANGELOG.md`.
- **Long discussion threads** (design decisions, future plans) belong in
  GitHub issues/PRs; pull only the conclusions into `docs/`.

Documentation in this directory is for **users of the suite**. Internal
development context lives next to the code it describes.

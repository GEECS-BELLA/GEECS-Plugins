# docs/ — Developer Context for Claude

This directory holds the **mkdocs-material** site published at the project's
docs URL. Site config is `mkdocs.yml` at the repo root; everything served is
authored as Markdown (`.md`) or Jupyter notebooks (`.ipynb`) under `docs/`,
with one exception: `docs/sites/` holds self-contained HTML pages (see
"When to put what where" below).

The site is organised into top-level tabs in `mkdocs.yml`'s `nav:`. The
canonical ordering is:

```
Home → Tutorials → Acquisition → Analysis → Platform → Agentic Tooling
```

The middle three are **purpose groups**, not one-tab-per-package. Each
groups the packages that serve a shared audience, and each opens on a
short section-index landing page (`docs/<group>/index.md`, surfaced via the
`navigation.indexes` theme feature) that orients the reader and links to the
constituent packages:

- **Acquisition** — running scans on the beamline: the GEECS Scanner (the
  web scanner console over the Bluesky queueserver).
- **Analysis** — turning acquired data into results: Image Analysis, Scan
  Analysis, and the Data Utils path/loading layer they build on.
- **Platform** — the access-and-contract layer everything sits on:
  GEECS-Core (device transport + DB + the GeecsDevice client), the GEECS
  Gateway (the EPICS access
  layer — central CA soft-IOC for scalars plus the distributed PVA image
  gateways on the camera servers), and GEECS Schemas (the typed
  scan-config contract).

Inside a group, each package is its own nav **section** (`navigation.sections`)
and follows roughly Diátaxis: Overview (explanation), Tutorial (when
applicable), How-To pages, Examples (notebooks), API Reference
(mkdocstrings-generated). To document a new package, add a section under the
group that matches its audience rather than creating a new top-level tab —
that one-tab-per-package sprawl is exactly what the purpose grouping replaced.

The Tutorials tab is the cross-package, user-facing entry point — it holds
the end-to-end walkthroughs (currently Analysis; Acquisition is stubbed
— see `tutorials/acquisition.md`). Agentic Tooling sits last: the
AI-agent surfaces rather than core suite, in two sections with distinct
audiences — the GEECS MCP Server (agents operating the lab) and Skills
(agents developing the code) — behind a landing page at
`docs/agentic/index.md`.

## When to put what where

| Kind of content | Lives in |
|---|---|
| Cross-package end-to-end tutorial | `docs/tutorials/` |
| Single-package overview, how-to, or reference | `docs/<package>/` |
| API reference auto-generated from docstrings | `docs/<package>/api/` (uses mkdocstrings) |
| Hands-on example using real data | `docs/<package>/examples/*.ipynb` |
| Hero landing surface | `docs/index.md` |
| Self-contained interactive page (a diagram or tool that needs its own script and layout) | `docs/sites/<name>/index.html`, linked from its purpose group's landing page; conventions in `docs/sites/README.md` |

`docs/sites/` is the one place HTML is authored directly. It exists for
pages that are a picture or a tool first (the platform data-flow map is
the first); anything that can be Markdown stays Markdown. A site page is
never in `nav:`; its purpose group's landing page links to it, and the
page links back to the docs root. `docs/sites/README.md` carries the
rules and the page table; add a row when you add a page.

If you find yourself wanting to document a workflow that spans two packages
(say, "configure analysis in the config editor then run from the data portal"), it
belongs under `tutorials/`, not under either constituent package.

## Building & previewing locally

From the repo root:

```bash
poetry run mkdocs serve     # live-reload preview at http://127.0.0.1:8000
poetry run mkdocs build     # static site → ./site/, clean output expected
```

`mkdocs build` should exit 0. Treat any new `ERROR` in the output as a
regression to fix before merging, even when the exit code is 0 (some
errors are non-fatal but indicate broken pages).

The build pulls in path-installed dev packages (GeecsBluesky, ImageAnalysis,
ScanAnalysis, etc.) because mkdocstrings imports them to read docstrings.
Missing imports surface as `griffe:` or `mkdocstrings:` warnings.

## Screenshots — author them, don't placeholder them

When documenting a web surface, capture real screenshots of representative
states in the browser. Store PNGs under `docs/<tab>/assets/`, named
`<app>_NN_<state>.png`, and regenerate them when the UI changes. Do not
ship screenshot placeholders. The former LiveWatch screenshot generator
was retired with that GUI.

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

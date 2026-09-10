# docs/sites — self-contained interactive pages

This directory holds pages that are **authored as HTML, not Markdown**:
interactive diagrams and small single-page tools that need their own
script and layout, which the Markdown pipeline cannot express. MkDocs
copies them to the published site verbatim, so `docs/sites/<name>/index.html`
is served at `<docs URL>/sites/<name>/`.

They are deliberately few. Everything that *can* be a Markdown page stays
a Markdown page under the purpose groups described in `docs/CLAUDE.md`;
a site here is the exception for content that is a picture or a tool first
and prose second.

## Rules

- **One directory per page**, `docs/sites/<name>/index.html`, with any
  assets beside it. Never a bare `.html` file at this level.
- **Self-contained.** Inline CSS, inline SVG, inline script. External
  resources only from Google Fonts. No build step, no bundler.
- **Reachable from the Markdown site.** Every page here is linked from
  the landing page of the purpose group it belongs to (the data-flow map
  is linked from the Platform tab), and links back to the docs root
  (`../../`) from its header. Nothing here appears in `mkdocs.yml`'s
  `nav:`; the link from the group landing page is the way in.
- **Both themes.** The docs site follows the reader's OS theme; a page
  here must render in light and dark via `prefers-color-scheme`.
- **This README is not published** (`exclude_docs` in `mkdocs.yml`).

## Pages

| Page | Linked from | What it is |
|---|---|---|
| `data_flow/` | Platform landing page, Fleet Map | The platform data-flow map: devices, gateways, scan engine, storage and people, with a clickable detail panel per block (what it does, how it works, status, next steps). The block text lives in the `INFO` object at the bottom of the file; the status dots are the maintainer's read and carry a date in the caption. Versions are deliberately not stated (they rot in days); each panel links to the package CHANGELOG. When an arc lands or a status changes, update the block's `INFO` entry in the same PR. |

## Publishing a page as a claude.ai artifact

The committed file is the source. A Claude Code session can publish it as
an artifact for a meeting link by stripping the `<!doctype>`/`<html>`/
`<head>`/`<body>` skeleton (the artifact host adds its own) and keeping
the `<title>`, the `<link>`, the `<style>` block and the body content.
The docs URL is the durable link; the artifact is a convenience copy.

# Changelog

All notable changes to `geecs-web-theme` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.

## [0.1.2] - 2026-09-12

### Changed

- The literal-colour guard walks the logbook's navigation script
  (`static/nav.js`) too.

## [0.1.1] - 2026-09-11

### Changed

- The literal-colour guard's surface list follows the logbook package
  rename (`GeecsLogbook/geecs_logbook/`).

## [0.1.0] - 2026-09-11

### Added

- `theme.css` — one token vocabulary (surfaces, text, rules, accent,
  semantic trio, type stacks) in three palettes, each with a light and a
  dark variant: `bella` (the BELLA Center's red on black), `laser`
  (532 nm pump green, the breadboards' red for criticals) and `plasma`
  (hydrogen Balmer — H-beta cyan accent, H-alpha for the agent voice).
  Selected by `data-theme` and `data-mode` on the root element; with no
  `data-mode` the palette follows `prefers-color-scheme`.
- `theme.js` — applies the stored choice and injects the picker wherever a
  page puts `<div data-theme-picker>`. One implementation for every
  surface, so a viewer's choice follows them between `/day`, `/run`,
  `/configs` and `/log`. The choice is per-viewer, so it lives in
  `localStorage` and every access is guarded.
- `static_dir()`, `THEMES`, `DEFAULT_THEME` — the Python surface, which is
  a path helper and two constants. No runtime dependencies at all: anything
  that can serve a static directory can use this.
- `theme-boot.js` — the one definition of the theme list and default. Loaded
  in `<head>` (not deferred) it validates the stored choice, resolves
  "follow the system" via `matchMedia`, and stamps the *effective* mode
  before first paint. Because the effective mode is stamped, the stylesheet
  carries exactly one dark block per theme and no `@media` copy to drift.
- `color-scheme` per block, so native widgets follow the chosen mode
  rather than the OS.
- `--trace-1..4` per palette for plot marks, and `--on-accent` for text on
  the accent.
- Contrast: every muted/paper, muted/surface, muted/surface-2,
  accent/paper and on-accent/accent pair is >= 4.5:1 in all six palettes,
  pinned by `tests/test_contrast.py`. The first cut failed this on every
  palette and nobody saw it by eye.
- `theme.js` dispatches `geecs:theme` on every change so anything painting
  outside CSS (the run page's Plotly figure) can repaint, and persists only
  a choice, never the default.
- `tests/test_no_literal_colours.py` — the guardrail. A token layer only
  works while components use it, and one hardcoded colour silently opts a
  component out of theming. This walks every registered web surface and
  fails on any colour literal outside a small allowlist, each entry of
  which carries its reason. It reads only real CSS (the `<style>` blocks of
  an HTML file, comments stripped) because `#765` in a comment referencing
  a pull request is a valid hex colour to a regex, and a guardrail that
  cries wolf gets switched off. A literal beside a token is still a
  literal (`var(...)` is stripped before judging), a missing surface fails
  rather than skips, and a second test checks every `var(--x)` any surface
  references is actually defined — the check that would have caught the
  first cut's `--surface2`/`--surface-2` mismatch, which shipped every
  hover fill and button ground as transparent.

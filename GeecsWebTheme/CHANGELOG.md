# Changelog

All notable changes to `geecs-web-theme` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.

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
- `tests/test_no_literal_colours.py` — the guardrail. A token layer only
  works while components use it, and one hardcoded colour silently opts a
  component out of theming. This walks every registered web surface and
  fails on any colour literal outside a small allowlist, each entry of
  which carries its reason. It reads only real CSS (the `<style>` blocks of
  an HTML file, comments stripped) because `#765` in a comment referencing
  a pull request is a valid hex colour to a regex, and a guardrail that
  cries wolf gets switched off.

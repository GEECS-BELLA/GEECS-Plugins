# Changelog

All notable changes to `geecs-web-theme` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.

## [0.2.0] - 2026-09-12

### Added

- **`kit.css` — the layout vocabulary.** `theme.css` settled colour and
  nothing else, so the portal and the logbook each answered the layout
  questions separately and disagreed on all of them: two rails (17rem
  fixed vs 228px sticky), fifteen status class names for about five
  states, three overlay mechanisms, and a run page with no breakpoint at
  all. The kit settles the shell (topbar / 216px rail / pane, one
  breakpoint at 900px), three containers by role (`.panel`, `.group`,
  `.well`), one status chip over six words, controls, tables, the five
  states a pane owes its reader, and the overlay ladder. No consumer is
  changed by this release — adoption is a separate step per surface.
- **The overlay ladder**, as components rather than prose: `details.disc`
  (rung 0), `.inspector` (1), `.drawer` (2), `<dialog>` (3), and a route
  (4, which needs no CSS). The rule is to take the lowest rung that fits;
  the decider is whether the user can lose work by pressing Esc.
- **`kit.html` — the kit's reference page.** Every component in the real
  theme, at whichever palette and density is picked. A static file beside
  the stylesheets, so a host already mounting this package serves it at
  `<mount>/kit.html` with no route of its own.
- **`kit.js`** — the drawer (Esc, scrim, focus return), a `<dialog>`
  helper with a fallback, and the density control, wired declaratively
  through `data-drawer-open` / `data-dialog-open`. Entirely optional: with
  it absent the page still renders and `<details>` still opens.
- **Density as a viewer preference.** `theme-boot.js` stamps
  `data-density` before first paint alongside the palette, and `kit.css`
  redefines `--pad` / `--row-h` / `--gap` under `[data-density="compact"]`.
  `geecs_web_theme.DENSITIES` / `DEFAULT_DENSITY` mirror the boot script
  the way `THEMES` / `DEFAULT_THEME` already do, pinned by a test.
- Structural tokens in `theme.css`: `--r-lg` (the panel corner, distinct
  from `--r`, the control corner), `--bw`, `--tk`, `--pad`, `--row-h`,
  `--gap`, `--shell-max`, `--scrim`, `--lift`. Declared in the bare
  `:root` only, so every palette shares them until one wants its own.
- `kit_css()`, `kit_js()`, `kit_html()` path helpers, and `STATES` /
  `PANE_STATES` constants so a host renders a vocabulary from a constant
  rather than hand-typing a `data-state` that silently matches no rule.
- A package `CLAUDE.md`: the kit's doctrine — the overlay ladder, the
  container roles, the vocabularies — reachable from where an agent
  building the next web surface actually looks.

### Changed

- The literal-colour guard walks `kit.css`, `kit.html` and `kit.js`.
- New tests: the kit introduces no token `theme.css` does not declare (one
  vocabulary, not two); every asset `kit.html` references exists; the
  density list agrees across Python, the boot script and the CSS; both
  vocabularies are pinned to `kit.css` in both directions; every kit rule
  is scoped to `.kit`.
- `README.md` and the root `CLAUDE.md` describe both layers, not just the
  palette.

### Fixed (from adversarial review, before first release)

- **Every kit rule is scoped to `.kit` on `<body>`.** Ungated component
  classes collided with both adoption targets, one load-bearingly: the
  portal's run page is `.pane{display:none}` / `.pane.on{display:block}` —
  its tab mechanism — which ties on specificity with an ungated `.pane`,
  leaving which wins to stylesheet order. Every tab pane would have
  rendered at once. Scoping also lets a surface convert one page at a time
  instead of every page changing when the `<link>` lands.
- The drawer honours `autofocus`. A selector list carries no priority, so
  `querySelector("[autofocus],…")` returned the first match in *tree*
  order — the header's Close button — while the adjacent `<dialog>` rung
  honoured `autofocus` natively, so identical markup behaved differently
  on two rungs.
- A `data-drawer-open` / `data-dialog-open` naming a missing id warns
  instead of producing a permanently dead control in silence.
- `Esc` no longer closes the drawer underneath an open `<dialog>`.
- The density picker builds into a host containing whitespace (ordinary
  Jinja formatting previously counted as "already built").
- The density pin asserted only that the selector existed — an *empty*
  compact block passed it. It now asserts the block overrides exactly the
  spacing tokens `theme.css` declares.

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

# Changelog

All notable changes to `geecs-web-theme` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.

## [0.6.1] - 2026-09-13

### Changed

- Merge of `master` into `feature/web-scanner`: the 0.4.1 surface addition
  below (GeecsScanner's `console.html`, `scanner.css`, `scanner.js` in
  `_SURFACES`) now runs over the tinycss2 guards of 0.5.0.

## [0.6.0] - 2026-09-13

### Added

- `testing.token_indirection_map(css)` — `{selector: {--local: --target}}`
  for every token set to exactly one `var(--x)`, read structurally (any
  spacing, comma lists split). The structural form of
  `token_indirections`; the logbook's `.tone-` guard is its first
  consumer.

### Changed

- `ForwardedPrefixMiddleware`'s docstring is the one-copy home for the
  proxy rule every surface's runbook points at: a prefix-stripping proxy
  sends `X-Forwarded-Prefix`; a prefix-preserving proxy is what
  `--root-path` is for — and the two are not interchangeable, because
  under a static `root_path` the `/static` and `/theme` mounts answer at
  the prefixed path only (a stripping proxy without the header serves a
  styleless page, not a 404). Found by the Codex review of #877 on the
  logbook; verified on the portal too.
- The `testing` extra's note spells out how a consumer that already takes
  the theme at runtime (`web`) also takes `testing` for its tests: the
  same path dependency re-declared in the dev group with
  `extras = ["testing"]`. One version floor, one owner.

## [0.5.0] - 2026-09-13

### Changed

- **The CSS guards read stylesheets through tinycss2, not regular
  expressions.** `tests/test_no_literal_colours.py` is rewritten over new
  parser-based helpers in `geecs_web_theme.testing` (`css_rules`,
  `rule_selectors`, `colour_literals`, `html_style_sources`,
  `js_colour_literals`, `referenced_tokens`, `defined_tokens`,
  `styled_classes`, `attribute_selector_values`, `classes_used`); HTML is
  read with the standard library's parser. Comments, `@media` blocks,
  nested braces and quoted strings are structure before a check looks at
  them, so each check reads like the rule it enforces and a wrong check
  fails loudly instead of matching nothing — the six regex scanners this
  replaces each had a silent hole found by review. 703 lines → ~430, and
  every kept guard was proven to bite by breaking a real file (a literal
  in `kit.css`, in a `style=` attribute, in an inline-script string; an
  unknown status styled; an unscoped rule; an unstyled class on the
  reference page; an undefined token; a palette missing a token; a token
  introduced by the kit; an emptied density block).
- **Pruned.** Kept: no literal colours; every referenced token defined and
  every palette complete; the kit introduces no token; the vocabularies
  agree across Python, `theme-boot.js` and the CSS; every kit rule scoped
  to `.kit`; the reference page shows only what the kit styles. Deleted:
  the `[hidden]` ordering check (a one-time bug, now a comment beside the
  rule), the per-component specimen list for the reference page (a
  maintenance list, not an invariant), and the probe cases that pinned
  holes in the old regexes.
- The allowlist is now by **selector** (`.themepick .sw`, `img.plot`,
  `.ce-preview img`), each with its reason, instead of by line substring.

### Added

- The `testing` extra (tinycss2) for consumers whose test suite uses a CSS
  helper — the scanner's "every class on the page is styled" check becomes
  `classes_used(template) - styled_classes(kit, theme, own_css)`. The HTML
  guards still need nothing.

## [0.4.1] - 2026-09-13

### Changed

- The literal-colour walk covers the third surface: GeecsScanner's
  `console.html`, `scanner.css` and `scanner.js` joined `_SURFACES`
  (adding a surface means adding it there — this package's CLAUDE.md).
## [0.4.0] - 2026-09-13

### Added

- **`geecs_web_theme.web` — the FastAPI glue, written once** (behind the
  new `web` extra: fastapi + jinja2; the static-only use stays
  dependency-free). Three surfaces each carried a copy of the same three
  things, verbatim by intent, and they drifted anyway — the portal computed
  `root` in Python, the logbook in Jinja, the scanner in a context
  processor. This module is the one copy: `ForwardedPrefixMiddleware`
  (the portal's `X-Forwarded-Prefix` → `root_path` middleware, path
  re-prefixed so a mount named like a route head still routes and
  trailing-slash redirects keep the prefix), `clean_prefix`, `root_of`,
  `mount_theme(app)` (the `/theme` mount, named for `url_for`), and
  `make_templates(dir, globals=, filters=, context_processors=)` (a
  `Jinja2Templates` with `root` in every context). The portal's prefix
  tests are re-homed here as the module's own.
- **`geecs_web_theme.testing` — the template guards as helpers**
  (standard library only): `bare_url_for_calls` (a `url_for(...)` without
  `.path` is a mixed-content block behind TLS), `unknown_data_states` (a
  literal `data-state` the kit does not colour renders a plausible neutral
  chip), `inline_scripts` + `javascript_syntax_error` (`node --check` over
  a template's inline scripts, Jinja blanked, JSON payloads skipped) and
  `node_available`. The logbook and the scanner each wrote these three
  tests by hand; the next surface writes three one-line tests instead.
- No consumer changes in this release. The portal, the logbook, the
  scanner and the analysis config editor (`scan_analysis.config_editor`,
  which hand-mounts the theme and builds a `root`-less template
  environment) switch to the shared copies in their own PRs and delete
  theirs.

## [0.3.0] - 2026-09-13

### Added

- **Live controls** — what a surface that *writes and watches* needs, found
  by building the web scanner's mock (the arc brief is GEECS-Plugins#869;
  this is its PR 1) and written in the kit's own idiom so it is not a
  third dialect:
  - `.live` — a reading with its label, value (+ unit) and **age**; the
    surface sets `data-age="stale"` past its threshold and the value says
    "stale" instead of continuing to look confident. The pane-level `stale`
    doctrine at value granularity. `.grid.tight` for a row of them.
  - `.meter` — determinate progress (`.bar` is indeterminate): track, fill,
    a two-ended label; `data-state` colours the fill by state.
  - `.field` validation — `aria-invalid="true"` on the control colours it
    and shows the `.err` slot that follows it (the state is the
    accessibility attribute, as with `.picklist`); the hint stays, since
    it often carries the unit; `.req` marks a required label; disabled
    inputs are styled. Focus was the only state a field had.
  - `.chip.lg` — the one state a room watches, at a size it can read.
  - `.tscroll.sticky` — a capped, scrolling table that keeps its header.
  - `dialog.ack` — rung 3 widened exactly once: a list of tickable
    preflight questions under one decision, Submit held until all are ticked.
- **`paused` joins the status words** (`STATES`, chip and dot): a run
  holding between steps is neither running nor degraded. The Qt console had
  its own amber pill for it; the kit had no word, so a web surface would
  have rendered it as one of the other two. Warn wash, no pulse.
- **`denied` has a rule of its own** on `.state` and `.banner` — a dashed
  edge on the recessed ground. It was named-but-neutral; ownership refusal
  in the scanner is its first real use.
- The reference page demonstrates every addition (a "Live controls" section,
  a validation specimen, the sticky table, the `denied` banner, the
  acknowledgement dialog behind "Submit scan…"), and a test pins that it
  keeps doing so.

## [0.2.2] - 2026-09-12

### Fixed

- **`hidden` did not hide.** The browser's `[hidden]{display:none}` is a
  *user-agent* rule, so any author rule setting `display` beats it — and
  this kit sets `display` on `.btn`, `.chip`, `.row`, `.picklist`, `.seg`,
  `.panel > header` and more. A `<button class="btn" hidden>` therefore
  rendered. That shipped: the logbook's Discard button, meant to appear
  only after an attachment autosaves, sat on screen from page load and did
  nothing when pressed, because there was nothing to discard.

  One `.kit [hidden]{display:none!important}` ahead of every component, so
  the platform attribute means what it says. Pinned, including that it
  keeps its `!important` and stays ahead of the first component that sets
  a `display`.

## [0.2.1] - 2026-09-12

### Fixed

- The `running` chip's pulse had **no reduced-motion escape**. It is the
  kit's only animation that never ends, so it is the one that most needed
  one — and a surface's own `prefers-reduced-motion` rule typically kills
  `transition` only, which does not touch it. Found while reviewing the
  logbook's adoption, where a permanent "today" label had been given that
  state.

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
- The scoping guard itself had three holes, found by re-review and each
  reproduced before fixing. A regex over selectors consumed the `{` of
  every `@media` prelude, so the **first rule inside each media block was
  never examined** — four blocks, four unguarded rules, one of them the
  mobile shell collapse. It also waved through `.kitchen` (a string
  prefix, not a class token) and `:root .pane` (a fully global descendant
  selector). The guard now walks braces instead of matching a regex, skips
  `@keyframes` stops, and allows exactly two unscoped shapes. Each hole is
  pinned as a probe case.
- The vocabulary pins only recognised double-quoted attribute selectors,
  so `[data-state='aborted']` slipped past — inconsistent with
  `_INLINE_STYLE` in the same file, which already handled both.
- `kit.html`'s dialog no longer carries `class="kit-dialog"`, which stopped
  matching anything when the rules were scoped. A dead class on the page
  people copy from is the wrong thing to copy.
- **`.picklist`**, extracted from `.rail nav` (Codex review of #856). The
  selectable-list styling was scoped to where it sat, so the inspector on
  the reference page rendered three native browser buttons — a component
  shown on the copy-from page that the kit did not actually style. It is
  now a named component the rail, the inspector and the console's device
  list all use, and a new guard fails when `kit.html` shows any class
  nothing styles.

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

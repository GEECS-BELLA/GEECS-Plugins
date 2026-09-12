# GeecsWebTheme — Developer Context for Claude

The shared look of every GEECS web surface. Two layers: `theme.css` settles
**colour**, `kit.css` settles **everything else** — shell, containers,
status, controls, tables, pane states, overlays.

**Read this before building a new web surface.** The whole point of the
package is that the next surface inherits these answers instead of inventing
a fourth set; a surface that hand-rolls its own rail, chip or modal has
already lost the thing this package exists to protect.

## Why the kit exists (do not re-litigate)

Before it, the Data Portal and the GeecsLogbook each answered the layout
questions separately, and disagreed on all of them: two rails (`17rem`
fixed vs `228px` sticky), **zero** lines of shared layout, **fifteen**
status class names for about five states, three overlay mechanisms — none a
real `<dialog>`, so the portal's modal had no focus trap and ignored `Esc`.
The forcing function was the third surface: the console, once web-based, is
the one with the hardest requirements (live values, write actions, hazards,
keyboard), so it sets the vocabulary rather than inheriting a compromise.

## The rules

- **Surfaces style through tokens, never with a literal colour.** Enforced
  by `tests/test_no_literal_colours.py`, which walks *every* surface in the
  repo — not just this package. Adding a web surface means adding it to
  `_SURFACES` there.
- **`theme.css` is the single token authority.** `kit.css` consumes tokens
  and overrides them (the density blocks, `--shell-max`); it never
  introduces one. Pinned by `test_kit_defines_no_token_the_theme_does_not`.
- **Every kit rule is scoped to `.kit` on `<body>`.** Both adoption targets
  already use several of these class names, one load-bearingly — the
  portal's run page is `.pane{display:none}` / `.pane.on{display:block}`,
  its tab mechanism, which ties on specificity with an ungated `.pane`.
  Scoping also lets a surface convert one template at a time. Pinned by
  `test_kit_rules_are_scoped_to_the_kit_class`.
- **Vocabularies are constants, not strings you type.** `THEMES`,
  `DENSITIES`, `STATES`, `PANE_STATES` are exported from `__init__.py` and
  pinned to the CSS. A hand-typed `data-state="no_data"` matches no rule and
  still renders a plausible neutral chip — it survives review *and* the
  browser, which is why the pin runs both directions.
- **`kit.js` is optional.** Every rule works with scripts off; the script
  adds only the drawer, the dialog helper and the density control. Do not
  make a component depend on it.
- **`theme-boot.js` is the one definition** of which themes and densities
  exist and what the defaults are. It loads in `<head>`, NOT deferred, so
  the palette and density are stamped before first paint. `theme.js` reads
  `window.GEECS_THEME` rather than carrying a copy, and the Python
  constants are pinned to it by test.

## The overlay ladder

The rule for what happens when someone clicks a thing. **Take the lowest
rung that fits.** The decider: *can the user lose work by pressing Esc?* If
yes it is a route, whatever it looks like.

| Rung | Component | Use it when |
|---|---|---|
| 0 | `details.disc` | The detail belongs to one row and the reader is still scanning. Native `<details>` — keyboard, no-JS and find-in-page all work. Tracking open state in JS means you are on the wrong rung. |
| 1 | `.inspector` | The user steps through many items and compares them. The selection belongs in the URL; if the link cannot carry it, it is not finished. |
| 2 | `.drawer` | A focused sub-task on the object already on screen. Never opens another drawer, never holds a destructive confirm, must survive `Esc` without losing typing. |
| 3 | `<dialog>` | A question that must be answered before anything else proceeds. Real `showModal()` — that is where the focus trap, `Esc` and the inert background come from. One question, at most two buttons, no scrolling, no form. Focus never lands on the destructive button. |
| 4 | a route | Long enough that someone will link to it, reload it, or return tomorrow. Needs no CSS, which is the point. |

## Containers, by role

Border, fill, radius and shadow each say *separate object*; spending all
four on every block is why dense screens go flat.

- `.panel` — a thing with an identity and its own state (a device, a scan, a
  queue): border, header, shadow.
- `.group` — controls belonging to the panel around them. **No edges.**
- `.well` — recessed fill, no border: output the reader did not write (a log
  tail, YAML, a repro snippet).

## Status and pane states

`queued · running · ok · degraded · failed · unknown`, plus `agent` — not a
severity but *who wrote this*, because the logbook already separates what an
analyzer wrote from what a person wrote and the console will want the same
for agent-submitted actions. Colour is never the only carrier: every chip is
a dot **and** a word.

Every pane owes five states: `loading · empty · error · stale · denied`.
`stale` is the one that matters for hardware — a live number that silently
stops updating is worse than no number, so any live value carries its own
age and the surface says so past a threshold rather than continuing to look
confident.

## The reference page

`kit.html` is a static file beside the stylesheets, so any host mounting
`static_dir()` already serves it (`/theme/kit.html` on the portal). Every
component in the real theme, at whichever palette and density. **Look there
before adding a component**, and change it in the same commit when you add
one — `test_kit_reference_page_assets_all_exist` pins its assets, and it is
the page people will copy from.

## Sharp edges

- The literal-colour guard blanks `/* */` comments but not `//` line
  comments in a standalone `.js` file, so a `#765`-style reference in a JS
  comment reads as a three-digit hex and fails the guard. Write it without
  the `#`. (Naive `//` handling would blank the `//` in every URL, which is
  why the guard does not try.)
- `_blocks()` in that test file reads raw CSS text, so a `:root` mentioned
  in a header comment runs together with the real selector. Strip comments
  before keying its result by selector name.
- Adding a spacing token means adding it to `_DENSITY_TOKENS` in the test
  too, or the density blocks that forgot it will not fail.

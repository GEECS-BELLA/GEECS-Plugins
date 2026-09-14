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

## Adoption status (owner rulings, 2026-09-13)

Theme = paint (tokens, palettes, picker); kit = furniture (shell,
containers, status words, controls, overlay ladder). A surface can be on
the theme without being on the kit.

| Surface | Theme | Kit | Glue (`geecs_web_theme.web`) |
|---|---|---|---|
| GeecsLogbook | yes | yes | yes (0.10.0) |
| GeecsScanner (web console, its branch) | yes | yes | yes |
| GEECS-DataPortal | yes | **no** — its own rail, tabs, badges, overlay, a 125-line inline style block and a 971-line inline script | no (own middleware copy) |
| ScanAnalysis config editor (`/configs`) | yes | no | no |

**Portal onto the kit is deliberately LAST** — after the Qt console is
deleted and HTU is quiet — because it is the kit's acceptance test: the
run page uses plots, tabs and toasts, which the kit does not have yet,
so that adoption will grow the kit rather than just consume it. Take the
portal's glue copy (`_ForwardedPrefixMiddleware`, its own `root`
context processor) out in the same change. The config editor follows the
portal (it is a router inside it); it never gets a kit pass of its own.
Do not start either early to "tidy up".

## The FastAPI glue and the template guards

`geecs_web_theme.web` (the `web` extra) is the one copy of what every
FastAPI surface needs around the theme: `ForwardedPrefixMiddleware`,
`clean_prefix`, `root_of`, `mount_theme`, `make_templates`. The portal,
the logbook, the scanner and the analysis config editor each carried a
copy and drifted (three mechanisms for `root`, one surface with none). **A new surface imports these; it does not copy
them.** `geecs_web_theme.testing` is the same for the template guards —
`bare_url_for_calls`, `unknown_data_states`, `inline_scripts` +
`javascript_syntax_error` — so a surface's `tests/test_page.py` is three
one-line tests over its templates. Both modules are exercised by
`tests/test_web.py` and `tests/test_testing.py`.

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

## `.picklist`

A vertical list of selectable things — the rail's navigation, the
inspector's item list, whatever the console's device list becomes. It is a
**named component, not a location**: the first cut styled `.rail nav`,
which left the inspector on the reference page rendering native browser
buttons and guaranteed a third use would need a third rule. Mark the
current item with `aria-current="page"` (navigation) or
`aria-pressed="true"` (selection); the state is the accessibility
attribute, never a class.

## Status and pane states

`queued · running · paused · ok · degraded · failed · unknown`, plus `agent` — not a
severity but *who wrote this*, because the logbook already separates what an
analyzer wrote from what a person wrote and the console will want the same
for agent-submitted actions. Colour is never the only carrier: every chip is
a dot **and** a word.

Every pane owes five states: `loading · empty · error · stale · denied`.
`stale` is the one that matters for hardware — a live number that silently
stops updating is worse than no number, so any live value carries its own
age and the surface says so past a threshold rather than continuing to look
confident. `.live` is that doctrine as a component (0.3.0): label, value,
age, and `data-age="stale"` set by the surface — the attribute's one value;
a fresh reading carries none. `paused` joined the status words with the
scanner — a run holding between steps is neither running nor degraded (the
Qt console had an amber pill for it; the kit had no word); it takes the warn
wash and does not pulse. `denied` has a rule of
its own, a dashed edge on the recessed ground: not an alarm, a door someone
else is holding.

## Live controls

What a surface that writes and watches needs, found by building the
scanner's mock and written in the kit's own idiom: `.meter` (determinate,
two-ended label, `data-state` colours the fill by state), `.field`
validation (`aria-invalid="true"` on the control colours it and shows the
`.err` slot that follows it — the accessibility attribute is the state, as
with `.picklist`; the `.hint` stays because it carries the unit; `.req` marks
required; disabled inputs are styled), `.chip.lg`
for the one state a room watches, `.tscroll.sticky` for a live table, and
`dialog.ack` — rung 3 widened exactly once, to admit a list of tickable
preflight questions under one decision. Anything further is its own small
PR against `kit.html`, not an inline style in a surface.

## The reference page

`kit.html` is a static file beside the stylesheets, so any host mounting
`static_dir()` already serves it (`/theme/kit.html` on the portal). Every
component in the real theme, at whichever palette and density. **Look there
before adding a component**, and change it in the same commit when you add
one — `test_kit_reference_page_assets_all_exist` pins its assets, and it is
the page people will copy from.

## The guards read CSS through a parser

`tests/test_no_literal_colours.py` and the CSS helpers in
`geecs_web_theme.testing` (`colour_literals`, `referenced_tokens`,
`defined_tokens`, `styled_classes`, `attribute_selector_values`,
`rule_selectors`, `html_style_sources`, `js_colour_literals`,
`classes_used`) read stylesheets through **tinycss2** and HTML through the
standard library's parser. Comments, `@media` blocks, nested braces and
quoted strings are structure by the time a check looks at them, so a check
reads like the rule it enforces. **Never add a regex CSS scanner** — six of
them had silent holes across three review rounds, and each hole passed
green. If a new check needs to see CSS, add a helper on `css_rules()` and
prove it bites by breaking a real file first.

What is pinned: no literal colours; every referenced token defined and
every palette complete; the kit introduces no token; the four vocabularies
agree across Python, `theme-boot.js` and the CSS; every kit rule scoped to
`.kit`; the reference page shows only what the kit styles. Deleted on
purpose: the `[hidden]` ordering check and the per-component specimen list.

Remaining edges:

- The `web` and `testing` extras are separate on purpose: a consumer's
  test suite takes `testing` (tinycss2) only if it uses a CSS helper; the
  HTML guards need nothing.
- Adding a spacing token means adding it to `_DENSITY_TOKENS` in the test,
  or the density block that forgot it will not fail.
- `js_colour_literals` judges string literals only: a colour built by
  concatenation is not seen. Scripts read colours through
  `getPropertyValue("--x")`, which is what the token check looks for.

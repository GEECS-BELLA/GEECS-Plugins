# GeecsWebTheme

The shared look of every GEECS web surface — the Data Portal, the scan
logbook, the analysis config editor, and the console once it is web-based.

Two layers, because they answer two different questions.

## `theme.css` — colour

One token vocabulary, three themes, each light and dark:

| Theme | Where it comes from |
|---|---|
| `bella` | The BELLA Center's red on black |
| `laser` | 532 nm pump green, with the breadboards' red for criticals |
| `plasma` | Hydrogen Balmer — H-β cyan accent, H-α for the agent voice |

A surface styles **through the tokens** — never with a literal colour. That
one discipline is what makes a new theme a block of overrides instead of a
hunt, and `tests/test_no_literal_colours.py` enforces it across every
surface, not just this package.

## `kit.css` — everything else

The layout vocabulary: the page shell, three containers by role, one status
chip, controls, tables, the states a pane owes its reader, and the overlay
ladder. It exists because the portal and the logbook each answered those
questions separately and disagreed on all of them — two rails, fifteen
status class names for about five states, three overlay mechanisms none of
which is a real `<dialog>`.

- **Shell** — topbar / 216 px rail / pane, one breakpoint at 900 px.
- **Containers** — `.panel` (a thing with its own state), `.group` (no
  edges; controls belonging to the panel around them), `.well` (recessed;
  output the reader did not write).
- **Status** — `queued · running · ok · degraded · failed · unknown`, plus
  `agent` for *who wrote this*. Named on `data-state`; every chip is a dot
  **and** a word, so colour is never the only carrier.
- **Pane states** — `loading · empty · error · stale · denied`.
- **The overlay ladder** — `details.disc` → `.inspector` → `.drawer` →
  `<dialog>` → a route. Take the lowest rung that fits; the decider is
  whether the user can lose work by pressing Esc.
- **Density** — a viewer preference stamped before first paint, driving
  `--pad` / `--row-h` / `--gap`.

Every kit rule is scoped to `.kit` on `<body>`, so a surface converts one
page at a time and the kit's class names cannot collide with a surface's
own. `kit.js` is optional: without it the page still renders and
`<details>` still opens; only the drawer, the dialog helper and the density
control go missing.

## Using it

```python
from geecs_web_theme import static_dir
app.mount("/theme", StaticFiles(directory=str(static_dir())), name="theme")
```

```html
<!-- in <head>: the boot script NOT deferred, so the palette and density
     are stamped before first paint -->
<script src="/theme/theme-boot.js"></script>
<link rel="stylesheet" href="/theme/theme.css">
<link rel="stylesheet" href="/theme/kit.css">
<script src="/theme/theme.js" defer></script>
<script src="/theme/kit.js" defer></script>

<body class="kit">
  <div data-theme-picker></div>
  <div data-density-picker></div>
```

Behind a reverse proxy, build those URLs from the request's `root_path` —
the portal does this with its `{{ root }}` idiom.

`THEMES`, `DENSITIES`, `STATES` and `PANE_STATES` are exported from the
package so a host renders a vocabulary from a constant rather than
hand-typing a `data-state` that silently matches no rule.

## The reference page

`kit.html` is a static file beside the stylesheets, so any host mounting
this package already serves it — the portal has it at `/theme/kit.html`.
It shows every component in the real theme at whichever palette and density
you pick. It is the place to look before adding a component, and the place
to argue with one.

## No dependencies

Deliberately: this package is stylesheets, two small scripts and a path
helper. Anything that can serve a static directory can use it, and nothing
it serves needs a Python import to work.

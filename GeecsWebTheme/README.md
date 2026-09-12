# GeecsWebTheme

One palette vocabulary shared by every GEECS web surface — the Data Portal,
the analysis config editor, and the scan logbook.

Three themes, each with a light and a dark variant:

| Theme | Where it comes from |
|---|---|
| `bella` | The BELLA Center's red on black |
| `laser` | 532 nm pump green, with the breadboards' red for criticals |
| `plasma` | Hydrogen Balmer — H-β cyan accent, H-α for the agent voice |

A surface links `theme.css`, adds `theme.js`, and styles **through the
tokens** — never with a literal colour. That one discipline is what makes a
new theme a block of overrides instead of a hunt, and it is enforced by
`tests/test_no_literal_colours.py`.

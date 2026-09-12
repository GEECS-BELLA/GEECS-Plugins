# Seed templates — the logbook's type buttons

Copy this directory to `logbook_templates/` at the top of the configs
checkout the portal reads (the parent of the `--processing-configs`
tree). Every `*.md` file here becomes a button on the composers; the file
stem is what an entry records as its `template`.

Format — a `key: value` header between `---` lines, then the prefill:

```markdown
---
label: Laser
colour: ok
book: ops
order: 10
---
#laser
```

| Key | Meaning | Default |
|---|---|---|
| `label` | button text | the stem, title-cased |
| `colour` | a theme **token name**: `accent`, `ok`, `warn`, `crit`, `agent`, `muted`, `trace-1`…`trace-4` — never a literal colour | `accent` |
| `book` | `scans`, `ops` or `both` — which composers offer it | `both` |
| `order` | sort key for the button row | `100` |

The body is the prefill. Keep the type's `#tag` in it: tags are read from
the text at save, so a button press and a typed tag are the same thing.
Templates seed, they never enforce — editing a file changes no stored
entry. See `GeecsLogbook/CLAUDE.md` § "Templates seed, they never enforce".

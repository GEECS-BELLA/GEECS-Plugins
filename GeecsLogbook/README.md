# GeecsLogbook

The logbook: two books in one store, its own web service (port 8400).

- **Scans** — `/day/2026-09-11`, a day-document view over GEECS scan
  folders. A day is a **query**, not a document: the page lists whatever
  scans exist for that date at request time. Nothing creates a log; scans
  appear because their folders do.
- **Ops** — `/month/2026-09`, routine operations read by month, from
  the notes store alone (never the share), with tag filters.

Notes cite each other: every entry's **Link** tool copies its permalink
(`/entry/{id}`, which resolves to whichever page draws it), and pasting
one into a composer writes a labelled reference. Bodies store the
relative form, so no mount prefix or host reaches the record.

Type buttons on every composer come from `logbook_templates/*.md` in the
configs checkout — see `examples/logbook_templates/README.md`.

Notes are Markdown; `$E = mc^2$` inline and `$$…$$` on lines of its own
are typeset (KaTeX, vendored — no internet needed) on the page and in the
composer's preview, and stay plain `$` syntax in the store and the mirror.

This package is a **consumer of scan folders** and never a producer. See
the repository `CLAUDE.md` for the invariant.

Run it: `geecs-logbook --experiment <name> [--notes-db PATH]
[--templates-dir DIR]`; deployment (the unit, the state directory, the
move out of the portal) in `deploy/DEPLOYMENT.md`. The Data Portal links
to it (`--logbook-url`) and no longer mounts it.

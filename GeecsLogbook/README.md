# GeecsLogbook

The logbook: two books in one store, mounted by the Data Portal at `/log`.

- **Scans** — `/log/day/2026-09-11`, a day-document view over GEECS scan
  folders. A day is a **query**, not a document: the page lists whatever
  scans exist for that date at request time. Nothing creates a log; scans
  appear because their folders do.
- **Ops** — `/log/month/2026-09`, routine operations read by month, from
  the notes store alone (never the share), with tag filters.

Type buttons on every composer come from `logbook_templates/*.md` in the
configs checkout — see `examples/logbook_templates/README.md`.

This package is a **consumer of scan folders** and never a producer. See
the repository `CLAUDE.md` for the invariant.

Mounted by GEECS-DataPortal at `/log`; no separate service, port, or unit.

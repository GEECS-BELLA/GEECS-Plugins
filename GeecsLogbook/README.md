# GeecsLogbook

The scan logbook: a day-document view over GEECS scan folders.

A day is a **query**, not a document — `/log/day/2026-09-11` lists whatever
scans exist for that date at request time. Nothing creates a log; scans
appear because their folders do.

This package is a **consumer of scan folders** and never a producer. See
the repository `CLAUDE.md` for the invariant.

Mounted by GEECS-DataPortal at `/log`; no separate service, port, or unit.

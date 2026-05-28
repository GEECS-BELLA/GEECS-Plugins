# Acquisition Tutorial

!!! warning "Under construction"
    The dedicated acquisition tutorial in this tab is a stub. For now,
    the canonical end-to-end walkthrough lives at
    **[Scanner GUI → Tutorial — Your First Scan](../geecs_scanner/tutorial.md)**.
    It covers the same ground this stub will eventually expand into:
    configuring an experiment, building a save element, running a NOSCAN,
    running a 1D scan, and finding the resulting data on disk.

## Why this page exists

The [analysis tutorial](analysis.md) covers the analysis half of the data
lifecycle end-to-end. The acquisition half is currently documented inside
the Scanner GUI tab because that's where the GUI itself is documented; it
predates the Tutorials tab.

When the acquisition tutorial is written here it'll be more comprehensive
than the existing Scanner GUI tutorial — covering optimization scans,
multi-scanner batches, action sequences, composite scan variables, and the
back-and-forth between Scanner GUI and the live analysis side. Until then,
the existing Scanner GUI tutorial is the right starting point.

## What the proper version will cover (roughly)

A non-exhaustive sketch, so the eventual writer has a starting outline:

1. **Setup** — `config.ini`, experiment selection, talking to the GEECS
   database.
2. **Save elements** — building one from scratch, composing multiple, the
   `setup_action` / `closeout_action` hooks.
3. **NOSCAN** — your first scan, where the data lands, sanity-checks.
4. **1D scan** — picking a scan variable, ranges and steps, composite scan
   variables.
5. **Multi-scanner** — queuing a batch of scans with different
   configurations.
6. **Optimization scans** — Xopt-driven scans with a custom evaluator.
7. **Pre/post-scan action sequences** — calibration shots, shutter
   handling, "leave it as you found it" closeouts.
8. **Operations** — what to check when something goes wrong, how the
   structured logs pair with the `/triage` skill.

## Until then

- Start with [Scanner GUI → Tutorial — Your First Scan](../geecs_scanner/tutorial.md).
- For the engine internals, [Scanner GUI → Architecture](../geecs_scanner/architecture.md).
- For specific scan-failure modes, [Scanner GUI → Troubleshooting](../geecs_scanner/troubleshooting.md).
- Once data is on disk, head to the [analysis tutorial](analysis.md).

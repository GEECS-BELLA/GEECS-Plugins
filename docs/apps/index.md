# Apps

!!! note "Under construction"
    This page is a stub. Full content lands as part of the docs sweep.

Three GUI applications are the canonical, user-friendly entry points to the
GEECS Plugin Suite. Each is a polished reference implementation built on top
of the underlying library APIs — meaning you can do everything the GUIs do
from a script if you need to, but the GUIs are the easiest path for most
day-to-day workflows.

- **[ConfigFileGUI](config_file_gui.md)** — point-and-click editor for the
  YAML configs that drive Image Analysis and Scan Analysis.
- **[LiveWatch](live_watch.md)** — automated per-scan analysis runner that
  watches a data directory and dispatches configured analyzers as scans
  complete.
- **[Scanner GUI](../geecs_scanner/overview.md)** — the data-acquisition
  application that runs scans on the beamline. Lives in its own tab.

For a guided walk through all three, see the
[end-to-end tutorial](tutorial.md).

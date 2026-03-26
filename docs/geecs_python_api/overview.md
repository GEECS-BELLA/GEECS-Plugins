# GEECS Python API

The GEECS Python API is the low-level Python interface to the GEECS control system — the software that manages hardware devices, coordinates timing, and records experiment data at BELLA Center.

Most users interact with this package **indirectly**, through the Scanner GUI or Scan Analysis packages. Direct use is most relevant when building new GUI components, writing custom device drivers, or scripting hardware control outside of the standard tools.

---

## What It Provides

**Controls** — Connect to GEECS devices, set and read device variables, handle timing and synchronization. This is the foundation for all hardware communication in the plugin suite.

**Analysis** — Utilities for loading and interpreting data that GEECS records — device variable logs, scan metadata, and shot-level data files.

**Tools** — Helper functions for common tasks: unit conversions, data formatting, configuration file parsing, and interfacing with the broader GEECS ecosystem.

---

## Relationship to Other Packages

```
GEECS Scanner GUI  ──┐
                     ├──▶  GEECS Python API  ──▶  GEECS hardware / data files
Scan Analysis      ──┘
```

If you are starting out with data analysis or scan automation, see [Scan Analysis](../scan_analysis/overview.md) or [Image Analysis](../image_analysis/overview.md) instead — they build on this package and provide higher-level workflows.

---

## API Reference

- [Analysis](api/analysis.md)
- [Controls](api/controls.md)
- [Tools](api/tools.md)

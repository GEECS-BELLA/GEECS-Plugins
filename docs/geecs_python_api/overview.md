# GEECS Python API

`geecs-python-api` is the low-level Python interface to the GEECS control system. It speaks the GEECS wire protocol (UDP for command/response, TCP for live data subscription) and gives you Python objects that represent devices, variables, and the experiment database.

If you're acquiring data through the [Scanner GUI](../geecs_scanner/overview.md), you're already using this package indirectly. The reasons to reach for it directly are:

- **You want to script device interaction outside the GUI** — write your own one-off measurement, run a Jupyter notebook that talks to a device, or drive an experiment from a script. See the [Scripting Guide](scripting_guide.md).
- **You're writing a custom GUI component or analyzer** that needs to read or set a device variable.
- **You're integrating GEECS hardware with a non-GEECS framework** — a Bluesky plan, a custom acquisition loop, an external monitoring script.

## What it provides

The package is `geecs_python_api.controls`: device classes, the database lookup, and the UDP/TCP transport layer. The two classes you'll touch most are `GeecsDevice` (for any GEECS device) and `ScanDevice` (a `GeecsDevice` subclass that adds support for composite scan variables). `GeecsDatabase` provides the experiment-info lookup that resolves device names to network endpoints.

The previous `analysis` and `tools` subpackages were removed in v0.4.0. Data-loading utilities now live in [Data Utils](../geecs_data_utils/overview.md) — that's where to reach for `ScanPaths`, `ScanData`, and s-file loaders.

## Status

**This package is being refactored.** Other packages in the suite (Scanner GUI, Data Utils, ScanAnalysis) use it primarily for `ScanDevice`, the database lookup, and the shared `config.ini`. New features should not be added to this package; if you find yourself wanting to extend it, raise the question of whether the new code belongs in the higher-level package that's calling into the API.

The intent is that long-term, the device transport layer either consolidates here cleanly or migrates into a Bluesky-style architecture. Either way, the API surface above will become smaller and more focused.

## Where to start

- **[Scripting Guide](scripting_guide.md)** — for the most common use case ("I want to write a script that talks to a device"), this walks through the basics.
- **[API Reference: Controls](api/controls.md)** — `GeecsDevice`, `ScanDevice`, `GeecsDatabase`, and the transport layer.

If you're not sure whether to start here or in [Data Utils](../geecs_data_utils/overview.md): if you want to **acquire** data live from devices, start here. If you want to **load** data that's already on disk, start in Data Utils.

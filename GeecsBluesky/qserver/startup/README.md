`startup.py` is the bluesky-queueserver RE Manager startup profile for a
GEECS worker. It builds the worker-wide `GeecsSession`, defines the
module-level `RE` (see below), subscribes the Tiled/s-file/scan-log
callbacks, registers the optimization loader when the `optimize` extra is
installed, and registers `geecs_scan_request_plan` — the one plan every
`ScanRequest` (step, noscan, optimize) runs through. See its module
docstring for the import-order (`geecs_bluesky` first, before any `aioca`
import) and experiment-resolution (`QS_EXPERIMENT` env, falling back to
`config.ini`) contracts.

The profile MUST define `RE = RunEngine({})` — the launcher passes
`--keep-re`, and without a startup-defined RE the manager silently bounces
every `queue start` (the failure appears only in the manager log as
"Run Engine is not found in the RE Worker environment"). `startup.py`
satisfies this by exposing `session.RE` as the top-level `RE` — the plan
preamble requires running on that exact session's RunEngine, so the two
must be the same object rather than two independently-constructed
RunEngines.

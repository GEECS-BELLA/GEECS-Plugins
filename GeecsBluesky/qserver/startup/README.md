Startup profile content arrives with the plan-preamble task; this directory intentionally contains no Python yet.

The profile MUST define `RE = RunEngine({})` — the launcher passes
`--keep-re`, and without a startup-defined RE the manager silently bounces
every `queue start` (the failure appears only in the manager log as
"Run Engine is not found in the RE Worker environment").

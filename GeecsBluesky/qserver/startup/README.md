`startup.py` is the bluesky-queueserver RE Manager startup profile for a
GEECS worker. It builds the worker-wide `RE` through
`geecs_bluesky.run_engine.make_run_engine` (the Tiled spool + the s-file export
subscribed, `connect_on_demand` installed outermost), publishes documents
to the 0MQ proxy, exports every device of the experiment as a noun
(`GeecsNamespace`, from the GEECS DB) and registers the plans listed in
`geecs_bluesky.plan_names.GEECS_PLAN_NAMES` over them (`count`, `sweep`,
`optimize`, `mv`, `run_action` and the two shot-offset calibration plans,
bound by `geecs_bluesky.plans.registry.bind_plans`). See its module
docstring for the import-order (`geecs_bluesky`
first, before any `aioca` import) and experiment-resolution
(`QS_EXPERIMENT` env, falling back to `config.ini`) contracts.

The profile MUST define a top-level `RE` — the launcher passes
`--keep-re`, and without a startup-defined RE the manager silently bounces
every `queue start` (the failure appears only in the manager log as
"Run Engine is not found in the RE Worker environment").

The scan plans are bound with the GEECS `take_reading` hook, so a
detector's shot is fired by the trigger box; a `GeecsDetector` under a
bare stock plan is refused at prepare — it cannot self-trigger.

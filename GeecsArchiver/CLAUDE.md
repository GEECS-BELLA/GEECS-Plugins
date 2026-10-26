# GeecsArchiver — Developer Context for Claude

The EPICS Archiver Appliance for GEECS. Upstream's appliance runs unchanged
in its official container; it archives the CA gateway's PVs as any Channel
Access client would. This package is the GEECS-shaped glue: the rule that
says *which* PVs, the client that tells the appliance, and the deploy recipe
that puts the appliance on a service host under the site profile.

`PLAN.md` is the arc's plan (decisions, phases, the 2026-10-02 pilot record);
`DEPLOYMENT.md` is the runbook. When the phases are done, the plan is deleted
and the runbook carries the record.

## Layout

```
geecs_archiver/
  config.py        appliance URL (GEECS_ARCHIVER_URL / config.ini [archiver] url),
                   experiment, thin wrappers over geecs_core.configs_repo and the
                   schemas' from_path loaders
  archive_set.py   THE RULE — derive_candidates (pure; where both passes meet,
                   under the exclude globs) + build_archive_set (DB, then the
                   registry pass; use_db=False = --no-db); classify() mirrors
                   geecs_core.db.variable_types; sampling_for()
  native.py        THE REGISTRY PASS — derive_native_candidates (pure, no DB):
                   native_devices.yaml + panel_devices.yaml + the family files
  mgmt_client.py   MgmtClient over <url>/mgmt/bpl; PVStatus; one MgmtError
  onboard.py       plan_onboarding → apply → verify (never-connected = drift alarm)
  cli.py           geecs-archiver list [--no-db] | onboard | status | export-config
deploy/
  geecs-archiver.service   unit TEMPLATE (User=@SERVICE_USER@, EnvironmentFile=@SITE_ENV@)
  compose.yaml.in          upstream image pinned by tag; host networking; JDBM2 persistence
                           (container pieces UNVERIFIED until the Phase 3 install — DEPLOYMENT §2)
  appliances.xml.in        single appliance; data_retrieval_url = @ARCHIVER_HOST@
  server.xml context.xml policies.py archappl.properties   static conf
  render_conf.sh           fills the .in files from site.env (render_units.sh does units only)
```

## Rules that hold here

- **A client of the gateway's contract, never of its code.** The archive
  set is derived from the GEECS DB with the gateway's own three queries
  (`GeecsDb.get_experiment_devices`, `get_experiment_device_variables`,
  `get_subscribed_variables`) and `geecs_core.pv_naming` /
  `geecs_core.db.variable_types` — the shared library both sides import. If
  the gateway's served-set rule changes, `derive_candidates` changes with
  it, and the appliance's never-connected list (read by `onboard` on every
  run) is what tells you they drifted. The pieces both sides need already
  live in the shared packages: `geecs_core.configs_repo` (the configs-repo
  lookup), `geecs_core.pv_naming.device_status_pv` and `gateway_status_pv`,
  `DerivedChannel.pv_parts` and `VersionedSchemaModel.from_path` in
  GEECS-Schemas. If the served-set rule itself ever bites, it moves to
  GEECS-Core the same way (the `variable_types` precedent) — never import
  the gateway.
- **Native devices come from the registry, never the DB.** A migrated
  device is disabled in the GEECS DB, so the DB pass drops it; the registry
  pass (`native.py`) archives every `native_devices.yaml` device and every
  `panel_devices.yaml` device the DB does not own. Per device: its family
  (the panel data's, else `geecs_schemas.REGISTRY_KIND_FAMILIES`, the
  bridge's table too), the
  family's `archive: true` roles — or, for a family marking none, the
  kind's rule (`default_archive_roles`, `NATIVE_DEVICES.md` §10) — plus the
  device's `archive`, minus `no_archive` and `omit`; scalars only. A role
  whose record is its own write target (`.VAL`, `current:SP`) is a
  setpoint, under `include_setpoints`; a setting's `_RBV` is a readback.
  Under `include_status`, the kind's liveness record (`LIVENESS_SUFFIXES`:
  the camera's `cam1:AsynIO.CNCT`) and one `<IOC>:HEARTBEAT` per IOC a
  device names (`ioc:`, devIocStats). PV names come from
  `geecs_core.pv_naming.native_device_prefix` (or a bench device's own
  `prefix`) + the suffix, per axis for a per-axis family — the prefix rule
  has one home, never re-derive it. A registry device the DB also owns
  (enabled) is a **conflict**: reported, archived from the DB only, exit 1;
  a registry device the pass cannot resolve (panel data naming a family of
  another kind, no axes) is **unresolved**, exit 1 too — a migrated device
  never drops silently, or `onboard` would pause its history.
  `exclude` globs apply to both passes; `include` keeps its gateway
  spelling (natives are derived, never hand-listed). Every candidate carries
  its `source` (`db` | `registry`), and the never-connected line names it.
  **No deadband is sent for a native PV**, as for a DB PV: the archive
  deadband is the record's own `ADEL` (the IOC's template), which the
  appliance's `DBE_LOG` monitor honours; the sampling overrides are the
  archiver-side lever. The heartbeat sits outside the experiment's prefix,
  so a run never pauses one the rule drops — pause it by hand.
- **The configuration of record is the rule + the committed overlay**, not
  the appliance's store. `onboard` is idempotent; rerunning it after a DB
  edit is the same reflex as restarting the gateway. The appliance's own
  store (a JDBM2 file) only has to survive restarts; `export-config` is the
  byte-exact snapshot for the hand-tuned exceptions.
- **Never delete data.** A PV the rule no longer wants is *paused*. Only
  PVs under the experiment's own prefix are ever touched, and pausing more
  than `PAUSE_GUARD` PVs in one run needs `--yes`. A hand-added PV belongs in
  the policy's `include` list, or the next run pauses it.
- **The overlay is the one sampling table.** Every request carries its
  period and method (the appliance's user-specified sampling, which wins
  over `policies.py`); there are no named policies to keep in step.
- **Excluded by construction:** the timestamp variables — `acq_timestamp`,
  `systimestamp` and any device's own `…timestamp` (disk burn), image / array variables (not CA), `path` long-strings (the
  appliance cannot type the gateway's char arrays — pilot finding, PLAN §12).
- **Archive-rate control** is the appliance's (sampling period / policy /
  ETL decimation) plus the DB's `archive_deadband` column, which the
  gateway applies to `DBE_LOG` events only — the appliance's CA monitor
  subscribes `DBE_LOG | DBE_ALARM`, so it needs no setting for it. The
  gateway's *monitor* deadband stays 0.0 — never reach for it
  (`GeecsCAGateway/PV_CONTRACT.md` §4).
- **Scalars only, never in the scan path.** The archiver is history and
  trends; it is never a source of s-file columns (GeecsBluesky's rule) and
  never read while a scan runs.
- **One appliance per facility; every site value in `site.env`**
  (`GEECS_ARCHIVER_HOST`, `GEECS_ARCHIVER_DATA_ROOT`, `GEECS_ARCHIVER_JAVA_OPTS`,
  plus the shared `EPICS_CA_*` and `TZ`). Port 17665, the state directory,
  the image tag and `appliance0` are fleet facts, not site values. Nothing
  in this package may carry a lab literal outside an example.
- **The analysis-side folder invariant does not apply** — nothing here
  touches scan folders — but the same spirit does: the only files this
  package writes are its own conf render and the archive under
  `GEECS_ARCHIVER_DATA_ROOT`.

## Testing

```bash
cd GeecsArchiver && poetry install --with dev && poetry run pytest tests -q
```

Hermetic: the DB rule runs on hand-built rows, the registry pass on the
installed family files and hand-written registries (expected PV names written
out, never computed), the client against an
`httpx.MockTransport`, the CLI with both faked, the renderer as a bash
subprocess against `deploy/site.env.example` plus the archiver keys. Nothing
needs the lab. The live checks are in `DEPLOYMENT.md`.

## Upgrading the appliance

Bump the image tag in `deploy/compose.yaml.in` (a reviewed PR — the tag is a
fleet pin), re-render, reinstall `/etc/geecs/archiver`, restart the unit.
Read upstream's release notes for persistence-format or Tomcat changes
first. Weekly upstream images are never deployed.

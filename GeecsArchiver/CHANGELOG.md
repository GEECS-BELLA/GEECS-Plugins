# Changelog

All notable changes to `geecs-archiver` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-10-02

### Fixed

- **Every archive request failed with HTTP 500 on the first production
  start**: the appliance reads `deploy/policies.py` with Jython (Python 2),
  which rejects a non-ASCII character without an encoding declaration, and
  the file's comments carried em-dashes. The whole `deploy/` conf directory
  is now ASCII (`archappl.properties` is read as ISO-8859-1 too), pinned by
  `test_every_conf_file_is_ascii`.

## [0.1.1] - 2026-10-02

### Changed

- **Fleet wiring (documentation in this package).** `DEPLOYMENT.md` § 2
  installs through `deploy/bootstrap_host.sh --only archiver`, which now
  owns the archiver: the clone, the CLI's environment, the Docker
  prerequisite check with its root lines, the unit through
  `render_units.sh`, the conf through this package's `render_conf.sh`,
  the `[archiver] url` in the rendered `config.ini`, and the root steps.
  `PLAN.md` § 6 marks the bootstrap item done. The repo-side wiring
  (`site.env.example` keys, `/fleet-status` and `/lab-status` rows, the
  fleet map and site profile pages) lands in the same PR.

## [0.1.0] - 2026-10-02

### Added

- **The package.** Phase 2 of `PLAN.md`: the glue a facility needs around
  upstream's EPICS Archiver Appliance, which archives the CA gateway's PVs
  as a stock Channel Access client (pilot verified 2026-10-02, `PLAN.md`
  §12).
- `geecs_archiver.archive_set`: the rule that derives an experiment's
  archive set from the GEECS database with the gateway's own three
  queries — monitored (`get='yes'`) readbacks, `:SP` setpoints of settable
  variables, each device's `connected` status, the gateway's derived
  channels; never the timestamp variables, images/arrays or `path`
  long-strings — then the experiment's `ArchivePolicy` overlay (GEECS-Schemas
  0.43.0): `exclude` globs, explicit `include` PVs and per-glob sampling
  (sent with every request as the appliance's user-specified sampling —
  the overlay is the one table; `policies.py` names only the stores). The
  configs-repo lookup, the status-PV name and the derived-channel name
  parts come from GEECS-Core 0.13.0 / GEECS-Schemas, shared with the gateway.
- `geecs_archiver.mgmt_client.MgmtClient`: a typed client of the
  appliance's management API (`getAllPVs`, `getPVStatus`, `archivePV`,
  `pauseArchivingPV`, `resumeArchivingPV`, `changeArchivalParameters`,
  `exportConfig`, `getApplianceMetrics`, `getVersions`, the disconnected
  lists), every failure one `MgmtError`.
- `geecs_archiver.onboard`: idempotent reconciliation — archive the new,
  resume the paused, retune the drifted (period or method), **pause** (never
  delete) what the rule no longer wants under this experiment's prefix —
  `verify`, which waits for new PVs to be *archived and connected*, and
  `stuck_requests`: the appliance's never-connected list, read on every
  run, is the drift alarm between the rule and the gateway.
- `geecs-archiver` CLI: `list`, `onboard [--dry-run] [--no-pause] [--yes]
  [--wait]`, `status`, `export-config`. Exit 0 ok / 1 drift / 2 usage (also
  a pause of more than ten PVs without `--yes`) / 3 appliance unreachable.
  The appliance URL comes from `GEECS_ARCHIVER_URL` or `config.ini
  [archiver] url`; the experiment from `config.ini [Experiment] expt`.
- `deploy/`: the systemd unit template (`geecs-archiver.service`, site-profile
  shape) wrapping upstream's official `singletomcat-2.4.1` container under
  `docker compose` with host networking on port 17665; `compose.yaml.in` and
  `appliances.xml.in` rendered from `site.env` by `render_conf.sh`
  (`GEECS_ARCHIVER_HOST`, `GEECS_ARCHIVER_DATA_ROOT`, the service account's
  ids); a site-neutral `policies.py` (STS hourly → LTS yearly, one policy —
  the request's own sampling sets the rate), `server.xml` (connector on
  17665), `context.xml`
  (no MariaDB datasource — configuration persists in a JDBM2 file under
  `/var/lib/geecs-archiver`), `archappl.properties`.
- `DEPLOYMENT.md`: install (the container pieces marked unverified until
  the Phase 3 install), onboarding, operations. Fleet wiring
  (`site.env.example` keys, `bootstrap_host.sh`, `/fleet-status`, the fleet
  map) follows in its own PR.

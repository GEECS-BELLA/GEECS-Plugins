# Changelog

All notable changes to `geecs-archiver` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  channels; never the intrinsic timestamps, images/arrays or `path`
  long-strings — then the experiment's `ArchivePolicy` overlay (GEECS-Schemas
  0.43.0): `exclude` globs and per-glob sampling.
- `geecs_archiver.mgmt_client.MgmtClient`: a typed client of the
  appliance's management API (`getAllPVs`, `getPVStatus`, `archivePV`,
  `pauseArchivingPV`, `resumeArchivingPV`, `changeArchivalParameters`,
  `exportConfig`, `getApplianceMetrics`, `getVersions`, the disconnected
  lists), every failure one `MgmtError`.
- `geecs_archiver.onboard`: idempotent reconciliation — archive the new,
  resume the paused, retune the drifted, **pause** (never delete) what the
  rule no longer wants under this experiment's prefix — and `verify`, which
  waits for new PVs to be *archived and connected*; a never-connected PV is
  the drift alarm between the rule and the gateway.
- `geecs-archiver` CLI: `list`, `onboard [--dry-run] [--no-pause] [--wait]`,
  `status`, `export-config`. The appliance URL comes from
  `GEECS_ARCHIVER_URL` or `config.ini [archiver] url`; the experiment from
  `config.ini [Experiment] expt`.
- `deploy/`: the systemd unit template (`geecs-archiver.service`, site-profile
  shape) wrapping upstream's official `singletomcat-2.4.1` container under
  `docker compose` with host networking on port 17665; `compose.yaml.in` and
  `appliances.xml.in` rendered from `site.env` by `render_conf.sh`
  (`GEECS_ARCHIVER_HOST`, `GEECS_ARCHIVER_DATA_ROOT`, the service account's
  ids); a site-neutral `policies.py` (STS hourly → LTS yearly, `Default` /
  `Slow` / `Fast`), `server.xml` (connector on 17665), `context.xml`
  (no MariaDB datasource — configuration persists in a JDBM2 file under
  `/var/lib/geecs-archiver`), `archappl.properties`.
- `DEPLOYMENT.md`: install, onboarding, operations. Fleet wiring
  (`site.env.example` keys, `bootstrap_host.sh`, `/fleet-status`, the fleet
  map) follows in its own PR.

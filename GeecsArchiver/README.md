# GeecsArchiver

Continuous, between-scan history of GEECS control variables — the EPICS
Archiver Appliance, deployed as a client of the GEECS CA gateway, plus the
tool that keeps its archive set in step with the experiment database and,
for devices migrated to native EPICS IOCs, the configs repository's
device registry.

Tiled records what happened *during a scan*. This records what every
served readback did *all day*: the vacuum last Tuesday at 03:00, the magnet
setpoints someone changed at 17:40, when a device dropped off the network.
The appliance is upstream's, untouched (an official container); this
package is the GEECS-shaped recipe around it.

## What is in here

| | |
|---|---|
| `geecs_archiver/archive_set.py` | **The rule.** From the GEECS DB, with the gateway's own queries: monitored readbacks, `:SP` setpoints, each device's `connected`, the derived channels. Never timestamps, images or path strings. Then the experiment's overlay |
| `geecs_archiver/native.py` | **The registry pass.** A migrated device is disabled in the DB, so its history comes from here: every device in `native_devices.yaml` and every non-DB device in `panel_devices.yaml`, archived by its family file's `archive` roles (or the kind's rule), the device's own opt-ins/opt-outs, the kind's liveness record and its IOC's devIocStats heartbeat. No DB needed. Every PV carries its source, `db` or `registry` |
| `geecs_archiver/mgmt_client.py` | A typed client of the appliance's management API |
| `geecs_archiver/onboard.py` | Idempotent reconciliation: archive / resume / retune / pause (never delete; `--yes` guards a mass pause), then verify every new PV is *archived and connected*; the appliance's never-connected list is the drift alarm, read on every run |
| `geecs_archiver/cli.py` | `geecs-archiver list \| onboard \| status \| export-config` |
| `deploy/` | The systemd unit template, the compose + appliance conf templates, `render_conf.sh` |
| `DEPLOYMENT.md` | The runbook |
| `PLAN.md` | The arc's plan and the pilot record (deleted when `DEPLOYMENT.md` carries everything) |

## Quick use

```bash
# what would be archived for this experiment (DB pass + registry pass)
geecs-archiver list --experiment Undulator

# the registry pass alone, offline, from a configs-repo checkout (no DB)
GEECS_PLUGINS_CONFIGS=/path/to/configs-checkout geecs-archiver list --experiment Undulator --no-db

# reconcile the appliance with that set; wait for the new PVs to connect
geecs-archiver onboard --experiment Undulator --dry-run
geecs-archiver onboard --experiment Undulator

# the appliance's health and this experiment's share of its PVs
geecs-archiver status
```

The appliance URL comes from `config.ini` `[archiver] url` (or
`GEECS_ARCHIVER_URL`); the experiment from `[Experiment] expt`. The
optional overlay, `scanner_configs/experiments/<Experiment>/archiver/archive_policy.yaml`
in the configs repo, is a `geecs_schemas.ArchivePolicy` document. The
registry pass reads `<Experiment>/native_devices.yaml` and
`<Experiment>/panel_devices.yaml` beside it (a device's `ioc:` names the IOC
whose `<IOC>:HEARTBEAT` is archived; `no_archive:` drops a family role on one
device). A device both in the registry and enabled in the DB is a conflict:
reported, archived from the DB only, exit 1; a registry device the pass
cannot resolve (its panel data names a family of another kind) exits 1 too.

## Reading the history

- **Phoebus Data Browser:** `org.csstudio.trends.databrowser3/urls=pbraw\://<host>:17665/retrieval`
  (and `archives=` the same) in `settings.ini`; then right-click any live
  PV → Data Browser.
- **Browser:** `http://<host>:17665/mgmt/ui/index.html` (management) and
  `http://<host>:17665/retrieval/ui/viewer/archViewer.html` (a plot).
- **Python:** `GET /retrieval/data/getData.json?pv=…&from=…&to=…` — plain
  JSON; `requests` + `pandas` and nothing appliance-specific.

## Boundaries

A client of the gateway's `PV_CONTRACT.md`; never imports the gateway's
code. Depends on GEECS-Core (the DB, `pv_naming`, the variable-type rule)
and GEECS-Schemas (the overlay, the registry, the family files). Never read in the scan path, never a source of s-file
columns (GeecsBluesky's rule). One appliance per facility.

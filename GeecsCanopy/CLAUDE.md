# GeecsCanopy — Developer Context for Claude

SLAC's Canopy runs here as the display backend, unpatched, in an image
derived from upstream's by digest. This package holds what is GEECS's: the
deploy recipe, the Raccoon directory adapter and the in-process device
module at `/geecs` (push 1). The decision and its conditions (a)–(f)
are `Planning/geecs_canopy.md` §7; `DEPLOYMENT.md` is the runbook.

## Layout

```
geecs_canopy/
  pin.py         CANOPY_IMAGE (digest) + CANOPY_COMMIT: the one record of the pin
  directory.py   GeecsDirectoryAdapter: Raccoon's three list calls, the grammar,
                 a per-source cache (cache_ttl_s, failures too) under a lock;
                 imports canopy.raccoon.adapters.base lazily
  native.py      NativeSource: native_devices.yaml + panel_devices.yaml joined
                 by device name, validated through GEECS-Schemas, families from
                 geecs_schemas.load_families()
  gateway.py     GatewaySource (TRANSITIONAL): the CA gateway's served set from
                 GeecsDb, credentials from GEECS_CANOPY_DB_URL
  app.py         the api command: self-test, canopy.main:app + /geecs mounted,
                 tagged, Canopy's error handler; the ONE Hare implementation
  canopy_surface.py  every Canopy-internal name used + VERIFIED_IMAGE; self_test
  hare.py        the Hare Protocol (get, describe, put, watch): no Canopy import
  module.py      ModuleSettings (env GEECS_CANOPY_*), Module: devices from
                 data (NativeSource.resolve), the operator policy, the tracker
  freshness.py   Tracker (internal broker client) and the verdicts
  gates.py       put_role: policy, allow, hazard, run lock, bounds, Hare, journal
  journal.py     SQLite of GEECS-Schemas JournalEntry rows
  routes.py      /v1/devices, /v1/devices/{name}, .../put, /v1/journal
deploy/
  Dockerfile (+ .dockerignore)   FROM the pin; pip --python into /app/.venv
  compose.yaml.in   api + 2 workers + Postgres 16 + Redis 7, host networking
  geecs-canopy.service   unit TEMPLATE (rendered by deploy/render_units.sh)
  render_conf.sh    fills compose.yaml.in from site.env and pin.py
  secrets.env.example   the secrets file's keys, placeholders only
```

## Rules that hold here

- **From Canopy outward.** Raccoon's shapes, Hare's protocol and Squirrel's
  model are Canopy's. Where they lack something, the gap is an upstream
  candidate in the decision page (§7.4), never a bend of this package
  toward the parked bridge's catalog. Canopy is the host process, never a
  dependency: type against local Protocols and import Canopy lazily, so the
  package installs and tests without it.
- **Condition (a): no operator or front-end key ever carries Canopy
  `write`.** A key's `write` is key-wide and reaches Hare's PUT and
  Squirrel's restore directly, around every module guard. Operators and
  pages get `read` keys; `write` and `admin` are service and ops keys only.
  Squirrel is storage and compare, never the executor of an operator's
  restore.
- **One identity: the GEECS device name.** It is the Raccoon device name,
  the journal's key, a Squirrel label and a screen binding. A PV prefix is
  derived by the naming rule (`geecs_core.pv_naming.pv_name(experiment,
  device) + ":"`), never stored a second time; `panel_devices.yaml`'s
  `prefix` is for a bench device outside the registry only, and the adapter
  refuses it on a registry device. DB device names do not change while
  LabVIEW runs them; a rule for renaming native devices is the maintainer's
  to choose before the first production registration.
- **Two files, one directory.** `native_devices.yaml` (what the worker
  binds) and `panel_devices.yaml` (panel data) stay two documents, joined
  by device name here; a device in both names one family in both, which
  only a reader of both can enforce (`native.py`). Folding them is a push-2
  decision.
- **The GEECS DB is transitional.** It enters Canopy in `gateway.py` alone
  (`elementType=gateway`), with the gateway's own read-only queries and a
  read-only MySQL user from the secrets file, never a `config.ini`. It is
  deleted with the CA gateway when the last LabVIEW device retires. No
  other module imports `geecs_core.db.geecs_db` (the
  `canopy-db-is-transitional` import contract); the native path depends on
  nothing in the DB.
- **Routes and gates import no Canopy name** (condition (b)): `app.py`
  alone does, through `canopy_surface.NAMES`; a new Canopy name goes into
  that list in the same change. Every route needs Canopy read access only;
  the module's own policy decides writes. Every write, a push-2 restore's
  rows included, goes through `gates.put_role` and leaves one outcome row in the
  journal, after an `attempted` row when it reached Hare (paired by
  `attempt`); no put goes out without its attempted row.
  Verdicts follow `Planning/operator_surfaces.md` §3 and Appendix A of
  `Planning/epics_estate.md` (rules 1, 6, 8, 9, 10, 11, 18), each pinned by
  a test in `tests/test_module.py`.
- **The pin is a fleet constant.** `pin.py` is the one record of the image
  digest and commit, bumped only in a reviewed PR after a rehearsal on the
  sim (DEPLOYMENT.md). The startup self-test and `canopy_surface.py`
  (every Canopy-internal import, condition (d)) read it; never `latest`.
- **Facility values have one home.** Non-secret values are `site.env` keys
  (`GEECS_CANOPY_BIND`, `_PORT`, `_ROOT_PATH`, `_DB`, plus the shared
  `GEECS_EXPERIMENT`, `GEECS_CONFIGS_ROOT`, EPICS and `TZ`); secrets are in
  `/etc/geecs/canopy/secrets.env`. No lab literal outside an example.

## Testing

`poetry run pytest tests -q` runs without Canopy (`tests/conftest.py`
stands in for its adapter base). Canopy's own contract suite runs where
Canopy is importable and `CANOPY_SOURCE` names a clone at the pin;
DEPLOYMENT.md §6 has the command.
